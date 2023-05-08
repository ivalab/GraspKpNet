#!/usr/bin/env python3
"""
This script creates an OpenCV window to draw bounding boxes around objects on a
topic. We publish the bounding boxes ObjectFilterList messages on a topic. This
will allow us to rank grasps on a per-object basis as opposed to a per-scene
basis.

https://docs.opencv.org/4.x/db/d5b/tutorial_py_mouse_handling.html
http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
"""

from argparse import ArgumentParser
from copy import deepcopy
from functools import partial

import cv2 as cv
import numpy as np
import rospy
from cv_bridge import CvBridge
from gknet_msgs.msg import ObjectFilter, ObjectFilterList
from sensor_msgs.msg import Image


class State:
    def __init__(self):
        self.img = None
        self.bbox = None
        self.bbox_callback = []

        # opencv state
        self._reset_drawing_state()

        # boxes that have already been drawn
        self.bboxes = []

    def _reset_drawing_state(self):
        self.drawing = False
        self.initial_point = (-1, -1)
        self.current_point = (-1, -1)

    def register_bbox_callback(self, callback):
        """Register a callback that is called whenever a bounding box is drawn
        or modified.

        The callback should take a single argument, which is a list of bounding
        boxes with two 2-tuples corresponding to the top left and bottom right
        corners in pixel coordinates.
        """
        self.bbox_callback.append(callback)

    def draw_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_RBUTTONUP and not self.drawing:
            # right click to delete a box
            self.bboxes = [
                bbox
                for bbox in self.bboxes
                if not (bbox[0][0] < x < bbox[1][0] and bbox[0][1] < y < bbox[1][1])
            ]
        if event == cv.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.initial_point = (x, y)
        elif event == cv.EVENT_MOUSEMOVE and self.drawing:
            self.current_point = (x, y)
        elif event == cv.EVENT_LBUTTONUP:
            # minimum size of 10x10 pixels
            if (
                abs(x - self.initial_point[0]) > 10
                and abs(y - self.initial_point[1]) > 10
            ):
                self.bboxes.append([deepcopy(self.initial_point), (x, y)])
            self._reset_drawing_state()

    def draw_bboxes(self, img):
        for bbox in self.bboxes:
            # blue for already drawn boxes
            cv.rectangle(img, bbox[0], bbox[1], (255, 0, 0), 3)
        if (
            self.drawing
            and self.initial_point != (-1, -1)
            and self.current_point != (-1, -1)
        ):
            # green for current box
            cv.rectangle(img, self.initial_point, self.current_point, (0, 255, 0), 3)

    def draw_helptext(self, img):
        """Draws text at the bottom of the screen as help/usage.

        Left click drag to draw filter areas.
        Right click to delete filter areas.
        Q/Esc to quit.
        """
        helptext = [
            "Left click drag to draw filter areas.",
            "Right click to delete filter areas.",
            "Q/Esc to quit.",
        ]
        for i, line in enumerate(reversed(helptext)):
            cv.putText(
                img,
                line,
                (10, img.shape[0] - 10 - i * 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    def draw_loop(self, title="image"):
        # https://stackoverflow.com/questions/56623487/why-does-a-right-click-open-a-drop-down-menu-in-my-opencv-imshow-window
        cv.namedWindow(
            title, flags=cv.WINDOW_AUTOSIZE | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL
        )
        cv.setMouseCallback(title, partial(self.draw_callback))
        while True:
            annotated_img = deepcopy(self.img)
            self.draw_bboxes(annotated_img)
            for cb in self.bbox_callback:
                cb(self.bboxes)
            self.draw_helptext(annotated_img)

            cv.imshow(title, annotated_img)
            # quit if escape key or q is pressed, or if the exit button is pressed
            if (
                cv.waitKey(20) & 0xFF in [27, ord("q")]
                or cv.getWindowProperty(title, cv.WND_PROP_VISIBLE) < 1
            ):
                break
        cv.destroyAllWindows()


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--color-image-topic", type=str, default="/camera/color/image_raw"
    )
    parser.add_argument(
        "--object-filter-topic", type=str, default="/gknet/object_filter"
    )
    parser.add_argument("--spin-when-done", action="store_true")
    # ignore any other args
    args, _ = parser.parse_known_args()
    return args


def color_image_callback(state, cv_bridge, msg):
    state.img = cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


def publish_object_filter_callback(pub, bboxes):
    msg = ObjectFilterList()
    for bbox in bboxes:
        # our message translates the pair of points into a flat list
        msg.objects.append(ObjectFilter(bbox=bbox[0] + bbox[1]))

    msg.header.stamp = rospy.Time.now()
    pub.publish(msg)


def main():
    args = parse_args()
    print("starting filter gui with args: ", args)

    rospy.init_node("filter_gui", anonymous=True, log_level=rospy.INFO)
    cv_bridge = CvBridge()
    state = State()

    rospy.Subscriber(
        args.color_image_topic,
        Image,
        partial(color_image_callback, state, cv_bridge),
    )

    pub = rospy.Publisher(
        args.object_filter_topic, ObjectFilterList, queue_size=1, latch=True
    )
    state.register_bbox_callback(partial(publish_object_filter_callback, pub))

    # wait for the first message to arrive
    print(f"waiting for message on {args.color_image_topic}")
    while state.img is None:
        rospy.sleep(0.1)
    print(f"got our first image with shape {state.img.shape}")

    state.draw_loop(title=args.color_image_topic)

    if args.spin_when_done:
        print("closed window, but continuing to spin and publish")
        rospy.spin()


if __name__ == "__main__":
    main()
