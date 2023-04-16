#!/usr/bin/env python3
"""
This script will run the detector on a stream of images from a camera.

We do not perform any post-processing on the detected keypoints, so it is up to
the consumer of the module to transform keypoints into the world frame relative
to the camera and manipulator.
"""
from argparse import ArgumentParser
from functools import partial

import cv2
import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge
from gknet_msgs.msg import Keypoint, KeypointList, ObjectFilter, ObjectFilterList
from sensor_msgs.msg import Image

from gknet.detectors.detector_factory import detector_factory
from gknet.opts import opts

IMG_LEN = 512


def preprocess(rgb_img, depth_img):
    """Merge rgb and depth images into a single image, and resize it."""
    img = rgb_img.copy()
    img[:, :, 0] = depth_img.copy()
    return cv2.resize(img, (IMG_LEN, IMG_LEN))


def intersects(rect, point):
    """Determine if a point is in a rectangle.

    Rect is an array representing the top-left and bottom-right corners of the
    rectangle ([xtl, ytl, xbr, ybr]).
    """
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def postprocess(detector_output, shape, object_filter_list, num_keypoints):
    """Convert the detector output into a KeypointList message."""
    kps_pr = []
    for _, preds in detector_output.items():
        if len(preds) == 0:
            continue
        for pred in preds:
            kps = pred[:4]
            score = pred[-1]
            kps_pr.append([kps[0], kps[1], kps[2], kps[3], score])

    # no detection
    if len(kps_pr) == 0:
        return KeypointList(keypoints=[])

    if not object_filter_list.objects:
        # add a simulated filter that extends the entire image
        object_filter_list.objects.append(ObjectFilter(bbox=[0, 0, shape[0], shape[1]]))

    # keep track of which points belong to which filter
    kp_per_filer = [[] for _ in range(len(object_filter_list.objects))]
    kps_pr = sorted(kps_pr, key=lambda x: x[-1], reverse=True)
    for kp_pr in kps_pr:
        # NOTE: wonder what this transformation actually does...
        f_w, f_h = shape[1] / IMG_LEN, shape[0] / IMG_LEN
        kp_lm = [int(kp_pr[0] * f_w), int(kp_pr[1] * f_h)]
        kp_rm = [int(kp_pr[2] * f_w), int(kp_pr[3] * f_h)]
        center = [int((kp_lm[0] + kp_rm[0]) / 2), int((kp_lm[1] + kp_rm[1]) / 2)]
        kp_msg = Keypoint(
            left_middle=kp_lm, right_middle=kp_rm, center=center, score=kp_pr[-1]
        )
        # check for intersection with each filter
        for i, obj_filter in enumerate(object_filter_list.objects):
            if len(kp_per_filer[i]) >= num_keypoints:
                continue
            if intersects(obj_filter.bbox, center):
                kp_per_filer[i].append(kp_msg)
        # break early if we have enough keypoints
        if all(len(kps) >= num_keypoints for kps in kp_per_filer):
            break

    # merge keypoints
    kps_msg = []
    for kps in kp_per_filer:
        kps_msg.extend(kps)

    return KeypointList(keypoints=kps_msg)


def detect_callback(
    detector,
    cv_bridge,
    keypoint_pub,
    rgb_msg,
    depth_msg,
    object_filter_msg,
    num_keypoints=1,
):
    rgb_img = np.array(
        cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8"), dtype=np.uint8
    )
    depth_img = np.array(
        cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1"), dtype=np.float32
    )

    # replace blue channel with the depth channel
    img = preprocess(rgb_img, (depth_img * 1000.0).astype(np.uint8))
    detector_results = detector.run(img[:, :, :])["results"]
    keypoint_list = postprocess(
        detector_results, rgb_img.shape, object_filter_msg, num_keypoints
    )

    # publish information
    keypoint_pub.publish(keypoint_list)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--color-image-topic", type=str, default="/camera/color/image_raw"
    )
    parser.add_argument(
        "--depth-image-topic",
        type=str,
        default="/camera/aligned_depth_to_color/image_raw",
    )
    parser.add_argument("--keypoints-topic", type=str, default="/gknet/keypoints")
    parser.add_argument(
        "--annotated-image-topic", type=str, default="/gknet/annotated_image"
    )
    parser.add_argument(
        "--object-filter-topic", type=str, default="/gknet/object_filter"
    )
    parser.add_argument("--num-keypoints", type=int, default=5)
    parser.add_argument("--model", type=str, default="dbmctdet_cornell")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/opt/models/model_dla34_cornell.pth",
    )
    parser.add_argument("--publisher-queue-size", type=int, default=1)
    parser.add_argument("--subscriber-queue-size", type=int, default=10)
    # ignore any other args
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    print("starting detector with args: ", args)
    # also parse opts
    opt = opts().init(args=f"{args.model} " f"--load_model {args.checkpoint}".split())
    detector = detector_factory[opt.task](opt)

    rospy.init_node("detect", anonymous=True, log_level=rospy.INFO)
    cv_bridge = CvBridge()

    # let's wait for the first message to arrive
    print(f"waiting for message on {args.color_image_topic}")
    rospy.wait_for_message(args.color_image_topic, Image)
    print(f"waiting for message on {args.depth_image_topic}")
    rospy.wait_for_message(args.depth_image_topic, Image)
    print(f"waiting for message on {args.object_filter_topic} for 1 second")
    try:
        rospy.wait_for_message(args.object_filter_topic, ObjectFilterList, timeout=1)
    except rospy.ROSException:
        print("no object filter received, using default")
        msg = ObjectFilterList(objects=[])
        object_filter_pub = rospy.Publisher(
            args.object_filter_topic, ObjectFilterList, queue_size=1, latch=True
        )
        object_filter_pub.publish(msg)
        print(f"waiting for message on {args.object_filter_topic}")
        rospy.wait_for_message(args.object_filter_topic, ObjectFilterList)
    print("input topics ready for processing")

    keypoint_pub = rospy.Publisher(
        args.keypoints_topic,
        KeypointList,
        queue_size=args.publisher_queue_size,
        latch=True,
    )

    image_sub = message_filters.Subscriber(args.color_image_topic, Image)
    depth_sub = message_filters.Subscriber(args.depth_image_topic, Image)
    object_filter_sub = message_filters.Subscriber(
        args.object_filter_topic, ObjectFilterList
    )
    ts = message_filters.ApproximateTimeSynchronizer(
        [image_sub, depth_sub, object_filter_sub], args.subscriber_queue_size, 0.1
    )
    ts.registerCallback(
        partial(
            detect_callback,
            detector,
            cv_bridge,
            keypoint_pub,
            num_keypoints=args.num_keypoints,
        )
    )
    print(
        f"registered callbacks, publishing to {args.keypoints_topic} and {args.annotated_image_topic}"
    )
    rospy.spin()


if __name__ == "__main__":
    main()
