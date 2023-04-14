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
from gknet_msgs.msg import Keypoint, KeypointList
from sensor_msgs.msg import Image

from gknet.detectors.detector_factory import detector_factory
from gknet.opts import opts

IMG_LEN = 512


def preprocess(rgb_img, depth_img):
    """Merge rgb and depth images into a single image, and resize it."""
    img = rgb_img.copy()
    img[:, :, 0] = depth_img.copy()
    return cv2.resize(img, (IMG_LEN, IMG_LEN))


def postprocess(detector_output, rgb_img, depth_img, num_keypoints):
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

    kps_msg = []
    kps_pr = sorted(kps_pr, key=lambda x: x[-1], reverse=True)
    for kp_pr in kps_pr[:num_keypoints]:
        # NOTE: wonder what this transformation actually does...
        shape = rgb_img.shape
        f_w, f_h = shape[1] / IMG_LEN, shape[0] / IMG_LEN
        kp_lm = [int(kp_pr[0] * f_w), int(kp_pr[1] * f_h)]
        kp_rm = [int(kp_pr[2] * f_w), int(kp_pr[3] * f_h)]
        center = [int((kp_lm[0] + kp_rm[0]) / 2), int((kp_lm[1] + kp_rm[1]) / 2)]
        kp_msg = Keypoint(
            left_middle=kp_lm, right_middle=kp_rm, center=center, score=kp_pr[-1]
        )
        kps_msg.append(kp_msg)

    return KeypointList(keypoints=kps_msg)


def annotate(rgb_img, keypoint_list):
    """Annotate the image with the detected keypoints."""
    img = rgb_img.copy()
    for kp in keypoint_list.keypoints:
        kp_lm = kp.left_middle
        kp_rm = kp.right_middle
        center = kp.center
        # draw arrow for left-middle and right-middle key-points
        lm_ep = (
            int(kp_lm[0] + (kp_rm[0] - kp_lm[0]) / 5.0),
            int(kp_lm[1] + (kp_rm[1] - kp_lm[1]) / 5.0),
        )
        rm_ep = (
            int(kp_rm[0] + (kp_lm[0] - kp_rm[0]) / 5.0),
            int(kp_rm[1] + (kp_lm[1] - kp_rm[1]) / 5.0),
        )
        img = cv2.arrowedLine(img, kp_lm, lm_ep, (0, 0, 0), 2)
        img = cv2.arrowedLine(img, kp_rm, rm_ep, (0, 0, 0), 2)
        # draw left-middle, right-middle and center key-points
        img = cv2.circle(img, (int(kp_lm[0]), int(kp_lm[1])), 2, (0, 0, 255), 2)
        img = cv2.circle(img, (int(kp_rm[0]), int(kp_rm[1])), 2, (0, 0, 255), 2)
        img = cv2.circle(img, (int(center[0]), int(center[1])), 2, (0, 0, 255), 2)
    return img


def detect_callback(
    detector,
    cv_bridge,
    keypoint_pub,
    annotated_image_pub,
    rgb_msg,
    depth_msg,
    num_keypoints=5,
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
    keypoint_list = postprocess(detector_results, rgb_img, depth_img, num_keypoints)
    annotated_img = annotate(rgb_img, keypoint_list)

    # publish information
    keypoint_pub.publish(keypoint_list)
    annotated_image_pub.publish(cv_bridge.cv2_to_imgmsg(annotated_img, encoding="bgr8"))


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
    parser.add_argument("--num-keypoints", type=int, default=5)
    parser.add_argument("--model", type=str, default="dbmctdet_cornell")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/opt/models/model_dla34_cornell.pth",
    )
    parser.add_argument("--publisher-queue-size", type=int, default=10)
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
    print("waiting for first message...")
    rospy.wait_for_message(args.color_image_topic, Image)
    rospy.wait_for_message(args.depth_image_topic, Image)
    print("first message arrived")

    keypoint_pub = rospy.Publisher(
        args.keypoints_topic,
        KeypointList,
        queue_size=args.publisher_queue_size,
        latch=True,
    )
    annotated_image_pub = rospy.Publisher(
        args.annotated_image_topic,
        Image,
        queue_size=args.publisher_queue_size,
        latch=True,
    )

    image_sub = message_filters.Subscriber(args.color_image_topic, Image)
    depth_sub = message_filters.Subscriber(args.depth_image_topic, Image)
    ts = message_filters.ApproximateTimeSynchronizer(
        [image_sub, depth_sub], args.subscriber_queue_size, 0.1
    )
    ts.registerCallback(
        partial(
            detect_callback,
            detector,
            cv_bridge,
            keypoint_pub,
            annotated_image_pub,
            num_keypoints=args.num_keypoints,
        )
    )
    print("registered callbacks, waiting for messages...")
    rospy.spin()


if __name__ == "__main__":
    main()
