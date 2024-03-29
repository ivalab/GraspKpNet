#!/usr/bin/env python3
"""
This script will read aligned color and depth images from a directory and create
a latched topic for the images to simulate a camera. This script is primarily
for testing detector behavior.
"""
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--color-image-filename", type=str, default="rgb.jpg")
    parser.add_argument("--depth-image-filename", type=str, default="depth.jpg")
    parser.add_argument(
        "--color-image-topic", type=str, default="/camera/color/image_raw"
    )
    parser.add_argument(
        "--depth-image-topic",
        type=str,
        default="/camera/aligned_depth_to_color/image_raw",
    )
    parser.add_argument("--rate", type=float, default=60, help="publish rate in Hz")
    # ignore any other args
    args, _ = parser.parse_known_args()
    return args


def _normalize(x, min_d, max_min_diff):
    return 255 * (x - min_d) / max_min_diff


def main():
    args = parse_args()

    # load images into memory
    print("starting static image publisher with arguments: ", args)
    image_path = Path(args.image_path)
    color_image = cv2.imread(str(image_path / args.color_image_filename))
    depth_image = cv2.imread(
        str(image_path / args.depth_image_filename), cv2.IMREAD_ANYDEPTH
    )

    min_depth = np.min(depth_image[:, :])
    max_depth = np.max(depth_image[:, :])
    max_min_diff = max_depth - min_depth
    normalize = np.vectorize(_normalize, otypes=[np.float32])
    depth_image = normalize(depth_image, min_depth, max_min_diff)

    rospy.init_node("static_image_topic", anonymous=True)
    cv_bridge = CvBridge()
    color_pub = rospy.Publisher(args.color_image_topic, Image, queue_size=1)
    depth_pub = rospy.Publisher(args.depth_image_topic, Image, queue_size=1)

    # publish in a loop
    rate = rospy.Rate(args.rate)
    while not rospy.is_shutdown():
        rate.sleep()
        msg = cv_bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
        msg.header.stamp = rospy.Time.now()
        color_pub.publish(msg)
        msg = cv_bridge.cv2_to_imgmsg(depth_image, encoding="32FC1")
        msg.header.stamp = rospy.Time.now()
        depth_pub.publish(msg)


if __name__ == "__main__":
    main()
