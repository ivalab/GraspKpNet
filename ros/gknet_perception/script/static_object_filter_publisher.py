#!/usr/bin/env python3
"""
Publish an empty object filter list, primarily for testing.
"""
from argparse import ArgumentParser

import rospy
from gknet_msgs.msg import ObjectFilterList


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object-filter-topic", type=str, default="/gknet/object_filter"
    )
    parser.add_argument("--rate", type=float, default=60, help="publish rate in Hz")
    # ignore any other args
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    print("starting static object filter publisher with arguments: ", args)
    rospy.init_node("static_object_filter_publisher", anonymous=True)
    pub = rospy.Publisher(args.object_filter_topic, ObjectFilterList, queue_size=1)
    msg = ObjectFilterList(objects=[])
    while not rospy.is_shutdown():
        # check if there are other publishers
        try:
            rospy.wait_for_message(
                args.object_filter_topic, ObjectFilterList, timeout=1.0 / args.rate
            )
        except rospy.ROSException:
            msg.header.stamp = rospy.Time.now()
            pub.publish(msg)


if __name__ == "__main__":
    main()
