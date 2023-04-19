import numpy as np
import pytest
import rospy
from cv_bridge import CvBridge
from gknet_msgs.msg import KeypointList
from sensor_msgs.msg import Image


@pytest.fixture(autouse=True)
def ros_node():
    rospy.init_node(__name__, anonymous=True)


@pytest.fixture()
def bridge():
    return CvBridge()


def test_static_image_publisher():
    msg = rospy.wait_for_message("/camera/color/image_raw", Image)
    assert msg, "no message received"


def test_gknet_keypoints():
    msg = rospy.wait_for_message("/gknet/keypoints", KeypointList)
    assert msg.keypoints, "no keypoints received"
    # assert we have 10 keypoints in descending order of score
    assert len(msg.keypoints) == 10, "wrong number of keypoints"
    assert all(
        [msg.keypoints[i].score >= msg.keypoints[i + 1].score for i in range(9)]
    ), "keypoints not in descending order of score"
    print(msg.keypoints)


def test_gknet_annotated_image(bridge):
    msg = rospy.wait_for_message("/gknet/annotated_image", Image)
    assert msg, "no message received"
