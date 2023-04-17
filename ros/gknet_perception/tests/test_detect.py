import numpy as np
import pytest
import rospy
from cv_bridge import CvBridge
from gknet_msgs.msg import KeypointList, ObjectFilter, ObjectFilterList
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
    # assert we have 3 keypoints in descending order of score
    assert len(msg.keypoints) == 3, "wrong number of keypoints"
    assert all(
        [msg.keypoints[i].score >= msg.keypoints[i + 1].score for i in range(2)]
    ), "keypoints not in descending order of score"
    print(msg.keypoints)


def test_gknet_annotated_image():
    msg = rospy.wait_for_message("/gknet/annotated_image", Image)
    assert msg, "no message received"


def test_gknet_object_filter_topic_empty():
    msg = rospy.wait_for_message("/gknet/object_filter", ObjectFilterList)
    assert not msg.objects, "object filter should be empty"


@pytest.fixture()
def object_filter_pub():
    return rospy.Publisher(
        "/gknet/object_filter", ObjectFilterList, queue_size=1, latch=True
    )


@pytest.mark.parametrize("num_corners", [1, 2, 3])
def test_gknet_keypoints_with_object_filter(object_filter_pub, num_corners):
    # 3 seconds worth of retries before failing
    num_retries = 30
    # see assets/tabletop_01/test_detect_bbox.png for what this should look
    # like. we purposely overlap corners to get the correct number of detected
    # keypoints.
    corners = [
        [156, 184, 306, 312],
        [132, 257, 369, 364],
        [330, 102, 562, 302],
    ]
    of = ObjectFilterList()
    of.objects = [ObjectFilter(bbox=corners[i]) for i in range(num_corners)]
    object_filter_pub.publish(of)

    for _ in range(num_retries):
        msg = rospy.wait_for_message("/gknet/object_filter", ObjectFilterList)
        if len(msg.objects) == num_corners:
            break
        rospy.sleep(0.1)
    assert len(msg.objects) == num_corners, "wrong number of objects"

    # top 3 keypoints from each object
    for _ in range(num_retries):
        msg = rospy.wait_for_message("/gknet/keypoints", KeypointList)
        if len(msg.keypoints) == num_corners * 3:
            break
        rospy.sleep(0.1)
    assert len(msg.keypoints) == num_corners * 3, "wrong number of keypoints"
