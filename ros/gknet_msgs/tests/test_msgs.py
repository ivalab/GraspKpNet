import numpy as np
from gknet_msgs.msg import Keypoint, KeypointList


def test_keypoint():
    lm = np.array([1, 2])
    rm = np.array([2, 3])
    center = (lm + rm) / 2
    kp = Keypoint(
        left_middle=lm.tolist(),
        right_middle=rm.tolist(),
        center=center.tolist(),
        score=1,
    )
    assert kp.left_middle == lm.tolist()
    assert kp.right_middle == rm.tolist()
    assert kp.center == center.tolist()
    assert kp.score == 1


def test_keypoint_list():
    lm = np.array([1, 2])
    rm = np.array([2, 3])
    center = (lm + rm) / 2
    kp = Keypoint(
        left_middle=lm.tolist(),
        right_middle=rm.tolist(),
        center=center.tolist(),
        score=1,
    )
    kpl = KeypointList(keypoints=[kp])
    assert kpl.keypoints[0].left_middle == lm.tolist()
    assert kpl.keypoints[0].right_middle == rm.tolist()
    assert kpl.keypoints[0].center == center.tolist()
    assert kpl.keypoints[0].score == 1
