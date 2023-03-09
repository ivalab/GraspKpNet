import sys

import cv2
import cv2.aruco as aruco
import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64MultiArray

from gknet.datasets.dataset_factory import dataset_factory
from gknet.detectors.detector_factory import detector_factory
from gknet.opts import opts

# transformation from the robot base to aruco tag
M_BL = np.array(
    [
        [1.0, 0.0, 0.0, 0.30000],
        [0.0, 1.0, 0.0, 0.32000],
        [0.0, 0.0, 1.0, -0.0450],
        [0.0, 0.0, 0.0, 1.00000],
    ]
)

# default transformation from the camera to aruco tag
default_M_CL = np.array(
    [
        [-0.07134498, -0.99639369, 0.0459293, -0.13825178],
        [-0.8045912, 0.03027403, -0.59305689, 0.08434352],
        [0.58952768, -0.07926594, -0.8038495, 0.66103522],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# camera intrinsic matrix of Realsense D435
cameraMatrix = np.array(
    [[607.47165, 0.0, 325.90064], [0.0, 606.30420, 240.91934], [0.0, 0.0, 1.0]]
)

# distortion of Realsense D435
distCoeffs = np.array([0.08847, -0.04283, 0.00134, -0.00102, 0.0])

# initialize GKNet Detector
opt = opts().parse()
Dataset = dataset_factory[opt.dataset]
opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
print(opt)
Detector = detector_factory[opt.task]
detector = Detector(opt)

# Publisher of perception result
pub_res = rospy.Publisher("/result", Float64MultiArray, queue_size=10)


def get_M_CL_info(gray, image_init, visualize=False):
    # parameters
    markerLength_CL = 0.093
    aruco_dict_CL = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    # aruco_dict_CL = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    corners_CL, ids_CL, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict_CL, parameters=parameters
    )

    # for the first frame, it may contain nothing
    if ids_CL is None:
        return default_M_CL, None

    rvec_CL, tvec_CL, _objPoints_CL = aruco.estimatePoseSingleMarkers(
        corners_CL[0], markerLength_CL, cameraMatrix, distCoeffs
    )
    dst_CL, jacobian_CL = cv2.Rodrigues(rvec_CL)
    M_CL = np.zeros((4, 4))
    M_CL[:3, :3] = dst_CL
    M_CL[:3, 3] = tvec_CL
    M_CL[3, :] = np.array([0, 0, 0, 1])

    if visualize:
        # print('aruco is located at mean position (%d, %d)' %(mean_x ,mean_y))
        aruco.drawAxis(
            image_init, cameraMatrix, distCoeffs, rvec_CL, tvec_CL, markerLength_CL
        )
    return M_CL, corners_CL[0][0, :, :]


def aruco_tag_remove(rgb_image, corners):
    img_out = rgb_image.copy()

    # find the top-left and right-bottom corners
    min = sys.maxsize
    max = -sys.maxsize
    tl_pxl = None
    br_pxl = None
    for corner in corners:
        if corner[0] + corner[1] < min:
            min = corner[0] + corner[1]
            tl_pxl = [int(corner[0]), int(corner[1])]

        if corner[0] + corner[1] > max:
            max = corner[0] + corner[1]
            br_pxl = [int(corner[0]), int(corner[1])]

    # get the replacement pixel value
    rep_color = img_out[tl_pxl[0] - 10, tl_pxl[1] - 10, :]

    for h in range(tl_pxl[1] - 45, br_pxl[1] + 46):
        for w in range(tl_pxl[0] - 45, br_pxl[0] + 46):
            img_out[h, w, :] = rep_color

    return img_out


def project(pixel, depth_image, M_CL, M_BL, cameraMatrix):
    """
    project 2d pixel on the image to 3d by depth info
    :param pixel: x, y
    :param M_CL: trans from camera to aruco tag
    :param cameraMatrix: camera intrinsic matrix
    :param depth_image: depth image
    :param depth_scale: depth scale that trans raw data to mm
    :return:
    q_B: 3d coordinate of pixel with respect to base frame
    """
    depth = depth_image[pixel[1], pixel[0]]

    # if the depth of the detected pixel is 0, check the depth of its neighbors
    # by counter-clock wise
    nei_range = 1
    while depth == 0:
        for delta_x in range(-nei_range, nei_range + 1):
            for delta_y in range(-nei_range, nei_range + 1):
                nei = [pixel[0] + delta_x, pixel[1] + delta_y]
                depth = depth_image[nei[1], nei[0]]

                if depth != 0:
                    break

            if depth != 0:
                break

        nei_range += 1

    pxl = np.linalg.inv(cameraMatrix).dot(
        np.array([pixel[0] * depth, pixel[1] * depth, depth])
    )
    q_C = np.array([pxl[0], pxl[1], pxl[2], 1])
    q_L = np.linalg.inv(M_CL).dot(q_C)
    q_B = M_BL.dot(q_L)

    return q_B


def pre_process(rgb_img, depth_img):
    inp_image = rgb_img
    inp_image[:, :, 0] = depth_img

    inp_image = cv2.resize(inp_image, (256, 256))

    return inp_image


def kinect_rgbd_callback(rgb_data, depth_data):
    """
    Save raw RGB and depth input from Kinect V1
    :param rgb_data: RGB image
    :param depth_data: raw depth image
    :return: None
    """
    try:
        cv_rgb = cv_bridge.imgmsg_to_cv2(rgb_data, "bgr8")
        cv_depth = cv_bridge.imgmsg_to_cv2(depth_data, "32FC1")

        cv_rgb_arr = np.array(cv_rgb, dtype=np.uint8)
        cv_depth_arr = np.array(cv_depth, dtype=np.float32)
        # cv_depth_arr = np.nan_to_num(cv_depth_arr)

        cv2.imshow("Depth", cv_depth)
        cv2.imshow("RGB", cv_rgb)

        img = cv_rgb_arr.copy()
        depth_raw = cv_depth_arr.copy()

        gray = img.astype(np.uint8)
        depth = (depth_raw * 1000).astype(np.uint8)

        # get the current transformation from the camera to aruco tag
        M_CL, corners = get_M_CL_info(gray, img, False)

        # remove aruco tag from input image to avoid mis-detection
        if corners is not None:
            img_wo_at = aruco_tag_remove(img, corners)

        # replace blue channel with the depth channel
        inp_image = pre_process(img_wo_at, depth)

        # pass the image into the network
        ret = detector.run(inp_image[:, :, :])
        ret = ret["results"]

        loc_ori = KpsToGrasppose(ret, img, depth_raw, M_CL, M_BL, cameraMatrix)
        pub_res.publish(loc_ori)

    except CvBridgeError as e:
        print(e)


def isWithinRange(pxl, w, h):
    x, y = pxl[:]

    return w / 12.0 <= x <= 11 * w / 12 and h / 12.0 <= y <= 11 * h / 12


def KpsToGrasppose(
    net_output, rgb_img, depth_map, M_CL, M_BL, cameraMatrix, visualize=True
):
    kps_pr = []
    for category_id, preds in net_output.items():
        if len(preds) == 0:
            continue

        for pred in preds:
            kps = pred[:4]
            score = pred[-1]
            kps_pr.append([kps[0], kps[1], kps[2], kps[3], score])

    # no detection
    if len(kps_pr) == 0:
        return [0, 0, 0, 0]

    # sort by the confidence score
    kps_pr = sorted(kps_pr, key=lambda x: x[-1], reverse=True)
    # select the top 1 grasp prediction within the workspace
    res = None
    for kp_pr in kps_pr:
        f_w, f_h = 640.0 / 512.0, 480.0 / 512.0
        kp_lm = (int(kp_pr[0] * f_w), int(kp_pr[1] * f_h))
        kp_rm = (int(kp_pr[2] * f_w), int(kp_pr[3] * f_h))

        if isWithinRange(kp_lm, 640, 480) and isWithinRange(kp_rm, 640, 480):
            res = kp_pr
            break

    if res is None:
        return [0, 0, 0, 0]

    f_w, f_h = 640.0 / 512.0, 480.0 / 512.0
    kp_lm = (int(res[0] * f_w), int(res[1] * f_h))
    kp_rm = (int(res[2] * f_w), int(res[3] * f_h))
    center = (int((kp_lm[0] + kp_rm[0]) / 2), int((kp_lm[1] + kp_rm[1]) / 2))

    kp_lm_3d = project(kp_lm, depth_map, M_CL, M_BL, cameraMatrix)
    kp_rm_3d = project(kp_rm, depth_map, M_CL, M_BL, cameraMatrix)
    center_3d = project(center, depth_map, M_CL, M_BL, cameraMatrix)

    orientation = np.arctan2(kp_rm_3d[1] - kp_lm_3d[1], kp_rm_3d[0] - kp_lm_3d[0])
    # motor 7 is clockwise
    if orientation > np.pi / 2:
        orientation = np.pi - orientation
    elif orientation < -np.pi / 2:
        orientation = -np.pi - orientation
    else:
        orientation = -orientation

    # compute the open width
    dist = np.linalg.norm(kp_lm_3d[:2] - kp_rm_3d[:2])

    # draw arrow for left-middle and right-middle key-points
    lm_ep = (
        int(kp_lm[0] + (kp_rm[0] - kp_lm[0]) / 5.0),
        int(kp_lm[1] + (kp_rm[1] - kp_lm[1]) / 5.0),
    )
    rm_ep = (
        int(kp_rm[0] + (kp_lm[0] - kp_rm[0]) / 5.0),
        int(kp_rm[1] + (kp_lm[1] - kp_rm[1]) / 5.0),
    )
    rgb_img = cv2.arrowedLine(rgb_img, kp_lm, lm_ep, (0, 0, 0), 2)
    rgb_img = cv2.arrowedLine(rgb_img, kp_rm, rm_ep, (0, 0, 0), 2)
    # draw left-middle, right-middle and center key-points
    rgb_img = cv2.circle(rgb_img, (int(kp_lm[0]), int(kp_lm[1])), 2, (0, 0, 255), 2)
    rgb_img = cv2.circle(rgb_img, (int(kp_rm[0]), int(kp_rm[1])), 2, (0, 0, 255), 2)
    rgb_img = cv2.circle(rgb_img, (int(center[0]), int(center[1])), 2, (0, 0, 255), 2)

    if visualize:
        cv2.namedWindow("visual", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("visual", rgb_img)

    return [center_3d[0], center_3d[1], center_3d[2], orientation, dist]


if __name__ == "__main__":
    # initialize ros node
    rospy.init_node("Static_grasping")

    # Bridge to convert ROS Image type to OpenCV Image type
    cv_bridge = CvBridge()
    cv2.WITH_QT = False
    # Get camera calibration parameters
    cam_param = rospy.wait_for_message(
        "/camera/rgb/camera_info", CameraInfo, timeout=None
    )

    # Subscribe to rgb and depth channel
    image_sub = message_filters.Subscriber("/camera/rgb/image_rect_color", Image)
    depth_sub = message_filters.Subscriber("/camera/depth_registered/image", Image)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 1, 0.1)
    ts.registerCallback(kinect_rgbd_callback)

    rospy.spin()
