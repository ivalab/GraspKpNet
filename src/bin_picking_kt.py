from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import cv2.aruco as aruco
import numpy as np
import sys

import rospy
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters

import torch

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
from utils.kinect_depth_smoother import HoleFilling_Filter, Denoising_Filter

# transformation from the robot base to aruco tag
M_BL = np.array([[1., 0., 0.,  0.30000],
                 [0., 1., 0.,  0.32000],
                 [0., 0., 1.,  -0.0450],
                 [0., 0., 0.,  1.00000]])

# default transformation from the camera to aruco tag
default_M_CL = np.array([[-0.07134498, -0.99639369,  0.0459293,  -0.13825178],
                         [-0.8045912,   0.03027403, -0.59305689,  0.08434352],
                         [ 0.58952768, -0.07926594, -0.8038495,   0.66103522],
                         [ 0.,          0.,          0.,          1.        ]]
                        )

# camera intrinsic matrix of Realsense D435
cameraMatrix = np.array([[607.47165, 0.0,  325.90064],
                         [0.0, 606.30420, 240.91934],
                         [0.0, 0.0, 1.0]])

# distortion of Realsense D435
distCoeffs = np.array([0.08847, -0.04283, 0.00134, -0.00102, 0.0])

# gripper parameters
grip_bbx_w = 0.015
grip_bbx_h = 0.03

### experimental setup
# height of the table
z_table = -0.045

# top-left of the pick bin
pb_tl = [120, 40]
# bottom-right of the pick bin
pb_br = [600, 380]

# initialize GKNet Detector
opt = opts().parse()
Dataset = dataset_factory[opt.dataset]
opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
print(opt)
Detector = detector_factory[opt.task]
detector = Detector(opt)

# Publisher of perception result
pub_res = rospy.Publisher('/result', Float64MultiArray, queue_size=10)
pub_end = rospy.Publisher('/clear', Bool, queue_size=10)

# initialize smoother for kinect depth image
hole_filter = HoleFilling_Filter(flag='mean')
noise_filter = Denoising_Filter(flag='anisotropic', theta=60)

def get_M_CL(gray, image_init, visualize=False):
    # parameters
    markerLength_CL = 0.093
    aruco_dict_CL = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    parameters = aruco.DetectorParameters_create()

    corners_CL, ids_CL, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict_CL, parameters=parameters)

    # for the first frame, it may contain nothing
    if ids_CL is None:
        return default_M_CL

    rvec_CL, tvec_CL, _objPoints_CL = aruco.estimatePoseSingleMarkers(corners_CL[0], markerLength_CL,
                                                                      cameraMatrix, distCoeffs)
    dst_CL, jacobian_CL = cv2.Rodrigues(rvec_CL)
    M_CL = np.zeros((4, 4))
    M_CL[:3, :3] = dst_CL
    M_CL[:3, 3] = tvec_CL
    M_CL[3, :] = np.array([0, 0, 0, 1])

    if visualize:
        # print('aruco is located at mean position (%d, %d)' %(mean_x ,mean_y))
        aruco.drawAxis(image_init, cameraMatrix, distCoeffs, rvec_CL, tvec_CL, markerLength_CL)
    return M_CL

def project_2d_3d(pixel, depth_image, M_CL):
    '''
     project 2d pixel on the image to 3d by depth info
     :param pixel: x, y
     :param M_CL: trans from camera to aruco tag
     :param cameraMatrix: camera intrinsic matrix
     :param depth_image: depth image
     :param depth_scale: depth scale that trans raw data to mm
     :return:
     q_B: 3d coordinate of pixel with respect to base frame
     '''
    depth = depth_image[int(pixel[1]), int(pixel[0])]

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
        np.array([pixel[0] * depth, pixel[1] * depth, depth]))
    q_C = np.array([pxl[0], pxl[1], pxl[2], 1])
    q_L = np.linalg.inv(M_CL).dot(q_C)
    q_B = M_BL.dot(q_L)

    return q_B

def project_3d_2d(point, M_CL):
    q_B = np.array([point[0], point[1], point[2], 1])
    q_L = np.linalg.inv(M_BL).dot(q_B)
    q_C = M_CL.dot(q_L)

    pxl = cameraMatrix.dot(q_C[:-1])
    pxl = [int(pxl[0]/pxl[-1]), int(pxl[1]/pxl[-1]), int(pxl[-1]/pxl[-1])]

    return pxl[:2]

def compute_collision_score(p1, p2, rot_max, center, depth_map, center_height, M_CL):
    count = 0
    score = 0
    for i in range(int(p1[0]), int(p2[0])+1):
        for j in range(int(p1[1]), int(p2[1])+1):
            count += 1

            # apply rotation
            p = [i, j]
            p -= center
            p = rot_max.dot([p[0], p[1], 1])[:2]
            p += center

            p_3d = project_2d_3d(p, depth_map, M_CL)
            score += np.heaviside(center_height - p_3d[2], 1)

    return count, score

def compute_occupancy_score(p1, p2, rot_max, center, depth_map, M_CL):
    count = 0
    score = 0
    for i in range(int(p1[0]), int(p2[0]) + 1):
        for j in range(int(p1[1]), int(p2[1]) + 1):
            count += 1

            # apply rotation
            p = [i, j]
            p -= center
            p = rot_max.dot([p[0], p[1], 1])[:2]
            p += center

            p_3d = project_2d_3d(p, depth_map, M_CL)
            score += np.heaviside(p_3d[2] - z_table, 0)

    score /= count
    return score

def compute_grasp_height(center_height):
    return np.abs(center_height - z_table) / np.abs(z_table)

def scoring(results, rgb_img, depth_map, M_CL):
    res = []
    for result in results:
        # pixel location over the RGB image
        f_w, f_h = (pb_br[0] - pb_tl[0]) / 512., (pb_br[1] - pb_tl[1]) / 512.
        kp_lm_r = np.array([int(result[0] * f_w) + pb_tl[0], int(result[1] * f_h) + pb_tl[1]])
        kp_rm_r = np.array([int(result[2] * f_w) + pb_tl[0], int(result[3] * f_h) + pb_tl[1]])
        center = np.array([int((kp_lm_r[0] + kp_rm_r[0]) / 2), int((kp_lm_r[1] + kp_rm_r[1]) / 2)])
        orientation_2d = -np.arctan2(kp_rm_r[1] - kp_lm_r[1], kp_rm_r[0] - kp_lm_r[0])
        rot_2d = np.array([[np.cos(orientation_2d), np.sin(orientation_2d), 0], [-np.sin(orientation_2d), np.cos(orientation_2d), 0], [0, 0, 1]])

        # project to the 3d location in the real world
        kp_lm_r_3d = project_2d_3d(kp_lm_r, depth_map, M_CL)
        kp_rm_r_3d = project_2d_3d(kp_rm_r, depth_map, M_CL)
        center_3d = project_2d_3d(center, depth_map, M_CL)

        # compute the corner points of bbx of grippers
        orientation = np.arctan2(kp_rm_r_3d[1] - kp_lm_r_3d[1], kp_rm_r_3d[0] - kp_lm_r_3d[0])

        # rotate the grasp bbx to horizontal one
        dst = np.linalg.norm(kp_lm_r_3d[:2]-center_3d[:2])
        kp_lm_3d = center_3d.copy()
        kp_lm_3d[1] += dst
        kp_rm_3d = center_3d.copy()
        kp_rm_3d[1] -= dst

        # compute the two corner points (left-top and right-bottom with respect to the image) for collision bbx
        l_lt_3d = [kp_lm_3d[0] + grip_bbx_h/2, kp_lm_3d[1] + grip_bbx_w/2, kp_lm_3d[2], 1]
        l_rb_3d = [kp_lm_3d[0] - grip_bbx_h/2, kp_lm_3d[1] - grip_bbx_w/2, kp_lm_3d[2], 1]
        r_lt_3d = [kp_rm_3d[0] + grip_bbx_h / 2, kp_rm_3d[1] + grip_bbx_w / 2, kp_rm_3d[2], 1]
        r_rb_3d = [kp_rm_3d[0] - grip_bbx_h / 2, kp_rm_3d[1] - grip_bbx_w / 2, kp_rm_3d[2], 1]

        l_lt = project_3d_2d(l_lt_3d, M_CL)
        l_rb = project_3d_2d(l_rb_3d, M_CL)
        r_lt = project_3d_2d(r_lt_3d, M_CL)
        r_rb = project_3d_2d(r_rb_3d, M_CL)

        # compute collision score
        l_pxl_cnt, l_score = compute_collision_score(l_lt, l_rb, rot_2d, center, depth_map, center_3d[2], M_CL)
        r_pxl_cnt, r_score = compute_collision_score(r_lt, r_rb, rot_2d, center, depth_map, center_3d[2], M_CL)
        c_s = (l_score + r_score) / (l_pxl_cnt + r_pxl_cnt)

        # compute the corner points (top-left and bottom-right) of the occupant bbx
        o_lt_3d = [kp_lm_3d[0] + grip_bbx_h / 2, kp_lm_3d[1], kp_lm_3d[2], 1]
        o_rb_3d = [kp_rm_3d[0] - grip_bbx_h / 2, kp_rm_3d[1], kp_rm_3d[2], 1]

        # project to 2D pixles
        o_lt = project_3d_2d(o_lt_3d, M_CL)
        o_rb = project_3d_2d(o_rb_3d, M_CL)

        # compute occupancy score
        o_s = compute_occupancy_score(o_lt, o_rb,  rot_2d, center, depth_map, M_CL)

        # compute height score
        h_s = compute_grasp_height(center_3d[2])

        # motor 7 is clockwise
        if orientation > np.pi / 2:
            orientation = np.pi - orientation
        elif orientation < -np.pi / 2:
            orientation = -np.pi - orientation
        else:
            orientation = -orientation

        # compute the open width
        dist = np.linalg.norm(kp_lm_3d[:2] - kp_rm_3d[:2])

        res.append([center_3d[0], center_3d[1], center_3d[2], orientation, dist,
                    kp_lm_r[0], kp_lm_r[1], kp_rm_r[0], kp_rm_r[1], c_s+o_s+h_s])

    return res

def KpsToGrasppose(net_output, rgb_img, depth_map, M_CL, visualize=True):
    kps_pr = []
    for category_id, preds in net_output.items():
        if len(preds) == 0:
            continue

        for pred in preds:
            kps_pr.append(pred[:4])

    grasps = scoring(kps_pr, rgb_img, depth_map, M_CL)
    grasps = sorted(grasps, key=lambda x: x[-1], reverse=True)

    grasp = grasps[0]

    if visualize:
        rgb_img = cv2.circle(rgb_img, (int(grasp[5]), int(grasp[6])), 2, (0, 0, 255), 3)
        rgb_img = cv2.circle(rgb_img, (int(grasp[7]), int(grasp[8])), 2, (0, 0, 255), 3)

        cv2.namedWindow('visual', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('visual', rgb_img)
        k = cv2.waitKey(1)

    return grasp[:5]

def pre_process(rgb_img, depth_img):
    inp_image = rgb_img.copy()
    inp_image[:, :, 0] = depth_img

    # crop the region of the picking bin
    inp_image = inp_image[pb_tl[1]:pb_br[1], pb_tl[0]:pb_br[0], :]
    inp_image = cv2.resize(inp_image, (512, 512))
    inp_image = inp_image[:, :, ::-1]

    return inp_image

def isPickbinClear(M_CL, depth_map):
    p_tl = project_3d_2d(pb_tl, M_CL)
    p_br = project_3d_2d(pb_br, M_CL)

    min_z = sys.maxsize
    max_z = -sys.maxsize
    for i in range(int(p_tl[0]), int(p_br[0])+1):
        for j in range(int(p_tl[1]), int(p_br[1])+1):
            p_3d = project_2d_3d([i, j], depth_map, M_CL)

            min_z = min(min_z, p_3d[2])
            max_z = max(max_z, p_3d[2])

    # consider it is cleared if the difference between object heights is less than 5mm
    return (max_z-min_z) < 0.01

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

        hf_depth = hole_filter.smooth_image(depth_raw[pb_tl[1]:pb_br[1], pb_tl[0]:pb_br[0]] * 1000)
        denoise_depth = noise_filter.smooth_image(hf_depth)
        denoise_depth /= 1000
        depth_raw[pb_tl[1]:pb_br[1], pb_tl[0]:pb_br[0]] = denoise_depth

        gray = img.astype(np.uint8)
        depth = (depth_raw * 1000).astype(np.uint8)

        # get the current transformation from the camera to aruco tag
        M_CL, corners = get_M_CL(gray, img, False)

        # check if pick bin has been cleared
        if isPickbinClear(M_CL, depth_raw):
            pub_end.publish(True)
        else:
            pub_end.publish(False)

        # replace blue channel with the depth channel
        inp_image = pre_process(img_wo_at, depth)

        # pass the image into the network
        ret = detector.run(inp_image[:, :, :])
        ret = ret["results"]

        pose = KpsToGrasppose(ret, img, depth_raw, M_CL, M_BL, cameraMatrix)
        msg = Float64MultiArray()
        msg.data = pose
        pub_res.publish(msg)

    except CvBridgeError as e:
        print(e)

if __name__ == '__main__':
    # initialize ros node
    rospy.init_node("Bin_picking")

    # Bridge to convert ROS Image type to OpenCV Image type
    cv_bridge = CvBridge()
    cv2.WITH_QT = False
    # Get camera calibration parameters
    cam_param = rospy.wait_for_message('/camera/rgb/camera_info', CameraInfo, timeout=None)

    # Subscribe to rgb and depth channel
    image_sub = message_filters.Subscriber("/camera/rgb/image_rect_color", Image)
    depth_sub = message_filters.Subscriber("/camera/depth_registered/image", Image)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 1, 0.1)
    ts.registerCallback(kinect_rgbd_callback)

    rospy.spin()