from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs

import rospy
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray

import torch

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
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

def get_M_CL(gray, image_init, visualize=False):
    # parameters
    markerLength_CL = 0.076
    # aruco_dict_CL = aruco.Dictionary_get(aruco.DICT_5X5_250)
    aruco_dict_CL = aruco.Dictionary_get(aruco.DICT_6X6_250)
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

def project(pixel, depth_image, M_CL, M_BL, cameraMatrix):
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
    depth = depth_image[pixel[1], pixel[0]]
    moving_pixel = [pixel[0], pixel[1]]
    while depth == 0:
        moving_pixel = [moving_pixel[0], moving_pixel[1]+1]
        depth = depth_image[moving_pixel[1], moving_pixel[0]]

    pxl = np.linalg.inv(cameraMatrix).dot(
        np.array([pixel[0] * depth, pixel[1] * depth, depth]))
    q_C = np.array([pxl[0], pxl[1], pxl[2], 1])
    q_L = np.linalg.inv(M_CL).dot(q_C)
    q_B = M_BL.dot(q_L)

    return q_B

def depth_process(depth_image):
    depth_image[depth_image > 1887] = 1887
    depth_image = (depth_image - 493) * 254 / (1887 - 493)
    depth_image[depth_image < 0] = 0

    return depth_image

def KpsToGrasppose(net_output, rgb_img, depth_map, M_CL, M_BL, cameraMatrix, visualize=True):
    kps_pr = []
    for category_id, preds in net_output.items():
        if len(preds) == 0:
            continue

        for pred in preds:
            kps = pred[:4]
            score = pred[-1]
            kps_pr.append([kps[0], kps[1], kps[2], kps[3], score])

    # sort by the confidence score
    kps_pr = sorted(kps_pr, key=lambda x: x[-1])
    # select the top 1
    kp_pr = kps_pr[0]

    kp_lm = [int(kp_pr[0]), int(kp_pr[1])]
    kp_rm = [int(kp_pr[2]), int(kp_pr[3])]
    kp_lm_3d = project(kp_lm, depth_map, M_CL, M_BL, cameraMatrix)
    kp_rm_3d = project(kp_rm, depth_map, M_CL, M_BL, cameraMatrix)
    center = [int((kp_lm[0]+kp_rm[0])/2), int((kp_lm[1]+kp_rm[1])/2)]
    center_3d = project(center, depth_map, M_CL, M_BL, cameraMatrix)

    orientation = np.arctan2(kp_rm_3d[1] - kp_lm_3d[1], p_4_3d[0] - kp_lm_3d[0])
    # motor 7 is clockwise
    if orientation > np.pi / 2:
        orientation = np.pi - orientation
    elif orientation < -np.pi / 2:
        orientation = -np.pi - orientation
    else:
        orientation = -orientation

    if visualize:
        rgb_img = cv2.circle(rgb_img, (int(kp_lm[0]), int(kp_lm[1])), 2, (0, 0, 255), 3)
        rgb_img = cv2.circle(rgb_img, (int(kp_rm[0]), int(kp_rm[1])), 2, (0, 0, 255), 3)

        cv2.namedWindow('visual', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('visual', rgb_img)
        k = cv2.waitKey(1)

    return [center_3d[0], center_3d[1], center_3d[2], orientation]

def run(opt, pipeline, align, depth_scale, pub_res):
    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Detector = detector_factory[opt.task]

    detector = Detector(opt)

    while not rospy.is_shutdown():
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Convert images to numpy arrays
        depth_raw = np.array(depth_frame.get_data()) * depth_scale
        depth = (depth_raw / depth_scale).astype(np.uint8)
        img = np.array(color_frame.get_data())
        gray = img.astype(np.uint8)

        # get the current transformation from the camera to aruco tag
        M_CL = get_M_CL(gray, img, True)

        inp_image = img.copy()
        depth_processed = depth.copy()

        # convert raw depth to 0-255
        depth_processed = depth_process(depth_processed)

        # replace blue channel with the depth channel
        inp_image[:, :, 2] = depth_processed

        # pass the image into the network
        ret = detector.run(inp_image)
        ret = ret["results"]

        loc_ori = KpsToGrasppose(ret, img, depth_raw, M_CL, M_BL, cameraMatrix)

        pub_res.publish(loc_ori)

    # Stop streaming
    pipeline.stop()

if __name__ == '__main__':
    opt = opts().parse()

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # initialize ros node
    rospy.init_node("Grasp experiment")
    # Publisher of perception result
    pub_res = rospy.Publisher('/result', Float64MultiArray, queue_size=10)

    run(opt, pipeline, align, depth_scale, pub_res)