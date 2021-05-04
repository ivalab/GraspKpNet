import numpy as np
import math
from shapely.geometry import Polygon

##########################################
# return the polygon after rotation
##########################################
# def rotate_bbox(box, angle, image_w, image_h):
#   xmin = box[0]
#   ymin = box[1]
#   xmax = box[2]
#   ymax = box[3]
#
#   x_center = (xmax - xmin + 1 / image_w) / 2 + xmin
#   y_center = (ymax - ymin + 1 / image_h) / 2 + ymin
#   width = (xmax - xmin + 1 / image_w)
#   height = (ymax - ymin + 1 / image_h)
#   diagonal = math.sqrt(width * width + height * height) / 2
#
#   theta = math.atan(height / width)
#
#   # jacquard
#   theta_1_grasp = theta + angle
#   theta_2_grasp = np.pi - theta + angle
#
#   p1 = (math.cos(theta_1_grasp) * diagonal + x_center, -math.sin(theta_1_grasp) * diagonal + y_center)
#   p2 = (math.cos(theta_2_grasp) * diagonal + x_center, -math.sin(theta_2_grasp) * diagonal + y_center)
#   p3 = (-math.cos(theta_1_grasp) * diagonal + x_center, math.sin(theta_1_grasp) * diagonal + y_center)
#   p4 = (-math.cos(theta_2_grasp) * diagonal + x_center, math.sin(theta_2_grasp) * diagonal + y_center)
#   p = Polygon([p1, p2, p3, p4])
#
#   return p1, p2, p3, p4, p

def rotate_bbox(box, angle):
  xmin = box[0]
  ymin = box[1]
  xmax = box[2]
  ymax = box[3]

  tl_0 = np.array([xmin, ymin])
  br_0 = np.array([xmax, ymax])
  bl_0 = np.array([xmin, ymax])
  tr_0 = np.array([xmax, ymin])
  center = np.array([(xmin+xmax)/2, (ymin+ymax)/2])

  T = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
  tl_1 = np.dot(T, (tl_0-center)) + center
  br_1 = np.dot(T, (br_0-center)) + center
  bl_1 = np.dot(T, (bl_0-center)) + center
  tr_1 = np.dot(T, (tr_0-center)) + center

  p_tl = (tl_1[0], tl_1[1])
  p_bl = (bl_1[0], bl_1[1])
  p_br = (br_1[0], br_1[1])
  p_tr = (tr_1[0], tr_1[1])

  p = Polygon([p_tl, p_bl, p_br, p_tr])

  return p_tl, p_bl, p_br, p_tr, p

def rotate_bbox_counterclock(box, angle):
  xmin = box[0]
  ymin = box[1]
  xmax = box[2]
  ymax = box[3]

  tl_0 = np.array([xmin, ymin])
  br_0 = np.array([xmax, ymax])
  bl_0 = np.array([xmin, ymax])
  tr_0 = np.array([xmax, ymin])
  center = np.array([(xmin+xmax)/2, (ymin+ymax)/2])

  T = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
  tl_1 = np.dot(T, (tl_0-center)) + center
  br_1 = np.dot(T, (br_0-center)) + center
  bl_1 = np.dot(T, (bl_0-center)) + center
  tr_1 = np.dot(T, (tr_0-center)) + center

  p_tl = (tl_1[0], tl_1[1])
  p_bl = (bl_1[0], bl_1[1])
  p_br = (br_1[0], br_1[1])
  p_tr = (tr_1[0], tr_1[1])

  p = Polygon([p_tl, p_bl, p_br, p_tr])

  return p_tl, p_bl, p_br, p_tr, p

############################################################
# compute the IOU between predicted bbx and groundtruth ROIs
############################################################
def _bbox_overlaps(boxes, rois_gt, angles_pr, angles_gt):
  K = rois_gt.shape[0]  # len(rois_gt)
  N = boxes.shape[0]
  overlaps = np.zeros((N, K))

  for k in range(K):
    # gt_rois_area = (rois_gt[k, 2] - rois_gt[k, 0] + 1/512) * (rois_gt[k, 3] - rois_gt[k, 1] + 1/512)
    ############################
    #    Jarqurad
    ############################
    angle_gt = angles_gt[k]
    _, _, _, _, p_gt = rotate_bbox(rois_gt[k], angle_gt)
    for n in range(N):
      ############################
      #    Jarqurad
      ############################
      angle_pr = angles_pr[n]
      _, _, _, _, p_pr = rotate_bbox(boxes[n], angle_pr)
      overlaps[n, k] = IoU(p_gt, p_pr)
  return overlaps

def _bbox_overlaps_counterclock(boxes, rois_gt, angles_pr, angles_gt):
  K = rois_gt.shape[0]  # len(rois_gt)
  N = boxes.shape[0]
  overlaps = np.zeros((N, K))

  for k in range(K):
    # gt_rois_area = (rois_gt[k, 2] - rois_gt[k, 0] + 1/512) * (rois_gt[k, 3] - rois_gt[k, 1] + 1/512)
    ############################
    #    Jarqurad
    ############################
    angle_gt = angles_gt[k]
    _, _, _, _, p_gt = rotate_bbox_counterclock(rois_gt[k], angle_gt)
    for n in range(N):
      ############################
      #    Jarqurad
      ############################
      angle_pr = angles_pr[n]
      _, _, _, _, p_pr = rotate_bbox_counterclock(boxes[n], angle_pr)
      overlaps[n, k] = IoU(p_gt, p_pr)
  return overlaps

def IoU(p1, p2):
    intersection = p1.intersection(p2)
    area_inter = intersection.area
    union = p1.union(p2)
    area_union = union.area

    return area_inter / area_union
