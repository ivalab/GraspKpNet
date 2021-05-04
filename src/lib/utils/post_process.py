from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy

from .image import transform_preds
from .ddd_utils import ddd2locrot


def get_pred_depth(depth):
  return depth

def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan(rot[:, 2] / rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan(rot[:, 6] / rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)

def dbmctdet_post_process(detections, c, s, h, w, scale, num_classes, ori_threshold):
  ret = []
  for i in range(detections.shape[0]):
    top_preds = {}

    # Apply transformation to predicteed bbox
    detections[i, :, 0:2] = transform_preds(
      detections[i, :, 0:2], c[i], s[i], (w, h))
    detections[i, :, 2:4] = transform_preds(
      detections[i, :, 2:4], c[i], s[i], (w, h))

    detections[i, :, 0:4] /= scale

    # classes = detections[i, :, -1]

    # Dump bbox whose central region has no center point
    detections = np.concatenate(detections, axis=1)

    # filter by orientation distance between quantized and continuous predicted angle
    classes = detections[..., -1]
    quant_ori = (5.0 * classes - 85.0) / 180 * np.pi
    lm_x = detections[..., 0]
    lm_y = detections[..., 1]
    rm_x = detections[..., 2]
    rm_y = detections[..., 3]
    vert_ind1 = (rm_x == lm_x) & (rm_y > lm_y)
    vert_ind2 = (rm_x == lm_x) & (rm_y < lm_y)
    cont_ori = np.arctan((rm_y - lm_y) / (rm_x - lm_x))
    cont_ori[vert_ind1] = np.pi / 2
    cont_ori[vert_ind2] = -np.pi / 2
    dist_ori = np.fabs(quant_ori - cont_ori)
    ind_over90 = dist_ori > np.pi / 2
    dist_ori[ind_over90] = np.pi - dist_ori[ind_over90]
    ori_ind = dist_ori < ori_threshold

    detections = detections[ori_ind]

    detections = np.expand_dims(detections, axis=0)
    classes = detections[i, :, -1]

    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        detections[i, inds, :4].astype(np.float32),
        detections[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret

def dbmctdet_cornell_post_process(detections, c, s, h, w, scale, num_classes, ori_threshold):
  ret = []
  for i in range(detections.shape[0]):
    top_preds = {}

    # Apply transformation to predicteed bbox
    detections[i, :, 0:2] = transform_preds(
      detections[i, :, 0:2], c[i], s[i], (w, h))
    detections[i, :, 2:4] = transform_preds(
      detections[i, :, 2:4], c[i], s[i], (w, h))

    detections[i, :, 0:4] /= scale

    # classes = detections[i, :, -1]

    # Dump bbox whose central region has no center point
    detections = np.concatenate(detections, axis=1)

    # filter by orientation distance between quantized and continuous predicted angle
    classes = detections[..., -1]
    quant_ori = (10.0 * (classes + 1) - 90.0) / 180 * np.pi
    lm_x = detections[..., 0]
    lm_y = detections[..., 1]
    rm_x = detections[..., 2]
    rm_y = detections[..., 3]
    vert_ind1 = (rm_x == lm_x) & (rm_y > lm_y)
    vert_ind2 = (rm_x == lm_x) & (rm_y < lm_y)
    cont_ori = np.arctan(-(rm_y - lm_y) / (rm_x - lm_x))
    cont_ori[vert_ind1] = -np.pi / 2
    cont_ori[vert_ind2] = np.pi / 2
    dist_ori = np.fabs(quant_ori - cont_ori)
    ind_over90 = dist_ori > np.pi / 2
    dist_ori[ind_over90] = np.pi - dist_ori[ind_over90]
    ori_ind = dist_ori < ori_threshold

    detections = detections[ori_ind]

    detections = np.expand_dims(detections, axis=0)
    classes = detections[i, :, -1]

    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        detections[i, inds, :4].astype(np.float32),
        detections[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret
