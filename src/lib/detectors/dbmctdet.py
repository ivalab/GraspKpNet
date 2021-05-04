from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from external.nms import soft_nms
from models.decode import dbmctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import dbmctdet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

class DbMCtdetDetector(BaseDetector):
  def __init__(self, opt):
    super(DbMCtdetDetector, self).__init__(opt)
  
  def process(self, images, return_time=False):
    with torch.no_grad():
      output = self.model(images)[-1]
      lm = output['lm'].sigmoid_()
      rm = output['rm'].sigmoid_()
      ct = output['ct'].sigmoid_()

      lm_tag = output['lm_tag']
      rm_tag = output['rm_tag']

      lm_reg = output['lm_reg']
      rm_reg = output['rm_reg']
      ct_reg = output['ct_reg']

      torch.cuda.synchronize()
      forward_time = time.time()
      dets = dbmctdet_decode(lm, rm, ct, lm_tag, rm_tag, lm_reg, rm_reg, ct_reg, \
                                kernel=self.opt.nms_kernel, ae_threshold=self.opt.ae_threshold, scores_thresh=0.1,
                                center_thresh=self.opt.center_threshold, scores_weight=self.opt.scores_weight,
                                K=self.opt.K, num_dets=self.opt.num_dets)
      
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  # apply transformation
  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()

    dets = dets.reshape(1, -1, dets.shape[2])

    dets = dbmctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], scale, self.opt.num_classes, self.opt.ori_threshold)

    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)

    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
         soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4], 
                                 img_id='out_pred_{:.1f}'.format(scale))

  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] > self.opt.vis_thresh:
          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    debugger.show_all_imgs(pause=self.pause)
