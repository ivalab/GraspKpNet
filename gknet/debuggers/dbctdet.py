from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import torch

from .base_debugger import BaseDebugger

from gknet.models.utils import _tranpose_and_gather_feat
from gknet.models.decode import _topk_original


class DbCtdetDebugger(BaseDebugger):
    def __init__(self, opt):
        super(DbCtdetDebugger, self).__init__(opt)

    def forward(self, images):
        with torch.no_grad():
            output = self.model(images)[-1]

            tl = output["tl"].sigmoid_()
            br = output["br"].sigmoid_()
            ct = output["ct"].sigmoid_()

            tl_tag = output["tl_tag"]
            br_tag = output["br_tag"]

            tl_reg = output["tl_reg"]
            br_reg = output["br_reg"]
            ct_reg = output["ct_reg"]

            detections = {
                "tl_heatmap": tl,
                "br_heatmap": br,
                "ct_heatmap": ct,
                "tl_reg": tl_reg,
                "br_reg": br_reg,
                "ct_reg": ct_reg,
                "tl_tag": tl_tag,
                "br_tag": br_tag,
            }

            return detections

    def debug(self, detections, targets, ae_threshold):
        tl_heat = detections["tl_heatmap"]
        br_heat = detections["br_heatmap"]
        ct_heat = detections["ct_heatmap"]

        tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk_original(tl_heat, K=128)
        br_scores, br_inds, br_clses, br_ys, br_xs = _topk_original(br_heat, K=128)
        ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = _topk_original(ct_heat, K=128)

        tl_tag = detections["tl_tag"]
        br_tag = detections["br_tag"]

        # gather by gt
        # tl_tag = _tranpose_and_gather_feat(tl_tag, targets['tl_tag'].to(torch.device("cuda")))
        # br_tag = _tranpose_and_gather_feat(br_tag, targets['br_tag'].to(torch.device("cuda")))
        # gather by top k
        tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
        br_tag = _tranpose_and_gather_feat(br_tag, br_inds)

        dists_tl_br = torch.abs(tl_tag - br_tag)

        dist_inds = dists_tl_br < ae_threshold
        dist_inds = dist_inds.squeeze(2)

        # get tl, bl and br index of heatmap after grouping
        tl_inds = tl_inds[dist_inds].to(torch.device("cpu")).numpy()
        br_inds = br_inds[dist_inds].to(torch.device("cpu")).numpy()
        # tl bl br index of heatmap groundtruth
        tl_tag_gt = targets["tl_tag"].to(torch.device("cpu")).numpy()
        br_tag_gt = targets["br_tag"].to(torch.device("cpu")).numpy()

        tl_intersect, _, tl_inds = np.intersect1d(
            tl_inds, tl_tag_gt[0], return_indices=True
        )
        br_intersect, _, br_inds = np.intersect1d(
            br_inds, br_tag_gt[0], return_indices=True
        )

        tl_br_intersect = np.intersect1d(tl_inds, br_inds)

        # true_positive = (dist_inds & targets['reg_mask'].to(torch.device("cuda")))
        # true_positive = true_positive.to(torch.device("cpu")).numpy()

        # print("Recall is {} out of {}".format(true_positive.sum(), targets['reg_mask'].numpy().sum()))

        # return true_positive.sum() / targets['reg_mask'].numpy().sum()
        return len(tl_br_intersect)
