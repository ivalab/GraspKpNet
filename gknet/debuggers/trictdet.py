import copy
import math

import numpy as np
import torch

from gknet.datasets.dataset.utils import _bbox_overlaps
from gknet.models.decode import _nms, _topk
from gknet.models.utils import _gather_feat, _tranpose_and_gather_feat

from .base_debugger import BaseDebugger


class TriCtdetDebugger(BaseDebugger):
    def __init__(self, opt):
        super(TriCtdetDebugger, self).__init__(opt)

    def forward(self, images):
        with torch.no_grad():
            output = self.model(images)[-1]

            tl = output["tl"].sigmoid_()
            bl = output["bl"].sigmoid_()
            br = output["br"].sigmoid_()
            ct = output["ct"].sigmoid_()

            tl_tag = output["tl_tag"]
            bl_tag = output["bl_tag"]
            br_tag = output["br_tag"]

            tl_reg = output["tl_reg"]
            bl_reg = output["bl_reg"]
            br_reg = output["br_reg"]
            ct_reg = output["ct_reg"]

            detections = {
                "tl_heatmap": tl,
                "bl_heatmap": bl,
                "br_heatmap": br,
                "ct_heatmap": ct,
                "tl_reg": tl_reg,
                "bl_reg": bl_reg,
                "br_reg": br_reg,
                "ct_reg": ct_reg,
                "tl_tag": tl_tag,
                "bl_tag": bl_tag,
                "br_tag": br_tag,
            }

            return detections

    def debug(self, detections, targets, ae_threshold):
        tl_heat = detections["tl_heatmap"]
        bl_heat = detections["bl_heatmap"]
        br_heat = detections["br_heatmap"]
        ct_heat = detections["ct_heatmap"]

        targets["tl_tag"] = targets["tl_tag"][targets["reg_mask"]].unsqueeze(0)
        targets["bl_tag"] = targets["bl_tag"][targets["reg_mask"]].unsqueeze(0)
        targets["br_tag"] = targets["br_tag"][targets["reg_mask"]].unsqueeze(0)
        targets["ct_tag"] = targets["ct_tag"][targets["reg_mask"]].unsqueeze(0)
        targets["tl_reg"] = targets["tl_reg"][targets["reg_mask"]].unsqueeze(0)
        targets["bl_reg"] = targets["bl_reg"][targets["reg_mask"]].unsqueeze(0)
        targets["br_reg"] = targets["br_reg"][targets["reg_mask"]].unsqueeze(0)
        targets["ct_reg"] = targets["ct_reg"][targets["reg_mask"]].unsqueeze(0)

        batch, cat, height, width = tl_heat.size()

        # tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=256)
        # bl_scores, bl_inds, bl_clses, bl_ys, bl_xs = _topk(bl_heat, K=256)
        # br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=256)
        # ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = _topk(ct_heat, K=256)

        tl_tag = detections["tl_tag"]
        bl_tag = detections["bl_tag"]
        br_tag = detections["br_tag"]
        tl_reg = detections["tl_reg"]
        bl_reg = detections["bl_reg"]
        br_reg = detections["br_reg"]
        ct_reg = detections["ct_reg"]

        # gather by gt
        tl_tag = _tranpose_and_gather_feat(
            tl_tag, targets["tl_tag"].to(torch.device("cuda"))
        )
        bl_tag = _tranpose_and_gather_feat(
            bl_tag, targets["bl_tag"].to(torch.device("cuda"))
        )
        br_tag = _tranpose_and_gather_feat(
            br_tag, targets["br_tag"].to(torch.device("cuda"))
        )
        # gather by top k
        # tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
        # bl_tag = _tranpose_and_gather_feat(bl_tag, bl_inds)
        # br_tag = _tranpose_and_gather_feat(br_tag, br_inds)

        avg_tag = (tl_tag + bl_tag + br_tag) / 3

        dists_tl = torch.abs(avg_tag - tl_tag).to(torch.device("cpu")).numpy()
        dists_bl = torch.abs(bl_tag - avg_tag).to(torch.device("cpu")).numpy()
        dists_br = torch.abs(avg_tag - br_tag).to(torch.device("cpu")).numpy()
        dists_avg = (
            (dists_tl.sum() + dists_bl.sum() + dists_br.sum()) / dists_tl.shape[1] / 3
        )
        min_tl = dists_tl.min()
        max_tl = dists_tl.max()
        min_bl = dists_bl.min()
        max_bl = dists_bl.max()
        min_br = dists_br.min()
        max_br = dists_br.max()

        # gather by gt
        tl_reg = _tranpose_and_gather_feat(
            tl_reg, targets["tl_tag"].to(torch.device("cuda"))
        )
        bl_reg = _tranpose_and_gather_feat(
            bl_reg, targets["bl_tag"].to(torch.device("cuda"))
        )
        br_reg = _tranpose_and_gather_feat(
            br_reg, targets["br_tag"].to(torch.device("cuda"))
        )
        ct_reg = _tranpose_and_gather_feat(
            ct_reg, targets["ct_tag"].to(torch.device("cuda"))
        )

        # reg_diff_tl = tl_reg - targets['tl_reg'].to(torch.device("cuda"))
        # reg_diff_tl = torch.sqrt(reg_diff_tl[..., 0]*reg_diff_tl[..., 0] + reg_diff_tl[..., 1]*reg_diff_tl[..., 1])
        # reg_diff_bl = bl_reg - targets['bl_reg'].to(torch.device("cuda"))
        # reg_diff_bl = torch.sqrt(reg_diff_bl[..., 0] * reg_diff_bl[..., 0] + reg_diff_bl[..., 1] * reg_diff_bl[..., 1])
        # reg_diff_br = br_reg - targets['br_reg'].to(torch.device("cuda"))
        # reg_diff_br = torch.sqrt(reg_diff_br[..., 0] * reg_diff_br[..., 0] + reg_diff_br[..., 1] * reg_diff_br[..., 1])
        # reg_diff_ct = ct_reg - targets['ct_reg'].to(torch.device("cuda"))
        # reg_diff_ct = torch.sqrt(reg_diff_ct[..., 0] * reg_diff_ct[..., 0] + reg_diff_ct[..., 1] * reg_diff_ct[..., 1])

        tl_xs = (
            ((targets["tl_tag"] % (width * height)) % width)
            .int()
            .float()
            .to(torch.device("cuda"))
        )
        tl_ys = (
            ((targets["tl_tag"] % (width * height)) / width)
            .int()
            .float()
            .to(torch.device("cuda"))
        )
        bl_xs = (
            ((targets["bl_tag"] % (width * height)) % width)
            .int()
            .float()
            .to(torch.device("cuda"))
        )
        bl_ys = (
            ((targets["bl_tag"] % (width * height)) / width)
            .int()
            .float()
            .to(torch.device("cuda"))
        )
        br_xs = (
            ((targets["br_tag"] % (width * height)) % width)
            .int()
            .float()
            .to(torch.device("cuda"))
        )
        br_ys = (
            ((targets["br_tag"] % (width * height)) / width)
            .int()
            .float()
            .to(torch.device("cuda"))
        )
        ct_xs = (
            ((targets["ct_tag"] % (width * height)) % width)
            .int()
            .float()
            .to(torch.device("cuda"))
        )
        ct_ys = (
            ((targets["ct_tag"] % (width * height)) / width)
            .int()
            .float()
            .to(torch.device("cuda"))
        )

        tl_xs_pr = (tl_xs + tl_reg[..., 0]).squeeze(0).to(torch.device("cpu")).numpy()
        tl_ys_pr = (tl_ys + tl_reg[..., 1]).squeeze(0).to(torch.device("cpu")).numpy()
        bl_xs_pr = (bl_xs + bl_reg[..., 0]).squeeze(0).to(torch.device("cpu")).numpy()
        bl_ys_pr = (bl_ys + bl_reg[..., 1]).squeeze(0).to(torch.device("cpu")).numpy()
        br_xs_pr = (br_xs + br_reg[..., 0]).squeeze(0).to(torch.device("cpu")).numpy()
        br_ys_pr = (br_ys + br_reg[..., 1]).squeeze(0).to(torch.device("cpu")).numpy()
        ct_xs_pr = (ct_xs + ct_reg[..., 0]).squeeze(0).to(torch.device("cpu")).numpy()
        ct_ys_pr = (ct_ys + ct_reg[..., 1]).squeeze(0).to(torch.device("cpu")).numpy()

        tl_xs_gt = (
            (tl_xs + targets["tl_reg"][..., 0].to(torch.device("cuda")))
            .squeeze(0)
            .to(torch.device("cpu"))
            .numpy()
        )
        tl_ys_gt = (
            (tl_ys + targets["tl_reg"][..., 1].to(torch.device("cuda")))
            .squeeze(0)
            .to(torch.device("cpu"))
            .numpy()
        )
        bl_xs_gt = (
            (bl_xs + targets["bl_reg"][..., 0].to(torch.device("cuda")))
            .squeeze(0)
            .to(torch.device("cpu"))
            .numpy()
        )
        bl_ys_gt = (
            (bl_ys + targets["bl_reg"][..., 1].to(torch.device("cuda")))
            .squeeze(0)
            .to(torch.device("cpu"))
            .numpy()
        )
        br_xs_gt = (
            (br_xs + targets["br_reg"][..., 0].to(torch.device("cuda")))
            .squeeze(0)
            .to(torch.device("cpu"))
            .numpy()
        )
        br_ys_gt = (
            (br_ys + targets["br_reg"][..., 1].to(torch.device("cuda")))
            .squeeze(0)
            .to(torch.device("cpu"))
            .numpy()
        )
        ct_xs_gt = (
            (ct_xs + targets["ct_reg"][..., 0].to(torch.device("cuda")))
            .squeeze(0)
            .to(torch.device("cpu"))
            .numpy()
        )
        ct_ys_gt = (
            (ct_ys + targets["ct_reg"][..., 1].to(torch.device("cuda")))
            .squeeze(0)
            .to(torch.device("cpu"))
            .numpy()
        )

        bboxes_gt = targets["bbox"][targets["reg_mask"]]

        nm_instances = tl_xs_pr.shape[0]

        for i in range(nm_instances):
            bbox_gt = bboxes_gt[i, :]
            # prediction
            bbox_coord_pr = []
            tl_x_pr = tl_xs_pr[i]
            tl_y_pr = tl_ys_pr[i]
            bl_x_pr = bl_xs_pr[i]
            bl_y_pr = bl_ys_pr[i]
            br_x_pr = br_xs_pr[i]
            br_y_pr = br_ys_pr[i]

            # center
            x_c = (tl_x_pr + br_x_pr) / 2.0
            y_c = (tl_y_pr + br_y_pr) / 2.0

            if bl_x_pr == br_x_pr:
                p_y = tl_y_pr
                p_x = br_x_pr
                if br_y_pr > bl_y_pr:
                    angle = np.pi / 2.0
                else:
                    angle = -np.pi / 2.0
            elif bl_y_pr == br_y_pr:
                p_x = tl_x_pr
                p_y = br_y_pr
                angle = 0.0
            else:
                # angle
                angle = math.atan2(-(br_y_pr - bl_y_pr), br_x_pr - bl_x_pr)
                # find intersected point
                a = (br_x_pr - bl_x_pr) / (br_y_pr - bl_y_pr)
                b = br_y_pr - a * br_x_pr
                delta_x = br_x_pr - bl_x_pr
                delta_y = br_y_pr - bl_y_pr
                p_x = (delta_x * tl_x_pr + delta_y * tl_y_pr - delta_x * b) / (
                    delta_x + delta_x * a
                )
                p_y = a * p_x + b
                # w, h
            w = np.sqrt(
                (br_x_pr - p_x) * (br_x_pr - p_x) + (br_y_pr - p_y) * (br_y_pr - p_y)
            )
            h = np.sqrt(
                (tl_x_pr - p_x) * (tl_x_pr - p_x) + (tl_y_pr - p_y) * (tl_y_pr - p_y)
            )

            bbox_coord_pr.append(
                [x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2, angle]
            )
            bbox_coord_pr = np.array(bbox_coord_pr)

            # groundtruth
            boxes_coord_gt = []
            tl_x_gt = tl_xs_gt[i]
            tl_y_gt = tl_ys_gt[i]
            bl_x_gt = bl_xs_gt[i]
            bl_y_gt = bl_ys_gt[i]
            br_x_gt = br_xs_gt[i]
            br_y_gt = br_ys_gt[i]
            if bl_x_gt == br_x_gt:
                p_y = tl_y_gt
                p_x = bl_x_gt
                if br_y_gt > bl_y_gt:
                    angle = np.pi / 4
                else:
                    angle = -np.pi / 4
            else:
                # center
                x_c = (tl_x_gt + br_x_gt) / 2.0
                y_c = (tl_y_gt + br_y_gt) / 2.0
                # angle
                angle = math.atan(-(br_y_gt - bl_y_gt) / (br_x_gt - bl_x_gt))
                # find intersected point
                a = (br_y_gt - bl_y_gt) / (br_x_gt - bl_x_gt)
                b = br_y_gt - a * br_x_gt
                delta_x = br_x_gt - bl_x_gt
                delta_y = br_y_gt - bl_y_gt
                p_x = (delta_x * tl_x_gt + delta_y * tl_y_gt - delta_y * b) / (
                    delta_x + delta_y * a
                )
                p_y = a * p_x + b
                # w, h
            w = np.sqrt(
                (br_x_gt - p_x) * (br_x_gt - p_x) + (br_y_gt - p_y) * (br_y_gt - p_y)
            )
            h = np.sqrt(
                (tl_x_gt - p_x) * (tl_x_gt - p_x) + (tl_y_gt - p_y) * (tl_y_gt - p_y)
            )
            boxes_coord_gt.append(
                [x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2, angle]
            )
            boxes_coord_gt = np.array(boxes_coord_gt)
            # print(np.array_equal(bbox_gt, boxes_coord_gt))

            overlaps = _bbox_overlaps(
                np.ascontiguousarray(bbox_coord_pr[:, :4], dtype=np.float32),
                np.ascontiguousarray(boxes_coord_gt[:, :4], dtype=np.float32),
                bbox_coord_pr[:, -1],
                boxes_coord_gt[:, -1],
                128,
                128,
            )

            flag_suc = False
            flag_exit = 0
            for i in range(overlaps.shape[0]):
                for j in range(overlaps.shape[1]):
                    value_overlap = overlaps[i, j]
                    angle_diff = math.fabs(bbox_coord_pr[i, -1] - boxes_coord_gt[j, -1])

                    if value_overlap > 0.25 and angle_diff < np.pi / 6:
                        flag_suc = True
                        flag_exit = 1
                        break
                if flag_exit:
                    break
            if flag_exit:
                break

        return min_tl, max_tl, min_bl, max_bl, min_br, max_br, dists_avg, flag_suc

    def process(self, images, kernel=1, ae_threshold=1, K=100, num_dets=100):
        with torch.no_grad():
            output = self.model(images)[-1]

            tl_heat = output["tl"].sigmoid_()
            bl_heat = output["bl"].sigmoid_()
            br_heat = output["br"].sigmoid_()
            ct_heat = output["ct"].sigmoid_()

            tl_tag = output["tl_tag"]
            bl_tag = output["bl_tag"]
            br_tag = output["br_tag"]

            tl_reg = output["tl_reg"]
            bl_reg = output["bl_reg"]
            br_reg = output["br_reg"]
            ct_reg = output["ct_reg"]

            batch, cat, height, width = tl_heat.size()

            tl_heat = _nms(tl_heat, kernel=3)
            bl_heat = _nms(bl_heat, kernel=3)
            br_heat = _nms(br_heat, kernel=3)
            ct_heat = _nms(ct_heat, kernel=3)

            tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
            bl_scores, bl_inds, bl_clses, bl_ys, bl_xs = _topk(bl_heat, K=K)
            br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
            ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = _topk(ct_heat, K=K)

            tl_ys = tl_ys.view(batch, K, 1, 1).expand(batch, K, K, K)
            tl_xs = tl_xs.view(batch, K, 1, 1).expand(batch, K, K, K)
            bl_ys = bl_ys.view(batch, 1, K, 1).expand(batch, K, K, K)
            bl_xs = bl_xs.view(batch, 1, K, 1).expand(batch, K, K, K)
            br_ys = br_ys.view(batch, 1, 1, K).expand(batch, K, K, K)
            br_xs = br_xs.view(batch, 1, 1, K).expand(batch, K, K, K)
            ct_ys = ct_ys.view(batch, 1, K).expand(batch, K, K)
            ct_xs = ct_xs.view(batch, 1, K).expand(batch, K, K)

            if tl_reg is not None and bl_reg is not None and br_reg is not None:
                tl_reg = _tranpose_and_gather_feat(tl_reg, tl_inds)
                tl_reg = tl_reg.view(batch, K, 1, 1, 2)
                bl_reg = _tranpose_and_gather_feat(bl_reg, bl_inds)
                bl_reg = bl_reg.view(batch, 1, K, 1, 2)
                br_reg = _tranpose_and_gather_feat(br_reg, br_inds)
                br_reg = br_reg.view(batch, 1, 1, K, 2)
                ct_reg = _tranpose_and_gather_feat(ct_reg, ct_inds)
                ct_reg = ct_reg.view(batch, 1, K, 2)

                tl_xs = tl_xs + tl_reg[..., 0]
                tl_ys = tl_ys + tl_reg[..., 1]
                bl_xs = bl_xs + bl_reg[..., 0]
                bl_ys = bl_ys + bl_reg[..., 1]
                br_xs = br_xs + br_reg[..., 0]
                br_ys = br_ys + br_reg[..., 1]
                ct_xs = ct_xs + ct_reg[..., 0]
                ct_ys = ct_ys + ct_reg[..., 1]

            # all possible boxes based on top k corners (ignoring class)
            bboxes = torch.stack((tl_xs, tl_ys, bl_xs, bl_ys, br_xs, br_ys), dim=4)

            tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
            tl_tag = tl_tag.view(batch, K, 1, 1)
            bl_tag = _tranpose_and_gather_feat(bl_tag, bl_inds)
            bl_tag = bl_tag.view(batch, 1, K, 1)
            br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
            br_tag = br_tag.view(batch, 1, 1, K)
            avg_tag = (tl_tag + bl_tag + br_tag) / 3
            dists = (
                torch.abs(tl_tag - avg_tag)
                + torch.abs(bl_tag - avg_tag)
                + torch.abs(br_tag - avg_tag)
            ) / 3

            tl_scores = tl_scores.view(batch, K, 1, 1).expand(batch, K, K, K)
            bl_scores = bl_scores.view(batch, 1, K, 1).expand(batch, K, K, K)
            br_scores = br_scores.view(batch, 1, 1, K).expand(batch, K, K, K)
            # reject boxes based on corner scores
            # sc_inds = (tl_scores < scores_thresh) | (bl_scores < scores_thresh) | (br_scores < scores_thresh)
            scores = (tl_scores + bl_scores + br_scores) / 3

            # reject boxes based on classes
            tl_clses = tl_clses.view(batch, K, 1, 1).expand(batch, K, K, K)
            bl_clses = bl_clses.view(batch, 1, K, 1).expand(batch, K, K, K)
            br_clses = br_clses.view(batch, 1, 1, K).expand(batch, K, K, K)
            cls_inds = (
                (tl_clses != bl_clses) | (bl_clses != br_clses) | (tl_clses != br_clses)
            )

            # reject boxes based on distances
            dist_inds = dists > ae_threshold

            scores[cls_inds] = -1
            scores[dist_inds] = -1
            # scores[sc_inds] = -1

            scores = scores.view(batch, -1)
            scores, inds = torch.topk(scores, num_dets)
            scores = scores.unsqueeze(2)

            bboxes = bboxes.view(batch, -1, 6)
            bboxes = _gather_feat(bboxes, inds)

            clses = bl_clses.contiguous().view(batch, -1, 1)
            clses = _gather_feat(clses, inds).float()
            tl_scores = tl_scores.contiguous().view(batch, -1, 1)
            tl_scores = _gather_feat(tl_scores, inds).float()
            bl_scores = bl_scores.contiguous().view(batch, -1, 1)
            bl_scores = _gather_feat(bl_scores, inds).float()
            br_scores = br_scores.contiguous().view(batch, -1, 1)
            br_scores = _gather_feat(br_scores, inds).float()

            ct_xs = ct_xs[:, 0, :]
            ct_ys = ct_ys[:, 0, :]

            centers = torch.cat(
                [
                    ct_xs.unsqueeze(2),
                    ct_ys.unsqueeze(2),
                    ct_clses.float().unsqueeze(2),
                    ct_scores.unsqueeze(2),
                ],
                dim=2,
            )
            detections = torch.cat(
                [bboxes, scores, tl_scores, bl_scores, br_scores, clses], dim=2
            )

            # tl_heat = output['tl'].sigmoid_()
            # bl_heat = output['bl'].sigmoid_()
            # br_heat = output['br'].sigmoid_()
            # ct_heat = output['ct'].sigmoid_()
            #
            # tl_tag = output['tl_tag']
            # bl_tag = output['bl_tag']
            # br_tag = output['br_tag']
            #
            # tl_reg = output['tl_reg']
            # bl_reg = output['bl_reg']
            # br_reg = output['br_reg']
            # ct_reg = output['ct_reg']
            #
            # kernel = self.opt.nms_kernel
            # ae_threshold = self.opt.ae_threshold
            # K = self.opt.K
            #
            # batch, cat, height, width = tl_heat.size()
            #
            # # perform nms on heatmaps
            # tl_heat = _nms(tl_heat, kernel=kernel)
            # bl_heat = _nms(bl_heat, kernel=kernel)
            # br_heat = _nms(br_heat, kernel=kernel)
            # ct_heat = _nms(ct_heat, kernel=kernel)
            #
            # tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
            # bl_scores, bl_inds, bl_clses, bl_ys, bl_xs = _topk(bl_heat, K=K)
            # br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
            # ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = _topk(ct_heat, K=K)
            #
            # tl_ys = tl_ys.view(batch, K, 1, 1).expand(batch, K, K, K)
            # tl_xs = tl_xs.view(batch, K, 1, 1).expand(batch, K, K, K)
            # bl_ys = bl_ys.view(batch, 1, K, 1).expand(batch, K, K, K)
            # bl_xs = bl_xs.view(batch, 1, K, 1).expand(batch, K, K, K)
            # br_ys = br_ys.view(batch, 1, 1, K).expand(batch, K, K, K)
            # br_xs = br_xs.view(batch, 1, 1, K).expand(batch, K, K, K)
            # ct_ys = ct_ys.view(batch, 1, K).expand(batch, K, K)
            # ct_xs = ct_xs.view(batch, 1, K).expand(batch, K, K)
            #
            # if tl_reg is not None and bl_reg is not None and br_reg is not None:
            #     tl_reg = _tranpose_and_gather_feat(tl_reg, tl_inds)
            #     tl_reg = tl_reg.view(batch, K, 1, 1, 2)
            #     bl_reg = _tranpose_and_gather_feat(bl_reg, bl_inds)
            #     bl_reg = bl_reg.view(batch, 1, K, 1, 2)
            #     br_reg = _tranpose_and_gather_feat(br_reg, br_inds)
            #     br_reg = br_reg.view(batch, 1, 1, K, 2)
            #     ct_reg = _tranpose_and_gather_feat(ct_reg, ct_inds)
            #     ct_reg = ct_reg.view(batch, 1, K, 2)
            #
            #     tl_xs = tl_xs + tl_reg[..., 0]
            #     tl_ys = tl_ys + tl_reg[..., 1]
            #     bl_xs = bl_xs + bl_reg[..., 0]
            #     bl_ys = bl_ys + bl_reg[..., 1]
            #     br_xs = br_xs + br_reg[..., 0]
            #     br_ys = br_ys + br_reg[..., 1]
            #     ct_xs = ct_xs + ct_reg[..., 0]
            #     ct_ys = ct_ys + ct_reg[..., 1]
            #
            # # all possible boxes based on top k corners (ignoring class)
            # bboxes = torch.stack((tl_xs, tl_ys, bl_xs, bl_ys, br_xs, br_ys), dim=4)
            #
            # tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
            # tl_tag = tl_tag.view(batch, K, 1, 1).expand(batch, K, K, K)
            # bl_tag = _tranpose_and_gather_feat(bl_tag, bl_inds)
            # bl_tag = bl_tag.view(batch, 1, K, 1).expand(batch, K, K, K)
            # br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
            # br_tag = br_tag.view(batch, 1, 1, K).expand(batch, K, K, K)
            # avg_tag = (tl_tag + bl_tag + br_tag) / 3
            # dists = (torch.abs(tl_tag - avg_tag) + torch.abs(bl_tag - avg_tag) + torch.abs(br_tag - avg_tag)) / 3
            #
            # tl_scores = tl_scores.view(batch, K, 1, 1).expand(batch, K, K, K)
            # bl_scores = bl_scores.view(batch, 1, K, 1).expand(batch, K, K, K)
            # br_scores = br_scores.view(batch, 1, 1, K).expand(batch, K, K, K)
            # scores = (tl_scores + bl_scores + br_scores) / 3
            #
            # # reject boxes based on classes
            # tl_clses = tl_clses.view(batch, K, 1, 1).expand(batch, K, K, K)
            # bl_clses = bl_clses.view(batch, 1, K, 1).expand(batch, K, K, K)
            # br_clses = br_clses.view(batch, 1, 1, K).expand(batch, K, K, K)
            # cls_inds = (tl_clses != bl_clses) | (bl_clses != br_clses) | (tl_clses != br_clses)
            #
            # # reject boxes based on distances
            # dist_inds = (dists > ae_threshold)
            #
            # # instead of filtering prediction according to the out-of-bound rotation, do data augmentation to mirror groundtruth
            #
            # scores[cls_inds] = -1
            # scores[dist_inds] = -1
            #
            # scores = scores.view(batch, -1)
            # scores, inds = torch.topk(scores, 100)
            # scores = scores.unsqueeze(2)
            #
            # bboxes = bboxes.view(batch, -1, 6)
            # bboxes = _gather_feat(bboxes, inds)
            #
            # tl_tag = tl_tag.contiguous().view(batch, -1, 1)
            # tl_tag = _gather_feat(tl_tag, inds)
            # bl_tag = bl_tag.contiguous().view(batch, -1, 1)
            # bl_tag = _gather_feat(bl_tag, inds)
            # br_tag = br_tag.contiguous().view(batch, -1, 1)
            # br_tag = _gather_feat(br_tag, inds)
            # avg_tag = avg_tag.contiguous().view(batch, -1, 1)
            # avg_tag = _gather_feat(avg_tag, inds)
            #
            # clses = bl_clses.contiguous().view(batch, -1, 1)
            # clses = _gather_feat(clses, inds).float()
            #
            # tl_scores = tl_scores.contiguous().view(batch, -1, 1)
            # tl_scores = _gather_feat(tl_scores, inds).float()
            # bl_scores = bl_scores.contiguous().view(batch, -1, 1)
            # bl_scores = _gather_feat(bl_scores, inds).float()
            # br_scores = br_scores.contiguous().view(batch, -1, 1)
            # br_scores = _gather_feat(br_scores, inds).float()
            #
            # ct_xs = ct_xs[:, 0, :]
            # ct_ys = ct_ys[:, 0, :]
            #
            # centers = torch.cat(
            #     [ct_xs.unsqueeze(2), ct_ys.unsqueeze(2), ct_clses.float().unsqueeze(2), ct_scores.unsqueeze(2)], dim=2)
            # detections = torch.cat([bboxes, scores, tl_scores, bl_scores, br_scores, clses, tl_tag, bl_tag, br_tag, avg_tag], dim=2)

        return detections, centers

    def post_process(
        self, detections, centers, num_classes, bbox_size_threshold, ori_threshold
    ):
        detections = detections.detach().cpu().numpy()
        centers = centers.detach().cpu().numpy()

        detections = detections.reshape(1, -1, detections.shape[2])
        centers = centers.reshape(1, -1, centers.shape[2])

        ret = []
        for i in range(detections.shape[0]):
            top_preds = {}
            detections[i, :, 0:2] *= 4.0
            detections[i, :, 2:4] *= 4.0
            detections[i, :, 4:6] *= 4.0
            centers[i, :, 0:2] *= 4.0

            # Dump bbox whose central region has no center point
            detections = np.concatenate(detections, axis=1)
            centers = np.concatenate(centers, axis=1)

            # filter by orientation distance between quantized and continuous predicted angle
            classes = detections[..., -1]
            quant_ori = (5.0 * classes - 85.0) / 180 * np.pi
            bl_x = detections[..., 2]
            bl_y = detections[..., 3]
            br_x = detections[..., 4]
            br_y = detections[..., 5]
            cont_ori = np.arctan(-(br_y - bl_y) / (br_x - bl_x))
            dist_ori = np.fabs(quant_ori - cont_ori)
            ori_ind = dist_ori < ori_threshold

            valid_detections = detections[ori_ind]

            valid_ind = valid_detections[:, 6] > -1
            valid_detections = valid_detections[valid_ind]
            # valid_ind = detections[:, 6] > -1
            # valid_detections = detections[valid_ind]

            box_width = np.sqrt(
                np.power(valid_detections[:, 2] - valid_detections[:, 4], 2)
                + np.power(valid_detections[:, 3] - valid_detections[:, 5], 2)
            )
            box_height = np.sqrt(
                np.power(valid_detections[:, 2] - valid_detections[:, 0], 2)
                + np.power(valid_detections[:, 3] - valid_detections[:, 1], 2)
            )

            s_ind = box_width * box_height <= bbox_size_threshold
            l_ind = box_width * box_height > bbox_size_threshold

            s_detections = valid_detections[s_ind]
            l_detections = valid_detections[l_ind]

            # pro-process for small bounding box
            s_tl_x = (2 * s_detections[:, 0] + s_detections[:, 4]) / 3
            s_br_x = (s_detections[:, 0] + 2 * s_detections[:, 4]) / 3
            s_tl_y = (2 * s_detections[:, 1] + s_detections[:, 5]) / 3
            s_br_y = (s_detections[:, 1] + 2 * s_detections[:, 5]) / 3

            s_temp_score = copy.copy(s_detections[:, 6])
            s_detections[:, 6] = -1

            center_x = centers[:, 0][:, np.newaxis]
            center_y = centers[:, 1][:, np.newaxis]
            s_tl_x = s_tl_x[np.newaxis, :]
            s_br_x = s_br_x[np.newaxis, :]
            s_tl_y = s_tl_y[np.newaxis, :]
            s_br_y = s_br_y[np.newaxis, :]

            ind_x1 = (center_x > s_tl_x) & (center_x < s_br_x)
            ind_x2 = (center_x < s_tl_x) & (center_x > s_br_x)
            ind_y1 = (center_y > s_tl_y) & (center_y < s_br_y)
            ind_y2 = (center_y < s_tl_y) & (center_y > s_br_y)
            ind_cls = (
                centers[:, 2][:, np.newaxis] - s_detections[:, -1][np.newaxis, :]
            ) == 0
            ind_s_new_score = (
                np.max(
                    (
                        ((ind_x1 + 0) & (ind_y1 + 0) & (ind_cls + 0))
                        | ((ind_x1 + 0) & (ind_y2 + 0) & (ind_cls + 0))
                        | ((ind_x2 + 0) & (ind_y2 + 0) & (ind_cls + 0))
                    ),
                    axis=0,
                )
                == 1
            )
            index_s_new_score = np.argmax(
                (
                    ((ind_x1 + 0) & (ind_y1 + 0) & (ind_cls + 0))
                    | ((ind_x1 + 0) & (ind_y2 + 0) & (ind_cls + 0))
                    | ((ind_x2 + 0) & (ind_y2 + 0) & (ind_cls + 0))
                )[:, ind_s_new_score],
                axis=0,
            )
            s_corner_score = s_temp_score[ind_s_new_score]
            s_center_score = centers[index_s_new_score, 3]
            s_detections[:, 6][ind_s_new_score] = (
                s_corner_score * 3 + s_center_score
            ) / 4

            # pro-process for large bounding box
            l_tl_x = (2 * l_detections[:, 0] + l_detections[:, 4]) / 3
            l_br_x = (l_detections[:, 0] + 2 * l_detections[:, 4]) / 3
            l_tl_y = (2 * l_detections[:, 1] + l_detections[:, 5]) / 3
            l_br_y = (l_detections[:, 1] + 2 * l_detections[:, 5]) / 3

            l_temp_score = copy.copy(l_detections[:, 6])
            l_detections[:, 6] = -1

            center_x = centers[:, 0][:, np.newaxis]
            center_y = centers[:, 1][:, np.newaxis]
            l_tl_x = l_tl_x[np.newaxis, :]
            l_br_x = l_br_x[np.newaxis, :]
            l_tl_y = l_tl_y[np.newaxis, :]
            l_br_y = l_br_y[np.newaxis, :]

            ind_x1 = (center_x > l_tl_x) & (center_x < l_br_x)
            ind_x2 = (center_x < l_tl_x) & (center_x > l_br_x)
            ind_y1 = (center_y > l_tl_y) & (center_y < l_br_y)
            ind_y2 = (center_y < l_tl_y) & (center_y > l_br_y)
            ind_cls = (
                centers[:, 2][:, np.newaxis] - l_detections[:, -1][np.newaxis, :]
            ) == 0
            ind_l_new_score = (
                np.max(
                    (
                        ((ind_x1 + 0) & (ind_y1 + 0) & (ind_cls + 0))
                        | ((ind_x1 + 0) & (ind_y2 + 0) & (ind_cls + 0))
                        | ((ind_x2 + 0) & (ind_y2 + 0) & (ind_cls + 0))
                    ),
                    axis=0,
                )
                == 1
            )
            index_l_new_score = np.argmax(
                (
                    ((ind_x1 + 0) & (ind_y1 + 0) & (ind_cls + 0))
                    | ((ind_x1 + 0) & (ind_y2 + 0) & (ind_cls + 0))
                    | ((ind_x2 + 0) & (ind_y2 + 0) & (ind_cls + 0))
                )[:, ind_l_new_score],
                axis=0,
            )
            l_corner_score = l_temp_score[ind_l_new_score]
            l_center_score = centers[index_l_new_score, 3]
            l_detections[:, 6][ind_l_new_score] = (
                l_corner_score * 3 + l_center_score
            ) / 4

            detections = np.concatenate([l_detections, s_detections], axis=0)
            detections = detections[np.argsort(-detections[:, 6])]
            classes = detections[..., -1]

            # reject detections with negative scores
            keep_inds = detections[:, 6] > -1
            detections = detections[keep_inds]
            classes = classes[keep_inds]

            detections = np.expand_dims(detections, axis=0)

            for j in range(num_classes):
                inds = classes == j
                top_preds[j + 1] = detections[i, inds, :].astype(np.float32).tolist()
            ret.append(top_preds)

        for j in range(1, num_classes + 1):
            ret[0][j] = np.array(ret[0][j], dtype=np.float32).reshape(-1, 11)

        return ret[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0
            ).astype(np.float32)
            if len(self.scales) > 1 or self.opt.nms:
                soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack([results[j][:, 6] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = results[j][:, 6] >= thresh
                results[j] = results[j][keep_inds]
        return results
