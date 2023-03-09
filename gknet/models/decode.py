



import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat
import numpy as np

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _left_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    shape = heat.shape 
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape) 

def _right_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    shape = heat.shape 
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i +1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape) 

def _top_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    heat = heat.transpose(3, 2) 
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)

def _bottom_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    heat = heat.transpose(3, 2) 
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i + 1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)

def _h_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * _left_aggregate(heat) + \
           aggr_weight * _right_aggregate(heat) + heat

def _v_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * _top_aggregate(heat) + \
           aggr_weight * _bottom_aggregate(heat) + heat

'''
# Slow for large number of categories
def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
'''
def _topk_channel(scores, K=40):
      batch, cat, height, width = scores.size()
      
      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = (topk_inds / width).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _topk_original(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def dbmctdet_decode(lm_heat, rm_heat, ct_heat, lm_tag, rm_tag, lm_reg, rm_reg, ct_reg, \
                       kernel = 1, ae_threshold = 1, scores_thresh = 0.1, center_thresh = 0.1, scores_weight = 1.0,
                       K = 100, num_dets = 100):
    batch, cat, height, width = lm_heat.size()

    # perform nms on heatmaps
    lm_heat = _nms(lm_heat, kernel=kernel)
    rm_heat = _nms(rm_heat, kernel=kernel)
    # ct_heat = _nms(ct_heat, kernel=kernel)

    lm_heat[lm_heat > 1] = 1
    rm_heat[rm_heat > 1] = 1

    lm_scores, lm_inds, lm_clses, lm_ys, lm_xs = _topk(lm_heat, K=K)
    rm_scores, rm_inds, rm_clses, rm_ys, rm_xs = _topk(rm_heat, K=K)

    lm_ys = lm_ys.view(batch, K, 1).expand(batch, K, K)
    lm_xs = lm_xs.view(batch, K, 1).expand(batch, K, K)
    rm_ys = rm_ys.view(batch, 1, K).expand(batch, K, K)
    rm_xs = rm_xs.view(batch, 1, K).expand(batch, K, K)

    lm_clses = lm_clses.view(batch, K, 1).expand(batch, K, K)
    rm_clses = rm_clses.view(batch, 1, K).expand(batch, K, K)

    lm_tag = _tranpose_and_gather_feat(lm_tag, lm_inds)
    lm_tag = lm_tag.view(batch, K, 1)
    rm_tag = _tranpose_and_gather_feat(rm_tag, rm_inds)
    rm_tag = rm_tag.view(batch, 1, K)

    box_ct_xs = ((lm_xs + rm_xs + 0.5) / 2).long()
    box_ct_ys = ((lm_ys + rm_ys + 0.5) / 2).long()
    ct_inds = box_ct_ys * width + box_ct_xs
    ct_inds = ct_inds.view(batch, -1)
    ct_heat = ct_heat.view(batch, -1, 1)
    ct_scores = _gather_feat(ct_heat, ct_inds)

    lm_scores = lm_scores.view(batch, K, 1).expand(batch, K, K)
    rm_scores = rm_scores.view(batch, 1, K).expand(batch, K, K)
    ct_scores = ct_scores.view(batch, K, K)
    scores = (scores_weight * (lm_scores + rm_scores) + 0.5 * ct_scores) / (2 * scores_weight + 0.5)
    # scores = (lm_scores + rm_scores) / 2

    cls_inds = lm_clses != rm_clses
    cls_inds = cls_inds > 0
    dists = torch.abs(lm_tag - rm_tag)
    dist_inds = (dists > ae_threshold)
    sc_inds = (lm_scores < scores_thresh) + (rm_scores < scores_thresh) + (ct_scores < center_thresh)
    sc_inds = sc_inds > 0

    scores[dist_inds] = -1
    scores[sc_inds] = -1
    scores[cls_inds] = -1
    # if dist_inds.long().cpu().numpy().sum() > 9900:
    #     print("distance policy filters {} cases".format(dist_inds.long().cpu().numpy().sum()))
    # if sc_inds.long().cpu().numpy().sum() > 9900:
    #     print("score policy filters {} cases".format(sc_inds.long().cpu().numpy().sum()))
    # if cls_inds.long().cpu().numpy().sum() > 9900:
    #     print("class policy filters {} cases".format(cls_inds.long().cpu().numpy().sum()))

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    if lm_reg is not None and rm_reg is not None:
        lm_reg = _tranpose_and_gather_feat(lm_reg, lm_inds)
        lm_reg = lm_reg.view(batch, K, 1, 2)
        rm_reg = _tranpose_and_gather_feat(rm_reg, rm_inds)
        rm_reg = rm_reg.view(batch, 1, K, 2)

        lm_xs = lm_xs + lm_reg[..., 0]
        lm_ys = lm_ys + lm_reg[..., 1]
        rm_xs = rm_xs + rm_reg[..., 0]
        rm_ys = rm_ys + rm_reg[..., 1]

    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((lm_xs, lm_ys, rm_xs, rm_ys), dim=3)
    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses = lm_clses.contiguous().view(batch, -1, 1)
    clses = _gather_feat(clses, inds).float()

    lm_xs = lm_xs.contiguous().view(batch, -1, 1)
    lm_xs = _gather_feat(lm_xs, inds).float()
    lm_ys = lm_ys.contiguous().view(batch, -1, 1)
    lm_ys = _gather_feat(lm_ys, inds).float()
    rm_xs = rm_xs.contiguous().view(batch, -1, 1)
    rm_xs = _gather_feat(rm_xs, inds).float()
    rm_ys = rm_ys.contiguous().view(batch, -1, 1)
    rm_ys = _gather_feat(rm_ys, inds).float()

    detections = torch.cat([bboxes, scores, lm_xs, lm_ys, rm_xs, rm_ys, clses], dim=2)

    return detections

