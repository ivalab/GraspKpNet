# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import _tranpose_and_gather_feat


def _ori_loss(bl_reg, br_reg, bl_tag, br_tag, ori_reg_gt, mask, width, height):
    bl_x = ((bl_tag % (width * height)) % width).int().float()
    bl_y = ((bl_tag % (width * height)) / width).int().float()
    br_x = ((br_tag % (width * height)) % width).int().float()
    br_y = ((br_tag % (width * height)) / width).int().float()

    bl_xs = bl_x + bl_reg[..., 0]
    bl_ys = bl_y + bl_reg[..., 1]
    br_xs = br_x + br_reg[..., 0]
    br_ys = br_y + br_reg[..., 1]

    ori_reg_pr = torch.atan(-(br_ys - bl_ys) / (br_xs - bl_xs))

    ori_loss = nn.functional.smooth_l1_loss(
        ori_reg_pr * mask.float(), ori_reg_gt * mask.float(), size_average=False
    )
    ori_loss = ori_loss / (mask.float().sum() + 1e-4)

    return ori_loss


def _ip_loss(tl_reg, bl_reg, br_reg, tl_tag, bl_tag, br_tag, mask, width, height):
    num = mask.sum(dim=1, keepdim=True).float()

    tl_x = ((tl_tag % (width * height)) % width).int().float()
    tl_y = ((tl_tag % (width * height)) / width).int().float()
    bl_x = ((bl_tag % (width * height)) % width).int().float()
    bl_y = ((bl_tag % (width * height)) / width).int().float()
    br_x = ((br_tag % (width * height)) % width).int().float()
    br_y = ((br_tag % (width * height)) / width).int().float()

    tl_xs = tl_x + tl_reg[..., 0]
    tl_ys = tl_y + tl_reg[..., 1]
    bl_xs = bl_x + bl_reg[..., 0]
    bl_ys = bl_y + bl_reg[..., 1]
    br_xs = br_x + br_reg[..., 0]
    br_ys = br_y + br_reg[..., 1]

    w = torch.sqrt(torch.pow(bl_xs - br_xs, 2) + torch.pow(bl_ys - br_ys, 2))
    h = torch.sqrt(torch.pow(bl_xs - tl_xs, 2) + torch.pow(bl_ys - tl_ys, 2))
    inner_product = (bl_xs - tl_xs) * (bl_xs - br_xs) + (bl_ys - tl_ys) * (
        bl_ys - br_ys
    )
    inner_product = inner_product / w
    inner_product = inner_product / h
    inner_product = inner_product / (num + 1e-4)
    inner_product = torch.abs(inner_product)
    inner_product = inner_product[mask]
    ip_loss = inner_product.sum()

    return ip_loss


def _ae_loss_2(tag0, tag1, mask):
    num = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    tag_mean = (tag0 + tag1) / 2

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask.bool()].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask.bool()].sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num = num.unsqueeze(2)
    num2 = (num - 1) * num
    if len(tag_mean.size()) < 2:
        tag_mean = tag_mean.unsqueeze(0)
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push


def _ae_loss_3(tag0, tag1, tag2, mask):
    num = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()
    tag2 = tag2.squeeze()

    tag_mean = (tag0 + tag1 + tag2) / 3

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    tag2 = torch.pow(tag2 - tag_mean, 2) / (num + 1e-4)
    tag2 = tag2[mask].sum()
    pull = tag0 + tag1 + tag2

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num = num.unsqueeze(2)
    num2 = (num - 1) * num
    if len(tag_mean.size()) < 2:
        tag_mean = tag_mean.unsqueeze(0)
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push


def _slow_neg_loss(pred, gt):
    """focal loss from CornerNet"""
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss(pred, gt):
    """Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    """
    pred_inds = pred.lt(0).float()

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    num_pos = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -= all_loss
    return loss


def _slow_reg_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr[mask]
    gt_regr = gt_regr[mask]

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def _reg_loss(regr, gt_regr, mask):
    """L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class OriRegLoss(nn.Module):
    """nn.Module wrapper for orientation regression loss"""

    def __init__(self):
        super(OriRegLoss, self).__init__()
        self.ori_loss = _ori_loss

    def forward(self, bl_reg, br_reg, bl_tag, br_tag, ori_reg, mask):
        batch, _, height, width = bl_reg.size()

        bl_reg = _tranpose_and_gather_feat(bl_reg, bl_tag)
        br_reg = _tranpose_and_gather_feat(br_reg, br_tag)

        return self.ori_loss(
            bl_reg, br_reg, bl_tag, br_tag, ori_reg, mask, width, height
        )


class InnerProductLoss(nn.Module):
    """nn.Module wrapper for inner product loss"""

    def __init__(self):
        super(InnerProductLoss, self).__init__()
        self.ip_loss = _ip_loss

    def forward(self, tl_reg, bl_reg, br_reg, tl_tag, bl_tag, br_tag, mask):
        batch, _, height, width = tl_reg.size()

        tl_reg = _tranpose_and_gather_feat(tl_reg, tl_tag)
        bl_reg = _tranpose_and_gather_feat(bl_reg, bl_tag)
        br_reg = _tranpose_and_gather_feat(br_reg, br_tag)

        return self.ip_loss(
            tl_reg, bl_reg, br_reg, tl_tag, bl_tag, br_tag, mask, width, height
        )


class TagLoss_2(nn.Module):
    """nn.Module wrapper for pull and push loss"""

    def __init__(self):
        super(TagLoss_2, self).__init__()
        self.ae_loss = _ae_loss_2

    def forward(self, tag1, tag2, ind1, ind2, mask):
        tag1 = _tranpose_and_gather_feat(tag1, ind1)
        tag2 = _tranpose_and_gather_feat(tag2, ind2)

        return self.ae_loss(tag1, tag2, mask)


class TagLoss_3(nn.Module):
    """nn.Module wrapper for pull and push loss"""

    def __init__(self):
        super(TagLoss_3, self).__init__()
        self.ae_loss = _ae_loss_3

    def forward(self, tag1, tag2, tag3, ind1, ind2, ind3, mask):
        tag1 = _tranpose_and_gather_feat(tag1, ind1)
        tag2 = _tranpose_and_gather_feat(tag2, ind2)
        tag3 = _tranpose_and_gather_feat(tag3, ind3)

        return self.ae_loss(tag1, tag2, tag3, mask)


class FocalLoss(nn.Module):
    """nn.Module wrapper for focal loss"""

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class RegLoss(nn.Module):
    """Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
    """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        # loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class NormRegL1Loss(nn.Module):
    def __init__(self):
        super(NormRegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        pred = pred / (target + 1e-4)
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction="elementwise_mean")
        return loss


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction="elementwise_mean")


# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction="elementwise_mean")


def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0])
        )
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0])
        )
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1])
        )
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1])
        )
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res
