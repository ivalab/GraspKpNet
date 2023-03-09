import numpy as np
import torch

from gknet.models.losses import (
    FocalLoss,
    NormRegL1Loss,
    RegL1Loss,
    RegLoss,
    RegWeightedL1Loss,
    TagLoss_2,
)
from gknet.models.utils import _sigmoid
from gknet.utils.debugger import Debugger
from gknet.utils.oracle_utils import gen_oracle_map

from .base_trainer import BaseTrainer


class DbMCtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(DbMCtdetLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = (
            RegL1Loss()
            if opt.regr_loss == "l1"
            else RegLoss()
            if opt.reg_loss == "sl1"
            else None
        )
        self.crit_tag = TagLoss_2()
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        focal_loss, pull_loss, push_loss, reg_loss = 0, 0, 0, 0
        lm_focal_loss, rm_focal_loss, ct_focal_loss = 0, 0, 0
        lm_reg_loss, rm_reg_loss, ct_reg_loss = 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output["lm"] = _sigmoid(output["lm"])
                output["rm"] = _sigmoid(output["rm"])
                output["ct"] = _sigmoid(output["ct"])

            if opt.eval_oracle_lm:
                output["lm"] = batch["lm"]
            if opt.eval_oracle_rm:
                output["rm"] = batch["rm"]
            if opt.eval_oracle_ct:
                output["ct"] = batch["ct"]
            if opt.eval_oracle_ae:
                output["lm_tag"] = torch.from_numpy(
                    gen_oracle_map(
                        batch["lm_tag"].detach().cpu().numpy(),
                        batch["lm_tag"].detach().cpu().numpy(),
                        output["lm_tag"].shape[3],
                        output["lm_tag"].shape[2],
                    )
                ).to(opt.device)
                output["rm_tag"] = torch.from_numpy(
                    gen_oracle_map(
                        batch["rm_tag"].detach().cpu().numpy(),
                        batch["rm_tag"].detach().cpu().numpy(),
                        output["rm_tag"].shape[3],
                        output["rm_tag"].shape[2],
                    )
                ).to(opt.device)
            if opt.eval_oracle_offset:
                output["lm_reg"] = torch.from_numpy(
                    gen_oracle_map(
                        batch["lm_reg"].detach().cpu().numpy(),
                        batch["lm_tag"].detach().cpu().numpy(),
                        output["lm_reg"].shape[3],
                        output["lm_reg"].shape[2],
                    )
                ).to(opt.device)
                output["rm_reg"] = torch.from_numpy(
                    gen_oracle_map(
                        batch["rm_reg"].detach().cpu().numpy(),
                        batch["rm_tag"].detach().cpu().numpy(),
                        output["rm_reg"].shape[3],
                        output["rm_reg"].shape[2],
                    )
                ).to(opt.device)
                output["ct_reg"] = torch.from_numpy(
                    gen_oracle_map(
                        batch["ct_reg"].detach().cpu().numpy(),
                        batch["ct_tag"].detach().cpu().numpy(),
                        output["ct_reg"].shape[3],
                        output["ct_reg"].shape[2],
                    )
                ).to(opt.device)

            # focal loss
            lm_focal_loss = self.crit(output["lm"], batch["lm"]) / opt.num_stacks
            rm_focal_loss = self.crit(output["rm"], batch["rm"]) / opt.num_stacks
            ct_focal_loss = self.crit(output["ct"], batch["ct"]) / opt.num_stacks
            focal_loss += lm_focal_loss
            focal_loss += rm_focal_loss
            focal_loss += ct_focal_loss

            # tag loss
            pull, push = self.crit_tag(
                output["rm_tag"],
                output["lm_tag"],
                batch["rm_tag"],
                batch["lm_tag"],
                batch["reg_mask"],
            )
            pull_loss += opt.pull_weight * pull / opt.num_stacks
            push_loss += opt.push_weight * push / opt.num_stacks

            # reg loss
            lm_reg_loss = (
                opt.regr_weight
                * self.crit_reg(
                    output["lm_reg"],
                    batch["reg_mask"],
                    batch["lm_tag"],
                    batch["lm_reg"],
                )
                / opt.num_stacks
            )
            rm_reg_loss = (
                opt.regr_weight
                * self.crit_reg(
                    output["rm_reg"],
                    batch["reg_mask"],
                    batch["rm_tag"],
                    batch["rm_reg"],
                )
                / opt.num_stacks
            )
            ct_reg_loss = (
                opt.regr_weight
                * self.crit_reg(
                    output["ct_reg"],
                    batch["reg_mask"],
                    batch["ct_tag"],
                    batch["ct_reg"],
                )
                / opt.num_stacks
            )
            reg_loss += lm_reg_loss
            reg_loss += rm_reg_loss
            reg_loss += ct_reg_loss

        loss = focal_loss + pull_loss + push_loss + reg_loss
        loss_stats = {
            "loss": loss,
            "focal_loss": focal_loss,
            "pull_loss": pull_loss,
            "push_loss": push_loss,
            "reg_loss": reg_loss,
            "lm_focal_loss": lm_focal_loss,
            "rm_focal_loss": rm_focal_loss,
            "ct_focal_loss": ct_focal_loss,
            "lm_reg_loss": lm_reg_loss,
            "rm_reg_loss": rm_reg_loss,
            "ct_reg_loss": ct_reg_loss,
        }
        return loss, loss_stats


class DbMCtdetTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(DbMCtdetTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = [
            "loss",
            "focal_loss",
            "pull_loss",
            "push_loss",
            "reg_loss",
            "lm_focal_loss",
            "rm_focal_loss",
            "ct_focal_loss",
            "lm_reg_loss",
            "rm_reg_loss",
            "ct_reg_loss",
        ]
        loss = DbMCtdetLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        """haven't implemented"""

    def save_result(self, output, batch, results):
        """haven't implemented"""
