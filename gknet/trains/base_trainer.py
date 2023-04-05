import time

import torch
import tqdm

from gknet.models.data_parallel import DataParallel
from gknet.utils.utils import AverageMeter


class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch["input"])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats


class BaseTrainer:
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModleWithLoss(model, self.loss)

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus, chunk_sizes=chunk_sizes
            ).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == "train":
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        start = time.time()

        with tqdm.tqdm(
            data_loader, total=num_iters, desc=f"{opt.task}/{opt.exp_id}"
        ) as batches, tqdm.tqdm(bar_format="{desc}") as loss_bar:
            for iter_id, batch in enumerate(batches):
                if iter_id >= num_iters:
                    break

                for k in batch:
                    if k == "meta":
                        continue
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

                output, loss, loss_stats = model_with_loss(batch)
                loss = loss.mean()
                if phase == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                for l in avg_loss_stats:
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch["input"].size(0)
                    )

                # format the loss description and update the bar, there are a
                # total of 11 loss terms
                desc = "|".join([f"{k} {v.avg:.4f}" for k, v in avg_loss_stats.items()])
                loss_bar.set_description_str(desc, refresh=False)
                loss_bar.update()

                if opt.print_iter > 0:
                    if iter_id % opt.print_iter == 0:
                        print(f"{opt.task}/{opt.exp_id}| {desc}")

                if opt.debug > 0:
                    self.debug(batch, output, iter_id)

                if opt.test:
                    self.save_result(output, batch, results)
                del output, loss, loss_stats

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        end = time.time()
        ret["time"] = (end - start) / 60.0
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        return self.run_epoch("val", epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch("train", epoch, data_loader)
