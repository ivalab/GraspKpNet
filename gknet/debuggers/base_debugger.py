import numpy as np
import torch

from gknet.models.model import create_model, load_model


class BaseDebugger(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device("cuda")
        else:
            opt.device = torch.device("cpu")

        print("Creating model...")
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True

    def forward(self, images):
        raise NotImplementedError

    def debug(self, detections, targets):
        raise NotImplementedError

    def process(self, images, return_time=False):
        raise NotImplementedError

    def post_process(self, dets, meta, scale=1):
        raise NotImplementedError

    def merge_outputs(self, detections):
        raise NotImplementedError

    def run(self, targets):
        images = targets["input"]
        images = images.to(self.opt.device)

        detections = self.forward(images)

        (
            min_tl,
            max_tl,
            min_bl,
            max_bl,
            min_br,
            max_br,
            avg,
            flag_found_match,
        ) = self.debug(detections, targets, self.opt.ae_threshold)

        min = np.minimum(np.minimum(min_tl, min_bl), min_br)
        max = np.maximum(np.maximum(max_tl, max_bl), max_br)

        return min, max, avg, flag_found_match

    def statistics(self, targets):
        images = targets["input"]
        images = images.to(self.opt.device)

        dets, centers = self.process(
            images,
            self.opt.nms_kernel,
            self.opt.ae_threshold,
            self.opt.K,
            self.opt.num_dets,
        )

        detections = []
        dets = self.post_process(
            dets,
            centers,
            self.opt.num_classes,
            self.opt.bbox_size_threshold,
            self.opt.ori_threshold,
        )
        detections.append(dets)

        results = self.merge_outputs(detections)

        return {"results": results}
