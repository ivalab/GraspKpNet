import os

import cv2
import numpy as np

from gknet.detectors.detector_factory import detector_factory
from gknet.opts import opts
from pathlib import Path

time_stats = ["tot", "load", "pre", "net", "dec", "post", "merge"]


def _normalize(x, min_d, max_min_diff):
    return 255 * (x - min_d) / max_min_diff


def demo(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    root = Path("data/test_sim")

    depth_image = cv2.imread(str(root / "depth.jpg"), cv2.IMREAD_ANYDEPTH)
    color_image = cv2.imread(str(root / "rgb.jpg"))

    min_depth = np.min(depth_image[:, :])
    max_depth = np.max(depth_image[:, :])
    max_min_diff = max_depth - min_depth

    normalize = np.vectorize(_normalize, otypes=[np.float32])
    depth_image = normalize(depth_image, min_depth, max_min_diff)

    inp = color_image.copy()
    inp[:, :, 2] = depth_image
    inp = inp[:, :, ::-1]

    ret = detector.run(inp)

if __name__ == "__main__":
    opt = opts().init()
    demo(opt)
