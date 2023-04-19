"""
A script to annotate a single image with the GraspKpNet detector.
It accepts a directory with a depth and rgb image and saves the annotated image
to the same directory.

You can save images from a gazebo simulation with the following command:

```bash
rosrun image_view image_view image:=/camera/color/image_raw
rosrun image_view image_view image:=/camera/aligned_depth_to_color/image_raw
```

Save these as rgb.jpg and depth.jpg in a directory and run this script:

```bash
python scripts/demo_annotate.py dbmctdet_cornell \
    --load_model models/model_dla34_cornell.pth \
    --dataset_dir data/test_sim
```

For good measure, we can also run this via docker:

```
# NOTE: you may need to run xhost commands to allow access to local x11, but
# this wasn't necessary on a WSL2 machine
xhost +local:docker

docker compose run --rm gpu \
python scripts/demo_annotate.py dbmctdet_cornell \
    --load_model models/model_dla34_cornell.pth \
    --dataset_dir data/test_sim
```
"""
import os

import cv2
import numpy as np

from gknet.detectors.detector_factory import detector_factory
from gknet.opts import opts
from pathlib import Path

def _normalize(x, min_d, max_min_diff):
    return 255 * (x - min_d) / max_min_diff


def demo(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str
    opt.debug = max(opt.debug, 2)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    root = Path(opt.dataset_dir)
    print(root)

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

    detector.run(inp, save_dir=(root / "out").as_posix())

if __name__ == "__main__":
    opt = opts().init()
    demo(opt)
