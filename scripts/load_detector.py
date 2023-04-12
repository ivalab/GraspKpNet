"""
This script will load a detector and then immediately exit. This purpose of this
script is to populate the `.cache` directory with detector weights in the docker
container. This will help improve boot times on subsequent docker runs when this
image is used without volume mounts.

```bash
python scripts/load_detector.py dbmctdet_cornell --load_model models/model_dla34_cornell.pth
```
"""

from gknet.detectors.detector_factory import detector_factory
from gknet.opts import opts

if __name__ == "__main__":
    opt = opts().init()
    # if we don't have a gpu, we're going to fail, but we don't care
    try:
        detector_factory[opt.task](opt)
    except Exception as e:
        pass
    print("loaded detector")
