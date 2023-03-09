import os
import time

import cv2
import numpy as np
import torch
import tqdm

from gknet.datasets.dataset_factory import dataset_factory
from gknet.detectors.detector_factory import detector_factory
from gknet.logger import Logger
from gknet.opts import opts
from gknet.utils.utils import AverageMeter


class PreprocessDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.images = dataset.images
        if opt.dataset.split("_")[0] == "jac":
            self.load_image_func = dataset.coco.loadImgs
        self.img_dir = dataset.img_dir
        self.pre_process_func = pre_process_func
        self.opt = opt

    def __getitem__(self, index):
        img_id = self.images[index]
        if self.opt.dataset.split("_")[0] == "jac":
            img_info = self.load_image_func(ids=[img_id])[0]
            img_path = os.path.join(self.img_dir, img_info["file_name"])
            image = cv2.imread(img_path)
        elif self.opt.dataset.split("_")[0] == "cornell":
            img_path = os.path.join(self.img_dir, img_id.split("\n")[0] + ".png")
            image = cv2.imread(img_path)
            image = cv2.resize(image, (256, 256))
        images, meta = {}, {}
        for scale in opt.test_scales:
            images[scale], meta[scale] = self.pre_process_func(image, scale)
        return img_id, {"images": images, "image": image, "meta": meta}

    def __len__(self):
        return len(self.images)


def test(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    logger = Logger(opt)
    logger.write_json(opt, stdout=True)

    split = "test" if not opt.trainval else "test"
    dataset = Dataset(opt, split)
    detector = detector_factory[opt.task](opt)

    data_loader = torch.utils.data.DataLoader(
        PreprocessDataset(opt, dataset, detector.pre_process),
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        prefetch_factor=opt.prefetch_factor,
    )

    results = {}
    time_stats = ["tot", "load", "pre", "net", "dec", "post", "merge"]
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for img_id, pre_processed_images in tqdm.tqdm(
        data_loader, desc=opt.exp_id, total=len(dataset)
    ):
        ret = detector.run(pre_processed_images)
        if opt.dataset.split("_")[0] == "jac":
            results[img_id.numpy().astype(np.int32)[0]] = ret["results"]
        elif opt.dataset.split("_")[0] == "cornell":
            results[img_id[0].split("\n")[0]] = ret["results"]

        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])

    # compute time stats
    stats = {}
    for k, v in avg_time_stats.items():
        stats[f"{k}_avg"] = v.avg
        stats[f"{k}_sum"] = v.sum
    logger.write_json(stats, stdout=True)

    if opt.task.split("_")[0] == "dbmctdet":
        start = time.time()
        success, total = dataset.run_eval_db_middle(results)
        end = time.time()
        logger.write_json(
            {
                "task": opt.task,
                "success": success,
                "total": total,
                "wall_time": end - start,
            },
            stdout=True,
        )


if __name__ == "__main__":
    opt = opts().parse()
    test(opt)
