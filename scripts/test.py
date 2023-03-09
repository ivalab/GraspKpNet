import os

import cv2
import numpy as np
import torch
from progress.bar import Bar

from gknet.datasets.dataset_factory import dataset_factory
from gknet.detectors.detector_factory import detector_factory
from gknet.logger import Logger
from gknet.opts import opts
from gknet.utils.utils import AverageMeter


class PrefetchDataset(torch.utils.data.Dataset):
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


def prefetch_test(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)
    Detector = detector_factory[opt.task]

    split = "test" if not opt.trainval else "test"
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    data_loader = torch.utils.data.DataLoader(
        PrefetchDataset(opt, dataset, detector.pre_process),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    results = {}
    num_iters = len(dataset)
    bar = Bar("{}".format(opt.exp_id), max=num_iters)
    time_stats = ["tot", "load", "pre", "net", "dec", "post", "merge"]
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
        ret = detector.run(pre_processed_images)
        if opt.dataset.split("_")[0] == "jac":
            results[img_id.numpy().astype(np.int32)[0]] = ret["results"]
        elif opt.dataset.split("_")[0] == "cornell":
            results[img_id[0].split("\n")[0]] = ret["results"]
        Bar.suffix = "[{0}/{1}]|Tot: {total:} |ETA: {eta:} ".format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td
        )
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + "|{} {tm.val:.3f}s ({tm.avg:.3f}s) ".format(
                t, tm=avg_time_stats[t]
            )
        bar.next()
    bar.finish()

    if opt.task.split("_")[0] == "dbmctdet":
        dataset.run_eval_db_middle(results)


def test(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)
    Detector = detector_factory[opt.task]

    split = "val" if not opt.trainval else "test"
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    results = {}
    num_iters = len(dataset)
    bar = Bar("{}".format(opt.exp_id), max=num_iters)
    time_stats = ["tot", "load", "pre", "net", "dec", "post", "merge"]
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind in range(num_iters):
        img_id = dataset.images[ind]
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(dataset.img_dir, img_info["file_name"])

        if opt.task == "ddd":
            ret = detector.run(img_path, img_info["calib"])
        else:
            ret = detector.run(img_path)

        results[img_id] = ret["results"]

        Bar.suffix = "[{0}/{1}]|Tot: {total:} |ETA: {eta:} ".format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td
        )
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + "|{} {:.3f} ".format(t, avg_time_stats[t].avg)
        bar.next()
    bar.finish()
    dataset.run_eval(results, opt.save_dir)


if __name__ == "__main__":
    opt = opts().parse()
    if opt.not_prefetch_test:
        test(opt)
    else:
        prefetch_test(opt)
