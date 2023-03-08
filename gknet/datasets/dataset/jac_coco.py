from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import numpy as np
import json
import os
from progress.bar import Bar
import cv2
import math
import copy
import os

from gknet.datasets.dataset.utils import _bbox_overlaps, rotate_bbox

import torch.utils.data as data


class JAC_COCO_36(data.Dataset):
    num_classes = 36
    num_ct_classes = 1
    default_resolution = [512, 512]
    mean = np.array([0.711156, 0.709059, 0.197465], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.060364, 0.067899, 0.087001], dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(JAC_COCO_36, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, "Jacquard")

        if split:
            self.img_dir = os.path.join(
                self.data_dir,
                "coco/512_cnt_angle",
                split,
                "grasps_{}2018".format(split),
            )
            if opt.flag_test:
                self.annot_path = os.path.join(
                    self.data_dir,
                    "coco/512_cnt_angle",
                    split,
                    "instances_grasps_{}2018.json",
                ).format(split)
            else:
                self.annot_path = os.path.join(
                    self.data_dir,
                    "coco/512_cnt_angle",
                    split,
                    "instances_grasps_{}2018_filter.json",
                ).format(split)
        self.max_objs = 128
        self.avg_h = 20.0
        self.class_name = [
            "__background__",
            "orient01",
            "orient02",
            "orient03",
            "orient04",
            "orient05",
            "orient06",
            "orient07",
            "orient08",
            "orient09",
            "orient10",
            "orient11",
            "orient12",
            "orient13",
            "orient14",
            "orient15",
            "orient16",
            "orient17",
            "orient18",
            "orient19",
            "orient20",
            "orient21",
            "orient22",
            "orient23",
            "orient24",
            "orient25",
            "orient26",
            "orient27",
            "orient28",
            "orient29",
            "orient30",
            "orient31",
            "orient32",
            "orient33",
            "orient34",
            "orient35",
            "orient36",
        ]
        self._valid_ids = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
        ]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}  # rx

        self.voc_color = [
            (v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32)
            for v in range(1, self.num_classes + 1)
        ]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array(
            [
                [-0.58752847, -0.69563484, 0.41340352],
                [-0.5832747, 0.00994535, -0.81221408],
                [-0.56089297, 0.71832671, 0.41158938],
            ],
            dtype=np.float32,
        )

        self.split = split
        self.opt = opt

        print("==> initializing jacquard dataset in coco format {} data.".format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()[:]
        self.num_samples = len(self.images)

        print("Loaded {} {} samples".format(split, self.num_samples))

    def __len__(self):
        return self.num_samples

    def run_eval_db_middle(self, results):
        dataset_size = len(results)
        nm_suc_case = 0
        bar = Bar("jacquard evaluation", max=dataset_size)

        for image_id, result in results.items():
            Bar.suffix = "[{0}/{1}]|Tot: {total:} |ETA: {eta:} ".format(
                image_id, dataset_size, total=bar.elapsed_td, eta=bar.eta_td
            )

            # get the associated groundtruth for predicted_bbox
            ann_ids = self.coco.getAnnIds(imgIds=[image_id])
            annotations = self.coco.loadAnns(ids=ann_ids)

            # collect the ground-truth grasp detection
            boxes_gt = []
            for anno in annotations:
                # skip if predicted bbox is 37 or there is no bbox predicted for this category id
                if anno["category_id"] == 37:
                    continue

                bbox_gt = anno["bbox"]
                x_min, y_min, w, h = bbox_gt[0], bbox_gt[1], bbox_gt[2], bbox_gt[3]

                c_x, c_y = x_min + w / 2, y_min + h / 2

                if w < 0 or h < 0:
                    continue

                angle = bbox_gt[4] / 180 * np.pi
                boxes_gt.append(
                    [c_x - w / 2, c_y - h / 2, c_x + w / 2, c_y + h / 2, angle]
                )
            boxes_gt = np.array(boxes_gt)

            # collect the predicted grasp detection
            boxes_pr = []
            boxes_s_pr = []
            for category_id, pr_bboxs in result.items():
                if len(pr_bboxs) == 0:
                    continue
                for pr_bbox in pr_bboxs:
                    boxes_pr.append(pr_bbox[:4])
                    boxes_s_pr.append(pr_bbox[-1])

            bbox_pr = []
            max_s = 0.0
            for i in range(len(boxes_s_pr)):
                x_c, y_c = (boxes_pr[i][0] + boxes_pr[i][2]) / 2, (
                    boxes_pr[i][1] + boxes_pr[i][3]
                ) / 2

                w = np.sqrt(
                    np.power(boxes_pr[i][0] - boxes_pr[i][2], 2)
                    + np.power(boxes_pr[i][1] - boxes_pr[i][3], 2)
                )

                if boxes_pr[i][0] == boxes_pr[i][2] and boxes_pr[i][1] < boxes_pr[i][3]:
                    angle = np.pi / 2
                elif (
                    boxes_pr[i][0] == boxes_pr[i][2] and boxes_pr[i][1] > boxes_pr[i][3]
                ):
                    angle = -np.pi / 2
                elif boxes_pr[i][1] == boxes_pr[i][3]:
                    angle = 0
                else:
                    angle = np.arctan(
                        (boxes_pr[i][3] - boxes_pr[i][1])
                        / (boxes_pr[i][2] - boxes_pr[i][0])
                    )

                if max_s < boxes_s_pr[i]:
                    max_s = boxes_s_pr[i]
                    bbox_pr = []
                    bbox_pr.append(
                        [
                            x_c - w / 2,
                            y_c - self.avg_h / 2,
                            x_c + w / 2,
                            y_c + self.avg_h / 2,
                            angle,
                        ]
                    )

            if len(bbox_pr) == 0:
                continue

            bbox_pr = np.array(bbox_pr)

            overlaps = _bbox_overlaps(
                np.ascontiguousarray(bbox_pr[:, :4], dtype=np.float32),
                np.ascontiguousarray(boxes_gt[:, :4], dtype=np.float32),
                bbox_pr[:, -1],
                boxes_gt[:, -1],
            )

            if self.evaluate(overlaps, bbox_pr, boxes_gt):
                nm_suc_case += 1

            bar.next()

        bar.finish()

        print("Succ rate is {}".format(nm_suc_case / dataset_size))

    def evaluate(self, overlaps, bbox_pr, boxes_gt):
        for i in range(overlaps.shape[0]):
            for j in range(overlaps.shape[1]):
                value_overlap = overlaps[i, j]
                angle_diff = math.fabs(bbox_pr[i, -1] - boxes_gt[j, -1])

                if angle_diff > np.pi / 2:
                    angle_diff = math.fabs(np.pi - angle_diff)

                if value_overlap > 0.25 and angle_diff < np.pi / 6:
                    return True

        return False
