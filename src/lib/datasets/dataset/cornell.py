from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import os
from progress.bar import Bar
import cv2
import math
import copy
import os

import torch.utils.data as data
from datasets.dataset.utils import _bbox_overlaps_counterclock, rotate_bbox_counterclock

class CORNELL(data.Dataset):
    num_classes = 18
    num_ct_classes = 1
    default_resolution = [227, 227]
    mean = np.array([0.850092, 0.805317, 0.247344],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.104114, 0.113242, 0.089067],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(CORNELL, self).__init__()

        self.max_objs = 32
        self.avg_h = 23.33
        self.class_name = ["__background__", 1, 2, 3, 4, 5, 6,
                           7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        self._valid_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                           13, 14, 15, 16, 17, 18]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}  # rx

        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self.split = split
        self.opt = opt

        self.data_dir = os.path.join(opt.data_dir, 'Cornell/')
        self.img_dir = os.path.join(self.data_dir, 'Images')
        self.annot_path = os.path.join(self.data_dir, 'Annotations')
        self.filelist_dir = os.path.join(self.data_dir, 'ImageSets', '{}.txt'.format(split))

        self.images = self.readimageset(self.filelist_dir)
        self.num_samples = len(self.images)

    def readimageset(self, file_path):
        images = []
        with open(file_path, 'r') as f:
            line = f.readline()
            while line:
                line = line.split('\n')[0]
                images.append(line)
                line = f.readline()
        return images

    def __len__(self):
        return self.num_samples

    def run_eval_db_middle(self, results):
        dataset_size = len(results)
        nm_suc_case = 0
        bar = Bar('cornell evaluation', max=dataset_size)

        for ind, (image_id, result) in enumerate(results.items()):
            Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                ind, dataset_size, total=bar.elapsed_td, eta=bar.eta_td)

            img_path = image_id.split('\n')[0]
            template_name = img_path
            anno_path = os.path.join(self.annot_path, template_name + '.txt')

            boxes_gt = []
            with open(anno_path, 'r') as f:
                line = f.readline().split()
                while line:
                    xmin, ymin, xmax, ymax, angle = float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5])
                    x_c, y_c = (xmin + xmax) / 2, (ymin + ymax) / 2
                    width, height = xmax - xmin, ymax - ymin

                    boxes_gt.append([x_c-width/2, y_c-height/2,
                                     x_c+width/2, y_c+height/2,
                                     angle])

                    line = f.readline().split()

            boxes_gt = np.array(boxes_gt)

            # collect the detection with the highest score
            bboxes_pr = []
            bboxes_s_pr = []
            for category_id, pr_bboxs in result.items():
                if len(pr_bboxs) == 0:
                    continue
                for pr_bbox in pr_bboxs:
                    bboxes_pr.append(pr_bbox[:4])
                    bboxes_s_pr.append(pr_bbox[-1])

            bbox_pr = []
            max_s = 0.0
            for i in range(len(bboxes_s_pr)):
                x_c, y_c = (bboxes_pr[i][0] + bboxes_pr[i][2]) / 2, \
                           (bboxes_pr[i][1] + bboxes_pr[i][3]) / 2

                w = np.sqrt(np.power(bboxes_pr[i][0] - bboxes_pr[i][2], 2) +
                            np.power(bboxes_pr[i][1] - bboxes_pr[i][3], 2))

                if bboxes_pr[i][0] == bboxes_pr[i][2] and bboxes_pr[i][1] < bboxes_pr[i][3]:
                    angle = -np.pi / 2
                elif bboxes_pr[i][0] == bboxes_pr[i][2] and bboxes_pr[i][1] > bboxes_pr[i][3]:
                    angle = np.pi / 2
                elif bboxes_pr[i][1] == bboxes_pr[i][3]:
                    angle = 0
                else:
                    angle = np.arctan(-(bboxes_pr[i][3]-bboxes_pr[i][1]) / (bboxes_pr[i][2]-bboxes_pr[i][0]))

                if max_s < bboxes_s_pr[i]:
                    max_s = bboxes_s_pr[i]
                    bbox_pr = []
                    bbox_pr.append([x_c-w/2, y_c-self.avg_h/2,
                                    x_c+w/2, y_c+self.avg_h/2,
                                    angle])

            if len(bbox_pr) == 0:
                continue

            bbox_pr = np.array(bbox_pr)

            overlaps = _bbox_overlaps_counterclock(np.ascontiguousarray(bbox_pr[:, :4], dtype=np.float32),
                                                   np.ascontiguousarray(boxes_gt[:, :4], dtype=np.float32),
                                                   bbox_pr[:, -1], boxes_gt[:, -1])

            if evaluate(overlaps, bbox_pr, boxes_gt):
                nm_suc_case += 1

            bar.next()

        bar.finish()

        print('Succ rate is {}'.format(nm_suc_case / dataset_size))

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




