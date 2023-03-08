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
    default_resolution = [256, 256]
    mean = np.array([0.850092, 0.805317, 0.247344],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.104114, 0.113242, 0.089067],
                   dtype=np.float32).reshape(1, 1, 3)
    zero_anno_list = ['pcd1018r_rgd_preprocessed_98', 'pcd1023r_rgd_preprocessed_97', 'pcd1024r_rgd_preprocessed_52',
                      'pcd1028r_rgd_preprocessed_119', 'pcd1017r_rgd_preprocessed_120', 'pcd1021r_rgd_preprocessed_54',
                      'pcd1003r_rgd_preprocessed_76', 'pcd1029r_rgd_preprocessed_74', 'pcd1003r_rgd_preprocessed_24',
                      'pcd1000r_rgd_preprocessed_110', 'pcd1018r_rgd_preprocessed_34', 'pcd1001r_rgd_preprocessed_84',
                      'pcd1018r_rgd_preprocessed_123', 'pcd1018r_rgd_preprocessed_118', 'pcd1030r_rgd_preprocessed_110',
                      'pcd1001r_rgd_preprocessed_5', 'pcd1003r_rgd_preprocessed_102', 'pcd1003r_rgd_preprocessed_96',
                      'pcd1021r_rgd_preprocessed_113', 'pcd1003r_rgd_preprocessed_64', 'pcd1029r_rgd_preprocessed_86',
                      'pcd1003r_rgd_preprocessed_66', 'pcd1003r_rgd_preprocessed_100', 'pcd1021r_rgd_preprocessed_62',
                      'pcd0248r_rgd_preprocessed_11', 'pcd1024r_rgd_preprocessed_63', 'pcd1018r_rgd_preprocessed_107',
                      'pcd1018r_rgd_preprocessed_35', 'pcd0248r_rgd_preprocessed_85', 'pcd1018r_rgd_preprocessed_9',
                      'pcd1003r_rgd_preprocessed_54', 'pcd1023r_rgd_preprocessed_70', 'pcd1030r_rgd_preprocessed_7',
                      'pcd1018r_rgd_preprocessed_19', 'pcd1029r_rgd_preprocessed_92', 'pcd1003r_rgd_preprocessed_68',
                      'pcd1003r_rgd_preprocessed_81', 'pcd1003r_rgd_preprocessed_120', 'pcd1001r_rgd_preprocessed_63',
                      'pcd1003r_rgd_preprocessed_108', 'pcd1003r_rgd_preprocessed_28', 'pcd1003r_rgd_preprocessed_50',
                      'pcd1026r_rgd_preprocessed_16', 'pcd1024r_rgd_preprocessed_13', 'pcd1024r_rgd_preprocessed_75',
                      'pcd1024r_rgd_preprocessed_50', 'pcd1021r_rgd_preprocessed_104', 'pcd1024r_rgd_preprocessed_10',
                      'pcd1003r_rgd_preprocessed_67', 'pcd1021r_rgd_preprocessed_117', 'pcd1001r_rgd_preprocessed_86',
                      'pcd0117r_rgd_preprocessed_4', 'pcd1018r_rgd_preprocessed_74', 'pcd1003r_rgd_preprocessed_88',
                      'pcd1003r_rgd_preprocessed_94', 'pcd1029r_rgd_preprocessed_15', 'pcd1003r_rgd_preprocessed_1',
                      'pcd1003r_rgd_preprocessed_95', 'pcd1028r_rgd_preprocessed_26', 'pcd1024r_rgd_preprocessed_27',
                      'pcd1030r_rgd_preprocessed_75', 'pcd1003r_rgd_preprocessed_98', 'pcd1000r_rgd_preprocessed_26',
                      'pcd1001r_rgd_preprocessed_108', 'pcd1003r_rgd_preprocessed_59', 'pcd1021r_rgd_preprocessed_96',
                      'pcd1023r_rgd_preprocessed_122', 'pcd1003r_rgd_preprocessed_125', 'pcd1001r_rgd_preprocessed_80',
                      'pcd1030r_rgd_preprocessed_111', 'pcd1029r_rgd_preprocessed_34', 'pcd1024r_rgd_preprocessed_70',
                      'pcd1029r_rgd_preprocessed_94', 'pcd1001r_rgd_preprocessed_93', 'pcd1024r_rgd_preprocessed_84',
                      'pcd1024r_rgd_preprocessed_22', 'pcd1003r_rgd_preprocessed_111', 'pcd1003r_rgd_preprocessed_110',
                      'pcd1003r_rgd_preprocessed_63', 'pcd1021r_rgd_preprocessed_46', 'pcd0248r_rgd_preprocessed_95',
                      'pcd1003r_rgd_preprocessed_15', 'pcd1003r_rgd_preprocessed_17', 'pcd1023r_rgd_preprocessed_124',
                      'pcd1024r_rgd_preprocessed_73', 'pcd1003r_rgd_preprocessed_93', 'pcd0248r_rgd_preprocessed_81',
                      'pcd1001r_rgd_preprocessed_13', 'pcd1023r_rgd_preprocessed_18', 'pcd1023r_rgd_preprocessed_55',
                      'pcd1001r_rgd_preprocessed_66', 'pcd1024r_rgd_preprocessed_115', 'pcd1024r_rgd_preprocessed_24',
                      'pcd1001r_rgd_preprocessed_75', 'pcd1021r_rgd_preprocessed_7', 'pcd1003r_rgd_preprocessed_31',
                      'pcd1023r_rgd_preprocessed_35', 'pcd1003r_rgd_preprocessed_70', 'pcd1024r_rgd_preprocessed_40',
                      'pcd1030r_rgd_preprocessed_88', 'pcd1030r_rgd_preprocessed_13', 'pcd1003r_rgd_preprocessed_113',
                      'pcd0248r_rgd_preprocessed_75', 'pcd1003r_rgd_preprocessed_5', 'pcd0249r_rgd_preprocessed_64',
                      'pcd1024r_rgd_preprocessed_34', 'pcd1003r_rgd_preprocessed_27', 'pcd1003r_rgd_preprocessed_23',
                      'pcd1029r_rgd_preprocessed_64', 'pcd1003r_rgd_preprocessed_39', 'pcd1003r_rgd_preprocessed_43',
                      'pcd0248r_rgd_preprocessed_34', 'pcd1023r_rgd_preprocessed_88', 'pcd0117r_rgd_preprocessed_102',
                      'pcd1028r_rgd_preprocessed_57', 'pcd0117r_rgd_preprocessed_98', 'pcd1018r_rgd_preprocessed_71',
                      'pcd1003r_rgd_preprocessed_36', 'pcd1003r_rgd_preprocessed_57', 'pcd1003r_rgd_preprocessed_42',
                      'pcd1018r_rgd_preprocessed_56', 'pcd1029r_rgd_preprocessed_122', 'pcd0117r_rgd_preprocessed_18',
                      'pcd1001r_rgd_preprocessed_10', 'pcd0117r_rgd_preprocessed_43', 'pcd1024r_rgd_preprocessed_97',
                      'pcd1021r_rgd_preprocessed_109', 'pcd1029r_rgd_preprocessed_69', 'pcd1003r_rgd_preprocessed_116',
                      'pcd1028r_rgd_preprocessed_14', 'pcd0117r_rgd_preprocessed_40', 'pcd1028r_rgd_preprocessed_76',
                      'pcd1001r_rgd_preprocessed_14', 'pcd1003r_rgd_preprocessed_84', 'pcd1018r_rgd_preprocessed_113',
                      'pcd1003r_rgd_preprocessed_14', 'pcd1021r_rgd_preprocessed_102', 'pcd1003r_rgd_preprocessed_3',
                      'pcd1003r_rgd_preprocessed_75', 'pcd1001r_rgd_preprocessed_36']

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

        self.data_dir = os.path.join(opt.data_dir, 'Cornell/rgd_5_5_5_corner_p_full/data')
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

                if line in self.zero_anno_list:
                    line = f.readline()
                    continue

                images.append(line)
                line = f.readline()
        return images

    def __len__(self):
        return self.num_samples

    def run_eval_db_middle(self, results):
        dataset_size = len(results)
        nm_suc_case = 0
        bar = Bar('cornell evaluation', max=dataset_size)

        factor = 256. / 227.
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
                    xmin, ymin, xmax, ymax, angle = float(line[1]) * factor, float(line[2]) * factor, \
                                                    float(line[3]) * factor, float(line[4]) * factor, \
                                                    float(line[5]) * factor
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

            if self.evaluate(overlaps, bbox_pr, boxes_gt):
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




