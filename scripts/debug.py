import math
import os

import cv2
import numpy as np
import torch
from progress.bar import Bar

from gknet.datasets.dataset.utils import _bbox_overlaps
from gknet.datasets.dataset_factory import dataset_factory
from gknet.debuggers.debugger_factory import debugger_factory
from gknet.logger import Logger
from gknet.opts import opts
from gknet.utils.image import (
    affine_transform,
    draw_msra_gaussian,
    draw_umich_gaussian,
    gaussian_radius,
    get_affine_transform,
)


class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset):
        self.dataset = dataset
        self.images = dataset.images
        self.img_dir = dataset.img_dir
        self.opt = opt
        self.max_objs = 256
        self.split = "test"

    def _coco_box_to_bbox(self, box):
        bbox = np.array(
            [box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32
        )
        return bbox

    def _bbox_to_tripoints(self, bbox, angle):
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        diagonal = math.sqrt(width * width + height * height) / 2
        theta = math.atan(height / width)

        theta_1 = theta + angle
        theta_2 = np.pi - theta + angle

        """
        Polygon order:
        R(right) L(left) T(top) B(bottom)
        TR->TL->BL->BR
        """
        p1 = np.array(
            [
                math.cos(theta_1) * diagonal + x_center,
                -math.sin(theta_1) * diagonal + y_center,
            ],
            dtype=np.float32,
        )
        p2 = np.array(
            [
                math.cos(theta_2) * diagonal + x_center,
                -math.sin(theta_2) * diagonal + y_center,
            ],
            dtype=np.float32,
        )
        p3 = np.array(
            [
                -math.cos(theta_1) * diagonal + x_center,
                math.sin(theta_1) * diagonal + y_center,
            ],
            dtype=np.float32,
        )
        p4 = np.array(
            [
                -math.cos(theta_2) * diagonal + x_center,
                math.sin(theta_2) * diagonal + y_center,
            ],
            dtype=np.float32,
        )

        return p2, p3, p4

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.dataset.coco.loadImgs(ids=[img_id])[0]["file_name"]
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.dataset.coco.getAnnIds(imgIds=[img_id])
        anns = self.dataset.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2.0, img.shape[0] / 2.0], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        # flipped = False
        # if self.split == 'train':
        #     if not self.opt.not_rand_crop:
        #         s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        #         w_border = self._get_border(128, img.shape[1])
        #         h_border = self._get_border(128, img.shape[0])
        #         c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        #         c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
        #     else:
        #         sf = self.opt.scale
        #         cf = self.opt.shift
        #         c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
        #         c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
        #         s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        #
        #     if np.random.random() < self.opt.flip:
        #         flipped = True
        #         img = img[:, ::-1, :]
        #         c[0] = width - c[0] - 1

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(
            img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR
        )
        inp = inp.astype(np.float32) / 255.0
        # if self.split == 'train' and not self.opt.no_color_aug:
        #     color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.dataset.mean) / self.dataset.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.dataset.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        tl_heatmaps = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        bl_heatmaps = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        br_heatmaps = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        ct_heatmaps = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        tl_reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        bl_reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        br_reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ct_reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        tl_tag = np.zeros((self.max_objs), dtype=np.int64)
        bl_tag = np.zeros((self.max_objs), dtype=np.int64)
        br_tag = np.zeros((self.max_objs), dtype=np.int64)
        ct_tag = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        bbox_gt = np.zeros((self.max_objs, 5), dtype=np.float32)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian

        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann["bbox"])
            # skip the class for the whole object
            if ann["category_id"] == 37:
                continue
            cls_id = int(self.dataset.cat_ids[ann["category_id"]])

            # if flipped:
            #     bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            # convert bbox to tl, bl, br points
            ftl_p, fbl_p, fbr_p = self._bbox_to_tripoints(
                bbox[:4], ann["bbox"][-1] / 180 * np.pi
            )
            # compute center point
            fct_p = np.array(
                [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32
            )

            # skip the bounding box whose points beyond the border after affine transformation and rotation
            if (
                ftl_p[0] < 0
                or ftl_p[0] > output_w - 1
                or ftl_p[1] < 0
                or ftl_p[1] > output_h - 1
                or fbl_p[0] < 0
                or fbl_p[0] > output_w - 1
                or fbl_p[1] < 0
                or fbl_p[1] > output_h - 1
                or fbr_p[0] < 0
                or fbr_p[0] > output_w - 1
                or fbr_p[1] < 0
                or fbr_p[1] > output_h - 1
            ):
                continue

            tl_p = ftl_p.astype(np.int32)
            bl_p = fbl_p.astype(np.int32)
            br_p = fbr_p.astype(np.int32)
            ct_p = fct_p.astype(np.int32)

            # bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            # bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius

                draw_gaussian(tl_heatmaps[cls_id], tl_p, radius)
                draw_gaussian(bl_heatmaps[cls_id], bl_p, radius)
                draw_gaussian(br_heatmaps[cls_id], br_p, radius)
                draw_gaussian(ct_heatmaps[cls_id], ct_p, radius)

                bbox_gt[k] = [
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                    ann["bbox"][-1] / 180 * np.pi,
                ]

                tl_tag[k] = tl_p[1] * output_w + tl_p[0]
                bl_tag[k] = bl_p[1] * output_w + bl_p[0]
                br_tag[k] = br_p[1] * output_w + br_p[0]
                ct_tag[k] = ct_p[1] * output_w + ct_p[0]

                tl_reg[k] = ftl_p - tl_p
                bl_reg[k] = fbl_p - bl_p
                br_reg[k] = fbr_p - br_p
                ct_reg[k] = fct_p - ct_p

                reg_mask[k] = 1

        ret = {
            "input": inp,
            "tl": tl_heatmaps,
            "bl": bl_heatmaps,
            "br": br_heatmaps,
            "ct": ct_heatmaps,
            "tl_tag": tl_tag,
            "bl_tag": bl_tag,
            "br_tag": br_tag,
            "ct_tag": ct_tag,
            "tl_reg": tl_reg,
            "bl_reg": bl_reg,
            "br_reg": br_reg,
            "ct_reg": ct_reg,
            "reg_mask": reg_mask,
            "bbox": bbox_gt,
        }

        return img_id, ret

    def __len__(self):
        return len(self.images)


def prefetch_test(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)
    Debugger = debugger_factory[opt.task]

    split = "test" if not opt.trainval else "test"
    dataset = Dataset(opt, split)
    debugger = Debugger(opt)

    data_loader = torch.utils.data.DataLoader(
        PrefetchDataset(opt, dataset),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    num_iters = len(dataset)
    min_dist = 1e10
    max_dist = 0.0
    avg_dist = 0.0
    nm_match = 0
    results = {}
    bar = Bar("{}".format(opt.exp_id), max=num_iters)
    for ind, (img_id, targets) in enumerate(data_loader):
        # minimum, maximum, average, flag_found_match = debugger.run(targets)
        # # minimum, maximum, average = debugger.run(targets)
        # if minimum < min_dist:
        #     min_dist = minimum
        # if maximum > max_dist:
        #     max_dist = maximum
        # avg_dist += average / num_iters
        #
        # if flag_found_match:
        #     nm_match += 1
        ret = debugger.statistics(targets)
        results[img_id.numpy().astype(np.int32)[0]] = ret["results"]

        Bar.suffix = "[{0}/{1}]|Tot: {total:} |ETA: {eta:} ".format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td
        )
        bar.next()
    bar.finish()

    dataset_size = len(results)
    fail_case_image_id = []
    file_image_id = open(
        "/home/ruinianxu/IVA_Lab/Project/TripleNet/exp/trictdet/tri_dla_jac_coco_36_correctized/debug/fail_case_image_id.txt",
        "r",
    )
    line = file_image_id.readline()
    while line:
        fail_case_image_id.append(int(line))
        line = file_image_id.readline()
    file_image_id.close()

    nm_suc_case = 0
    nm_zero_detection = 0

    fail_cases = []
    for image_id, result in results.items():
        Bar.suffix = "[{0}/{1}]|Tot: {total:} |ETA: {eta:} ".format(
            image_id, dataset_size, total=bar.elapsed_td, eta=bar.eta_td
        )

        if image_id not in fail_case_image_id:
            continue

        # get the associated groundtruth for predicted_bbox
        img_info = dataset.coco.loadImgs(ids=[image_id])[0]
        image_w = img_info["width"]
        image_h = img_info["height"]
        ann_ids = dataset.coco.getAnnIds(imgIds=[image_id])
        annotations = dataset.coco.loadAnns(ids=ann_ids)

        # collect all detections and select the one with the highest score also
        bboxes_pr = []
        best_score = 0.0
        bbox_pr_highest = []
        for category_id, pr_bboxs in result.items():
            if len(pr_bboxs) == 0:
                continue
            for pr_bbox in pr_bboxs:
                pr_bbox[0] /= image_w
                pr_bbox[2] /= image_w
                pr_bbox[4] /= image_w
                pr_bbox[1] /= image_h
                pr_bbox[3] /= image_h
                pr_bbox[5] /= image_h
                bboxes_pr.append(pr_bbox[:])

                if pr_bbox[6] > best_score:
                    best_score = pr_bbox[6]
                    bbox_pr_highest.clear()
                    bbox_pr_highest.append(pr_bbox[:])

        if len(bboxes_pr) == 0:
            nm_zero_detection += 1
            continue

        # bbox type conversion
        bbox_coord_pr = []
        for bbox_pr in bboxes_pr:
            tl_x = bbox_pr[0]
            tl_y = bbox_pr[1]
            bl_x = bbox_pr[2]
            bl_y = bbox_pr[3]
            br_x = bbox_pr[4]
            br_y = bbox_pr[5]

            # center
            x_c = (tl_x + br_x) / 2.0
            y_c = (tl_y + br_y) / 2.0

            if bl_x == br_x:
                p_y = tl_y
                p_x = br_x
                if br_y > bl_y:
                    angle = np.pi / 2.0
                else:
                    angle = -np.pi / 2.0
            elif bl_y == br_y:
                p_x = tl_x
                p_y = br_y
                angle = 0.0
            else:
                # angle
                angle = math.atan(-(br_y - bl_y) / (br_x - bl_x))
                # find intersected point
                a = (br_y - bl_y) / (br_x - bl_x)
                b = br_y - a * br_x
                delta_x = br_x - bl_x
                delta_y = br_y - bl_y
                p_x = (delta_x * tl_x + delta_y * tl_y - delta_y * b) / (
                    delta_x + delta_y * a
                )
                p_y = a * p_x + b
            # w, h
            w = np.sqrt((br_x - p_x) * (br_x - p_x) + (br_y - p_y) * (br_y - p_y))
            h = np.sqrt((tl_x - p_x) * (tl_x - p_x) + (tl_y - p_y) * (tl_y - p_y))

            bbox_coord_pr.append(
                [x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2, angle]
            )
        bbox_coord_pr = np.array(bbox_coord_pr)

        boxes_coord_gt = []

        for anno in annotations:
            # skip if predicted bbox is 37 or there is no bbox predicted for this category id
            if anno["category_id"] == 37:
                continue
            bbox_gt = anno["bbox"]
            x_min = bbox_gt[0]
            y_min = bbox_gt[1]
            w = bbox_gt[2]
            h = bbox_gt[3]
            angle = bbox_gt[4] / 180 * np.pi
            boxes_coord_gt.append(
                [
                    int(x_min) / image_w,
                    int(y_min) / image_h,
                    int(x_min + w) / image_w,
                    int(y_min + h) / image_h,
                    angle,
                ]
            )
        boxes_coord_gt = np.array(boxes_coord_gt)

        overlaps = _bbox_overlaps(
            np.ascontiguousarray(bbox_coord_pr[:, :4], dtype=np.float32),
            np.ascontiguousarray(boxes_coord_gt[:, :4], dtype=np.float32),
            bbox_coord_pr[:, -1],
            boxes_coord_gt[:, -1],
            image_w,
            image_h,
        )

        flag_exit = 0
        best_overlap = 0.0
        result = {}
        closet_bbox_pr = []
        for i in range(overlaps.shape[0]):
            for j in range(overlaps.shape[1]):
                value_overlap = overlaps[i, j]
                angle_diff = math.fabs(bbox_coord_pr[i, -1] - boxes_coord_gt[j, -1])

                if value_overlap >= best_overlap:
                    best_overlap = value_overlap
                    closet_bbox_pr.clear()
                    closet_bbox_pr = bboxes_pr[i].tolist()

                if value_overlap > 0.25 and angle_diff < np.pi / 6:
                    bbox_pr_cor = bboxes_pr[i]

                    result["image_id"] = image_id
                    result["tl_x"] = bbox_pr_cor[0]
                    result["tl_y"] = bbox_pr_cor[1]
                    result["bl_x"] = bbox_pr_cor[2]
                    result["bl_y"] = bbox_pr_cor[3]
                    result["br_x"] = bbox_pr_cor[4]
                    result["br_y"] = bbox_pr_cor[5]
                    result["tl_highest_x"] = bbox_pr_highest[0][0]
                    result["tl_highest_y"] = bbox_pr_highest[0][1]
                    result["bl_highest_x"] = bbox_pr_highest[0][2]
                    result["bl_highest_y"] = bbox_pr_highest[0][3]
                    result["br_highest_x"] = bbox_pr_highest[0][4]
                    result["br_highest_y"] = bbox_pr_highest[0][5]
                    result["tl_score"] = bbox_pr_cor[7]
                    result["bl_score"] = bbox_pr_cor[8]
                    result["br_score"] = bbox_pr_cor[9]
                    result["ct_score"] = (
                        bbox_pr_cor[6] * 4
                        - bbox_pr_cor[7]
                        - bbox_pr_cor[8]
                        - bbox_pr_cor[9]
                    )
                    result["tl_highest_score"] = bbox_pr_highest[0][7]
                    result["bl_highest_score"] = bbox_pr_highest[0][8]
                    result["br_highest_score"] = bbox_pr_highest[0][9]
                    result["ct_highest_score"] = (
                        bbox_pr_highest[0][6] * 4
                        - bbox_pr_highest[0][7]
                        - bbox_pr_highest[0][8]
                        - bbox_pr_highest[0][9]
                    )

                    fail_cases.append(result)
                    nm_suc_case += 1
                    flag_exit = 1
                    break

            if flag_exit:
                break
        print(flag_exit)
        if not flag_exit:
            result["image_id"] = image_id
            result["tl_x"] = closet_bbox_pr[0]
            result["tl_y"] = closet_bbox_pr[1]
            result["bl_x"] = closet_bbox_pr[2]
            result["bl_y"] = closet_bbox_pr[3]
            result["br_x"] = closet_bbox_pr[4]
            result["br_y"] = closet_bbox_pr[5]
            result["tl_highest_x"] = bbox_pr_highest[0][0]
            result["tl_highest_y"] = bbox_pr_highest[0][1]
            result["bl_highest_x"] = bbox_pr_highest[0][2]
            result["bl_highest_y"] = bbox_pr_highest[0][3]
            result["br_highest_x"] = bbox_pr_highest[0][4]
            result["br_highest_y"] = bbox_pr_highest[0][5]
            result["tl_score"] = closet_bbox_pr[7]
            result["bl_score"] = closet_bbox_pr[8]
            result["br_score"] = closet_bbox_pr[9]
            result["ct_score"] = (
                closet_bbox_pr[6] * 4 - bbox_pr_cor[7] - bbox_pr_cor[8] - bbox_pr_cor[9]
            )
            result["tl_highest_score"] = bbox_pr_highest[0][7]
            result["bl_highest_score"] = bbox_pr_highest[0][8]
            result["br_highest_score"] = bbox_pr_highest[0][9]
            result["ct_highest_score"] = (
                bbox_pr_highest[0][6] * 4
                - bbox_pr_highest[0][7]
                - bbox_pr_highest[0][8]
                - bbox_pr_highest[0][9]
            )

            fail_cases.append(result)
        bar.next()

    bar.finish()

    file = open(
        "/home/ruinianxu/IVA_Lab/Project/TripleNet/exp/trictdet/tri_dla_jac_coco_36_correctized/debug/fail_case_ana_2.txt",
        "w+",
    )
    for result in fail_cases:
        file.write(
            str(result["image_id"])
            + " "
            + str(result["tl_x"])
            + " "
            + str(result["tl_y"])
            + " "
            + str(result["bl_x"])
            + " "
            + str(result["bl_y"])
            + " "
            + str(result["br_x"])
            + " "
            + str(result["br_y"])
            + " "
            + str(result["tl_highest_x"])
            + " "
            + str(result["tl_highest_y"])
            + " "
            + str(result["bl_highest_x"])
            + " "
            + str(result["bl_highest_y"])
            + " "
            + str(result["br_highest_x"])
            + " "
            + str(result["br_highest_y"])
            + " "
            + str(result["tl_score"])
            + " "
            + str(result["bl_score"])
            + " "
            + str(result["br_score"])
            + " "
            + str(result["ct_score"])
            + " "
            + str(result["tl_highest_score"])
            + " "
            + str(result["bl_highest_score"])
            + " "
            + str(result["br_highest_score"])
            + " "
            + str(result["ct_highest_score"])
            + " "
            + "\n"
        )
    file.close()


if __name__ == "__main__":
    opt = opts().parse()
    prefetch_test(opt)
