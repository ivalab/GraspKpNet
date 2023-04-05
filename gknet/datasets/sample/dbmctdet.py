import math
import os

import cv2
import numpy as np
import torch.utils.data as data

from gknet.utils.image import (
    affine_transform,
    color_aug,
    draw_msra_gaussian,
    draw_umich_gaussian,
    gaussian_radius,
    get_affine_transform,
)


class DbMCTDetDataset(data.Dataset):
    """
    Bbox format: left_middle_x, left_middle_y, center_x, center_y, right_middle_x, right_middle_y, top_middle_x, top_middle_y,
                 average_height, angle
    """

    def _coco_box_to_bbox(self, box):
        bbox = np.array(
            [box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32
        )
        return bbox

    def _bbox_to_points(self, bbox, angle):
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

        tl_0 = np.array([xmin, ymin])
        br_0 = np.array([xmax, ymax])
        bl_0 = np.array([xmin, ymax])
        tr_0 = np.array([xmax, ymin])
        center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])

        T = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        tl_1 = np.dot(T, (tl_0 - center)) + center
        br_1 = np.dot(T, (br_0 - center)) + center
        bl_1 = np.dot(T, (bl_0 - center)) + center
        tr_1 = np.dot(T, (tr_0 - center)) + center

        p_tl = np.array([tl_1[0], tl_1[1]], dtype=np.float32)
        p_bl = np.array([bl_1[0], bl_1[1]], dtype=np.float32)
        p_br = np.array([br_1[0], br_1[1]], dtype=np.float32)
        p_tr = np.array([tr_1[0], tr_1[1]], dtype=np.float32)

        return p_tl, p_bl, p_br, p_tr

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]["file_name"]
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
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

        if self.split == "train":
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(
            img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR
        )
        inp_show = inp.copy()
        inp = inp.astype(np.float32) / 255.0
        if self.split == "train" and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        num_ct_classes = self.num_ct_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        lm_heatmaps = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        rm_heatmaps = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        ct_heatmaps = np.zeros((num_ct_classes, output_h, output_w), dtype=np.float32)
        lm_reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        rm_reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ct_reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        lm_tag = np.zeros((self.max_objs), dtype=np.int64)
        rm_tag = np.zeros((self.max_objs), dtype=np.int64)
        ct_tag = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian

        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann["bbox"])
            cls_id = int(self.cat_ids[ann["category_id"]])
            width_origin = bbox[2] - bbox[0]

            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)

            ftl_p, fbl_p, fbr_p, ftr_p = self._bbox_to_points(
                bbox[:4], ann["bbox"][-1] / 180 * np.pi
            )

            flm_p = np.array(
                [(ftl_p[0] + fbl_p[0]) / 2, (ftl_p[1] + fbl_p[1]) / 2], dtype=np.float32
            )
            frm_p = np.array(
                [(ftr_p[0] + fbr_p[0]) / 2, (ftr_p[1] + fbr_p[1]) / 2], dtype=np.float32
            )

            fct_p = np.array(
                [(ftl_p[0] + fbr_p[0]) / 2, (ftl_p[1] + fbr_p[1]) / 2], dtype=np.float32
            )

            # skip the bounding box whose points beyond the border after affine transformation and rotation
            if (
                flm_p[0] < 0
                or flm_p[0] > output_w - 1
                or flm_p[1] < 0
                or flm_p[1] > output_h - 1
                or frm_p[0] < 0
                or frm_p[0] > output_w - 1
                or frm_p[1] < 0
                or frm_p[1] > output_h - 1
            ):
                continue

            lm_p = flm_p.astype(np.int32)
            rm_p = frm_p.astype(np.int32)
            ct_p = fct_p.astype(np.int32)

            w = np.sqrt(
                np.power(flm_p[0] - frm_p[0], 2) + np.power(flm_p[1] - frm_p[1], 2)
            )
            h = w / width_origin * 20.0
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius

                draw_gaussian(lm_heatmaps[cls_id], lm_p, radius)
                draw_gaussian(rm_heatmaps[cls_id], rm_p, radius)
                draw_gaussian(ct_heatmaps[0], ct_p, radius)

                lm_tag[k] = lm_p[1] * output_w + lm_p[0]
                rm_tag[k] = rm_p[1] * output_w + rm_p[0]
                ct_tag[k] = ct_p[1] * output_w + ct_p[0]
                if ct_p[1] * output_w + ct_p[0] > 16383:
                    print(file_name)
                    print("Out of upper bound!")
                elif ct_p[1] * output_w + ct_p[0] < 0:
                    print(file_name)
                    print("Out of lower bound!")

                lm_reg[k] = flm_p - lm_p
                rm_reg[k] = frm_p - rm_p
                ct_reg[k] = fct_p - ct_p

                reg_mask[k] = 1

        if (ct_reg > 1).any():
            print("Float precision error!")

        ret = {
            "input": inp,
            "lm": lm_heatmaps,
            "rm": rm_heatmaps,
            "ct": ct_heatmaps,
            "lm_tag": lm_tag,
            "rm_tag": rm_tag,
            "ct_tag": ct_tag,
            "lm_reg": lm_reg,
            "rm_reg": rm_reg,
            "ct_reg": ct_reg,
            "reg_mask": reg_mask,
        }

        return ret
