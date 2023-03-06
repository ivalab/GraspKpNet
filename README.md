# GKNet

Grasp Keypoint Network for Grasp Candidates Detection

![](https://github.com/ivalab/GraspKpNet/blob/main/demo/fig_ill_mul_resized.png)

> **GKNet: grasp keypoint network for Grasp Candidates Detection** <br>
> Ruinian Xu, Fu-Jen Chu and Patricio A. Vela

## Table of Contents

- [Abstract](#Abstract)
- [Highlights](#Highlights)
- [Vision Benchmark Results](#Vision-Benchmark-Results)
  - [Cornell](#Grasp-detection-on-the-Cornell-Dataset)
  - [AJD](#Grasp-detection-on-the-AJD)
- [Installation](#Installation)
- [Dataset](#Dataset)
- [Usage](#Usage)
- [Develop](#Develop)
- [Physical Experiments](#Physical-Experiments)
- [Supplemental Material](#Supplemental-Material)

## Abstract

Contemporary grasp detection approaches employ deep learning to achieve robustness to sensor and object model uncertainty.
The two dominant approaches design either grasp-quality scoring or anchor-based grasp recognition networks.
This paper presents a different approach to grasp detection by treating it as keypoint detection.
The deep network detects each grasp candidate as a pair of keypoints, convertible to the grasp representation g = {x, y, w, θ} T, rather than a triplet or quartet of corner points.
Decreasing the detection difficulty by grouping keypoints into pairs boosts performance.
The addition of a non-local module into the grasp keypoint detection architecture promotes dependencies between a keypoint and its corresponding grasp candidate keypoint.
A final filtering strategy based on discrete and continuous orientation prediction removes false correspondences and further improves grasp detection performance.
GKNet, the approach presented here, achieves the best balance of accuracy and speed on the Cornell and the abridged Jacquard dataset (96.9% and 98.39% at 41.67 and 23.26 fps).
Follow-up experiments on a manipulator evaluate GKNet using 4 types of grasping experiments reflecting different nuisance sources: static grasping, dynamic grasping, grasping at varied camera angles, and bin picking.
GKNet outperforms reference baselines in static and dynamic grasping experiments while showing robustness to grasp detection for varied camera viewpoints and bin picking experiments.
The results confirm the hypothesis that grasp keypoints are an effective output representation for deep grasp networks that provide robustness to expected nuisance factors.

## Highlights

- **Accurate:** The proposed method achieves _96.9%_ and _98.39%_ detection rate over the Cornell Dataset and AJD, respectively.
- **Fast:** The proposed method is capable of running at real-time speed of _41.67_ FPS and _23.26_ FPS over the Cornell Dataset and AJD, respectively.

## Vision Benchmark Results

### Grasp detection on the Cornell Dataset

| Backbone      | Acc (w o.f.) / % | Acc (w/o o.f.) / % | Speed / FPS |
| :------------ | :--------------: | :----------------: | :---------: |
| DLA           |       96.9       |        96.8        |    41.67    |
| Hourglass-52  |       94.5       |        93.6        |    33.33    |
| Hourglass-104 |       95.5       |        95.3        |    21.27    |
| Resnet-18     |       96.0       |        95.7        |    66.67    |
| Resnet-50     |       96.5       |        96.4        |    52.63    |
| VGG-16        |       96.8       |        96.4        |    55.56    |
| AlexNet       |       95.0       |        94.8        |    83.33    |

### Grasp detection on the AJD

| Backbone      | Acc (w o.f.) / % | Acc (w/o o.f.) / % | Speed / FPS |
| :------------ | :--------------: | :----------------: | :---------: |
| DLA           |      98.39       |       96.99        |    23.26    |
| Hourglass-52  |      97.21       |       93.81        |    15.87    |
| Hourglass-104 |      97.93       |       96.04        |    9.90     |
| Resnet-18     |      97.95       |       95.97        |    31.25    |
| Resnet-50     |      98.24       |       95.91        |    25.00    |
| VGG-16        |      98.36       |       96.13        |    21.28    |
| AlexNet       |      97.37       |       94.53        |    34.48    |

## Installation

Please refer to for [INSTALL.md](readme/INSTALL.md) installation instructions.

## Dataset

The two training datasets are provided here:

- Cornell: [Download link](https://www.dropbox.com/sh/x4t8p2wrqnfevo3/AAC2gLawRtm-986_JWxE0w0Za?dl=0).
  In case the download link expires in the future, you can also use the matlab scripts provided in the `GKNet_ROOT/scripts/data_aug` to generate your own dataset based on the original Cornell dataset.
  You will need to modify the corresponding path for loading the input images and output files.
- Abridged Jacquard Dataset (AJD): [Download link](https://smartech.gatech.edu/handle/1853/64897).

## Usage

After downloading datasets, place each dataset in the corresponding folder under `GKNet_ROOT/datasets/`.
The cornell dataset should be placed under `GKNet_ROOT/datasets/Cornell/` and the AJD should be placed under `GKNet_ROOT/datasets/Jacquard/`.
Download models [ctdet_coco_dla_2x](https://www.dropbox.com/sh/eicrmhhay2wi8fy/AAAGrToUcdp0tO-F732Xhsxwa?dl=0) and put it under `GKNet_ROOT/models/`.

### Training

For training the Cornell Dataset:

```bash
python main.py dbmctdet_cornell \
  --exp_id dla34 \
  --batch_size 4 \
  --lr 1.25e-4 \
  --arch dla_34 \
  --dataset cornell \
  --load_model ../models/ctdet_coco_dla_2x.pth \
  --num_epochs 15 \
  --val_intervals 1 \
  --save_all \
  --lr_step 5,10
```

For training AJD:

```bash
python main.py dbmctdet \
  --exp_id dla34 \
  --batch_size 4 \
  --lr 1.25e-4 \
  --arch dla_34 \
  --dataset jac_coco_36 \
  --load_model ../models/ctdet_coco_dla_2x.pth \
  --num_epochs 30 \
  --val_intervals 1 \
  --save_all
```

### Evaluation

You can evaluate your own trained models or download [pretrained models](https://www.dropbox.com/sh/eicrmhhay2wi8fy/AAAGrToUcdp0tO-F732Xhsxwa?dl=0) and put them under `GKNet_ROOT/models/`.

For evaluating the Cornell Dataset:

```
python test.py dbmctdet_cornell \
  --exp_id dla34_test \
  --arch dla_34 \
  --dataset cornell \
  --fix_res \
  --flag_test \
  --load_model ../models/model_dla34_cornell.pth \
  --ae_threshold 1.0 \
  --ori_threshold 0.24 \
  --center_threshold 0.05 \
  --dataset_dir ../datasets
```

For evaluating AJD:

```bash
python test.py dbmctdet \
  --exp_id dla34_test \
  --arch dla_34 \
  --dataset jac_coco_36 \
  --fix_res \
  --flag_test \
  --load_model ../models/model_dla34_ajd.pth \
  --ae_threshold 0.65 \
  --ori_threshold 0.1745 \
  --center_threshold 0.15 \
  --dataset_dir ../datasets
```

## Develop

If you are interested in training GKNet on a new or customized dataset, please refer to [DEVELOP.md](https://github.com/ivalab/GraspKpNet/blob/master/readme/DEVELOP.md).
Also you can leave your issues here if you meet some problems.

## Physical Experiments

To run physical experiments with GKNet and ROS, please follow the instructions provided in [Experiment.md](https://github.com/ivalab/GraspKpNet/blob/master/readme/experiment.md).

## Supplemental Material

This section collects results of some experiments or discussions which aren't documented in the manuscript due to the lack of enough scientific values.

### Keypoint representation

This [readme](https://github.com/ivalab/GraspKpNet/blob/main/readme/kp_rep.md) file documents some examples with visualiztions for Top-left, bottom-left and bottom-right (TlBlBr) grasp keypoint representation.
Theseexamples help clarify the effectiveness of grasp keypoint representation of less number of keypoints.

### Tuning hyper-parameters of alpha, beta and gamma.

The result is recorded in [tune_hp.md](https://github.com/ivalab/GraspKpNet/blob/main/readme/tune_kp.md)

### Demo video

The demo video of all physical experiments are uploaded on the [Youtube](https://www.youtube.com/watch?v=Q8-Kr8Q9vC0). Please watch it if you are interested.

### Detailed Result and Tables

Some of the source data was summarized with the raw source data not provided.
The links below provide access to the source material:

- [Trial results of bin picking](https://github.com/ivalab/GraspKpNet/blob/main/readme/bin_picking.md) experiment.
- [6-DoF summary results](https://github.com/ivalab/GraspKpNet/blob/main/readme/bin_picking_6DoF.md) for clutter clearance or bin-picking tasks.

### Implementation of GGCNN

Considering that GGCNN didn't provide the result of training and testing on the Cornell Dataset, we implemented their work based on their public repository.
The modified version is provided [here](https://github.com/ivalab/ggcnn).

## License

GKNet is released under the MIT License (refer to the LICENSE file for details).
Portions of the code are borrowed from [CenterNet](https://github.com/xingyizhou/CenterNet), [dla](https://github.com/ucbdrive/dla) (DLA network), [DCNv2](https://github.com/CharlesShang/DCNv2)(deformable convolutions).
Please refer to the original License of these projects (See [Notice](https://github.com/ivalab/GKNet/blob/master/NOTICE)).

## Citation

If you use GKNet in your work, please cite:

```
@article{xu2021gknet,
  title={GKNet: grasp keypoint network for grasp candidates detection},
  author={Xu, Ruinian and Chu, Fu-Jen and Vela, Patricio A},
  journal={arXiv preprint arXiv:2106.08497},
  year={2021}
}
```
