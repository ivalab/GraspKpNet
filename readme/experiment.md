# Physical Experiments
The physical experiments of GKNet is composed of four different grasping experiments: (a) static grasping, (b) grasping at varied camera viewpoints, (c) dynamic 
grasping and (d) bin picking.

This readme file documents how to re-produce the results of all physical experiments presented in the manuscript, which mainly consists of Python and ROS codes.

## Static Grasping
The static grasping experiment consists of grasping trials for individual objects. It is a baseline test for understanding expected performance in ideal situations. 
The test primarily evaluates accuracy of grasping candidate prediction, and secondly robustness to object variation.

### Command of running robotic grasping detection

```
python static_grasp.py dbmctdet_cornell --exp_id static_grasp --arch dla_34 --dataset cornell --fix_res --load_model ../models/model_dla34_cornell.pth --ae_threshold 1.0 --ori_threshold 0.24 --center_threshold 0.05

```

### Running ROS-side scripts

## Grasping at Varied Camera Angles
The Cornell and AJD datasets are created from annotating imagery with similar camera views, effectively a top-down view. Consequently, training of GKNet 
uses imagery with a top-down camera perspective, which need not be the configured perspective for a manipulation setup. 
In such cases, it is important for the grasping algorithm to provide consistent grasping performance insensitive to deviations
from the top-down perspective.

### Command of running robotic grasping detection
```
python static_grasp.py dbmctdet_cornell --exp_id static_grasp --arch dla_34 --dataset cornell --fix_res --load_model ../models/model_dla34_cornell.pth --ae_threshold 1.0 --ori_threshold 0.24 --center_threshold 0.05

```

### Running ROS-side scripts

## Dynamic Grasping
An additional nuisance factor that may occur during manipulator deployment is object movement in a consistent direction, such as when grasping from 
a conveyor belt or other consistently moving support surface. A dynamic grasping experiment following the protocol of Morrison et al. (2019) 
tests real-time robustness to object movement.

### Command of running robotic grasping detection

### Running ROS-side scripts

## Bin Picking
The last experiment to perform is a bin picking experiment similar to Mahler and Goldberg (2017) and Morrison et al. (2019). 
The experiment evaluates robustness of grasp recognition to environmental clutter, which would thereby examine the generalization capabilities of GKNet 
since the network is trained with images containing only individual objects. Multi-object and cluttered environments are scenarios anticipated to be 
experienced for some real-world deployments, thus providing reliable grasp prediction to other sensed objects is an essential capability of 
grasp detection approaches.

### Command of running robotic grasping detection
```
python bin_picking.py dbmctdet --exp_id bin_picking --arch dlanonlocal_34 --dataset jac_coco_36 --load_model ../models/model_dla34_ajd.pth --ae_threshold 0.65 --ori_threshold 0.1745 --center_threshold 0.15
```

### Running ROS-side scripts

