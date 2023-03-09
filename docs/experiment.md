# Physical Experiments

The physical experiments of GKNet is composed of four different grasping experiments: (a) static grasping, (b) grasping at varied camera viewpoints, (c) dynamic
grasping and (d) bin picking. The design purpose of each experiment has been clarified in the manuscript. Please refer to Section 7 for details if interested.

This readme file documents how to run all physical experiments presented in the manuscript, which mainly consists of Python and ROS codes. All experiments in the manuscript were conducted with Kinect Xbox 360 but considering the potential bugs happened with old drivers, codes with camera Realsense is provided here.

## Python

We are supporting two types of cameras: Kinect Xbox 360 and Realsense D435. Personally I will recommend Realsense since Kinect is pretty old and its driver
isn't very stable.
To use Realsense D435, you just need to follow the installation instruction in the official [website](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md).
To use Kinect Xbox 360, since ROS used Python 2.7 as default python library, to run the physical experiment, you will need to create another anaconda environment with python 2.7. Considering
Pytorch dropped their support of newer versions for Python 2.7, you might need to install pytorch with a version that can be found and fit your Cuda version. If not, you
might consider install another cuda with older version. For installing multiple Cuda, you can refer to this [tutorial](https://towardsdatascience.com/installing-multiple-cuda-cudnn-versions-in-ubuntu-fcb6aa5194e2).

### Static Grasping

This script will run GKNet to provide grasp detections via camera Realsense/Kinect. The grasp detection results will be published on the ROS topic for ROS-side scripts to subscribe.

```
python static_grasp_rl.py/static_ grasp_kt.py dbmctdet_cornell --exp_id static_grasp --arch dla_34 --dataset cornell --fix_res --load_model ../models/model_dla34_cornell.pth --ae_threshold 0.6 --ori_threshold 0.24 --center_threshold 0.10 --scores_threshold 0.15 --center_weight 1.0

```

### Grasping at Varied Camera Angles

This script will run GKNet to provide grasp detections via camera Realsense/Kinect. The grasp detection results will be published on the ROS topic for ROS-side scripts to subscribe.

```
python static_grasp_rl.py/static_ grasp_kt.py dbmctdet_cornell --exp_id grasp_varied_angle --arch dla_34 --dataset cornell --fix_res --load_model ../models/model_dla34_cornell.pth --ae_threshold 0.6 --ori_threshold 0.24 --center_threshold 0.10 --scores_threshold 0.15 --center_weight 1.0

```

### Dynamic Grasping

This script will run GKNet to provide contincontinuousous grasp detection via camera Realsense/Kinect. The grasp detection results will be published on the ROS topic for ROS-side scripts to subscribe.

```
python dynamic_grasp_rl.py/dynamic_ grasp_kt.py dbmctdet_cornell --exp_id dynamic_grasp --arch dla_34 --dataset cornell --fix_res --load_model ../models/model_dla34_cornell.pth --ae_threshold 0.6 --ori_threshold 0.24 --center_threshold 0.10 --scores_threshold 0.15 --center_weight 1.0

```

### Bin Picking

This script will run GKNet to provide grasp detections via camera Realsense. The grasp detection results will be published on the ROS topic for ROS-side scripts to subscribe.
Additionally, this code will check if the pick bin is clean to determine if it is time to end the task. The result will also be published through ROS topic.

```
python bin_picking_rl.py/bin_picking_kt.py dbmctdet --exp_id bin_picking --arch dlanonlocal_34 --dataset jac_coco_36 --load_model ../models/model_dla34_ajd.pth --ae_threshold 0.65 --ori_threshold 0.1745 --center_threshold 0.15 --scores_threshold 0.15
```

## ROS

1. Install [ROS](http://wiki.ros.org/ROS/Installation).
2. Install [MoveIt!](https://moveit.ros.org/install/).
3. Install camera driver for [Kinect](http://wiki.ros.org/openni_kinect) or [Realsense](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md).
4. Download [ivaHandy](https://github.com/ivaROS/ivaHandy/tree/master/ros) and compile it under your ROS workspace for experiment codebase.
5. Download [handy_experiment](https://github.com/ivaROS/ivaHandyExperiment) package and compile it under your ROS workspace for experiment codebase.
6. Run all launch files for setup step by step.

```
cd handy_ws
roslaunch finalarm_cotrol controller_manager.launch
roslaunch finalarm_control start_controller.launch
roslaunch finalarm_description robot_state_pub.launch
roslaunch finalarm_moveit_config move_group.launch
roslaunch finalarm_moveit_config moveit_rviz.launch
```

You might meet the error after trying to launch the controller for each motor. To fix the error, type

```
sudo chmod 666 /dev/ttyUSB0
```

7. Run camera driver

```
roslaunch openni_launch openni.launch depth_registration:=true
```

8. Run the corresponding script for each experiment.

```
roslaunch handy_experiment static_grasp.launch
roslaunch handy_expeirment dynamic_grasp.launch
roslaunch handy_experiment bin_picking.launch
```

Note:

1. static grasping and grasping at varied camera angles experiments share the same source code.
2. Running dynamic grasping experiment requires running dbrt for estimating the gripper's pose. All codes here stored in [here](https://github.com/ivalab/dbrt_for_handy). Please follow the instructions to setup everything before launch dynamic_grasp.launch.
