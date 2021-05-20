# Physical Experiments
The physical experiments of GKNet is composed of four different grasping experiments: (a) static grasping, (b) grasping at varied camera viewpoints, (c) dynamic 
grasping and (d) bin picking. The design purpose of each experiment has been clarified in the manuscript. Please refer to Section 7 for details if interested.

This readme file documents how to run all physical experiments presented in the manuscript, which mainly consists of Python and ROS codes. All experiments were conducted with Kinect Xbox 360 but considering the potential bugs happened with [Freenect](https://github.com/OpenKinect/libfreenect), codes with camera Realsense is provided here.

## Python

### Static Grasping
This script will run GKNet to provide grasp detections via camera Realsense. The grasp detection results will be published on the ROS topic for ROS-side scripts to subscribe. 

```
python static_grasp.py dbmctdet_cornell --exp_id static_grasp --arch dla_34 --dataset cornell --fix_res --load_model ../models/model_dla34_cornell.pth --ae_threshold 1.0 --ori_threshold 0.24 --center_threshold 0.05

```

### Grasping at Varied Camera Angles
This script will run GKNet to provide grasp detections via camera Realsense. The grasp detection results will be published on the ROS topic for ROS-side scripts to subscribe. 

```
python static_grasp.py dbmctdet_cornell --exp_id static_grasp --arch dla_34 --dataset cornell --fix_res --load_model ../models/model_dla34_cornell.pth --ae_threshold 1.0 --ori_threshold 0.24 --center_threshold 0.05

```

### Dynamic Grasping


### Bin Picking
This script will run GKNet to provide grasp detections via camera Realsense. The grasp detection results will be published on the ROS topic for ROS-side scripts to subscribe. 
Additionally, this code will check if the pick bin is clean to determine if it is time to end the task. The result will also be published through ROS topic.

```
python bin_picking.py dbmctdet --exp_id bin_picking --arch dlanonlocal_34 --dataset jac_coco_36 --load_model ../models/model_dla34_ajd.pth --ae_threshold 0.65 --ori_threshold 0.1745 --center_threshold 0.15
```

## ROS
1. Install [ROS](http://wiki.ros.org/ROS/Installation).
2. Install camera driver for [Realsense](https://github.com/IntelRealSense/librealsense).
3. Download [ivaHandy](https://github.com/ivaROS/ivaHandy/tree/master/ros) and compile it under your ROS workspace for experiment codebase. 
4. Download [handy_experiment](https://github.com/ivaROS/handy_experiment) package and compile it under your ROS workspace for experiment codebase.
5. Run all setups step by step
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
6. Run the corresponding script for each experiment.
```
roslaunch handy_experiment static_grasp.launch
roslaunch handy_expeirment dynamic_grasp.launch
roslaunch handy_experiment bin_picking.launch
```
Note: static grasping and grasping at varied camera angles experiments share the same source code.
