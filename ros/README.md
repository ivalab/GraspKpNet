# ros

This directory contains ROS packages for running the GraspKpNet code as part of a larger system.
The primary mode of deployment is via Docker to simplify dependencies and configuration.

You should include this repo in your catkin workspace to use the `gknet_msgs` package.
You will need to be able to run `gknet` on your system directly if you would like to use the `gknet_perception` package.
Read the `docker` directory for more information on how to install the necessary prerequisites, otherwise use the container directly.

## testing

We can do some very basic testing with a simulated image from the d435i camera.
Run each of the following commands in their own terminal.

```bash
# publish static images to a topic for testing
docker compose run --rm gpu roslaunch gknet_perception static_image_publisher.launch

# view images on a topic
docker compose run --rm gpu rosrun image_view image_view image:=/gknet/annotated_image
# or run this locally e.g. WSL2 graphics issues
rosdep install image_view
rosrun image_view image_view image:=/gknet/annotated_image

# run the gknet perception module
docker compose run --rm gpu roslaunch gknet_perception detect.launch
```

We can also look at th keypoint results via `rostopic`:

```yaml
$ rostopic echo /gknet/keypoints

header:
  seq: 1
  stamp:
    secs: 0
    nsecs:         0
  frame_id: ''
keypoints:
  -
    left_middle: [197.0, 305.0]
    right_middle: [222.0, 286.0]
    center: [209.0, 295.0]
    score: 0.45235133171081543
  -
    left_middle: [217.0, 256.0]
    right_middle: [253.0, 234.0]
    center: [235.0, 245.0]
    score: 0.39672884345054626
  -
    left_middle: [382.0, 125.0]
    right_middle: [412.0, 170.0]
    center: [397.0, 147.0]
    score: 0.3544672131538391
  ...
```
