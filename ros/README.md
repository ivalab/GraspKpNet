# ros

This directory contains ROS packages for running the GraspKpNet code as part of a larger system.
The primary mode of deployment is via Docker to simplify dependencies and configuration.

You should include this repo in your catkin workspace to use the `gknet_msgs` package.
You will need to be able to run `gknet` on your system directly if you would like to use the `gknet_perception` package.
Read the `docker` directory for more information on how to install the necessary prerequisites, otherwise use the container directly.

## testing

Ensure you have the ability to run X11 applications via docker on your host.
See [this ROS wiki article](http://wiki.ros.org/docker/Tutorials/GUI) for hints.
In most cases, you should be able to set xhost to allow connections from docker:

```bash
xhost +local:docker

# in case this doesn't work, try this
xhost +local:root
```

This assumes that you have created a docker group and added your user to it [as per the official docs](https://docs.docker.com/engine/install/linux-postinstall/).

We can do some very basic testing with a simulated image from the d435i camera.
Run each of the following commands in their own terminal.

```bash
# run the unit tests
docker compose run --rm gpu catkin test
```

Run the demo with the static publisher:

```bash
docker compose run --rm gpu roslaunch gknet_perception demo.launch
```

We can also run most of these nodes individually:

```bash
# publish static images to a topic for testing
docker compose run --rm gpu roslaunch gknet_perception static_image_publisher.launch

# run the gknet perception module
docker compose run --rm gpu roslaunch gknet_perception detect.launch

# view images on a topic
docker compose run --rm gpu rosrun gknet_perception stream_camera.py --image-topic=/gknet/annotated_image

# and launch our manual object filter gui
docker compose run --rm gpu rosrun gknet_perception filter_gui.py
```

We can also look at keypoint results via `rostopic`:

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

```yaml
$ rostopic echo /gknet/object_filter

header:
  seq: 533
  stamp:
    secs: 0
    nsecs:         0
  frame_id: ''
objects:
  -
    bbox: [164, 196, 297, 268]
  -
    bbox: [174, 268, 334, 362]
  -
    bbox: [318, 146, 570, 290]
```
