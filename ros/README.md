# ros

This directory contains ROS packages for running the GraspKpNet code as part of a larger system.
The primary mode of deployment is via Docker to simplify dependencies and configuration.

You should include this repo in your catkin workspace to use the `gknet_msgs` package.
You will need to be able to run `gknet` on your system directly if you would like to use the `gknet_perception` package.
Read the `docker` directory for more information on how to install the necessary prerequisites, otherwise use the container directly.
