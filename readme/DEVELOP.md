# Develop
This document provides the instruction for developing GKNet with the new dataset, task or architecture.

## New dataset
The implementation of new dataset can be composed of two main steps.

- Define a class for your own dataset which contains the information of (1) the number of class for orientation, (2) the input data resolution, (3) the mean of RGD channels and (4) the std of RGD channels. The initialization mainly does the job of loading the split file, the image path, and the annotation path. The dataset intilization file locates at `src/lib/datasets/dataset`. You can mimic either the definition of the Cornell or AJD. 

- Define a sampler. Since the detailed definition for grasps can be different in different datasets, it will be better redefine a new data sampler for your own dataset. For example, the grasp annotation in some datasets can be min_x, min_y, max_x and max_y while some can be center_x, center_y, width, height. Additionally, the definition of orientation can also be different. It can be clockwise or counter-clockwise, which will affect when you convert the horizontal bounding box to rotated one. The dataset sampler
files locate at `src/lib/datasets/sample`. You need to make sure all these informations are correct.

The easiest way might be converting the data format of your own dataset to COCO format and follow the definition of AJD.

## New task
As shown in the manuscript, there are a lot of other way to represent the grasp as a set of keypoints. To define a new task, for example representing the grasp as the set 
of three corner keypoints, you will need to add files to `src/lib/datasets/sample`, `src/lib/datasets/trains` and `src/lib/datasets/detectors`.
The sampler defines ground-truth generation, the trainer defines training targets and the detector defines testing targets.

## New architecture
- Add your model file to `src/lib/models/networks/`. The model should accept a dict `heads` of `{name: channels}`, which specify the name of each network output and its number of channels. Make sure your model returns a list (for multiple stages. Single stage model should return a list containing a single element.). The element of the list is a dict contraining the same keys with `heads`.
- Add your model in `model_factory` of `src/lib/models/model.py`.
