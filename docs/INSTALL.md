# Installation

## Manual Installation

The code was tested on Ubuntu 16.04, with [Anaconda](https://www.anaconda.com/download) Python 3.6 and [PyTorch](<(http://pytorch.org/)>) v0.4.1.
NVIDIA GPUs are needed for both training and testing.
The PyTorch v1.x is also supported.
The installation instruction can be followed by this [Issue](https://github.com/xingyizhou/CenterNet/issues/7).
The key difference comes from the Deconvolutional Network.
After install Anaconda:

0. [Optional but recommended] create a new conda environment.

   ```
   conda create --name GKNet python=3.6
   ```

   And activate the environment.

   ```
   conda activate GKNet
   ```

1. Install pytorch0.4.1:

   ```
   conda install pytorch=0.4.1 torchvision -c pytorch
   ```

   And disable cudnn batch normalization(Due to [this issue](https://github.com/xingyizhou/pytorch-pose-hg-3d/issues/16)).

   ```
   # PYTORCH=/path/to/pytorch # usually ~/anaconda3/envs/CenterNet/lib/python3.6/site-packages/
   # for pytorch v0.4.0
   sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
   # for pytorch v0.4.1
   sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
   ```

   For other pytorch version, you can manually open `torch/nn/functional.py` and find the line with `torch.batch_norm` and replace the `torch.backends.cudnn.enabled` with `False`.
   We observed slight worse training results without doing so.

2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   make
   python setup.py install --user
   ```

3. Clone this repo:

   ```
   GKNet_ROOT=/path/to/clone/GKNet
   git clone https://github.com/ruinianxu/GKNet $GKNet_ROOT
   ```

4. Install the requirements

   ```
   pip install -r requirements.txt
   ```

5. Compile deformable convolutional (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_0.4)).

   ```
   cd $GKNet_ROOT/src/lib/models/networks/DCNv2
   ./make.sh
   ```

6. [Optional, only required if you are using extremenet or multi-scale testing] Compile NMS if your want to use multi-scale testing or test ExtremeNet.

   ```
   cd $CenterNet_ROOT/src/lib/external
   make
   ```

## Docker Installation

We have built a docker image with Ubuntu 20.04, ROS Noetic, and PyTorch 1.13.1.
Make sure you have nvidia-docker installed.
Then build the docker image.
At the root of the repository, run the following command:

```bash
docker compose build base
```

Run the container:

```bash
docker compose run --rm gpu
```

The base container can build the code without having access to a GPU, and is useful for exploring the structure of the project and environment.
The actual code must be run from a container that has access to a GPU.
