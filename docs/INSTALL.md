# Installation

## Docker Installation (recommended)

Docker is a self-contained environment that can be used to run the code without having to install the dependencies.
We have built a docker image with Ubuntu 20.04, ROS Noetic, and PyTorch 1.13.1.
Make sure you have [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.
At the root of the repository, run the following command to build the container:

```bash
docker compose build base
```

Run the container:

```bash
docker compose run --rm gpu
```

The base container can build the code without having access to a GPU, and is useful for exploring the structure of the project and environment.
The actual code must be run from a container that has access to a GPU.

The docker-compose file at the root of the repository is configured to mount the current directory into the container.
This means that any changes you make to the code on your host machine will be reflected in the container.

## Manual Installation

The code was tested on Ubuntu 20.04 using Python 3.8 and [PyTorch](http://pytorch.org) v1.13.1.
NVIDIA GPUs are needed for both training and testing.
Reference the [Dockerfile](../docker/Dockerfile.noetic) for build instructions and dependencies.

1. [Optional but recommended] create a new virtual environment and activate it.

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

1. Install pytorch 1.13.1 with CUDA 11.7

   ```python
   pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
   ```

1. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

   ```bash
   cd /opt
   git clone https://github.com/cocodataset/cocoapi.git && \
      cd cocoapi &&\
      git checkout 8c9bcc3 && \
      cd PythonAPI && \
      make && \
      python setup.py install
   ```

1. Compile deformable convolutional (forked from [DCNv2](https://github.com/CharlesShang/DCNv2)).

   ```bash
   git clone https://github.com/acmiyaguchi/DCNv2.git && \
      cd DCNv2 && \
      git checkout pytorch_1.11 && \
      # https://github.com/facebookresearch/pytorch3d/issues/318
      FORCE_CUDA=1 \
      # https://github.com/pytorch/extension-cpp/issues/71
      TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX" \
      python setup.py build develop
   ```

1. Clone this repo:

   ```bash
   GKNet_ROOT=<path to clone>
   git clone https://github.com/ivalab/GraspKpNet.git $GKNet_ROOT
   ```

1. Install the requirements

   ```bash
   pip install -r requirements.txt
   ```

1. [Optional, only required if you are using extremenet or multi-scale testing] Compile NMS if your want to use multi-scale testing or test ExtremeNet.

   ```
   cd $GKNet_ROOT/vendor/nms
   make
   ```
