# Installation

## Docker Installation (recommended)

Docker is a self-contained environment that can be used to run the code without having to install the dependencies.
We have built a docker image with Ubuntu 20.04, ROS Noetic, and PyTorch 1.13.1.
Make sure you have [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.
The newer version of PyTorch means that the host machine must have a CUDA 11.7 compatible driver.
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

## Downloading models

In addition to the model weight and dataset links referenced in the main README, we have hosted the following on a public mirror.
The purpose is to provide a stable link for a subset of models used during model inference.
In general, prefix the filename with `https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/` to download the file.
Please use discretion when downloading the larger files.

- Pretrained_models (zipped, see https://www.dropbox.com/sh/eicrmhhay2wi8fy/AAAGrToUcdp0tO-F732Xhsxwa?dl=0)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/Pretrained_models.zip (5.1 GB, sha1 4ac3a5ded51c46b6abdd552d2b709d84663e4cfd)
- Pretrained_models
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/models/ctdet_coco_dla_2x.pth (77.2 MB, sha1 f77895ebefe56bd4a6abd173f919d17e6ef60c79)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/models/model_alexnet_ajd.pth (26.5 MB, sha1 5de7fdac3325f5012bc98a4e5516c4d463e8c1ea)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/models/model_alexnet_cornell.pth (26.5 MB, sha1 4aedef106634c2bb7242ef5aa2659c8deb69ccfd)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/models/model_dla34_ajd.pth (80 MB, sha1 cb874a470d3f54801bdd53468411da3ee2d2bbf7)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/models/model_dla34_cornell.pth (235.5 MB, sha1 0c19ab75c54a685514500b10336c84d9b81a5c56)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/models/model_hg104_ajd.pth (2258.8 MB, sha1 1b3ac9a363a9095608c8f952c6f7ae82ccdc3a9a)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/models/model_hg104_cornell.pth (753.3 MB, sha1 4d9235d2f85a61e902cf643db275d8a327f9806a)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/models/model_hg52_ajd.pth (376.2 MB, sha1 d9b9478a1f60dc3cc19c0dc576c9da293d5e191c)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/models/model_hg52_cornell.pth (376.2 MB, sha1 6aeca75cc4276c05933451ea8a2500853ebcc3b8)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/models/model_resnet18_ajd.pth (63.7 MB, sha1 f9bb4a0dff92197b19ad9d387c209abf7164bf2d)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/models/model_resnet18_cornell.pth (63.7 MB, sha1 14a6be95db8b032bc3aac4f14720b752d10c5b88)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/models/model_resnet50_ajd.pth (135 MB, sha1 3220c95b4dfbf8f0b49370517fea8f071f19b838)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/models/model_resnet50_cornell.pth (135 MB, sha1 58ef7b9d0a990faa2be53e1293ec4a5c0f876cca)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/models/model_vgg16_ajd.pth (77.3 MB, sha1 ccc57a04f20b304a4661c03c5bcabbb849118984)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/models/model_vgg16_cornell.pth (77.3 MB, sha1 f0d3d5275b46c923c6b5bb48171577a51f746ac5)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/models/model_dla34_cornell.pth (80 MB, sha1 efa86a33444c59e3bde21e4314104d133cda310c)
- Cornell_256_augmented (see https://www.dropbox.com/scl/fo/3tudn7uorjygxd040fcmn/h?dl=0)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/Cornell_256_augmented.tar.gz (5.8 GB, sha1 4385f47eeea8cc020f79f5c2248108606ea8b689)
- Abridged_Jacquard (see https://smartech.gatech.edu/handle/1853/64897)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/Abridged_Jacquard/instances_grasps_test2018.json (0.2 GB, sha1 e9bf723f3f914872084d05ba93e27e21e431b968)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/Abridged_Jacquard/instances_grasps_test2018_edge_denseanno_filter.json (0.3 GB, sha1 cff53c4d152ea7a7eb1fca7b9169e52a10b6b9ef)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/Abridged_Jacquard/instances_grasps_test2018_filter.json (0.2 GB, sha1 98ebbf969f37367d7e48b4437db09b9888df99c1)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/Abridged_Jacquard/instances_grasps_train2018_edge_denseanno_filter.json (1.3 GB, sha1 86698fbfb3f5f34c4344a3684bd43809652f53f4)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/Abridged_Jacquard/instances_grasps_train2018_filter.json (1 GB, sha1 47c5e116bd0172faa8e21a092fa55f60ce000ecc)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/Abridged_Jacquard/README.doc (0 GB, sha1 52a60290b299945527b3fe05666eb1c269347468)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/Abridged_Jacquard/README.md (0 GB, sha1 42bb95169a14bdc4f3142b5197faae05507dbe7a)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/Abridged_Jacquard/test_annotations_0_5.tar.gz (0.1 GB, sha1 bb532c43b54ed0e81151cf73996eaaecfc764e23)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/Abridged_Jacquard/test_annotations_6_11.tar.gz (0.1 GB, sha1 3fb969ccad48ec1d3ae51ec64cbf3d7b2957ac4a)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/Abridged_Jacquard/train_annotations_0_5.tar.gz (0.6 GB, sha1 813402cbaf4e769514be37854a8e4863fdfd3d1e)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/Abridged_Jacquard/train_annotations_6_11.tar.gz (0.5 GB, sha1 f8ac6f5afedd2dc688b737bbc9f997cac393f68d)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/Abridged_Jacquard/train_grasps_test2018_0_5.tar.gz (0.7 GB, sha1 c17bd238fd900e2129190c8109d1c271f5ca7dbd)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/Abridged_Jacquard/train_grasps_test2018_6_11.tar.gz (0.7 GB, sha1 5d100cba83fa4ca35be9fd22d5f06c5530cbf55e)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/Abridged_Jacquard/train_grasps_train2018_0_5.tar.gz (2.9 GB, sha1 a7af278749ecfc84dfa1e3f63bf7303aa110b92c)
  - https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/Abridged_Jacquard/train_grasps_train2018_6_11.tar.gz (2.7 GB, sha1 595fa7429ef613ac3caa7550dbc5052455f03be7)
