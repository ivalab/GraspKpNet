FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

# NOTE: We need cuda devel to build DCNv2, but we don't necessarily need it to
# run inference. This should probably become a multi-part build if we care about
# deployment size.

# take the core ros image on top of the cuda focal image
# https://github.com/osrf/docker_images/blob/master/ros/noetic/ubuntu/focal/ros-core/Dockerfile

# noninteractive to avoid tzdata prompt
ENV DEBIAN_FRONTEND noninteractive
ENV ROS_DISTRO noetic
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ros-noetic-ros-core=1.5.0-1* \
    python3-catkin-tools \
    python3-rosdep \
    && \
    apt clean

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    apt-utils \
    build-essential \
    git \
    python3-dev \
    python3-pip \
    python-is-python3 \
    ninja-build \
    libgl1-mesa-glx \
    libsm-dev \
    wget \
    && \
    apt clean

RUN pip install --upgrade pip wheel cython

RUN apt update && \
    apt install -y --no-install-recommends \
    # install ros packages
    # rosdep install --from-paths src -r -y --simulate
    ros-noetic-ros-pytest \
    ros-noetic-cv-bridge \
    # additional package for viewing images
    ros-noetic-image-view \
    && \
    apt clean

# install cocoapi
WORKDIR /opt
RUN git clone https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi &&\
    git checkout 8c9bcc3 && \
    cd PythonAPI && \
    make && \
    python setup.py install

# install nms
WORKDIR /app
ADD vendor/ vendor/
RUN cd vendor/nms && \
    python setup.py build install

# install requirements (most importantly pytorch) and DCNv2
ADD requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# this is fairly slow, and the ABI is dependent on both cuda and pytorch. We
# need to make sure we build and run with the same version of pytorch.
WORKDIR /opt
# fork of https://github.com/lbin/DCNv2.git with ability to compile with cuda
# without an nvidia gpu attached
RUN git clone https://github.com/acmiyaguchi/DCNv2.git && \
    cd DCNv2 && \
    git checkout pytorch_1.11 && \
    # https://github.com/facebookresearch/pytorch3d/issues/318
    FORCE_CUDA=1 \
    # https://github.com/pytorch/extension-cpp/issues/71
    TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX" \
    ./make.sh

# download the dla34 models
# NOTE: if these links are broken for some reason, this can safely be removed
# and instead have the models mounted from the host. You will also probably want
# to download the full set of pretrained models to mount onto the container
# during development.
RUN mkdir -p /opt/models && \
    cd /opt/models && \
    wget https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/models/model_dla34_cornell.pth && \
    wget https://f004.backblazeb2.com/file/acm-ivalab/GraspKpNet/models/model_dla34_ajd.pth

# add the source code to the image
# we also link /app to /catkin_ws/src/app as a shortcut to the source
WORKDIR /catkin_ws/src/app
RUN ln -s /catkin_ws/src/app /app

# let's only copy in the package, in case we don't change much of it

ADD gknet/ ./gknet/
ADD scripts/ ./scripts/
ADD setup.py ./setup.py
RUN pip install -e .

# cache the model weights for default inference
RUN python scripts/load_detector.py dbmctdet_cornell \
    --load_model /opt/models/model_dla34_cornell.pth

# Now we can add the rest of the app
ADD ./ .

WORKDIR /catkin_ws
RUN catkin config --init --extend /opt/ros/noetic && \
    rosdep init && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    catkin build

WORKDIR /catkin_ws/src/app

# setup entrypoint
COPY ./docker/bin/ros_entrypoint.sh /
ENTRYPOINT ["/ros_entrypoint.sh",  "stdbuf", "-o", "L"]
CMD ["bash"]
