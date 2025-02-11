# !/bin/bash

## Install apt packages
# sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
# libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
# xz-utils tk-dev

## Install pyenv latest
# curl https://pyenv.run | bash
# exec $SHELL

## Install Python 3.9.7 using pyenv
# pyenv install 3.9.7
# pyenv global 3.9.7

# Install miniconda 4.12.0 
curl -o conda_install.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
bash conda_install.sh

# Configure virtualenv
conda create -n otp python=3.9 -y
conda activate otp

# Install torch
# conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y
# conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia -y
# conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch

# Install the latest mmcv
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
# pip install mmcv-full==2.0.0rc1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
# pip install mmcv==2.0.0rc1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
# pip install mmcv==2.0.0rc1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12/index.html
# pip install mmcv==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html


# torch 1.13 & cuda 11.6
conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia -y
# pip install mmcv==2.0.0rc1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
pip install mmcv==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html

# torch 1.12 & cuda 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
# pip install mmcv==2.0.0rc1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
pip install mmcv==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html

# Install mmdetection
# pip install mmdet==3.0.0rc4
pip install mmdet==2.26.0

# Install mmtracking
# pip install mmtrack==1.0.0rc1
pip install mmtrack==0.14.0

# Build mmtracking
# git clone https://github.com/open-mmlab/mmtracking.git
# cd mmtracking
# pip install -r requirements/build.txt
# pip install -v -e .

# Install dataset packages
pip install git+https://github.com/votchallenge/toolkit.git
pip install git+https://github.com/JonathonLuiten/TrackEval.git
pip install git+https://github.com/lvis-dataset/lvis-api.git
pip install git+https://github.com/TAO-Dataset/tao.git

# Install other packages
pip install wandb click numba notebook



# Switch

pip install mmcv==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
root@ai-contest-203:/workspace/object_tracking.pytorch# pip install mmcv-full==1.7.0
pip install mmdet==2.26.0
pip install mmtrack==0.14.0

pip install mmcv==2.0.0rc1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
pip install mmdet==3.0.0rc4
pip install mmtrack==1.0.0rc1