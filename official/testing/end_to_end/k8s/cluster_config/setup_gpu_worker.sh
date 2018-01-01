#!/bin/bash
# Startup script to run on Google Cloud instances.
# This script is used by create_gpu_workers.sh.
set -e
set -x

# Install cuda
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda; then
  # The 16.04 installer works with 16.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  apt-get update
  apt-get install cuda=8.0.61-1 -y
fi
apt-get install -y nvidia-cuda-toolkit

# Install corresponding cuDNN
if ! ls /usr/local/cuda/lib64 | grep cudnn; then
  wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/cudnn-8.0-linux-x64-v6.0.tgz
  tar -xvzf cudnn-8.0-linux-x64-v6.0.tgz
  cp cuda/include/cudnn.h /usr/local/cuda/include
  cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
  chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
fi

