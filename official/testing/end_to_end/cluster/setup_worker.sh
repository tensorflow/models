#!/bin/bash
# Startup script to run on Google Cloud instances.
set -e
set -x

# TODO(karmel): Installing CUDA shouldn't be necessary if using nvidia-docker;
# just need the drivers apt-get install nvidia 387 384
# Install cuda if a CUDA-capable GPU is detected
if lspci | grep -i nvidia; then
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


  # Install nvidia-docker
  # Add the package repositories
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
    sudo apt-key add -
  curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt-get update

  # Install nvidia-docker2 and reload the Docker daemon configuration
  sudo apt-get install -y nvidia-docker2

  # Make nvidia-docker the default by appending to the daemon.js config
  apt-get install -y jq
  jq '. + {"default-runtime": "nvidia"}' /etc/docker/daemon.json > docker_daemon.json
  mv docker_daemon.json /etc/docker/daemon.json

  # Restart dockerd
  sudo pkill -SIGHUP dockerd

fi
