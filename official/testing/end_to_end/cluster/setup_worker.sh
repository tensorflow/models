#!/bin/bash
# Startup script to run on Google Cloud instances.
set -e
set -x

# Install cuda if a CUDA-capable GPU is detected
if lspci | grep -i nvidia; then
  echo "Installing NVIDIA drivers"

  apt-get purge nvidia-*
  add-apt-repository ppa:graphics-drivers -y
  apt-get update
  apt-get install nvidia-384 -y

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
