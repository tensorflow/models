#!/bin/bash

# This script sets up the required environment by installing necessary packages.

# Update the package lists for upgrades and new package installations.
sudo apt update

# Download the Docker installation script from Docker's official site.
curl -fsSL https://get.docker.com -o get-docker.sh

# Run the Docker installation script.
sudo sh get-docker.sh

# Install python3-pip, a package manager for Python, and ffmpeg, a multimedia 
# framework.
sudo apt install python3-pip ffmpeg

# Install Python packages:
# ffmpeg-python: Python bindings for FFmpeg
# opencv-python: Open source computer vision library for Python
# pandas: Data analysis library for Python
# pandas-gbq: Integration between pandas and Google BigQuery
# google-cloud-bigquery: Google BigQuery API client library
# google-auth: Authentication library for Google services
# trackpy: Particle-tracking toolkit
# google-cloud-storage: Google Cloud Storage API client library
pip3 install natsort absl-py opencv-python pandas pandas-gbq \
  google-cloud-bigquery google-auth trackpy google-cloud-storage tensorflow \
  scikit-image scikit-learn webcolors

# Cloning project directory from TF Model Garden for postprocessing
# and preprocessing functions.
git clone --depth 1 https://github.com/tensorflow/models.git