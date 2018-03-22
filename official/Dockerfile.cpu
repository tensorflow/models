# Docker image for running examples in Tensorflow models.
# base_image depends on whether we are running on GPUs or non-GPUs
FROM ubuntu:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    python \
    python-pip \
    python-setuptools

RUN pip install tf-nightly

# Checkout tensorflow/models at HEAD
RUN git clone https://github.com/tensorflow/models.git /tensorflow_models

