FROM tensorflow/tensorflow:1.15.2-gpu-py3

ARG DEBIAN_FRONTEND=noninteractive

# key rotation
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget \
    unzip

RUN pip install --upgrade pip
RUN pip install -U numpy==1.19.5
RUN pip install -U pycocotools==2.0.6
RUN pip install tf_slim
RUN pip install keras==2.2.5
RUN pip uninstall -y protobuf
RUN pip install protobuf==3.19.6
RUN apt-get install -y protobuf-compiler
RUN apt-get install -y libgl1-mesa-glx

# Install gcloud and gsutil commands
# https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu
RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y

# Add new user to avoid running as root
RUN useradd -ms /bin/bash tensorflow
USER tensorflow
WORKDIR /home/tensorflow

# Copy this version of of the model garden into the image
COPY --chown=tensorflow . /home/tensorflow/models

# Compile protobuf configs
RUN (cd /home/tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /home/tensorflow/models/research/

RUN cp object_detection/packages/tf1/setup.py ./
ENV PATH="/home/tensorflow/.local/bin:${PATH}"

RUN python -m pip install --user -U pip
RUN python -m pip install --user .

ENV TF_CPP_MIN_LOG_LEVEL 3
