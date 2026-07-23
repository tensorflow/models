# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# #==========================================================================

# Pull TF 1.15.0 docker image
FROM tensorflow/tensorflow:1.15.0

# Install required packages 
RUN apt-get -y update && apt-get -y install git curl gpg-agent lsb-release unzip 

# Get the tensorflow models research directory, and move it into tensorflow
# source folder to match recommendation of installation
RUN git clone --depth 1 https://github.com/tensorflow/models.git && \
    mkdir /tensorflow && mv models /tensorflow/models


# Install gcloud and gsutil commands
# https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu
RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y


# Install the Tensorflow Object Detection API from here
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

# Install object detection api dependencies
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y protobuf-compiler python-pil python-lxml python-tk && \
    pip install Cython && \
    pip install contextlib2 && \
    pip install jupyter && \
    pip install matplotlib

# Install pycocoapi
RUN git clone --depth 1 https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    make -j8 && \
    cp -r pycocotools /tensorflow/models/research && \
    cd ../../ && \
    rm -rf cocoapi

# Get protoc 3.0.0, rather than the old version already in the container
RUN curl -OL "https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip" && \
    unzip protoc-3.0.0-linux-x86_64.zip -d proto3 && \
    mv proto3/bin/* /usr/local/bin && \
    mv proto3/include/* /usr/local/include && \
    rm -rf proto3 protoc-3.0.0-linux-x86_64.zip

# Run protoc on the object detection repo
RUN cd /tensorflow/models/research && \
    protoc object_detection/protos/*.proto --python_out=.

# Set the PYTHONPATH to finish installing the API
ENV PYTHONPATH $PYTHONPATH:/tensorflow/models/research:/tensorflow/models/research/slim


# Install wget (to make life easier) and editors (to allow people to edit
# the files inside the container)
RUN apt-get install -y wget vim emacs nano


WORKDIR /tensorflow
