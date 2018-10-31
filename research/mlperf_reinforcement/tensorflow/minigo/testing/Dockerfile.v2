# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Docker image test-harness container for running python tests. Based off of
# https://github.com/kubeflow/kubeflow/tree/master/testing

FROM ubuntu:16.04

# add env we can debug with the image name:tag
ARG IMAGE_ARG
ENV IMAGE=${IMAGE_ARG}

RUN apt-get update && apt-get install -y \
    python \
    python3 \
    python3-pip \
    rsync \
    git \
    wget && \
    apt-get clean

ENV GCLOUD_VERSION 163.0.0
RUN wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-$GCLOUD_VERSION-linux-x86_64.tar.gz && \
    tar xf google-cloud-sdk-$GCLOUD_VERSION-linux-x86_64.tar.gz -C / && \
    rm google-cloud-sdk-$GCLOUD_VERSION-linux-x86_64.tar.gz && \
    /google-cloud-sdk/install.sh
ENV PATH "/google-cloud-sdk/bin:${PATH}"

WORKDIR /workspace
COPY bootstrap_v2.sh /workspace/bootstrap_v2.sh

COPY staging/requirements.txt /workspace/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r /workspace/requirements.txt
RUN pip3 install "tensorflow>=1.5,<1.6"

ENTRYPOINT ["/bin/sh", "/workspace/bootstrap_v2.sh"]
