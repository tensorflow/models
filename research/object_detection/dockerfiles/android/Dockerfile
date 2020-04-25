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

# Pull TF nightly-devel docker image
FROM tensorflow/tensorflow:nightly-devel

# Get the tensorflow models research directory, and move it into tensorflow
# source folder to match recommendation of installation
RUN git clone --depth 1 https://github.com/tensorflow/models.git && \
    mv models /tensorflow/models


# Install gcloud and gsutil commands
# https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu
RUN apt-get -y update && apt-get install -y gpg-agent && export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y


# Install the Tensorflow Object Detection API from here
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

# Install object detection api dependencies - use non-interactive mode to set default tzdata config during installation
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


# Install wget (to make life easier below) and editors (to allow people to edit
# the files inside the container)
RUN apt-get install -y wget vim emacs nano


# Grab various data files which are used throughout the demo: dataset,
# pretrained model, and pretrained TensorFlow Lite model. Install these all in
# the same directories as recommended by the blog post.

# Pets example dataset
RUN mkdir -p /tmp/pet_faces_tfrecord/ && \
    cd /tmp/pet_faces_tfrecord && \
    curl "http://download.tensorflow.org/models/object_detection/pet_faces_tfrecord.tar.gz" | tar xzf -

# Pretrained model
# This one doesn't need its own directory, since it comes in a folder.
RUN cd /tmp && \
    curl -O "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz" && \
    tar xzf ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz && \
    rm ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz

# Trained TensorFlow Lite model. This should get replaced by one generated from
# export_tflite_ssd_graph.py when that command is called.
RUN cd /tmp && \
    curl -L -o tflite.zip \
    https://storage.googleapis.com/download.tensorflow.org/models/tflite/frozengraphs_ssd_mobilenet_v1_0.75_quant_pets_2018_06_29.zip && \
    unzip tflite.zip -d tflite && \
    rm tflite.zip


# Install Android development tools
# Inspired by the following sources:
# https://github.com/bitrise-docker/android/blob/master/Dockerfile
# https://github.com/reddit/docker-android-build/blob/master/Dockerfile

# Set environment variables
ENV ANDROID_HOME /opt/android-sdk-linux
ENV ANDROID_NDK_HOME /opt/android-ndk-r14b
ENV PATH ${PATH}:${ANDROID_HOME}/tools:${ANDROID_HOME}/tools/bin:${ANDROID_HOME}/platform-tools

# Install SDK tools
RUN cd /opt && \
    curl -OL https://dl.google.com/android/repository/sdk-tools-linux-4333796.zip && \
    unzip sdk-tools-linux-4333796.zip -d ${ANDROID_HOME} && \
    rm sdk-tools-linux-4333796.zip

# Accept licenses before installing components, no need to echo y for each component
# License is valid for all the standard components in versions installed from this file
# Non-standard components: MIPS system images, preview versions, GDK (Google Glass) and Android Google TV require separate licenses, not accepted there
RUN yes | sdkmanager --licenses

# Install platform tools, SDK platform, and other build tools
RUN yes | sdkmanager \
    "tools" \
    "platform-tools" \
    "platforms;android-27" \
    "platforms;android-23" \
    "build-tools;27.0.3" \
    "build-tools;23.0.3"

# Install Android NDK (r14b)
RUN cd /opt && \
    curl -L -o android-ndk.zip http://dl.google.com/android/repository/android-ndk-r14b-linux-x86_64.zip && \
    unzip -q android-ndk.zip && \
    rm -f android-ndk.zip

# Configure the build to use the things we just downloaded
RUN cd /tensorflow && \
    printf '\n\n\nn\ny\nn\nn\nn\ny\nn\nn\nn\nn\nn\nn\n\ny\n%s\n\n\n' ${ANDROID_HOME}|./configure


WORKDIR /tensorflow
