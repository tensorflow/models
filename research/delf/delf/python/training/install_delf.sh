#!/bin/bash

# Copyright 2020 Google Inc. All Rights Reserved.
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
# ==============================================================================

# Installs the DELF package and all its dependencies.

protoc_folder="protoc"
protoc_url="https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip"
tf_slim_git_repo="https://github.com/google-research/tf-slim.git"

handle_exit_code() {
    # Fail gracefully in case of an exit code different than 0
    exit_code=$1
    error_message=$2
    if [ ${exit_code} -ne 0 ]; then
        echo "${error_message} Exiting."
        exit 1
    fi
}

install_tensorflow() {
    # Install TensorFlow 2.2. Exit if Python version is 3.7.x or it TensorFlow installation fails

    echo "Checking that Python version is different than 3.7.x"
    python -c "import sys; assert (sys.version_info < (3, 7, 0) or sys.version_info > (3, 8, 0))" 1>/dev/null 2>/dev/null
    local exit_code=$?
    handle_exit_code ${exit_code} "Python version different than 3.7 is a pre-requisite. Current version ${python_version}."
    
    echo "Installing TensorFlow 2.2"
    pip3 install --upgrade tensorflow==2.2.0
    local exit_code=$?
    handle_exit_code ${exit_code} "Unable to install Tensorflow 2.2."
    
    echo "Installing TensorFlow 2.2 for GPU"
    pip3 install --upgrade tensorflow-gpu==2.2.0
    local exit_code=$?
    handle_exit_code ${exit_code} "Unable to install Tensorflow for GPU 2.2.0."
}

install_tf_slim() {
    # Install TF-Slim from source
    echo "Installing TF-Slim from source: ${git_repo}"
    git clone ${tf_slim_git_repo}
    local exit_code=$?
    handle_exit_code ${exit_code} "Unable to clone TF-Slim repository ${tf_slim_git_repo}."
    pushd .
    cd tf-slim
    pip3 install .
    popd
    rm -rf tf-slim
}

download_protoc() {
    # Installs the Protobuf compiler protoc
    echo "Downloading Protobuf compiler from ${protoc_url}"
    curl -L -Os ${protoc_url}
    local exit_code=$?
    handle_exit_code ${exit_code} "Unable to download Protobuf compiler from ${tf_slim_git_repo}."

    mkdir ${protoc_folder}
    local protoc_archive=`basename ${protoc_url}`
    unzip ${protoc_archive} -d ${protoc_folder}
    local exit_code=$?
    handle_exit_code ${exit_code} "Unable to unzip Protobuf compiler from ${protoc_archive}."
    
    rm ${protoc_archive}
}

compile_delf_protos() {
    # Compiles DELF protobufs from tensorflow/models/research/delf/delf using the potoc compiler 
    echo "Compiling DELF Protobufs"
    PATH_TO_PROTOC="`pwd`/${protoc_folder}"
    pushd . > /dev/null
    cd ../..
    ${PATH_TO_PROTOC}/bin/protoc protos/*.proto --python_out=.
    local exit_code=$?
    handle_exit_code ${exit_code} "Unable to compile DELF Protobufs."
    popd > /dev/null
    exit 1
}

cleanup_protoc() {
    # Removes the downloaded Protobuf compiler protoc after the installation of the DELF package
    echo "Cleaning up Protobuf compiler download"
    rm -rf ${protoc_folder}
}

install_python_libraries() {
    # Installs Python libraries upon which the DELF package has dependencies
    echo "Installing matplotlib, numpy, scikit-image, scipy and python3-tk"
    pip3 install matplotlib numpy scikit-image scipy
    sudo apt-get -y install python3-tk
}

install_object_detection() {
    # Installs the object detection package from tensorflow/models/research
    echo "Installing object detection"
    pushd . > /dev/null
    cd ../../../..
    export PYTHONPATH=$PYTHONPATH:`pwd`
    pip3 install .
    popd > /dev/null
}

install_delf_package() {
    # Installs the object detection package from tensorflow/models/research/delf/delf
    echo "Installing DELF package"
    pushd . > /dev/null
    cd ../..
    pip3 install -e .
    popd > /dev/null
}

post_install_check() {
    # Checks the DEFL package has been successfully installed
    echo "Checking DELF package installation"
    pip3 install -e .
}

install_delf() {
    # Orchestrates DEFL package installation
    #install_tensorflow
    #install_tf_slim
    #download_protoc
    compile_delf_protos
    cleanup_protoc
    install_python_libraries
    install_object_detection
    install_delf_package
    post_install_check
}

install_delf