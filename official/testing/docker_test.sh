#!/usr/bin/env bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# 
# DO NOT MODIFY THIS FILE. Add tests to be executed in test_models.sh
# Usage: docker_test.sh [--docker-image <DOCKER_IMG_NAME>]
#
# DOCKERFILE_IMG_NAME: (Optional) The tensorflow docker container version
#                  If this optional value is not supplied (via the
#                  --docker-image flag), the default latest tensorflow docker
#                  will be used.
#


# SETUP
# Default exit status
EXIT=0

# Get current directory path to mount
export WORKSPACE=${PWD}

DOCKER_BINARY="docker"

# Decide docker image and tag
if [[ "$1" == "--docker-image" ]]; then
  DOCKER_IMG_NAME="$2"
  echo "Using specified docker tensorflow image and tag: ${DOCKER_IMG_NAME}"
  shift 2
else
  DOCKER_IMG_NAME="tensorflow/tensorflow:1.3.0"
  echo "Using the default docker tensorflow image and tag: ${DOCKER_IMG_NAME}"
fi

# Specify which test is to be run
COMMAND="./official/testing/test_models.sh"

# RUN
${DOCKER_BINARY} run \
    -v ${WORKSPACE}:/workspace \
    -w /workspace \
    -t \
    ${DOCKER_IMG_NAME} \
    ${COMMAND} \
    || EXIT=$?


# TEARDOWN
${DOCKER_BINARY} rmi \
  -f \
  ${DOCKER_IMG_NAME}

git clean -dfx

# Return exit status
exit ${EXIT}