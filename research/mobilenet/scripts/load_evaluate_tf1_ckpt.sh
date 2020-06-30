#!/usr/bin/env bash

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

# The checkpoints can be found here.
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md
# https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet


cd "$( dirname "${BASH_SOURCE[0]}" )" || exit
DIR="$( pwd )"
SRC_DIR=${DIR}"/../../../"
export PYTHONPATH=${PYTHONPATH}:${SRC_DIR}
echo "PYTHONPATH="${PYTHONPATH}

MODEL_NAME=mobilenet_v1
DATASET_NAME=imagenet2012
CHECKPOINT_PATH=/Users/luoshixin/Downloads/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt

GCS_DATA_DIR=gs://tf_mobilenet/imagenet/imagenet-2012-tfrecord
LOCAL_DATA_DIR=~/Downloads/imagenet/imagenet-2012-tfrecord

python -m research.mobilenet.tf1_loader.mobilenet_evaluator \
  --model_name ${MODEL_NAME} \
  --dataset_name ${DATASET_NAME} \
  --data_dir ${GCS_DATA_DIR} \
  --checkpoint_path ${CHECKPOINT_PATH}