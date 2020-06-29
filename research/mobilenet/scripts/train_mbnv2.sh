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

RUN_MODE=$1

cd "$( dirname "${BASH_SOURCE[0]}" )" || exit
DIR="$( pwd )"
SRC_DIR=${DIR}"/../../../"
export PYTHONPATH=${PYTHONPATH}:${SRC_DIR}
echo "PYTHONPATH="${PYTHONPATH}

GCS_MODEL_DIR=gs://tf_mobilenet/jobs
LOCAL_MODEL_DIR=${SRC_DIR}/mobilenet/jobs # for local testing

GCS_DATA_DIR=gs://tf_mobilenet/imagenet/imagenet-2012-tfrecord
LOCAL_DATA_DIR=~/Downloads/imagenet/imagenet-2012-tfrecord

NOW="$(date +"%Y%m%d_%H%M%S")"
JOB_PREFIX="MobilenetV2"
MODEL_NAME=mobilenet_v2
DATASET_NAME=imagenet2012

BATCH_SIZE=96
if [ "${RUN_MODE}" == "test" ]
then
    BATCH_SIZE=4
    MODEL_DIR=${LOCAL_MODEL_DIR}/${JOB_PREFIX}_${NOW}
    DATA_DIR=${LOCAL_DATA_DIR}
else
    MODEL_DIR=${GCS_MODEL_DIR}/${JOB_PREFIX}_${NOW}
    DATA_DIR=${GCS_DATA_DIR}
fi

# The training params are extracted from
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
# While the original implementation used a weight decay of 4e-5,
# tf.nn.l2_loss divides it by 2, so we halve this to compensate in Keras
TRAINER_ARGS="\
  --model_dir ${MODEL_DIR} \
  --optimizer_name rmsprop \
  --op_momentum 0.9 \
  --op_decay_rate 0.9 \
  --lr 0.045 \
  --lr_decay_rate 0.98 \
  --lr_decay_epochs 1.0 \
  --label_smoothing 0.1 \
  --dropout_rate 0.2 \
  --std_weight_decay 0.00002 \
  --truncated_normal_stddev 0.09 \
  --batch_norm_decay 0.9997 \
  --batch_size $BATCH_SIZE \
  --epochs 30
  "

echo $TRAINER_ARGS

python -m research.mobilenet.mobilenet_trainer \
  --model_name ${MODEL_NAME} \
  --dataset_name ${DATASET_NAME} \
  --data_dir ${DATA_DIR} \
  $TRAINER_ARGS
