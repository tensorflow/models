#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
# This script is used to run local training on DAVIS 2017. Users could also
# modify from this script for their use case. See eval.sh for an example of
# local inference with a pre-trained model.
#
# Note that this script runs local training with a single GPU and a smaller crop
# and batch size, while in the paper, we trained our models with 16 GPUS with
# --num_clones=2, --train_batch_size=6, --num_replicas=8,
# --training_number_of_steps=200000, --train_crop_size=465,
# --train_crop_size=465.
#
# Usage:
#   # From the tensorflow/models/research/feelvos directory.
#   sh ./train.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim:`pwd`/feelvos

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/feelvos"

# Set up the working directories.
DATASET_DIR="datasets"
DAVIS_FOLDER="davis17"
DAVIS_DATASET="${WORK_DIR}/${DATASET_DIR}/${DAVIS_FOLDER}/tfrecord"
EXP_FOLDER="exp/train"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DAVIS_FOLDER}/${EXP_FOLDER}/train"
mkdir -p ${TRAIN_LOGDIR}

# Go to datasets folder and download and convert the DAVIS 2017 dataset.
DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"
sh download_and_convert_davis17.sh

# Go to models folder and download and unpack the COCO pre-trained model.
MODELS_DIR="models"
mkdir -p "${WORK_DIR}/${MODELS_DIR}"
cd "${WORK_DIR}/${MODELS_DIR}"
if [ ! -d "xception_65_coco_pretrained" ]; then
  wget http://download.tensorflow.org/models/xception_65_coco_pretrained_2018_10_02.tar.gz
  tar -xvf xception_65_coco_pretrained_2018_10_02.tar.gz
  rm xception_65_coco_pretrained_2018_10_02.tar.gz
fi
INIT_CKPT="${WORK_DIR}/${MODELS_DIR}/xception_65_coco_pretrained/x65-b2u1s2p-d48-2-3x256-sc-cr300k_init.ckpt"

# Go back to orignal directory.
cd "${CURRENT_DIR}"

python "${WORK_DIR}"/train.py \
  --dataset=davis_2017 \
  --dataset_dir="${DAVIS_DATASET}" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --tf_initial_checkpoint="${INIT_CKPT}" \
  --logtostderr \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --decoder_output_stride=4 \
  --model_variant=xception_65 \
  --multi_grid=1 \
  --multi_grid=1 \
  --multi_grid=1 \
  --output_stride=16 \
  --weight_decay=0.00004 \
  --num_clones=1 \
  --train_batch_size=1 \
  --train_crop_size=300 \
  --train_crop_size=300
