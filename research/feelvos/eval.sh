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
# This script is used to locally run inference on DAVIS 2017. Users could also
# modify from this script for their use case. See train.sh for an example of
# local training.
#
# Usage:
#   # From the tensorflow/models/research/feelvos directory.
#   sh ./eval.sh
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

# Run embedding_utils_test first to make sure the PYTHONPATH is correctly set.
python "${WORK_DIR}"/utils/embedding_utils_test.py -v

# Go to datasets folder and download and convert the DAVIS 2017 dataset.
DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"
sh download_and_convert_davis17.sh

# Go to models folder and download and unpack the DAVIS 2017 trained model.
MODELS_DIR="models"
mkdir -p "${WORK_DIR}/${MODELS_DIR}"
cd "${WORK_DIR}/${MODELS_DIR}"
if [ ! -d "feelvos_davis17_trained" ]; then
  wget http://download.tensorflow.org/models/feelvos_davis17_trained.tar.gz
  tar -xvf feelvos_davis17_trained.tar.gz
  echo "model_checkpoint_path: \"model.ckpt-200004\"" > feelvos_davis17_trained/checkpoint
  rm feelvos_davis17_trained.tar.gz
fi
CHECKPOINT_DIR="${WORK_DIR}/${MODELS_DIR}/feelvos_davis17_trained/"

# Go back to orignal directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
DAVIS_FOLDER="davis17"
EXP_FOLDER="exp/eval_on_val_set"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DAVIS_FOLDER}/${EXP_FOLDER}/eval"
mkdir -p ${VIS_LOGDIR}

DAVIS_DATASET="${WORK_DIR}/${DATASET_DIR}/${DAVIS_FOLDER}/tfrecord"

python "${WORK_DIR}"/vis_video.py \
  --dataset=davis_2017 \
  --dataset_dir="${DAVIS_DATASET}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --checkpoint_dir="${CHECKPOINT_DIR}" \
  --logtostderr \
  --atrous_rates=12 \
  --atrous_rates=24 \
  --atrous_rates=36 \
  --decoder_output_stride=4 \
  --model_variant=xception_65 \
  --multi_grid=1 \
  --multi_grid=1 \
  --multi_grid=1 \
  --output_stride=8 \
  --save_segmentations
