# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Imagenet dataset.
# 2. Trains image compression model on patches from Imagenet.
# 3. Evaluates the models and writes sample images to disk.
#
# Usage:
# cd models/research/gan/image_compression
# ./launch_jobs.sh ${weight_factor} ${git_repo}
set -e

# Weight of the adversarial loss.
weight_factor=$1
if [[ "$weight_factor" == "" ]]; then
  echo "'weight_factor' must not be empty."
  exit
fi

# Location of the git repository.
git_repo=$2
if [[ "$git_repo" == "" ]]; then
  echo "'git_repo' must not be empty."
  exit
fi

# Base name for where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/compression-model

# Base name for where the evaluation images will be saved to.
EVAL_DIR=/tmp/compression-model/eval

# Where the dataset is saved to.
DATASET_DIR=/tmp/imagenet-data

export PYTHONPATH=$PYTHONPATH:$git_repo:$git_repo/research:$git_repo/research/slim:$git_repo/research/slim/nets

# A helper function for printing pretty output.
Banner () {
  local text=$1
  local green='\033[0;32m'
  local nc='\033[0m'  # No color.
  echo -e "${green}${text}${nc}"
}

# Download the dataset.
bazel build "${git_repo}/research/slim:download_and_convert_imagenet" 
"./bazel-bin/download_and_convert_imagenet" ${DATASET_DIR}

# Run the compression model.
NUM_STEPS=10000
MODEL_TRAIN_DIR="${TRAIN_DIR}/wt${weight_factor}"
Banner "Starting training an image compression model for ${NUM_STEPS} steps..."
python "${git_repo}/research/gan/image_compression/train.py" \
  --train_log_dir=${MODEL_TRAIN_DIR} \
  --dataset_dir=${DATASET_DIR} \
  --max_number_of_steps=${NUM_STEPS} \
  --weight_factor=${weight_factor} \
  --alsologtostderr
Banner "Finished training image compression model ${NUM_STEPS} steps."

# Run evaluation.
MODEL_EVAL_DIR="${TRAIN_DIR}/eval/wt${weight_factor}"
Banner "Starting evaluation of image compression model..."
python "${git_repo}/research/gan/image_compression/eval.py" \
  --checkpoint_dir=${MODEL_TRAIN_DIR} \
  --eval_dir=${MODEL_EVAL_DIR} \
  --dataset_dir=${DATASET_DIR} \
  --max_number_of_evaluations=1
Banner "Finished evaluation. See ${MODEL_EVAL_DIR} for output images."
