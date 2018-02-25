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
# 1. Downloads the MNIST dataset.
# 2. Trains an unconditional model on the MNIST training set using a
#    tf.Estimator.
# 3. Evaluates the models and writes sample images to disk.
#
#
# Usage:
# cd models/research/gan/mnist_estimator
# ./launch_jobs.sh ${git_repo}
set -e

# Location of the git repository.
git_repo=$1
if [[ "$git_repo" == "" ]]; then
  echo "'git_repo' must not be empty."
  exit
fi

# Base name for where the evaluation images will be saved to.
EVAL_DIR=/tmp/mnist-estimator

# Where the dataset is saved to.
DATASET_DIR=/tmp/mnist-data

export PYTHONPATH=$PYTHONPATH:$git_repo:$git_repo/research:$git_repo/research/gan:$git_repo/research/slim

# A helper function for printing pretty output.
Banner () {
  local text=$1
  local green='\033[0;32m'
  local nc='\033[0m'  # No color.
  echo -e "${green}${text}${nc}"
}

# Download the dataset.
python "${git_repo}/research/slim/download_and_convert_data.py" \
  --dataset_name=mnist \
  --dataset_dir=${DATASET_DIR}

# Run unconditional GAN.
NUM_STEPS=1600
Banner "Starting training GANEstimator ${NUM_STEPS} steps..."
python "${git_repo}/research/gan/mnist_estimator/train.py" \
  --max_number_of_steps=${NUM_STEPS} \
  --eval_dir=${EVAL_DIR} \
  --dataset_dir=${DATASET_DIR} \
  --alsologtostderr
Banner "Finished training GANEstimator ${NUM_STEPS} steps. See ${EVAL_DIR} for output images."
