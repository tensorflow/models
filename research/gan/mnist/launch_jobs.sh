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
# 2. Trains an unconditional, conditional, or InfoGAN model on the MNIST
#    training set.
# 3. Evaluates the models and writes sample images to disk.
#
# These examples are intended to be fast. For better final results, tune
# hyperparameters or train longer.
#
# NOTE: Each training step takes about 0.5 second with a batch size of 32 on
# CPU. On GPU, it takes ~5 milliseconds.
#
# With the default batch size and number of steps, train times are:
#
#   unconditional: CPU: 800  steps, ~10 min   GPU: 800  steps, ~1 min
#   conditional:   CPU: 2000 steps, ~20 min   GPU: 2000 steps, ~2 min
#   infogan:       CPU: 3000 steps, ~20 min   GPU: 3000 steps, ~6 min
#
# Usage:
# cd models/research/gan/mnist
# ./launch_jobs.sh ${gan_type} ${git_repo}
set -e

# Type of GAN to run. Right now, options are `unconditional`, `conditional`, or
# `infogan`.
gan_type=$1
if ! [[ "$gan_type" =~ ^(unconditional|conditional|infogan) ]]; then
  echo "'gan_type' must be one of: 'unconditional', 'conditional', 'infogan'."
  exit
fi

# Location of the git repository.
git_repo=$2
if [[ "$git_repo" == "" ]]; then
  echo "'git_repo' must not be empty."
  exit
fi

# Base name for where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/mnist-model

# Base name for where the evaluation images will be saved to.
EVAL_DIR=/tmp/mnist-model/eval

# Where the dataset is saved to.
DATASET_DIR=/tmp/mnist-data

# Location of the classifier frozen graph used for evaluation.
FROZEN_GRAPH="${git_repo}/research/gan/mnist/data/classify_mnist_graph_def.pb"

export PYTHONPATH=$PYTHONPATH:$git_repo:$git_repo/research:$git_repo/research/slim

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
if [[ "$gan_type" == "unconditional" ]]; then
  UNCONDITIONAL_TRAIN_DIR="${TRAIN_DIR}/unconditional"
  UNCONDITIONAL_EVAL_DIR="${EVAL_DIR}/unconditional"
  NUM_STEPS=3000
  # Run training.
  Banner "Starting training unconditional GAN for ${NUM_STEPS} steps..."
  python "${git_repo}/research/gan/mnist/train.py" \
    --train_log_dir=${UNCONDITIONAL_TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --max_number_of_steps=${NUM_STEPS} \
    --gan_type="unconditional" \
    --alsologtostderr
  Banner "Finished training unconditional GAN ${NUM_STEPS} steps."

  # Run evaluation.
  Banner "Starting evaluation of unconditional GAN..."
  python "${git_repo}/research/gan/mnist/eval.py" \
    --checkpoint_dir=${UNCONDITIONAL_TRAIN_DIR} \
    --eval_dir=${UNCONDITIONAL_EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --eval_real_images=false \
    --classifier_filename=${FROZEN_GRAPH} \
    --max_number_of_evaluation=1
  Banner "Finished unconditional evaluation. See ${UNCONDITIONAL_EVAL_DIR} for output images."
fi

# Run conditional GAN.
if [[ "$gan_type" == "conditional" ]]; then
  CONDITIONAL_TRAIN_DIR="${TRAIN_DIR}/conditional"
  CONDITIONAL_EVAL_DIR="${EVAL_DIR}/conditional"
  NUM_STEPS=3000
  # Run training.
  Banner "Starting training conditional GAN for ${NUM_STEPS} steps..."
  python "${git_repo}/research/gan/mnist/train.py" \
    --train_log_dir=${CONDITIONAL_TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --max_number_of_steps=${NUM_STEPS} \
    --gan_type="conditional" \
    --alsologtostderr
  Banner "Finished training conditional GAN ${NUM_STEPS} steps."

  # Run evaluation.
  Banner "Starting evaluation of conditional GAN..."
  python "${git_repo}/research/gan/mnist/conditional_eval.py" \
    --checkpoint_dir=${CONDITIONAL_TRAIN_DIR} \
    --eval_dir=${CONDITIONAL_EVAL_DIR} \
    --classifier_filename=${FROZEN_GRAPH} \
    --max_number_of_evaluation=1
  Banner "Finished conditional evaluation. See ${CONDITIONAL_EVAL_DIR} for output images."
fi

# Run InfoGAN.
if [[ "$gan_type" == "infogan" ]]; then
  INFOGAN_TRAIN_DIR="${TRAIN_DIR}/infogan"
  INFOGAN_EVAL_DIR="${EVAL_DIR}/infogan"
  NUM_STEPS=3000
  # Run training.
  Banner "Starting training infogan GAN for ${NUM_STEPS} steps..."
  python "${git_repo}/research/gan/mnist/train.py" \
    --train_log_dir=${INFOGAN_TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --max_number_of_steps=${NUM_STEPS} \
    --gan_type="infogan" \
    --alsologtostderr
  Banner "Finished training InfoGAN ${NUM_STEPS} steps."

  # Run evaluation.
  Banner "Starting evaluation of infogan..."
  python "${git_repo}/research/gan/mnist/infogan_eval.py" \
    --checkpoint_dir=${INFOGAN_TRAIN_DIR} \
    --eval_dir=${INFOGAN_EVAL_DIR} \
    --classifier_filename=${FROZEN_GRAPH} \
    --max_number_of_evaluation=1
  Banner "Finished InfoGAN evaluation. See ${INFOGAN_EVAL_DIR} for output images."
fi
