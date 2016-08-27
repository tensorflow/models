#!/bin/bash
#
# Before running this script, make sure you've followed the instructions for
# downloading and converting the MNIST dataset.
# See slim/datasets/download_and_convert_mnist.py.
#
# Usage:
# ./slim/scripts/train_lenet_on_mnist.sh

# Compile the training and evaluation binaries
bazel build slim:train
bazel build slim:eval

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/lenet-model

# Where the dataset was saved to.
DATASET_DIR=/tmp/mnist

# Run training.
./bazel-bin/slim/train \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=mnist \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet \
  --preprocessing_name=lenet \
  --max_number_of_steps=20000 \
  --learning_rate=0.01 \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --optimizer=sgd \
  --learning_rate_decay_factor=1.0
  --weight_decay=0

# Run evaluation.
./blaze-bin/slim/eval \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=mnist \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet
