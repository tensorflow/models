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
TRAIN_DIR=/tmp/cifarnet-model

# Where the dataset was saved to.
DATASET_DIR=/tmp/cifar10

# Run training.
./bazel-bin/slim/train \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=cifar10 \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=cifarnet \
  --preprocessing_name=cifarnet \
  --max_number_of_steps=100000 \
  --save_interval_secs=120 \
  --save_summaries_secs=120 \
  --optimizer=sgd \
  --learning_rate=0.1 \
  --learning_rate_decay_factor=0.1 \
  --num_epochs_per_decay=200 \
  --weight_decay=0.004

# Run evaluation.
./blaze-bin/slim/eval \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=cifar10 \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=cifarnet
