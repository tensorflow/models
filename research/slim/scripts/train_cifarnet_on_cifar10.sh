#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Cifar10 dataset
# 2. Trains a CifarNet model on the Cifar10 training set.
# 3. Evaluate the model on the Cifar10 testing set.
#
# Usage:
# cd slim
# ./scripts/train_cifar_net_on_mnist.sh

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/cifarnet-model

# Where the dataset was saved to.
DATASET_DIR=/tmp/cifar10

# Download the dataset
python datasets/download_and_convert_cifar10.py \
  --dataset_dir=${DATASET_DIR}

# Run training.
python train.py \
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
python eval.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=cifar10 \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=cifarnet
