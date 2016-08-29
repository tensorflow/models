#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the MNIST dataset
# 2. Trains a LeNet model on the MNIST training set.
# 3. Evaluate the model on the MNIST testing set.
#
# Usage:
# cd slim
# ./slim/scripts/train_lenet_on_mnist.sh

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/lenet-model

# Where the dataset was saved to.
DATASET_DIR=/tmp/mnist

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=mnist \
  --dataset_dir=${DATASET_DIR}

# Run training.
python train.py \
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
  --learning_rate_decay_factor=1.0 \
  --weight_decay=0

# Run evaluation.
python eval.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=mnist \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet
