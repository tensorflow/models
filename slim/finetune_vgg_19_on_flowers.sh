#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an vgg19 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd models
# ./slim/scripts/finetune_inceptionv3_on_flowers.sh
set -e

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/tmp/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/flowers-models/vgg_19

# Where the dataset is saved to.
DATASET_DIR=/tmp/data/flowers

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/vgg_19.ckpt ]; then
  wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
  tar -xvf vgg_19_2016_08_28.tar.gz
  mv vgg_19.ckpt ${PRETRAINED_CHECKPOINT_DIR}/vgg_19.ckpt
  rm vgg_19_2016_08_28.tar.gz
fi

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=flowers \
  --dataset_dir=${DATASET_DIR}

# Fine-tune only the new layers for 1000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=flowers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_19 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/vgg_19.ckpt \
  --checkpoint_exclude_scopes=vgg_19/fc8 \
  --trainable_scopes=vgg_19/fc8 \
  --max_number_of_steps=1000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004


# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=flowers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_19
