#!/bin/sh
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an MobilenetV1 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./scripts/finetune_mobilenet_v1_on_flowers.sh
#    default: 1.0 224
# ./scripts/finetune_mobilenet_v1_on_flowers.sh {1.0,0.75,0.50,0.25}
#    {1.0,0.75,0.50,0.25} 224
# ./scripts/finetune_mobilenet_v1_on_flowers.sh {1.0,0.75,0.50,0.25} {224,192,160,128}
set -e

MOBILENET_VERSION="1.0"
IMAGE_SIZE="224"

if [ $# -eq 1 ]; then
  MOBILENET_VERSION=$1
fi

if [ $# -eq 2 ]; then
  MOBILENET_VERSION=$1
  IMAGE_SIZE=$2
fi

if [ ${MOBILENET_VERSION} = "1.0" ]; then
   SLIM_NAME=mobilenet_v1
elif [ ${MOBILENET_VERSION} = "0.75" ]; then
   SLIM_NAME=mobilenet_v1_075
elif [ ${MOBILENET_VERSION} = "0.50" ]; then
   SLIM_NAME=mobilenet_v1_050
elif [ ${MOBILENET_VERSION} = "0.25" ]; then
   SLIM_NAME=mobilenet_v1_025
else
  echo "Bad mobilenet version, should be one of 1.0, 0.75, 0.50, or 0.25"
  exit 1
fi

if [ ${IMAGE_SIZE} -ne "224" ] && [ ${IMAGE_SIZE} -ne "192" ] && [ ${IMAGE_SIZE} -ne "160" ] && [ ${IMAGE_SIZE} -ne "128" ]; then
  echo "Bad input image size, should be one of 224, 192, 160, or 128"
  exit 1
fi

# Where the pre-trained MobilenetV1 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/tmp/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/flowers-models/${SLIM_NAME}_${IMAGE_SIZE}

# Where the dataset is saved to.
DATASET_DIR=/tmp/flowers

alpha=${MOBILENET_VERSION}
rho=${IMAGE_SIZE}

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/mobilenet_v1_${alpha}_${rho}.ckpt.index ]; then
  wget http://download.tensorflow.org/models/mobilenet_v1_${alpha}_${rho}_2017_06_14.tar.gz
  tar -xvf mobilenet_v1_${alpha}_${rho}_2017_06_14.tar.gz
  mv mobilenet_v1_${alpha}_${rho}.ckpt.* ${PRETRAINED_CHECKPOINT_DIR}/
  rm mobilenet_v1_${alpha}_${rho}_2017_06_14.tar.gz
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
  --model_name=${SLIM_NAME} \
  --preprocessing_name=mobilenet_v1 \
  --train_image_size=${IMAGE_SIZE} \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/mobilenet_v1_${alpha}_${rho}.ckpt \
  --checkpoint_exclude_scopes=MobilenetV1/Logits \
  --trainable_scopes=MobilenetV1/Logits \
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
  --model_name=${SLIM_NAME} \
  --preprocessing_name=mobilenet_v1 \
  --eva_limage_size=${IMAGE_SIZE}

# Fine-tune all the new layers for 500 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=flowers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${SLIM_NAME} \
  --preprocessing_name=mobilenet_v1 \
  --train_image_size=${IMAGE_SIZE} \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=500 \
  --batch_size=32 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=flowers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${SLIM_NAME} \
  --preprocessing_name=mobilenet_v1 \
  --eva_limage_size=${IMAGE_SIZE}
