#!/bin/bash
set -e
set -x

ROOT_WORKSPACE=/home/wew57/dataset/
DATA_DIR=/home/wew57/dataset/cifar10_data_1/
export CUDA_VISIBLE_DEVICES=1
OPTIMIZER=adam
GRAD_BITS=1
BASE_LR=0.0002
CLIP_FACTOR=2.5

if [ ! -d "$ROOT_WORKSPACE" ]; then
  echo "${ROOT_WORKSPACE} does not exsit!"
  exit
fi

TRAIN_WORKSPACE=${ROOT_WORKSPACE}/cifar10_training_data/
EVAL_WORKSPACE=${ROOT_WORKSPACE}/cifar10_eval_data/
INFO_WORKSPACE=${ROOT_WORKSPACE}/cifar10_info/
if [ ! -d "${INFO_WORKSPACE}" ]; then
  echo "Creating ${INFO_WORKSPACE} ..."
  mkdir -p ${INFO_WORKSPACE}
fi
current_time=$(date)
current_time=${current_time// /_}
current_time=${current_time//:/-}
FOLDER_NAME=${OPTIMIZER}_${GRAD_BITS}_${BASE_LR}_${CLIP_FACTOR}_${current_time}
TRAIN_DIR=${TRAIN_WORKSPACE}/${FOLDER_NAME}
EVAL_DIR=${EVAL_WORKSPACE}/${FOLDER_NAME}
if [ ! -d "$TRAIN_DIR" ]; then
  echo "Creating ${TRAIN_DIR} ..."
  mkdir -p ${TRAIN_DIR}
fi
if [ ! -d "$EVAL_DIR" ]; then
  echo "Creating ${EVAL_DIR} ..."
  mkdir -p ${EVAL_DIR}
fi

python cifar10_eval.py --eval_dir $EVAL_DIR  --data_dir ${DATA_DIR} --checkpoint_dir $TRAIN_DIR   >  ${INFO_WORKSPACE}/eval_${current_time}_info.txt 2>&1 &
python cifar10_train.py --optimizer ${OPTIMIZER} --grad_bits ${GRAD_BITS} --base_lr ${BASE_LR} --clip_factor ${CLIP_FACTOR} --data_dir ${DATA_DIR} --train_dir ${TRAIN_DIR}  > ${INFO_WORKSPACE}/training_${current_time}_info.txt 2>&1 &

