#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Fine-tunes a ResNetV1-50 model on the ImageNet training set using FakeQuant ops for 8b quantization.
# 2. Evaluates the model on the ImageNet validation set.
# 3. Exports an inference graph of the model.
# 4. Freezes the graph.
# 5. Converts the frozen graph to a quantized TF-Lite model using TOCO.
#
# Usage:
# cd slim
# ./slim/scripts/quantize_resnet_v1_50_on_imagenet.sh
set -e

# Which model.
NETWORK_NAME=resnet_v1_50

# Which dataset (i.e. imagenet or flowers).
DATASET_NAME=imagenet

# Where the pre-trained ResNetV1-50 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/tmp/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/${DATASET_NAME}-models/${NETWORK_NAME}
#rm -Rf ${TRAIN_DIR}

# Where the dataset is saved to.
DATASET_DIR=/tmp/${DATASET_NAME}

# Where the TensorFlow source code is.
TENSORFLOW_DIR=/tmp/tensorflow

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/resnet_v1_50.ckpt ]; then
  wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
  tar -xvf resnet_v1_50_2016_08_28.tar.gz
  mv resnet_v1_50.ckpt ${PRETRAINED_CHECKPOINT_DIR}/resnet_v1_50.ckpt
  rm resnet_v1_50_2016_08_28.tar.gz
fi

## Fine-tune 100 steps to learn the data ranges.
#python train_image_classifier.py \
#  --train_dir=${TRAIN_DIR} \
#  --dataset_name=${DATASET_NAME} \
#  --dataset_split_name=train \
#  --dataset_dir=${DATASET_DIR} \
#  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${NETWORK_NAME}.ckpt \
#  --ignore_missing_vars \
#  --model_name=${NETWORK_NAME} \
#  --labels_offset=1 \
#  --max_number_of_steps=100 \
#  --batch_size=64 \
#  --learning_rate=0.00001 \
#  --save_interval_secs=60 \
#  --save_summaries_secs=60 \
#  --log_every_n_steps=100 \
#  --optimizer=rmsprop \
#  --weight_decay=0.00004 \
#  --quantize
#
## Run evaluation.
#python eval_image_classifier.py \
#  --checkpoint_path=${TRAIN_DIR} \
#  --eval_dir=${TRAIN_DIR} \
#  --dataset_name=${DATASET_NAME} \
#  --dataset_split_name=validation \
#  --dataset_dir=${DATASET_DIR} \
#  --model_name=${NETWORK_NAME} \
#  --labels_offset=1 \
#  --quantize

# Export an inference graph.
python export_inference_graph.py \
  --alsologtostderr \
  --batch_size=1 \
  --dataset_name=${DATASET_NAME} \
  --model_name=${NETWORK_NAME} \
  --labels_offset=1 \
  --output_file=${TRAIN_DIR}/${NETWORK_NAME}_inf_graph.pb \
  --quantize

echo "** Generated inference graph: ${TRAIN_DIR}/${NETWORK_NAME}_inf_graph.pb"

# Freeze the graph (convert the weights to constants).
OUTPUT_NODES=${NETWORK_NAME}/predictions/Reshape_1
CHECKPOINT_ITER=100
(cd ${TENSORFLOW_DIR}; bazel run --config=opt //tensorflow/python/tools:freeze_graph -- \
  --input_graph=${TRAIN_DIR}/${NETWORK_NAME}_inf_graph.pb \
  --input_checkpoint=${TRAIN_DIR}/model.ckpt-${CHECKPOINT_ITER} \
  --input_binary=true \
  --output_graph=${TRAIN_DIR}/${NETWORK_NAME}_frozen_graph.pb \
  --output_node_names=${OUTPUT_NODES} \
)
echo "** Generated frozen inference graph: ${TRAIN_DIR}/${NETWORK_NAME}_frozen_graph.pb"


# Use TOCO to generate a PDF of the model.
(cd ${TENSORFLOW_DIR}; bazel run --config=opt //tensorflow/contrib/lite/toco:toco -- \
  --input_format=TENSORFLOW_GRAPHDEF \
  --input_file=${TRAIN_DIR}/${NETWORK_NAME}_frozen_graph.pb \
  --output_format=GRAPHVIZ_DOT \
  --output_file=${TRAIN_DIR}/${NETWORK_NAME}.quantized.dot \
  --inference_type=QUANTIZED_UINT8 \
  --input_shape=1,224,224,3 \
  --input_array=input \
  --output_array=${OUTPUT_NODES} \
  --mean_value=114.7936 \
  --std_value=0.9281\
)
dot -Tpdf -O ${TRAIN_DIR}/${NETWORK_NAME}.quantized.dot 
echo "** Generated PDF of graph: ${TRAIN_DIR}/${NETWORK_NAME}.quantized.dot.pdf"

# Use TOCO to generate a quantized TF-Lite model. 
(cd ${TENSORFLOW_DIR}; bazel run --config=opt //tensorflow/contrib/lite/toco:toco -- \
  --input_format=TENSORFLOW_GRAPHDEF \
  --input_file=${TRAIN_DIR}/${NETWORK_NAME}_frozen_graph.pb \
  --output_format=TFLITE \
  --output_file=${TRAIN_DIR}/${NETWORK_NAME}.quantized.tflite \
  --inference_type=QUANTIZED_UINT8 \
  --input_shape=1,224,224,3 \
  --input_array=input \
  --output_array=${OUTPUT_NODES} \
  --mean_value=114.7936 \
  --std_value=0.9281\
)
dot -Tpdf -O ${TRAIN_DIR}/${NETWORK_NAME}.quantized.dot 
echo "** Generated quantized TF-Lite model: ${TRAIN_DIR}/${NETWORK_NAME}.quantized.tflite"

# Test the TF-Lite model with the example "label_image" application. 
#  There are two issues:
#  1) This ResNet does not have a background class so reported classes need to be shifted up by one index.
#  2) This ResNet expects VGG-preprocessing so you should do mean channel subtraction (see "preprocessing/vgg_preprocessing.py"). 
(cd ${TENSORFLOW_DIR}; bazel run --config=opt //tensorflow/contrib/lite/examples/label_image:label_image -- \
  --tflite_model=${TRAIN_DIR}/${NETWORK_NAME}.quantized.tflite \
  --image=${TENSORFLOW_DIR}/tensorflow/contrib/lite/examples/label_image/testdata/grace_hopper.bmp \
  --labels=${DATASET_DIR}/labels.txt \
)
