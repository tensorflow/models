#!/usr/bin/env bash

tensor_root=/home/ubuntu/tensorflow-master

image_size=128

pb_name=/tmp/inception3.pb
frozen_pb_name=/tmp/frozen_inception3.pb


python export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v3 \
  --dataset_name=cifar100   \
  --image_size=${image_size} \
  --output_file=${pb_name}

python ${tensor_root}/tensorflow/python/tools/freeze_graph.py \
 --input_graph=${pb_name} \
 --input_checkpoint=/tmp/cifar100-models/inception_v3_small/all/model.ckpt-30000 \
 --input_binary=true \
 --output_graph=${frozen_pb_name} \
 --output_node_names=InceptionV3/Predictions/Reshape_1


python ${tensor_root}/tensorflow/examples/label_image/label_image.py \
  --image=97.jpg \
  --labels=/tmp/cifar100/labels.txt \
  --input_layer=input \
  --output_layer=InceptionV3/Predictions/Reshape_1 \
  --graph=${frozen_pb_name} \
  --input_mean=0 \
  --input_std=255 \
  --input_height=${image_size} \
  --input_width=${image_size}
