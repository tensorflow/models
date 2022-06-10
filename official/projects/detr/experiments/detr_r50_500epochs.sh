#!/bin/bash
python3 official/projects/detr/train.py \
  --experiment=detr_coco \
  --mode=train_and_eval \
  --model_dir=/tmp/logging_dir/ \
  --params_override=task.init_ckpt='gs://tf_model_garden/vision/resnet50_imagenet/ckpt-62400'
