#!/bin/bash
python3 train.py \
  --experiment=detr_coco \
  --mode=train_and_eval \
  --model_dir=gs://ghpark-ckpts/detr/detr_coco/ckpt_03_detr_coco_resnet101 \
  --tpu=postech-tpu \
  --params_override=runtime.distribution_strategy='tpu'