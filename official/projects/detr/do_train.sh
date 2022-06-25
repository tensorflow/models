#!/bin/bash
python3 train.py \
  --experiment=detr_coco \
  --mode=train_and_eval \
  --model_dir=gs://ghpark-ckpts/detr/detr_coco/ckpt_03_test \
  --tpu=postech-tpu \
  --params_override=runtime.distribution_strategy='tpu'