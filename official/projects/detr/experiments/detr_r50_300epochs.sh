#!/bin/bash
python3 official/projects/detr/train.py \
  --experiment=detr_coco \
  --mode=train_and_eval \
  --model_dir=/tmp/logging_dir/ \
  --params_override=task.init_checkpoint='gs://tf_model_garden/vision/resnet50_imagenet/ckpt-62400',trainer.train_steps=554400,trainer.optimizer_config.learning_rate.stepwise.boundaries="[369600]"
