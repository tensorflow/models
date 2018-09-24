#!/usr/bin/env bash
set -e

pushd ../..
export PYTHONPATH=$(pwd)
popd

TIME=$(date +%s)

python3 imagenet_main.py --clean --data_dir /imn/imagenet/combined/ \
  --model_dir /tmp/garden_imagenet --train_epochs 30 --batch_size 2048 --resnet_version 1 \
  --resnet_size 50 --dtype fp16 --num_gpus 8 |& tee imagenet_garden_${TIME}.log
