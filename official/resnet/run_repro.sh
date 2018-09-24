#!/usr/bin/env bash
set -e

pushd ../..
export PYTHONPATH=$(pwd)
popd

TIME=$(date +%s)


# Batch size of 2400 is in place to fail quickly
# A batch size of 2048 is most common, with 1024 also
# reasonably common. 2400 fails immediately, 2048 fails on
# epoch 22, and 1024 fails on epoch 46.
python3 imagenet_main.py --clean --data_dir /imn/imagenet/combined/ \
  --model_dir /tmp/garden_imagenet --train_epochs 30 --batch_size 2400 --resnet_version 1 \
  --resnet_size 50 --dtype fp16 --num_gpus 8 |& tee imagenet_garden_${TIME}.log
