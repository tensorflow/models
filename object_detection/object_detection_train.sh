#!/usr/bin/env bash
python train.py --logtostderr \
                --train_dir=models/startdt_pedestrian/train \
                --pipeline_config_path=models/startdt_pedestrian/faster_rcnn_resnet101_person.config
#                --pipeline_config_path=models/head_detector/faster_rcnn_inception_resnet_v2_atrous_head.config
