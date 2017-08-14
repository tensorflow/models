#!/usr/bin/env bash
model_root=models/head_detector/train
#model_root=/home/arkenstone/tensorflow/workspace/models/object_detection/models/faster_rcnn_resnet101_coco_11_06_2017
#python ~/tensorflow/tensorflow/python/tools/freeze_graph.py \
#        --input_graph=$model_root/graph.pbtxt \
#        --input_checkpoint=$model_root/model.ckpt \
#        --output_graph=$model_root/test_frozen_inference_graph.pb \
#        --output_node_names=image_tensor,detection_boxes,detection_scores,detection_classes,num_detections
python export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path models/head_detector/faster_rcnn_resnet101_head.config \
        --checkpoint_path $model_root/model.ckpt-200000 \
        --inference_graph_path $model_root/frozen_inference_graph.pb