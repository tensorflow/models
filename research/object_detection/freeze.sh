echo "Please specify your path to trained checkpoint in the shell script."
bazel build -c opt --config=cuda research/object_detection:export_inference_graph

bazel-bin/research/object_detection/export_inference_graph \
  --input_type image_tensor \
  --pipeline_config_path ../swift/depot/src/vision/detection/model_configs/faster_rcnn_resnet101_sw.config \
  --trained_checkpoint_prefix research/object_detection/my_train/model.ckpt-0 \
  --output_directory /home/deepdot/Dataset/check_point 
