bazel build -c opt --config=cuda src/third_party/tensorflow/models/research/object_detection:export_inference_graph

bazel-bin/src/third_party/tensorflow/models/research/object_detection/export_inference_graph \
  --input_type image_tensor \
  --pipeline_config_path src/third_party/tensorflow/models/research/object_detection/faster_rcnn_resnet101_sw.config \
  --trained_checkpoint_prefix src/third_party/tensorflow/models/research/object_detection/my_train/model.ckpt-5567 \
  --output_directory src/third_party/tensorflow/models/research/object_detection/my_train/
