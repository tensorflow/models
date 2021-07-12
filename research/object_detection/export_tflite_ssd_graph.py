# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Exports an SSD detection model to use with tf-lite.

Outputs file:
* A tflite compatible frozen graph - $output_directory/tflite_graph.pb

The exported graph has the following input and output nodes.

Inputs:
'normalized_input_image_tensor': a float32 tensor of shape
[1, height, width, 3] containing the normalized input image. Note that the
height and width must be compatible with the height and width configured in
the fixed_shape_image resizer options in the pipeline config proto.

In floating point Mobilenet model, 'normalized_image_tensor' has values
between [-1,1). This typically means mapping each pixel (linearly)
to a value between [-1, 1]. Input image
values between 0 and 255 are scaled by (1/128.0) and then a value of
-1 is added to them to ensure the range is [-1,1).
In quantized Mobilenet model, 'normalized_image_tensor' has values between [0,
255].
In general, see the `preprocess` function defined in the feature extractor class
in the object_detection/models directory.

Outputs:
If add_postprocessing_op is true: frozen graph adds a
  TFLite_Detection_PostProcess custom op node has four outputs:
  detection_boxes: a float32 tensor of shape [1, num_boxes, 4] with box
  locations
  detection_classes: a float32 tensor of shape [1, num_boxes]
  with class indices
  detection_scores: a float32 tensor of shape [1, num_boxes]
  with class scores
  num_boxes: a float32 tensor of size 1 containing the number of detected boxes
else:
  the graph has two outputs:
   'raw_outputs/box_encodings': a float32 tensor of shape [1, num_anchors, 4]
    containing the encoded box predictions.
   'raw_outputs/class_predictions': a float32 tensor of shape
    [1, num_anchors, num_classes] containing the class scores for each anchor
    after applying score conversion.

Example Usage:
--------------
python object_detection/export_tflite_ssd_graph.py \
    --pipeline_config_path path/to/ssd_mobilenet.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory

The expected output would be in the directory
path/to/exported_model_directory (which is created if it does not exist)
with contents:
 - tflite_graph.pbtxt
 - tflite_graph.pb
Config overrides (see the `config_override` flag) are text protobufs
(also of type pipeline_pb2.TrainEvalPipelineConfig) which are used to override
certain fields in the provided pipeline_config_path.  These are useful for
making small changes to the inference graph that differ from the training or
eval config.

Example Usage (in which we change the NMS iou_threshold to be 0.5 and
NMS score_threshold to be 0.0):
python object_detection/export_tflite_ssd_graph.py \
    --pipeline_config_path path/to/ssd_mobilenet.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory
    --config_override " \
            model{ \
            ssd{ \
              post_processing { \
                batch_non_max_suppression { \
                        score_threshold: 0.0 \
                        iou_threshold: 0.5 \
                } \
             } \
          } \
       } \
       "
"""

import tensorflow.compat.v1 as tf
from google.protobuf import text_format
from object_detection import export_tflite_ssd_graph_lib
from object_detection.protos import pipeline_pb2

flags = tf.app.flags
flags.DEFINE_string('output_directory', None, 'Path to write outputs.')
flags.DEFINE_string(
    'pipeline_config_path', None,
    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
    'file.')
flags.DEFINE_string('trained_checkpoint_prefix', None, 'Checkpoint prefix.')
flags.DEFINE_integer('max_detections', 10,
                     'Maximum number of detections (boxes) to show.')
flags.DEFINE_integer('max_classes_per_detection', 1,
                     'Maximum number of classes to output per detection box.')
flags.DEFINE_integer(
    'detections_per_class', 100,
    'Number of anchors used per class in Regular Non-Max-Suppression.')
flags.DEFINE_bool('add_postprocessing_op', True,
                  'Add TFLite custom op for postprocessing to the graph.')
flags.DEFINE_bool(
    'use_regular_nms', False,
    'Flag to set postprocessing op to use Regular NMS instead of Fast NMS.')
flags.DEFINE_string(
    'config_override', '', 'pipeline_pb2.TrainEvalPipelineConfig '
    'text proto to override pipeline_config_path.')

FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.
  flags.mark_flag_as_required('output_directory')
  flags.mark_flag_as_required('pipeline_config_path')
  flags.mark_flag_as_required('trained_checkpoint_prefix')

  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

  with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
  text_format.Merge(FLAGS.config_override, pipeline_config)
  export_tflite_ssd_graph_lib.export_tflite_graph(
      pipeline_config, FLAGS.trained_checkpoint_prefix, FLAGS.output_directory,
      FLAGS.add_postprocessing_op, FLAGS.max_detections,
      FLAGS.max_classes_per_detection, use_regular_nms=FLAGS.use_regular_nms)


if __name__ == '__main__':
  tf.app.run(main)
