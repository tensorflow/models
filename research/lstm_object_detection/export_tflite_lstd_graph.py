# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
r"""Exports an LSTM detection model to use with tf-lite.

Outputs file:
* A tflite compatible frozen graph - $output_directory/tflite_graph.pb

The exported graph has the following input and output nodes.

Inputs:
'input_video_tensor': a float32 tensor of shape
[unroll_length, height, width, 3] containing the normalized input image.
Note that the height and width must be compatible with the height and
width configured in the fixed_shape_image resizer options in the pipeline
config proto.

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
  the graph has three outputs:
   'raw_outputs/box_encodings': a float32 tensor of shape [1, num_anchors, 4]
    containing the encoded box predictions.
   'raw_outputs/class_predictions': a float32 tensor of shape
    [1, num_anchors, num_classes] containing the class scores for each anchor
    after applying score conversion.
   'anchors': a float32 constant tensor of shape [num_anchors, 4]
    containing the anchor boxes.

Example Usage:
--------------
python lstm_object_detection/export_tflite_lstd_graph.py \
    --pipeline_config_path path/to/lstm_pipeline.config \
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
python lstm_object_detection/export_tflite_lstd_graph.py \
    --pipeline_config_path path/to/lstm_pipeline.config \
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

from lstm_object_detection import export_tflite_lstd_graph_lib
from lstm_object_detection.utils import config_util

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

  pipeline_config = config_util.get_configs_from_pipeline_file(
      FLAGS.pipeline_config_path)

  export_tflite_lstd_graph_lib.export_tflite_graph(
      pipeline_config,
      FLAGS.trained_checkpoint_prefix,
      FLAGS.output_directory,
      FLAGS.add_postprocessing_op,
      FLAGS.max_detections,
      FLAGS.max_classes_per_detection,
      use_regular_nms=FLAGS.use_regular_nms)


if __name__ == '__main__':
  tf.app.run(main)
