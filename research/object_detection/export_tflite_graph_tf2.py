# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
r"""Exports TF2 detection SavedModel for conversion to TensorFlow Lite.

Link to the TF2 Detection Zoo:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
The output folder will contain an intermediate SavedModel that can be used with
the TfLite converter.

NOTE: This only supports SSD meta-architectures for now.

One input:
  image: a float32 tensor of shape[1, height, width, 3] containing the
  *normalized* input image.
  NOTE: See the `preprocess` function defined in the feature extractor class
  in the object_detection/models directory.

Four Outputs:
  detection_boxes: a float32 tensor of shape [1, num_boxes, 4] with box
  locations
  detection_classes: a float32 tensor of shape [1, num_boxes]
  with class indices
  detection_scores: a float32 tensor of shape [1, num_boxes]
  with class scores
  num_boxes: a float32 tensor of size 1 containing the number of detected boxes

Example Usage:
--------------
python object_detection/export_tflite_graph_tf2.py \
    --pipeline_config_path path/to/ssd_model/pipeline.config \
    --trained_checkpoint_dir path/to/ssd_model/checkpoint \
    --output_directory path/to/exported_model_directory

The expected output SavedModel would be in the directory
path/to/exported_model_directory (which is created if it does not exist).

Config overrides (see the `config_override` flag) are text protobufs
(also of type pipeline_pb2.TrainEvalPipelineConfig) which are used to override
certain fields in the provided pipeline_config_path.  These are useful for
making small changes to the inference graph that differ from the training or
eval config.

Example Usage 1 (in which we change the NMS iou_threshold to be 0.5 and
NMS score_threshold to be 0.0):
python object_detection/export_tflite_model_tf2.py \
    --pipeline_config_path path/to/ssd_model/pipeline.config \
    --trained_checkpoint_dir path/to/ssd_model/checkpoint \
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

Example Usage 2 (export CenterNet model for keypoint estimation task with fixed
shape resizer and customized input resolution):
python object_detection/export_tflite_model_tf2.py \
    --pipeline_config_path path/to/ssd_model/pipeline.config \
    --trained_checkpoint_dir path/to/ssd_model/checkpoint \
    --output_directory path/to/exported_model_directory \
    --keypoint_label_map_path path/to/label_map.txt \
    --max_detections 10 \
    --centernet_include_keypoints true \
    --config_override " \
            model{ \
              center_net { \
                image_resizer { \
                  fixed_shape_resizer { \
                    height: 320 \
                    width: 320 \
                  } \
                } \
              } \
            }" \
"""
from absl import app
from absl import flags

import tensorflow.compat.v2 as tf
from google.protobuf import text_format
from object_detection import export_tflite_graph_lib_tf2
from object_detection.protos import pipeline_pb2

tf.enable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'pipeline_config_path', None,
    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
    'file.')
flags.DEFINE_string('trained_checkpoint_dir', None,
                    'Path to trained checkpoint directory')
flags.DEFINE_string('output_directory', None, 'Path to write outputs.')
flags.DEFINE_string(
    'config_override', '', 'pipeline_pb2.TrainEvalPipelineConfig '
    'text proto to override pipeline_config_path.')
flags.DEFINE_integer('max_detections', 10,
                     'Maximum number of detections (boxes) to return.')
# SSD-specific flags
flags.DEFINE_bool(
    'ssd_use_regular_nms', False,
    'Flag to set postprocessing op to use Regular NMS instead of Fast NMS '
    '(Default false).')
# CenterNet-specific flags
flags.DEFINE_bool(
    'centernet_include_keypoints', False,
    'Whether to export the predicted keypoint tensors. Only CenterNet model'
    ' supports this flag.'
)
flags.DEFINE_string(
    'keypoint_label_map_path', None,
    'Path of the label map used by CenterNet keypoint estimation task. If'
    ' provided, the label map path in the pipeline config will be replaced by'
    ' this one. Note that it is only used when exporting CenterNet model for'
    ' keypoint estimation task.'
)


def main(argv):
  del argv  # Unused.
  flags.mark_flag_as_required('pipeline_config_path')
  flags.mark_flag_as_required('trained_checkpoint_dir')
  flags.mark_flag_as_required('output_directory')

  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

  with tf.io.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
    text_format.Parse(f.read(), pipeline_config)
  override_config = pipeline_pb2.TrainEvalPipelineConfig()
  text_format.Parse(FLAGS.config_override, override_config)
  pipeline_config.MergeFrom(override_config)

  export_tflite_graph_lib_tf2.export_tflite_model(
      pipeline_config, FLAGS.trained_checkpoint_dir, FLAGS.output_directory,
      FLAGS.max_detections, FLAGS.ssd_use_regular_nms,
      FLAGS.centernet_include_keypoints, FLAGS.keypoint_label_map_path)


if __name__ == '__main__':
  app.run(main)
