# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

r"""Tool to export an object detection model for inference.

Prepares an object detection tensorflow graph for inference using model
configuration and an optional trained checkpoint.

The inference graph contains one of two input nodes depending on the user
specified option.
  * `image_tensor`: Accepts a uint8 4-D tensor of shape [1, None, None, 3]
  * `tf_example`: Accepts a serialized TFExample proto. The batch size in this
    case is always 1.

and the following output nodes:
  * `num_detections` : Outputs float32 tensors of the form [batch]
      that specifies the number of valid boxes per image in the batch.
  * `detection_boxes`  : Outputs float32 tensors of the form
      [batch, num_boxes, 4] containing detected boxes.
  * `detection_scores` : Outputs float32 tensors of the form
      [batch, num_boxes] containing class scores for the detections.
  * `detection_classes`: Outputs float32 tensors of the form
      [batch, num_boxes] containing classes for the detections.

Note that currently `batch` is always 1, but we will support `batch` > 1 in
the future.

Optionally, one can freeze the graph by converting the weights in the provided
checkpoint as graph constants thereby eliminating the need to use a checkpoint
file during inference.

Note that this tool uses `use_moving_averages` from eval_config to decide
which weights to freeze.

Example Usage:
--------------
python export_inference_graph \
    --input_type image_tensor \
    --pipeline_config_path path/to/ssd_inception_v2.config \
    --checkpoint_path path/to/model-ckpt \
    --inference_graph_path path/to/inference_graph.pb
"""
import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can be '
                    'one of [`image_tensor` `tf_example_proto`]')
flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')
flags.DEFINE_string('checkpoint_path', '', 'Optional path to checkpoint file. '
                    'If provided, bakes the weights from the checkpoint into '
                    'the graph.')
flags.DEFINE_string('inference_graph_path', '', 'Path to write the output '
                    'inference graph.')

FLAGS = flags.FLAGS


def main(_):
  assert FLAGS.pipeline_config_path, 'TrainEvalPipelineConfig missing.'
  assert FLAGS.inference_graph_path, 'Inference graph path missing.'
  assert FLAGS.input_type, 'Input type missing.'
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
  exporter.export_inference_graph(FLAGS.input_type, pipeline_config,
                                  FLAGS.checkpoint_path,
                                  FLAGS.inference_graph_path)


if __name__ == '__main__':
  tf.app.run()
