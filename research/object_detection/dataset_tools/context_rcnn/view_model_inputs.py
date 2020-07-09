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
r"""A Beam job to generate embedding data for camera trap images.

This tool runs inference with an exported Object Detection model in
`saved_model` format and produce raw embeddings for camera trap data. These
embeddings contain an object-centric feature embedding from Faster R-CNN, the
datetime that the image was taken (normalized in a specific way), and the
position of the object of interest. By default, only the highest-scoring object
embedding is included.

Steps to generate a embedding dataset:
1. Use object_detection/export_inference_graph.py to get a Faster R-CNN
  `saved_model` for inference. The input node must accept a tf.Example proto.
2. Run this tool with `saved_model` from step 1 and an TFRecord of tf.Example
  protos containing images for inference.

Example Usage:
--------------
python tensorflow_models/object_detection/export_inference_graph.py \
    --alsologtostderr \
    --input_type tf_example \
    --pipeline_config_path path/to/faster_rcnn_model.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory \
    --additional_output_tensor_names detection_features

python generate_embedding_data.py \
    --alsologtostderr \
    --embedding_input_tfrecord path/to/input_tfrecords* \
    --embedding_output_tfrecord path/to/output_tfrecords \
    --embedding_model_dir path/to/exported_model_directory/saved_model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
import threading

import numpy as np
import six
import tensorflow.compat.v1 as tf

try:
  import apache_beam as beam  # pylint:disable=g-import-not-at-top
except ModuleNotFoundError:
  pass

def _load_inference_model(args):
# Because initialization of the tf.Session is expensive we share
# one instance across all threads in the worker. This is possible since
# tf.Session.run() is thread safe.
    print(args)
    args = vars(args)
    session_lock = threading.Lock()
    session = None
    with session_lock:
      if session is None:
        graph = tf.Graph()
        session = tf.Session(graph=graph)
        with graph.as_default():
          meta_graph = tf.saved_model.loader.load(
              session, [tf.saved_model.tag_constants.SERVING],
              args['embedding_model_dir'])
        signature = meta_graph.signature_def['serving_default']
        print(signature.inputs)
        print(type(signature.inputs))
        input_tensor_name = signature.inputs['input_tensor'].name
        print(input_tensor_name)
        _input = graph.get_tensor_by_name(input_tensor_name)
        print(_input.shape)

        detection_features_name = signature.outputs['detection_features'].name
        detection_boxes_name = signature.outputs['detection_boxes'].name
        num_detections_name = signature.outputs['num_detections'].name
        
        self._embedding_node = graph.get_tensor_by_name(detection_features_name)
        self._box_node = graph.get_tensor_by_name(detection_boxes_name)
        self._scores_node = graph.get_tensor_by_name(
            signature.outputs['detection_scores'].name)
        self._num_detections = graph.get_tensor_by_name(num_detections_name)
        tf.logging.info(signature.outputs['detection_features'].name)
        tf.logging.info(signature.outputs['detection_boxes'].name)
        tf.logging.info(signature.outputs['num_detections'].name)
        print("Hello")

def parse_args(argv):
  """Command-line argument parser.

  Args:
    argv: command line arguments
  Returns:
    beam_args: Arguments for the beam pipeline.
    pipeline_args: Arguments for the pipeline options, such as runner type.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--embedding_input_tfrecord',
      dest='embedding_input_tfrecord',
      required=True,
      help='TFRecord containing images in tf.Example format for object '
      'detection.')
  parser.add_argument(
      '--embedding_output_tfrecord',
      dest='embedding_output_tfrecord',
      required=True,
      help='TFRecord containing embeddings in tf.Example format.')
  parser.add_argument(
      '--embedding_model_dir',
      dest='embedding_model_dir',
      required=True,
      help='Path to directory containing an object detection SavedModel with'
      'detection_box_classifier_features in the output.')
  parser.add_argument(
      '--top_k_embedding_count',
      dest='top_k_embedding_count',
      default=1,
      help='The number of top k embeddings to add to the memory bank.')
  parser.add_argument(
      '--bottom_k_embedding_count',
      dest='bottom_k_embedding_count',
      default=0,
      help='The number of bottom k embeddings to add to the memory bank.')
  parser.add_argument(
      '--num_shards',
      dest='num_shards',
      default=0,
      help='Number of output shards.')
  beam_args, pipeline_args = parser.parse_known_args(argv)
  return beam_args, pipeline_args


def main(argv=None, save_main_session=True):
  """Runs the Beam pipeline that performs inference.

  Args:
    argv: Command line arguments.
    save_main_session: Whether to save the main session.
  """
  args, pipeline_args = parse_args(argv)
  _load_inference_model(args)

if __name__ == '__main__':
  main()

