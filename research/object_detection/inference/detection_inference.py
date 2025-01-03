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
"""Utility functions for detection inference."""
from __future__ import division

import tensorflow.compat.v1 as tf

from object_detection.core import standard_fields


def build_input(tfrecord_paths):
  """Builds the graph's input.

  Args:
    tfrecord_paths: List of paths to the input TFRecords

  Returns:
    serialized_example_tensor: The next serialized example. String scalar Tensor
    image_tensor: The decoded image of the example. Uint8 tensor,
        shape=[1, None, None,3]
  """
  filename_queue = tf.train.string_input_producer(
      tfrecord_paths, shuffle=False, num_epochs=1)

  tf_record_reader = tf.TFRecordReader()
  _, serialized_example_tensor = tf_record_reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example_tensor,
      features={
          standard_fields.TfExampleFields.image_encoded:
              tf.FixedLenFeature([], tf.string),
      })
  encoded_image = features[standard_fields.TfExampleFields.image_encoded]
  image_tensor = tf.image.decode_image(encoded_image, channels=3)
  image_tensor.set_shape([None, None, 3])
  image_tensor = tf.expand_dims(image_tensor, 0)

  return serialized_example_tensor, image_tensor


def build_inference_graph(image_tensor, inference_graph_path):
  """Loads the inference graph and connects it to the input image.

  Args:
    image_tensor: The input image. uint8 tensor, shape=[1, None, None, 3]
    inference_graph_path: Path to the inference graph with embedded weights

  Returns:
    detected_boxes_tensor: Detected boxes. Float tensor,
        shape=[num_detections, 4]
    detected_scores_tensor: Detected scores. Float tensor,
        shape=[num_detections]
    detected_labels_tensor: Detected labels. Int64 tensor,
        shape=[num_detections]
  """
  with tf.gfile.Open(inference_graph_path, 'rb') as graph_def_file:
    graph_content = graph_def_file.read()
  graph_def = tf.GraphDef()
  graph_def.MergeFromString(graph_content)

  tf.import_graph_def(
      graph_def, name='', input_map={'image_tensor': image_tensor})

  g = tf.get_default_graph()

  num_detections_tensor = tf.squeeze(
      g.get_tensor_by_name('num_detections:0'), 0)
  num_detections_tensor = tf.cast(num_detections_tensor, tf.int32)

  detected_boxes_tensor = tf.squeeze(
      g.get_tensor_by_name('detection_boxes:0'), 0)
  detected_boxes_tensor = detected_boxes_tensor[:num_detections_tensor]

  detected_scores_tensor = tf.squeeze(
      g.get_tensor_by_name('detection_scores:0'), 0)
  detected_scores_tensor = detected_scores_tensor[:num_detections_tensor]

  detected_labels_tensor = tf.squeeze(
      g.get_tensor_by_name('detection_classes:0'), 0)
  detected_labels_tensor = tf.cast(detected_labels_tensor, tf.int64)
  detected_labels_tensor = detected_labels_tensor[:num_detections_tensor]

  return detected_boxes_tensor, detected_scores_tensor, detected_labels_tensor


def infer_detections_and_add_to_example(
    serialized_example_tensor, detected_boxes_tensor, detected_scores_tensor,
    detected_labels_tensor, discard_image_pixels):
  """Runs the supplied tensors and adds the inferred detections to the example.

  Args:
    serialized_example_tensor: Serialized TF example. Scalar string tensor
    detected_boxes_tensor: Detected boxes. Float tensor,
        shape=[num_detections, 4]
    detected_scores_tensor: Detected scores. Float tensor,
        shape=[num_detections]
    detected_labels_tensor: Detected labels. Int64 tensor,
        shape=[num_detections]
    discard_image_pixels: If true, discards the image from the result
  Returns:
    The de-serialized TF example augmented with the inferred detections.
  """
  tf_example = tf.train.Example()
  (serialized_example, detected_boxes, detected_scores,
   detected_classes) = tf.get_default_session().run([
       serialized_example_tensor, detected_boxes_tensor, detected_scores_tensor,
       detected_labels_tensor
   ])
  detected_boxes = detected_boxes.T

  tf_example.ParseFromString(serialized_example)
  feature = tf_example.features.feature
  feature[standard_fields.TfExampleFields.
          detection_score].float_list.value[:] = detected_scores
  feature[standard_fields.TfExampleFields.
          detection_bbox_ymin].float_list.value[:] = detected_boxes[0]
  feature[standard_fields.TfExampleFields.
          detection_bbox_xmin].float_list.value[:] = detected_boxes[1]
  feature[standard_fields.TfExampleFields.
          detection_bbox_ymax].float_list.value[:] = detected_boxes[2]
  feature[standard_fields.TfExampleFields.
          detection_bbox_xmax].float_list.value[:] = detected_boxes[3]
  feature[standard_fields.TfExampleFields.
          detection_class_label].int64_list.value[:] = detected_classes

  if discard_image_pixels:
    del feature[standard_fields.TfExampleFields.image_encoded]

  return tf_example
