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
"""Tensorflow Example proto parser for data loading.

A parser to decode data containing serialized tensorflow.Example
protos into materialized tensors (numpy arrays).
"""

import numpy as np

from object_detection.core import data_parser
from object_detection.core import standard_fields as fields


class FloatParser(data_parser.DataToNumpyParser):
  """Tensorflow Example float parser."""

  def __init__(self, field_name):
    self.field_name = field_name

  def parse(self, tf_example):
    return np.array(
        tf_example.features.feature[self.field_name].float_list.value,
        dtype=np.float).transpose() if tf_example.features.feature[
            self.field_name].HasField("float_list") else None


class StringParser(data_parser.DataToNumpyParser):
  """Tensorflow Example string parser."""

  def __init__(self, field_name):
    self.field_name = field_name

  def parse(self, tf_example):
    return "".join(tf_example.features.feature[self.field_name]
                   .bytes_list.value) if tf_example.features.feature[
                       self.field_name].HasField("bytes_list") else None


class Int64Parser(data_parser.DataToNumpyParser):
  """Tensorflow Example int64 parser."""

  def __init__(self, field_name):
    self.field_name = field_name

  def parse(self, tf_example):
    return np.array(
        tf_example.features.feature[self.field_name].int64_list.value,
        dtype=np.int64).transpose() if tf_example.features.feature[
            self.field_name].HasField("int64_list") else None


class BoundingBoxParser(data_parser.DataToNumpyParser):
  """Tensorflow Example bounding box parser."""

  def __init__(self, xmin_field_name, ymin_field_name, xmax_field_name,
               ymax_field_name):
    self.field_names = [
        ymin_field_name, xmin_field_name, ymax_field_name, xmax_field_name
    ]

  def parse(self, tf_example):
    result = []
    parsed = True
    for field_name in self.field_names:
      result.append(tf_example.features.feature[field_name].float_list.value)
      parsed &= (
          tf_example.features.feature[field_name].HasField("float_list"))

    return np.array(result).transpose() if parsed else None


class TfExampleDetectionAndGTParser(data_parser.DataToNumpyParser):
  """Tensorflow Example proto parser."""

  def __init__(self):
    self.items_to_handlers = {
        fields.DetectionResultFields.key:
            StringParser(fields.TfExampleFields.source_id),
        # Object ground truth boxes and classes.
        fields.InputDataFields.groundtruth_boxes: (BoundingBoxParser(
            fields.TfExampleFields.object_bbox_xmin,
            fields.TfExampleFields.object_bbox_ymin,
            fields.TfExampleFields.object_bbox_xmax,
            fields.TfExampleFields.object_bbox_ymax)),
        fields.InputDataFields.groundtruth_classes: (
            Int64Parser(fields.TfExampleFields.object_class_label)),
        # Object detections.
        fields.DetectionResultFields.detection_boxes: (BoundingBoxParser(
            fields.TfExampleFields.detection_bbox_xmin,
            fields.TfExampleFields.detection_bbox_ymin,
            fields.TfExampleFields.detection_bbox_xmax,
            fields.TfExampleFields.detection_bbox_ymax)),
        fields.DetectionResultFields.detection_classes: (
            Int64Parser(fields.TfExampleFields.detection_class_label)),
        fields.DetectionResultFields.detection_scores: (
            FloatParser(fields.TfExampleFields.detection_score)),
    }

    self.optional_items_to_handlers = {
        fields.InputDataFields.groundtruth_difficult:
            Int64Parser(fields.TfExampleFields.object_difficult),
        fields.InputDataFields.groundtruth_group_of:
            Int64Parser(fields.TfExampleFields.object_group_of)
    }

  def parse(self, tf_example):
    """Parses tensorflow example and returns a tensor dictionary.

    Args:
      tf_example: a tf.Example object.

    Returns:
      A dictionary of the following numpy arrays:
      fields.DetectionResultFields.source_id - string containing original image
      id.
      fields.InputDataFields.groundtruth_boxes - a numpy array containing
      groundtruth boxes.
      fields.InputDataFields.groundtruth_classes - a numpy array containing
      groundtruth classes.
      fields.InputDataFields.groundtruth_group_of - a numpy array containing
      groundtruth group of flag (optional, None if not specified).
      fields.InputDataFields.groundtruth_difficult - a numpy array containing
      groundtruth difficult flag (optional, None if not specified).
      fields.DetectionResultFields.detection_boxes - a numpy array containing
      detection boxes.
      fields.DetectionResultFields.detection_classes - a numpy array containing
      detection class labels.
      fields.DetectionResultFields.detection_scores - a numpy array containing
      detection scores.
      Returns None if tf.Example was not parsed or non-optional fields were not
      found.
    """
    results_dict = {}
    parsed = True
    for key, parser in self.items_to_handlers.items():
      results_dict[key] = parser.parse(tf_example)
      parsed &= (results_dict[key] is not None)

    for key, parser in self.optional_items_to_handlers.items():
      results_dict[key] = parser.parse(tf_example)

    return results_dict if parsed else None
