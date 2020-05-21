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
r"""Utilities for creating TFRecords of TF examples for the Open Images dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from object_detection.core import standard_fields
from object_detection.utils import dataset_util


def tf_example_from_annotations_data_frame(annotations_data_frame, label_map,
                                           encoded_image):
  """Populates a TF Example message with image annotations from a data frame.

  Args:
    annotations_data_frame: Data frame containing the annotations for a single
      image.
    label_map: String to integer label map.
    encoded_image: The encoded image string

  Returns:
    The populated TF Example, if the label of at least one object is present in
    label_map. Otherwise, returns None.
  """

  filtered_data_frame = annotations_data_frame[
      annotations_data_frame.LabelName.isin(label_map)]
  filtered_data_frame_boxes = filtered_data_frame[
      ~filtered_data_frame.YMin.isnull()]
  filtered_data_frame_labels = filtered_data_frame[
      filtered_data_frame.YMin.isnull()]
  image_id = annotations_data_frame.ImageID.iloc[0]

  feature_map = {
      standard_fields.TfExampleFields.object_bbox_ymin:
          dataset_util.float_list_feature(
              filtered_data_frame_boxes.YMin.as_matrix()),
      standard_fields.TfExampleFields.object_bbox_xmin:
          dataset_util.float_list_feature(
              filtered_data_frame_boxes.XMin.as_matrix()),
      standard_fields.TfExampleFields.object_bbox_ymax:
          dataset_util.float_list_feature(
              filtered_data_frame_boxes.YMax.as_matrix()),
      standard_fields.TfExampleFields.object_bbox_xmax:
          dataset_util.float_list_feature(
              filtered_data_frame_boxes.XMax.as_matrix()),
      standard_fields.TfExampleFields.object_class_text:
          dataset_util.bytes_list_feature([
              six.ensure_binary(label_text)
              for label_text in filtered_data_frame_boxes.LabelName.as_matrix()
          ]),
      standard_fields.TfExampleFields.object_class_label:
          dataset_util.int64_list_feature(
              filtered_data_frame_boxes.LabelName.map(
                  lambda x: label_map[x]).as_matrix()),
      standard_fields.TfExampleFields.filename:
          dataset_util.bytes_feature(
              six.ensure_binary('{}.jpg'.format(image_id))),
      standard_fields.TfExampleFields.source_id:
          dataset_util.bytes_feature(six.ensure_binary(image_id)),
      standard_fields.TfExampleFields.image_encoded:
          dataset_util.bytes_feature(six.ensure_binary(encoded_image)),
  }

  if 'IsGroupOf' in filtered_data_frame.columns:
    feature_map[standard_fields.TfExampleFields.
                object_group_of] = dataset_util.int64_list_feature(
                    filtered_data_frame_boxes.IsGroupOf.as_matrix().astype(int))
  if 'IsOccluded' in filtered_data_frame.columns:
    feature_map[standard_fields.TfExampleFields.
                object_occluded] = dataset_util.int64_list_feature(
                    filtered_data_frame_boxes.IsOccluded.as_matrix().astype(
                        int))
  if 'IsTruncated' in filtered_data_frame.columns:
    feature_map[standard_fields.TfExampleFields.
                object_truncated] = dataset_util.int64_list_feature(
                    filtered_data_frame_boxes.IsTruncated.as_matrix().astype(
                        int))
  if 'IsDepiction' in filtered_data_frame.columns:
    feature_map[standard_fields.TfExampleFields.
                object_depiction] = dataset_util.int64_list_feature(
                    filtered_data_frame_boxes.IsDepiction.as_matrix().astype(
                        int))

  if 'ConfidenceImageLabel' in filtered_data_frame_labels.columns:
    feature_map[standard_fields.TfExampleFields.
                image_class_label] = dataset_util.int64_list_feature(
                    filtered_data_frame_labels.LabelName.map(
                        lambda x: label_map[x]).as_matrix())
    feature_map[standard_fields.TfExampleFields
                .image_class_text] = dataset_util.bytes_list_feature([
                    six.ensure_binary(label_text) for label_text in
                    filtered_data_frame_labels.LabelName.as_matrix()
                ]),
  return tf.train.Example(features=tf.train.Features(feature=feature_map))
