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

  image_id = annotations_data_frame.ImageID.iloc[0]

  feature_map = {
      standard_fields.TfExampleFields.object_bbox_ymin:
          dataset_util.float_list_feature(filtered_data_frame.YMin.as_matrix()),
      standard_fields.TfExampleFields.object_bbox_xmin:
          dataset_util.float_list_feature(filtered_data_frame.XMin.as_matrix()),
      standard_fields.TfExampleFields.object_bbox_ymax:
          dataset_util.float_list_feature(filtered_data_frame.YMax.as_matrix()),
      standard_fields.TfExampleFields.object_bbox_xmax:
          dataset_util.float_list_feature(filtered_data_frame.XMax.as_matrix()),
      standard_fields.TfExampleFields.object_class_text:
          dataset_util.bytes_list_feature(
              filtered_data_frame.LabelName.as_matrix()),
      standard_fields.TfExampleFields.object_class_label:
          dataset_util.int64_list_feature(
              filtered_data_frame.LabelName.map(lambda x: label_map[x])
              .as_matrix()),
      standard_fields.TfExampleFields.filename:
          dataset_util.bytes_feature('{}.jpg'.format(image_id)),
      standard_fields.TfExampleFields.source_id:
          dataset_util.bytes_feature(image_id),
      standard_fields.TfExampleFields.image_encoded:
          dataset_util.bytes_feature(encoded_image),
  }

  if 'IsGroupOf' in filtered_data_frame.columns:
    feature_map[standard_fields.TfExampleFields.
                object_group_of] = dataset_util.int64_list_feature(
                    filtered_data_frame.IsGroupOf.as_matrix().astype(int))
  if 'IsOccluded' in filtered_data_frame.columns:
    feature_map[standard_fields.TfExampleFields.
                object_occluded] = dataset_util.int64_list_feature(
                    filtered_data_frame.IsOccluded.as_matrix().astype(int))
  if 'IsTruncated' in filtered_data_frame.columns:
    feature_map[standard_fields.TfExampleFields.
                object_truncated] = dataset_util.int64_list_feature(
                    filtered_data_frame.IsTruncated.as_matrix().astype(int))
  if 'IsDepiction' in filtered_data_frame.columns:
    feature_map[standard_fields.TfExampleFields.
                object_depiction] = dataset_util.int64_list_feature(
                    filtered_data_frame.IsDepiction.as_matrix().astype(int))

  return tf.train.Example(features=tf.train.Features(feature=feature_map))


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
  """Opens all TFRecord shards for writing and adds them to an exit stack.

  Args:
    exit_stack: A context2.ExitStack used to automatically closed the TFRecords
      opened in this function.
    base_path: The base path for all shards
    num_shards: The number of shards

  Returns:
    The list of opened TFRecords. Position k in the list corresponds to shard k.
  """
  tf_record_output_filenames = [
      '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
      for idx in xrange(num_shards)
  ]

  tfrecords = [
      exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
      for file_name in tf_record_output_filenames
  ]

  return tfrecords
