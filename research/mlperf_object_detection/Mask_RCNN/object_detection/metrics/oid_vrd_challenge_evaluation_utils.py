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
r"""Converts data from CSV format to the VRDDetectionEvaluator format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import numpy as np
from object_detection.core import standard_fields
from object_detection.utils import vrd_evaluation


def build_groundtruth_vrd_dictionary(data, class_label_map,
                                     relationship_label_map):
  """Builds a groundtruth dictionary from groundtruth data in CSV file.

  Args:
    data: Pandas DataFrame with the groundtruth data for a single image.
    class_label_map: Class labelmap from string label name to an integer.
    relationship_label_map: Relationship type labelmap from string name to an
      integer.

  Returns:
    A dictionary with keys suitable for passing to
    VRDDetectionEvaluator.add_single_ground_truth_image_info:
        standard_fields.InputDataFields.groundtruth_boxes: A numpy array
          of structures with the shape [M, 1], representing M tuples, each tuple
          containing the same number of named bounding boxes.
          Each box is of the format [y_min, x_min, y_max, x_max] (see
          datatype vrd_box_data_type, single_box_data_type above).
        standard_fields.InputDataFields.groundtruth_classes: A numpy array of
          structures shape [M, 1], representing  the class labels of the
          corresponding bounding boxes and possibly additional classes (see
          datatype label_data_type above).
        standard_fields.InputDataFields.verified_labels: numpy array
          of shape [K] containing verified labels.
  """
  data_boxes = data[data.LabelName.isnull()]
  data_labels = data[data.LabelName1.isnull()]

  boxes = np.zeros(data_boxes.shape[0], dtype=vrd_evaluation.vrd_box_data_type)
  boxes['subject'] = data_boxes[['YMin1', 'XMin1', 'YMax1',
                                 'XMax1']].as_matrix()
  boxes['object'] = data_boxes[['YMin2', 'XMin2', 'YMax2', 'XMax2']].as_matrix()

  labels = np.zeros(data_boxes.shape[0], dtype=vrd_evaluation.label_data_type)
  labels['subject'] = data_boxes['LabelName1'].map(lambda x: class_label_map[x])
  labels['object'] = data_boxes['LabelName2'].map(lambda x: class_label_map[x])
  labels['relation'] = data_boxes['RelationshipLabel'].map(
      lambda x: relationship_label_map[x])

  return {
      standard_fields.InputDataFields.groundtruth_boxes:
          boxes,
      standard_fields.InputDataFields.groundtruth_classes:
          labels,
      standard_fields.InputDataFields.verified_labels:
          data_labels['LabelName'].map(lambda x: class_label_map[x]),
  }


def build_predictions_vrd_dictionary(data, class_label_map,
                                     relationship_label_map):
  """Builds a predictions dictionary from predictions data in CSV file.

  Args:
    data: Pandas DataFrame with the predictions data for a single image.
    class_label_map: Class labelmap from string label name to an integer.
    relationship_label_map: Relationship type labelmap from string name to an
      integer.

  Returns:
    Dictionary with keys suitable for passing to
    VRDDetectionEvaluator.add_single_detected_image_info:
        standard_fields.DetectionResultFields.detection_boxes: A numpy array of
          structures with shape [N, 1], representing N tuples, each tuple
          containing the same number of named bounding boxes.
          Each box is of the format [y_min, x_min, y_max, x_max] (as an example
          see datatype vrd_box_data_type, single_box_data_type above).
        standard_fields.DetectionResultFields.detection_scores: float32 numpy
          array of shape [N] containing detection scores for the boxes.
        standard_fields.DetectionResultFields.detection_classes: A numpy array
          of structures shape [N, 1], representing the class labels of the
          corresponding bounding boxes and possibly additional classes (see
          datatype label_data_type above).
  """
  data_boxes = data

  boxes = np.zeros(data_boxes.shape[0], dtype=vrd_evaluation.vrd_box_data_type)
  boxes['subject'] = data_boxes[['YMin1', 'XMin1', 'YMax1',
                                 'XMax1']].as_matrix()
  boxes['object'] = data_boxes[['YMin2', 'XMin2', 'YMax2', 'XMax2']].as_matrix()

  labels = np.zeros(data_boxes.shape[0], dtype=vrd_evaluation.label_data_type)
  labels['subject'] = data_boxes['LabelName1'].map(lambda x: class_label_map[x])
  labels['object'] = data_boxes['LabelName2'].map(lambda x: class_label_map[x])
  labels['relation'] = data_boxes['RelationshipLabel'].map(
      lambda x: relationship_label_map[x])

  return {
      standard_fields.DetectionResultFields.detection_boxes:
          boxes,
      standard_fields.DetectionResultFields.detection_classes:
          labels,
      standard_fields.DetectionResultFields.detection_scores:
          data_boxes['Score'].as_matrix()
  }


def write_csv(fid, metrics):
  """Writes metrics key-value pairs to CSV file.

  Args:
    fid: File identifier of an opened file.
    metrics: A dictionary with metrics to be written.
  """
  metrics_writer = csv.writer(fid, delimiter=',')
  for metric_name, metric_value in metrics.items():
    metrics_writer.writerow([metric_name, str(metric_value)])
