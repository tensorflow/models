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
r"""Converts data from CSV to the OpenImagesDetectionChallengeEvaluator format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from object_detection.core import standard_fields


def build_groundtruth_boxes_dictionary(data, class_label_map):
  """Builds a groundtruth dictionary from groundtruth data in CSV file.

  Args:
    data: Pandas DataFrame with the groundtruth data for a single image.
    class_label_map: Class labelmap from string label name to an integer.

  Returns:
    A dictionary with keys suitable for passing to
    OpenImagesDetectionChallengeEvaluator.add_single_ground_truth_image_info:
        standard_fields.InputDataFields.groundtruth_boxes: float32 numpy array
          of shape [num_boxes, 4] containing `num_boxes` groundtruth boxes of
          the format [ymin, xmin, ymax, xmax] in absolute image coordinates.
        standard_fields.InputDataFields.groundtruth_classes: integer numpy array
          of shape [num_boxes] containing 1-indexed groundtruth classes for the
          boxes.
        standard_fields.InputDataFields.verified_labels: integer 1D numpy array
          containing all classes for which labels are verified.
        standard_fields.InputDataFields.groundtruth_group_of: Optional length
          M numpy boolean array denoting whether a groundtruth box contains a
          group of instances.
  """
  data_boxes = data[data.ConfidenceImageLabel.isnull()]
  data_labels = data[data.XMin.isnull()]

  return {
      standard_fields.InputDataFields.groundtruth_boxes:
          data_boxes[['YMin', 'XMin', 'YMax', 'XMax']].as_matrix(),
      standard_fields.InputDataFields.groundtruth_classes:
          data_boxes['LabelName'].map(lambda x: class_label_map[x]).as_matrix(),
      standard_fields.InputDataFields.groundtruth_group_of:
          data_boxes['IsGroupOf'].as_matrix().astype(int),
      standard_fields.InputDataFields.groundtruth_image_classes:
          data_labels['LabelName'].map(lambda x: class_label_map[x])
          .as_matrix(),
  }


def build_predictions_dictionary(data, class_label_map):
  """Builds a predictions dictionary from predictions data in CSV file.

  Args:
    data: Pandas DataFrame with the predictions data for a single image.
    class_label_map: Class labelmap from string label name to an integer.

  Returns:
    Dictionary with keys suitable for passing to
    OpenImagesDetectionChallengeEvaluator.add_single_detected_image_info:
        standard_fields.DetectionResultFields.detection_boxes: float32 numpy
          array of shape [num_boxes, 4] containing `num_boxes` detection boxes
          of the format [ymin, xmin, ymax, xmax] in absolute image coordinates.
        standard_fields.DetectionResultFields.detection_scores: float32 numpy
          array of shape [num_boxes] containing detection scores for the boxes.
        standard_fields.DetectionResultFields.detection_classes: integer numpy
          array of shape [num_boxes] containing 1-indexed detection classes for
          the boxes.

  """
  return {
      standard_fields.DetectionResultFields.detection_boxes:
          data[['YMin', 'XMin', 'YMax', 'XMax']].as_matrix(),
      standard_fields.DetectionResultFields.detection_classes:
          data['LabelName'].map(lambda x: class_label_map[x]).as_matrix(),
      standard_fields.DetectionResultFields.detection_scores:
          data['Score'].as_matrix()
  }
