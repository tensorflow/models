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
r"""Converts data from CSV to the OpenImagesDetectionChallengeEvaluator format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import zlib

import numpy as np
import pandas as pd
from pycocotools import mask as coco_mask

from object_detection.core import standard_fields


def _to_normalized_box(mask_np):
  """Decodes binary segmentation masks into np.arrays and boxes.

  Args:
    mask_np: np.ndarray of size NxWxH.

  Returns:
    a np.ndarray of the size Nx4, each row containing normalized coordinates
    [YMin, XMin, YMax, XMax] of a box computed of axis parallel enclosing box of
    a mask.
  """
  coord1, coord2 = np.nonzero(mask_np)
  if coord1.size > 0:
    ymin = float(min(coord1)) / mask_np.shape[0]
    ymax = float(max(coord1) + 1) / mask_np.shape[0]
    xmin = float(min(coord2)) / mask_np.shape[1]
    xmax = float((max(coord2) + 1)) / mask_np.shape[1]

    return np.array([ymin, xmin, ymax, xmax])
  else:
    return np.array([0.0, 0.0, 0.0, 0.0])


def _decode_raw_data_into_masks_and_boxes(segments, image_widths,
                                          image_heights):
  """Decods binary segmentation masks into np.arrays and boxes.

  Args:
    segments: pandas Series object containing either
      None entries, or strings with
      base64, zlib compressed, COCO RLE-encoded binary masks.
      All masks are expected to be the same size.
    image_widths: pandas Series of mask widths.
    image_heights: pandas Series of mask heights.

  Returns:
    a np.ndarray of the size NxWxH, where W and H is determined from the encoded
    masks; for the None values, zero arrays of size WxH are created. If input
    contains only None values, W=1, H=1.
  """
  segment_masks = []
  segment_boxes = []
  ind = segments.first_valid_index()
  if ind is not None:
    size = [int(image_heights[ind]), int(image_widths[ind])]
  else:
    # It does not matter which size we pick since no masks will ever be
    # evaluated.
    return np.zeros((segments.shape[0], 1, 1), dtype=np.uint8), np.zeros(
        (segments.shape[0], 4), dtype=np.float32)

  for segment, im_width, im_height in zip(segments, image_widths,
                                          image_heights):
    if pd.isnull(segment):
      segment_masks.append(np.zeros([1, size[0], size[1]], dtype=np.uint8))
      segment_boxes.append(np.expand_dims(np.array([0.0, 0.0, 0.0, 0.0]), 0))
    else:
      compressed_mask = base64.b64decode(segment)
      rle_encoded_mask = zlib.decompress(compressed_mask)
      decoding_dict = {
          'size': [im_height, im_width],
          'counts': rle_encoded_mask
      }
      mask_tensor = coco_mask.decode(decoding_dict)

      segment_masks.append(np.expand_dims(mask_tensor, 0))
      segment_boxes.append(np.expand_dims(_to_normalized_box(mask_tensor), 0))

  return np.concatenate(
      segment_masks, axis=0), np.concatenate(
          segment_boxes, axis=0)


def merge_boxes_and_masks(box_data, mask_data):
  return pd.merge(
      box_data,
      mask_data,
      how='outer',
      on=['LabelName', 'ImageID', 'XMin', 'XMax', 'YMin', 'YMax', 'IsGroupOf'])


def build_groundtruth_dictionary(data, class_label_map):
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
  data_location = data[data.XMin.notnull()]
  data_labels = data[data.ConfidenceImageLabel.notnull()]

  dictionary = {
      standard_fields.InputDataFields.groundtruth_boxes:
          data_location[['YMin', 'XMin', 'YMax', 'XMax']].as_matrix(),
      standard_fields.InputDataFields.groundtruth_classes:
          data_location['LabelName'].map(lambda x: class_label_map[x]
                                        ).as_matrix(),
      standard_fields.InputDataFields.groundtruth_group_of:
          data_location['IsGroupOf'].as_matrix().astype(int),
      standard_fields.InputDataFields.groundtruth_image_classes:
          data_labels['LabelName'].map(lambda x: class_label_map[x]
                                      ).as_matrix(),
  }

  if 'Mask' in data_location:
    segments, _ = _decode_raw_data_into_masks_and_boxes(
        data_location['Mask'], data_location['ImageWidth'],
        data_location['ImageHeight'])
    dictionary[
        standard_fields.InputDataFields.groundtruth_instance_masks] = segments

  return dictionary


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
  dictionary = {
      standard_fields.DetectionResultFields.detection_classes:
          data['LabelName'].map(lambda x: class_label_map[x]).as_matrix(),
      standard_fields.DetectionResultFields.detection_scores:
          data['Score'].as_matrix()
  }

  if 'Mask' in data:
    segments, boxes = _decode_raw_data_into_masks_and_boxes(
        data['Mask'], data['ImageWidth'], data['ImageHeight'])
    dictionary[standard_fields.DetectionResultFields.detection_masks] = segments
    dictionary[standard_fields.DetectionResultFields.detection_boxes] = boxes
  else:
    dictionary[standard_fields.DetectionResultFields.detection_boxes] = data[[
        'YMin', 'XMin', 'YMax', 'XMax'
    ]].as_matrix()

  return dictionary
