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
"""Wrappers for third party lvis to be used within object_detection.

Usage example: given a set of images with ids in the list image_ids
and corresponding lists of numpy arrays encoding groundtruth (boxes,
masks and classes) and detections (masks, scores and classes), where
elements of each list correspond to detections/annotations of a single image,
then evaluation can be invoked as follows:

  groundtruth = lvis_tools.LVISWrapper(groundtruth_dict)
    detections = lvis_results.LVISResults(groundtruth, detections_list)
    evaluator = lvis_tools.LVISEvalWrapper(groundtruth, detections,
      iou_type='segm')
    summary_metrics = evaluator.ComputeMetrics()

TODO(jonathanhuang): Add support for exporting to JSON.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from lvis import eval as lvis_eval
from lvis import lvis
import numpy as np
from pycocotools import mask
import six
from six.moves import range


def RleCompress(masks):
  """Compresses mask using Run-length encoding provided by pycocotools.

  Args:
    masks: uint8 numpy array of shape [mask_height, mask_width] with values in
    {0, 1}.

  Returns:
    A pycocotools Run-length encoding of the mask.
  """
  rle = mask.encode(np.asfortranarray(masks))
  rle['counts'] = six.ensure_str(rle['counts'])
  return rle


def _ConvertBoxToCOCOFormat(box):
  """Converts a box in [ymin, xmin, ymax, xmax] format to COCO format.

  This is a utility function for converting from our internal
  [ymin, xmin, ymax, xmax] convention to the convention used by the COCO API
  i.e., [xmin, ymin, width, height].

  Args:
    box: a [ymin, xmin, ymax, xmax] numpy array

  Returns:
    a list of floats representing [xmin, ymin, width, height]
  """
  return [float(box[1]), float(box[0]), float(box[3] - box[1]),
          float(box[2] - box[0])]


class LVISWrapper(lvis.LVIS):
  """Wrapper for the lvis.LVIS class."""

  def __init__(self, dataset, detection_type='bbox'):
    """LVISWrapper constructor.

    See https://www.lvisdataset.org/dataset for a description of the format.
    By default, the coco.COCO class constructor reads from a JSON file.
    This function duplicates the same behavior but loads from a dictionary,
    allowing us to perform evaluation without writing to external storage.

    Args:
      dataset: a dictionary holding bounding box annotations in the COCO format.
      detection_type: type of detections being wrapped. Can be one of ['bbox',
        'segmentation']

    Raises:
      ValueError: if detection_type is unsupported.
    """
    self.logger = logging.getLogger(__name__)
    self.logger.info('Loading annotations.')
    self.dataset = dataset
    self._create_index()


class LVISEvalWrapper(lvis_eval.LVISEval):
  """LVISEval wrapper."""

  def __init__(self, groundtruth=None, detections=None, iou_type='bbox'):
    lvis_eval.LVISEval.__init__(
        self, groundtruth, detections, iou_type=iou_type)
    self._iou_type = iou_type

  def ComputeMetrics(self):
    self.run()
    summary_metrics = {}
    summary_metrics = self.results
    return summary_metrics


def ExportSingleImageGroundtruthToLVIS(image_id,
                                       next_annotation_id,
                                       category_id_set,
                                       groundtruth_boxes,
                                       groundtruth_classes,
                                       groundtruth_masks=None,
                                       groundtruth_area=None):
  """Export groundtruth of a single image to LVIS format.

  This function converts groundtruth detection annotations represented as numpy
  arrays to dictionaries that can be ingested by the LVIS evaluation API. Note
  that the image_ids provided here must match the ones given to
  ExportSingleImageDetectionMasksToLVIS. We assume that boxes, classes and masks
  are in correspondence - that is, e.g., groundtruth_boxes[i, :], and
  groundtruth_classes[i] are associated with the same groundtruth annotation.

  In the exported result, "area" fields are always set to the area of the
  groundtruth bounding box.

  Args:
    image_id: a unique image identifier either of type integer or string.
    next_annotation_id: integer specifying the first id to use for the
      groundtruth annotations. All annotations are assigned a continuous integer
      id starting from this value.
    category_id_set: A set of valid class ids. Groundtruth with classes not in
      category_id_set are dropped.
    groundtruth_boxes: numpy array (float32) with shape [num_gt_boxes, 4]
    groundtruth_classes: numpy array (int) with shape [num_gt_boxes]
    groundtruth_masks: optional uint8 numpy array of shape [num_detections,
      image_height, image_width] containing detection_masks.
    groundtruth_area: numpy array (float32) with shape [num_gt_boxes]. If
      provided, then the area values (in the original absolute coordinates) will
      be populated instead of calculated from bounding box coordinates.

  Returns:
    a list of groundtruth annotations for a single image in the COCO format.

  Raises:
    ValueError: if (1) groundtruth_boxes and groundtruth_classes do not have the
      right lengths or (2) if each of the elements inside these lists do not
      have the correct shapes or (3) if image_ids are not integers
  """

  if len(groundtruth_classes.shape) != 1:
    raise ValueError('groundtruth_classes is '
                     'expected to be of rank 1.')
  if len(groundtruth_boxes.shape) != 2:
    raise ValueError('groundtruth_boxes is expected to be of '
                     'rank 2.')
  if groundtruth_boxes.shape[1] != 4:
    raise ValueError('groundtruth_boxes should have '
                     'shape[1] == 4.')
  num_boxes = groundtruth_classes.shape[0]
  if num_boxes != groundtruth_boxes.shape[0]:
    raise ValueError('Corresponding entries in groundtruth_classes, '
                     'and groundtruth_boxes should have '
                     'compatible shapes (i.e., agree on the 0th dimension).'
                     'Classes shape: %d. Boxes shape: %d. Image ID: %s' % (
                         groundtruth_classes.shape[0],
                         groundtruth_boxes.shape[0], image_id))

  groundtruth_list = []
  for i in range(num_boxes):
    if groundtruth_classes[i] in category_id_set:
      if groundtruth_area is not None and groundtruth_area[i] > 0:
        area = float(groundtruth_area[i])
      else:
        area = float((groundtruth_boxes[i, 2] - groundtruth_boxes[i, 0]) *
                     (groundtruth_boxes[i, 3] - groundtruth_boxes[i, 1]))
      export_dict = {
          'id':
              next_annotation_id + i,
          'image_id':
              image_id,
          'category_id':
              int(groundtruth_classes[i]),
          'bbox':
              list(_ConvertBoxToCOCOFormat(groundtruth_boxes[i, :])),
          'area': area,
      }
      if groundtruth_masks is not None:
        export_dict['segmentation'] = RleCompress(groundtruth_masks[i])

      groundtruth_list.append(export_dict)
  return groundtruth_list


def ExportSingleImageDetectionMasksToLVIS(image_id,
                                          category_id_set,
                                          detection_masks,
                                          detection_scores,
                                          detection_classes):
  """Export detection masks of a single image to LVIS format.

  This function converts detections represented as numpy arrays to dictionaries
  that can be ingested by the LVIS evaluation API. We assume that
  detection_masks, detection_scores, and detection_classes are in correspondence
  - that is: detection_masks[i, :], detection_classes[i] and detection_scores[i]
    are associated with the same annotation.

  Args:
    image_id: unique image identifier either of type integer or string.
    category_id_set: A set of valid class ids. Detections with classes not in
      category_id_set are dropped.
    detection_masks: uint8 numpy array of shape [num_detections, image_height,
      image_width] containing detection_masks.
    detection_scores: float numpy array of shape [num_detections] containing
      scores for detection masks.
    detection_classes: integer numpy array of shape [num_detections] containing
      the classes for detection masks.

  Returns:
    a list of detection mask annotations for a single image in the COCO format.

  Raises:
    ValueError: if (1) detection_masks, detection_scores and detection_classes
      do not have the right lengths or (2) if each of the elements inside these
      lists do not have the correct shapes or (3) if image_ids are not integers.
  """

  if len(detection_classes.shape) != 1 or len(detection_scores.shape) != 1:
    raise ValueError('All entries in detection_classes and detection_scores'
                     'expected to be of rank 1.')
  num_boxes = detection_classes.shape[0]
  if not num_boxes == len(detection_masks) == detection_scores.shape[0]:
    raise ValueError('Corresponding entries in detection_classes, '
                     'detection_scores and detection_masks should have '
                     'compatible lengths and shapes '
                     'Classes length: %d.  Masks length: %d. '
                     'Scores length: %d' % (
                         detection_classes.shape[0], len(detection_masks),
                         detection_scores.shape[0]
                     ))
  detections_list = []
  for i in range(num_boxes):
    if detection_classes[i] in category_id_set:
      detections_list.append({
          'image_id': image_id,
          'category_id': int(detection_classes[i]),
          'segmentation': RleCompress(detection_masks[i]),
          'score': float(detection_scores[i])
      })
  return detections_list
