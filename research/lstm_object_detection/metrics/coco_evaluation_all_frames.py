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

"""Class for evaluating video object detections with COCO metrics."""

import tensorflow.compat.v1 as tf

from object_detection.core import standard_fields
from object_detection.metrics import coco_evaluation
from object_detection.metrics import coco_tools


class CocoEvaluationAllFrames(coco_evaluation.CocoDetectionEvaluator):
  """Class to evaluate COCO detection metrics for frame sequences.

  The class overrides two functions: add_single_ground_truth_image_info and
  add_single_detected_image_info.

  For the evaluation of sequence video detection, by iterating through the
  entire groundtruth_dict, all the frames in the unrolled frames in one LSTM
  training sample are considered. Therefore, both groundtruth and detection
  results of all frames are added for the evaluation. This is used when all the
  frames are labeled in the video object detection training job.
  """

  def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
    """Add groundtruth results of all frames to the eval pipeline.

    This method overrides the function defined in the base class.

    Args:
      image_id: A unique string/integer identifier for the image.
      groundtruth_dict: A list of dictionary containing -
        InputDataFields.groundtruth_boxes: float32 numpy array of shape
          [num_boxes, 4] containing `num_boxes` groundtruth boxes of the format
          [ymin, xmin, ymax, xmax] in absolute image coordinates.
        InputDataFields.groundtruth_classes: integer numpy array of shape
          [num_boxes] containing 1-indexed groundtruth classes for the boxes.
        InputDataFields.groundtruth_is_crowd (optional): integer numpy array of
          shape [num_boxes] containing iscrowd flag for groundtruth boxes.
    """
    for idx, gt in enumerate(groundtruth_dict):
      if not gt:
        continue

      image_frame_id = '{}_{}'.format(image_id, idx)
      if image_frame_id in self._image_ids:
        tf.logging.warning(
            'Ignoring ground truth with image id %s since it was '
            'previously added', image_frame_id)
        continue

      self._groundtruth_list.extend(
          coco_tools.ExportSingleImageGroundtruthToCoco(
              image_id=image_frame_id,
              next_annotation_id=self._annotation_id,
              category_id_set=self._category_id_set,
              groundtruth_boxes=gt[
                  standard_fields.InputDataFields.groundtruth_boxes],
              groundtruth_classes=gt[
                  standard_fields.InputDataFields.groundtruth_classes]))
      self._annotation_id += (
          gt[standard_fields.InputDataFields.groundtruth_boxes].shape[0])

      # Boolean to indicate whether a detection has been added for this image.
      self._image_ids[image_frame_id] = False

  def add_single_detected_image_info(self, image_id, detections_dict):
    """Add detection results of all frames to the eval pipeline.

    This method overrides the function defined in the base class.

    Args:
      image_id: A unique string/integer identifier for the image.
      detections_dict: A list of dictionary containing -
        DetectionResultFields.detection_boxes: float32 numpy array of shape
          [num_boxes, 4] containing `num_boxes` detection boxes of the format
          [ymin, xmin, ymax, xmax] in absolute image coordinates.
        DetectionResultFields.detection_scores: float32 numpy array of shape
          [num_boxes] containing detection scores for the boxes.
        DetectionResultFields.detection_classes: integer numpy array of shape
          [num_boxes] containing 1-indexed detection classes for the boxes.

    Raises:
      ValueError: If groundtruth for the image_id is not available.
    """
    for idx, det in enumerate(detections_dict):
      if not det:
        continue

      image_frame_id = '{}_{}'.format(image_id, idx)
      if image_frame_id not in self._image_ids:
        raise ValueError(
            'Missing groundtruth for image-frame id: {}'.format(image_frame_id))

      if self._image_ids[image_frame_id]:
        tf.logging.warning(
            'Ignoring detection with image id %s since it was '
            'previously added', image_frame_id)
        continue

      self._detection_boxes_list.extend(
          coco_tools.ExportSingleImageDetectionBoxesToCoco(
              image_id=image_frame_id,
              category_id_set=self._category_id_set,
              detection_boxes=det[
                  standard_fields.DetectionResultFields.detection_boxes],
              detection_scores=det[
                  standard_fields.DetectionResultFields.detection_scores],
              detection_classes=det[
                  standard_fields.DetectionResultFields.detection_classes]))
      self._image_ids[image_frame_id] = True
