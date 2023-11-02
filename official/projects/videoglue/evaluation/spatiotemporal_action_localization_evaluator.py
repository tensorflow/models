# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""The evaluator for the spatiotemporal action localization task."""
from typing import Mapping

from absl import logging
import numpy as np
import tensorflow as tf, tf_keras

from object_detection.utils import object_detection_evaluation

# 60 filtered classes used for reporting evaluation results.
_AVA_LABELS_60 = frozenset([
    1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 22, 24, 26, 27, 28,
    29, 30, 34, 36, 37, 38, 41, 43, 45, 46, 47, 48, 49, 51, 52, 54, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 76, 77, 78, 79,
    80
])

_AVA_LABELS_STR_80 = (
    'bend/bow (at the waist)', 'crawl', 'crouch/kneel', 'dance', 'fall down',
    'get up', 'jump/leap', 'lie/sleep', 'martial art', 'run/jog', 'sit',
    'stand', 'swim', 'walk', 'answer phone', 'brush teeth',
    'carry/hold (an object)', 'catch (an object)', 'chop',
    'climb (e.g., a mountain)', 'clink glass', 'close (e.g., a door, a box)',
    'cook', 'cut', 'dig', 'dress/put on clothing', 'drink',
    'drive (e.g., a car, a truck)', 'eat', 'enter', 'exit', 'extract',
    'fishing', 'hit (an object)', 'kick (an object)', 'lift/pick up',
    'listen (e.g., to music)', 'open (e.g., a window, a car door)', 'paint',
    'play board game', 'play musical instrument', 'play with pets',
    'point to (an object)', 'press', 'pull (an object)', 'push (an object)',
    'put down', 'read', 'ride (e.g., a bike, a car, a horse)', 'row boat',
    'sail boat', 'shoot', 'shovel', 'smoke', 'stir', 'take a photo',
    'text on/look at a cellphone', 'throw', 'touch (an object)',
    'turn (e.g., a screwdriver)', 'watch (e.g., TV)', 'work on a computer',
    'write', 'fight/hit (a person)', 'give/serve (an object) to (a person)',
    'grab (a person)', 'hand clap', 'hand shake', 'hand wave', 'hug (a person)',
    'kick (a person)', 'kiss (a person)', 'lift (a person)',
    'listen to (a person)', 'play with kids', 'push (another person)',
    'sing to (e.g., self, a person, a group)',
    'take (an object) from (a person)',
    'talk to (e.g., self, a person, a group)', 'watch (a person)'
)

_IMAGE_SIZE = 512


class SpatiotemporalActionLocalizationEvaluator(object):
  """Spatiotemporal action localization evaluation metric class."""

  def __init__(self, iou_threshold: float = 0.5):
    self._prediction_scores_list = []
    self._prediction_boxes_list = []
    self._groundtruth_classes_list = []
    self._groundtruth_boxes_list = []

    ava_categories = []
    for idx, name in enumerate(_AVA_LABELS_STR_80):
      ava_categories.append({'id': idx + 1, 'name': name})

    self._evaluator = object_detection_evaluation.PascalDetectionEvaluator(
        categories=ava_categories,
        matching_iou_threshold=iou_threshold)

  @property
  def name(self):
    return 'mAP@.5IOU'

  def _maybe_convert_to_numpy(self, outputs):
    """Converts tf.Tensor to numpy arrays."""
    if outputs:
      outputs = tf.nest.map_structure(
          lambda x: x.numpy() if isinstance(x, tf.Tensor) else x, outputs)
      numpy_outputs = {}
      for key, val in outputs.items():
        if isinstance(val, tuple):
          val = np.concatenate(val)
        numpy_outputs[key] = val
    else:
      numpy_outputs = outputs

    return numpy_outputs

  def _format_ava_eval_data(
      self, scores, boxes, groundtruth_classes, groundtruth_boxes):
    """Converts data in the correct evaluation format.

    Args:
      scores: float32 numpy array of shape [N, C] for prediction scores.
      boxes: float32 numpy array of shape [N, 4] for prediction boxes.
      groundtruth_classes: float32 numpy array of shape [N] indicates 0-index
        groundtruth classes.
      groundtruth_boxes: float32 numpy array of shape [N, 4] of corresponding
        groundtruth boxes.

    Returns:
      output_dict: the dictionary contains formatted numpy arrays.
    """
    instances_mask = np.sum(boxes, axis=-1) > 0
    scores = scores[instances_mask]
    boxes = boxes[instances_mask]

    groundtruth_mask = np.sum(groundtruth_boxes, axis=-1) > 0
    valid_gt_classes = groundtruth_classes[groundtruth_mask]
    valid_gt_boxes = groundtruth_boxes[groundtruth_mask]

    # There are circumstances that no groundtruth is provided for current clip.
    if valid_gt_classes.size == 0:
      return None

    formatted_groundtruth_boxes = []
    formatted_groundtruth_classes = []
    formatted_detection_boxes = []
    formatted_detection_classes = []
    formatted_detection_scores = []
    for i in range(valid_gt_boxes.shape[0]):
      # Only evaluate AVA 60-classes.
      if (valid_gt_classes[i] + 1) not in _AVA_LABELS_60:
        continue
      formatted_groundtruth_boxes.append(valid_gt_boxes[i] * _IMAGE_SIZE)
      formatted_groundtruth_classes.append(valid_gt_classes[i] + 1)

    for i in range(scores.shape[0]):
      one_scores = scores[i].tolist()
      for cls_idx, score in enumerate(one_scores):
        # Only evaluate AVA 60-classes.
        if (cls_idx + 1) not in _AVA_LABELS_60:
          continue
        formatted_detection_boxes.append(boxes[i] * _IMAGE_SIZE)
        formatted_detection_classes.append(cls_idx + 1)
        formatted_detection_scores.append(score)

    if not formatted_groundtruth_boxes or not formatted_detection_boxes:
      return None
    else:
      output_dict = {
          'groundtruth_boxes': formatted_groundtruth_boxes,
          'groundtruth_classes': formatted_groundtruth_classes,
          'detection_boxes': formatted_detection_boxes,
          'detection_classes': formatted_detection_classes,
          'detection_scores': formatted_detection_scores,
      }
      return output_dict

  def update_state(self, step_outputs: Mapping[str, tf.Tensor]):
    """Updates per-step evaluation states by aggregating prediction results.

    Args:
      step_outputs: A dictionary contains tensors for the evaluation.
        * predictions: the model prediction score in shape [B, N, C].
        * instances_position: the corresponding boxes for each predictions.
        * nonmerge_label: the 0-indexed groundtruth label.
        * nonmerge_instances_position: the corresponding groundtruth boxes for
          each label. Note that the boxes here could be duplicated due to the
          multi-labels.
    """
    filtered_step_outputs = {
        'scores': step_outputs['predictions'],
        'boxes': step_outputs['instances_position'],
        'groundtruth_classes': step_outputs['nonmerge_label'],
        'groundtruth_boxes': step_outputs['nonmerge_instances_position'],
    }
    outputs_np = self._maybe_convert_to_numpy(filtered_step_outputs)

    self._prediction_scores_list.append(outputs_np['scores'])
    self._prediction_boxes_list.append(outputs_np['boxes'])
    self._groundtruth_classes_list.append(outputs_np['groundtruth_classes'])
    self._groundtruth_boxes_list.append(outputs_np['groundtruth_boxes'])

  def reset_states(self):
    """Resets evaluation states."""
    self._evaluator.clear()

    self._prediction_scores_list = []
    self._prediction_boxes_list = []
    self._groundtruth_classes_list = []
    self._groundtruth_boxes_list = []

  def result(self):
    """Fetches the final evaluation results."""
    groundtruth_classes = np.concatenate(self._groundtruth_classes_list)
    groundtruth_boxes = np.concatenate(self._groundtruth_boxes_list)
    prediction_scores = np.concatenate(self._prediction_scores_list)
    prediction_boxes = np.concatenate(self._prediction_boxes_list)

    num_samples = groundtruth_classes.shape[0]
    skipped = 0
    for batch_id in range(num_samples):
      output_dict = self._format_ava_eval_data(
          scores=prediction_scores[batch_id],
          boxes=prediction_boxes[batch_id],
          groundtruth_classes=groundtruth_classes[batch_id],
          groundtruth_boxes=groundtruth_boxes[batch_id])
      if output_dict is None:
        skipped += 1
        continue

      groundtruth_dict = {
          'groundtruth_boxes': np.array(
              output_dict['groundtruth_boxes'], dtype=float),
          'groundtruth_classes': np.array(
              output_dict['groundtruth_classes'], dtype=int),
          'groundtruth_difficult': np.zeros(
              len(output_dict['groundtruth_boxes']), dtype=bool),
      }
      detections_dict = {
          'detection_boxes': np.array(
              output_dict['detection_boxes'], dtype=float),
          'detection_classes': np.array(
              output_dict['detection_classes'], dtype=int),
          'detection_scores': np.array(
              output_dict['detection_scores'], dtype=float),
      }
      self._evaluator.add_single_ground_truth_image_info(
          image_id=batch_id, groundtruth_dict=groundtruth_dict)
      self._evaluator.add_single_detected_image_info(
          image_id=batch_id, detections_dict=detections_dict)

    metrics = self._evaluator.evaluate()
    logging.info('Evaluated on %d videos, skipped %d videos.',
                 num_samples - skipped, skipped)
    return {'mAP@.5IOU': metrics['PascalBoxes_Precision/mAP@0.5IOU']}
