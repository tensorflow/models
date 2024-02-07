# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Detection evaluator for the Waymo Open Dataset."""

import abc
from typing import Any, Mapping

from absl import logging
import numpy as np
import tensorflow as tf, tf_keras

from official.projects.pointpillars.configs import pointpillars as cfg
from official.projects.pointpillars.utils import utils
from waymo_open_dataset import label_pb2
from waymo_open_dataset.metrics.python import wod_detection_evaluator

np.set_printoptions(precision=4, suppress=True)


class _AbstractEvaluator(
    wod_detection_evaluator.WODDetectionEvaluator, metaclass=abc.ABCMeta):
  """WOD detection evaluation metric base class."""

  def __init__(self, model_config, config=None):
    super().__init__(config=config)

    image_config = model_config.image
    self._resolution = image_config.resolution
    self._vehicle_xy = utils.get_vehicle_xy(image_config.height,
                                            image_config.width,
                                            image_config.x_range,
                                            image_config.y_range)
    self._classes = model_config.classes

  def _remove_padding(self, tensor_dict: Mapping[str, Any],
                      num_valid: int) -> Mapping[str, Any]:
    """Remove the paddings of the prediction/groundtruth data."""
    result_tensor_dict = {}
    gather_indices = tf.range(num_valid)
    for k, v in tensor_dict.items():
      if v.shape[0] < num_valid:
        raise ValueError(
            '{} does not have enough elements to gather, {} < {}'.format(
                k, v.shape[0], num_valid))
      result_tensor_dict[k] = tf.gather(v, gather_indices)
    return result_tensor_dict

  def _compact_tensors(self,
                       tensor_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    """Compact tensors by concatenating them in tuples."""
    compact_tensor_dict = {}
    for k, v in tensor_dict.items():
      if isinstance(v, tuple):
        compact_tensor_dict[k] = tf.concat(v, axis=0)
      elif isinstance(v, dict):
        compact_tensor_dict[k] = v
        for dk, dv in v.items():
          if isinstance(dv, tuple):
            compact_tensor_dict[k][dk] = tf.concat(dv, axis=0)
      else:
        compact_tensor_dict[k] = v
    return compact_tensor_dict

  def _adjust_class(self, tensor_dict: Mapping[str, Any]) -> tf.Tensor:
    """Change predicted class to what defiend by label.proto."""
    original_type = tf.cast(tensor_dict['classes'], tf.uint8)
    if self._classes == 'all':
      adjusted_type = tf.where(
          tf.equal(original_type, 3),
          tf.ones_like(original_type) * 4,
          original_type)
    else:
      adjusted_type = tf.where(
          tf.equal(original_type, 1),
          tf.ones_like(original_type) * utils.CLASSES[self._classes],
          original_type)
    return adjusted_type

  @abc.abstractmethod
  def _get_box(self, box2d: tf.Tensor, attributes: Mapping[str, tf.Tensor]):
    """Get box from yxyx and attributes.

    Args:
      box2d: a [N, 4] tensor encoding as (ymin, xmin, ymax, xmax)
      attributes: a {name: [N, 1]} dict
    Returns:
      box: a tensor representing a 2d or 3d box
    """

  def update_state(self,
                   groundtruths: Mapping[str, tf.Tensor],
                   predictions: Mapping[str, tf.Tensor]):
    """Update the metrics state with prediction and groundtruth data.

    Notations:
      B: batch size.
      N: number of ground truth boxes.
      M: number of predicted boxes.
      T: attribute size.

    Args:
      groundtruths: a dictionary of Tensors including the fields below.
        Required fields:
          - frame_id: a tensor of int64 of shape [B].
          - num_detections: a tensor of int32 of shape [B].
          - boxes: a tensor of float32 of shape [B, N, 4],
              (ymin, xmin, ymax, xmax).
          - classes: a tensor of int32 of shape [B, N].
          - attributes: a dict of tensor of float32 of shape [B, N, T].
          - difficulties: a tensor of int32 of shape [B, N].

      predictions: a dictionary of tensors including the fields below.
        Required fields:
          - num_detections: a tensor of int32 of shape [B].
          - boxes: a tensor of float32 of shape [B, M, 4],
              (ymin, xmin, ymax, xmax).
          - scores: a tensor of float32 of shape [B, M].
          - classes: a tensor of int32 of shape [B, M].
          - attributes: a dict of tensor of float32 of shape [B, M, T].
    """
    # Remove tuples from dataset.
    groundtruths = self._compact_tensors(groundtruths)
    predictions = self._compact_tensors(predictions)

    # Adjust type.
    gt_type = self._adjust_class(groundtruths)
    pred_type = self._adjust_class(predictions)

    batch_size = tf.shape(groundtruths['frame_id'])[0]
    for i in tf.range(batch_size):
      # Set ground truths
      gt_num_detections = groundtruths['num_detections'][i]
      gt_attributes = {}
      for k, v in groundtruths['attributes'].items():
        gt_attributes[k] = v[i]
      frame_groundtruths = {
          'ground_truth_frame_id':
              tf.tile([groundtruths['frame_id'][i]], [gt_num_detections]),
          'ground_truth_bbox':
              self._get_box(groundtruths['boxes'][i], gt_attributes),
          'ground_truth_type':
              gt_type[i],
          'ground_truth_difficulty':
              tf.cast(groundtruths['difficulty'][i], tf.uint8),
      }
      frame_groundtruths = self._remove_padding(
          frame_groundtruths, gt_num_detections)

      # Set predictions
      pred_num_detections = predictions['num_detections'][i]
      pred_attributes = {}
      for k, v in predictions['attributes'].items():
        pred_attributes[k] = v[i]
      frame_predictions = {
          'prediction_frame_id':
              tf.tile([groundtruths['frame_id'][i]], [pred_num_detections]),
          'prediction_bbox':
              self._get_box(predictions['boxes'][i], pred_attributes),
          'prediction_type':
              pred_type[i],
          'prediction_score':
              predictions['scores'][i],
          'prediction_overlap_nlz':
              tf.zeros_like(predictions['scores'][i], dtype=tf.bool)
      }
      frame_predictions = self._remove_padding(
          frame_predictions, pred_num_detections)

      # Update state for this frame.
      super().update_state(frame_groundtruths, frame_predictions)

  def evaluate(self) -> Mapping[str, Any]:
    """Compute the final metrics.

    Returns:
      metric_dict: A dict of metrics, contains following breakdown keys:
        mAP/{class}_level_1
        mAP/{class}_[0, 30)_level_1
        mAP/{class}_[30, 50)_level_1
        mAP/{class}_[50, +inf)_level_1
        mAP/{class}_level_2
        mAP/{class}_[0, 30)_level_2
        mAP/{class}_[30, 50)_level_2
        mAP/{class}_[50, +inf)_level_2
        mAPH/{class}_level_1
        mAPH/{class}_[0, 30)_level_1
        mAPH/{class}_[30, 50)_level_1
        mAPH/{class}_[50, +inf)_level_1
        mAPH/{class}_level_2
        mAPH/{class}_[0, 30)_level_2
        mAPH/{class}_[30, 50)_level_2
        mAPH/{class}_[50, +inf)_level_2
      It also contains following keys used as public NAS rewards.
        AP
        APH
    """
    ap, aph, _, _, _, _, _ = super().evaluate()
    metric_dict = {}
    for i, name in enumerate(self._breakdown_names):
      # Skip sign metrics since we don't use this type.
      if 'SIGN' in name:
        continue
      # Make metric name more readable.
      name = name.lower()
      for c in utils.CLASSES:
        pos = name.find(c)
        if pos != -1:
          name = name[pos:]
      if self._classes == 'all' or self._classes in name:
        metric_dict['mAP/{}'.format(name)] = ap[i]
        metric_dict['mAPH/{}'.format(name)] = aph[i]

    # Set public metrics as AP and APH.
    if self._classes == 'all':
      ap, aph = 0, 0
      for c in utils.CLASSES:
        ap += metric_dict['mAP/{}_level_1'.format(c)]
        aph += metric_dict['mAPH/{}_level_1'.format(c)]
      metric_dict['AP'] = ap / len(utils.CLASSES)
      metric_dict['APH'] = aph / len(utils.CLASSES)
    else:
      metric_dict['AP'] = metric_dict['mAP/{}_level_1'.format(self._classes)]
      metric_dict['APH'] = metric_dict['mAPH/{}_level_1'.format(self._classes)]
    return metric_dict


class Wod3dDetectionEvaluator(_AbstractEvaluator):
  """WOD 3D detection evaluation metric class."""

  def _get_box(self, box2d: tf.Tensor,
               attributes: Mapping[str, tf.Tensor]) -> tf.Tensor:
    """Get box from yxyx and attributes.

    Args:
      box2d: a float32 [N, 4] tensor encoding as (ymin, xmin, ymax, xmax)
      attributes: a float32 {name: [N, 1]} dict
    Returns:
      box: a float32 [N, 7] tensor representing a 3d box
    """
    box2d = utils.image_to_frame_boxes(box2d, self._vehicle_xy,
                                       self._resolution)
    values = []
    values.append(box2d[:, 0])  # center_x
    values.append(box2d[:, 1])  # center_y
    values.append(attributes['z'][:, 0])  # center_z
    values.append(box2d[:, 2])  # length
    values.append(box2d[:, 3])  # width
    values.append(attributes['height'][:, 0])  # height
    values.append(attributes['heading'][:, 0])  # heading
    box3d = tf.stack(values, axis=-1)
    return box3d


class Wod2dDetectionEvaluator(_AbstractEvaluator):
  """WOD 2D detection evaluation metric class."""

  def __init__(self, image_config: Any, config: Any = None):
    if config is None:
      config = self._get_default_config()
      config.box_type = label_pb2.Label.Box.TYPE_2D
    super().__init__(image_config, config)

  # use utils
  def _get_box(self, box2d: tf.Tensor,
               attributes: Mapping[str, tf.Tensor]) -> tf.Tensor:
    """Get box from yxyx and attributes.

    Args:
      box2d: a float32 [N, 4] tensor encoding as (ymin, xmin, ymax, xmax)
      attributes: a float32 {name: [N, 1]} dict
    Returns:
      box: a float32 [N, 5] tensor representing a 2d box with heading
    """
    box2d = utils.image_to_frame_boxes(box2d, self._vehicle_xy,
                                       self._resolution)
    values = []
    values.append(box2d[:, 0])  # center_x
    values.append(box2d[:, 1])  # center_y
    values.append(box2d[:, 2])  # length
    values.append(box2d[:, 3])  # width
    values.append(attributes['heading'][:, 0])  # heading
    box2d_h = tf.stack(values, axis=-1)
    return box2d_h


def create_evaluator(model_config: cfg.PointPillarsModel) -> _AbstractEvaluator:
  """Create either 2d or 3d evaluator."""
  attr_count = len(model_config.head.attribute_heads)
  if attr_count == 1:
    logging.info('Use 2D detection evaluator.')
    return Wod2dDetectionEvaluator(model_config)
  if attr_count == 3:
    logging.info('Use 3D detection evaluator.')
    return Wod3dDetectionEvaluator(model_config)
  raise ValueError(
      'The length of attribute_heads should be 1 or 3, found {}'.format(
          attr_count))
