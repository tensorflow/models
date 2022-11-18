# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Data decoder and parser for Pointpillars."""

from typing import Any, Dict, List, Tuple

import tensorflow as tf

from official.projects.pointpillars.utils import utils
from official.vision.dataloaders import parser
from official.vision.ops import anchor
from official.vision.ops import preprocess_ops


class Parser(parser.Parser):
  """The class to parse decoded tensors to features and labels.

  Notations:
    N: number of pillars in an example
    P: number of points in a pillar
    D: number of features in a point
    M: number of labeled boxes in an example
    L: number of anchor boxes per pixel/location
  """

  def __init__(self,
               classes: str,
               min_level: int,
               max_level: int,
               image_size: Tuple[int, int],
               anchor_sizes: List[Tuple[float, float]],
               match_threshold: float,
               unmatched_threshold: float,
               max_num_detections: int,
               dtype: str):
    """Initialize the parser.

    Args:
      classes: A str to indicate which classes should be predicted.
      min_level: An `int` minimum level of multiscale outputs.
      max_level: An `int` maximum level of multiscale outputs.
      image_size: A tuple (height, width) of image size.
      anchor_sizes: A list of tuple (length, width) of anchor boxes.
      match_threshold: A float number for positive anchor boxes.
      unmatched_threshold: A float number for negative anchor boxes.
      max_num_detections: An `int` number of maximum number of instances in an
        image. The groundtruth data will be clipped/padded to the number.
      dtype: `str`, data type. One of {`bfloat16`, `float32`, `float16`}.
    """
    self._classes = classes
    self._image_size = image_size
    self._match_threshold = match_threshold
    self._unmatched_threshold = unmatched_threshold
    self._max_num_detections = max_num_detections
    self._dtype = dtype

    # Generate anchors,
    # multi-level anchor dict, {level: [h_1, w_l, anchors_per_location * 4]}.
    self._anchor_boxes = utils.generate_anchors(min_level,
                                                max_level,
                                                image_size,
                                                anchor_sizes)

  def _fix_groundtruths_size(self, groundtruths: Dict[str, Any],
                             size: int) -> Dict[str, Any]:
    """Clips or pads the first dimension of groundtruths to the fixed size.

    Args:
      groundtruths: A dictionary of {`str`: `tf.Tensor`} that contains
        groundtruth annotations of `classes`, `boxes`, `attributes` and
        `difficulty`.
      size: An `int` that specifies the expected size of the first dimension of
        padded tensors.

    Returns:
      A dictionary of the same keys as input and padded tensors as values.
    """
    groundtruths['classes'] = preprocess_ops.clip_or_pad_to_fixed_size(
        groundtruths['classes'], size, -1)
    groundtruths['boxes'] = preprocess_ops.clip_or_pad_to_fixed_size(
        groundtruths['boxes'], size, -1)
    if 'attributes' in groundtruths:
      for k, v in groundtruths['attributes'].items():
        groundtruths['attributes'][
            k] = preprocess_ops.clip_or_pad_to_fixed_size(v, size, -1)
    groundtruths['difficulty'] = preprocess_ops.clip_or_pad_to_fixed_size(
        groundtruths['difficulty'], size, -1)
    return groundtruths

  def _filter_level_2_labels(
      self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Filter labels whose level is 2 [only for training]."""
    mask = tf.where(data['gt_difficulty'] < 2)
    data['gt_classes'] = tf.gather_nd(data['gt_classes'], mask)
    data['gt_boxes'] = tf.gather_nd(data['gt_boxes'], mask)
    for k, v in data['gt_attributes'].items():
      data['gt_attributes'][k] = tf.gather_nd(v, mask)
    data['gt_difficulty'] = tf.gather_nd(data['gt_difficulty'], mask)
    return data

  def _filter_non_class_labels(
      self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Filter labels whose class is not self._classes."""
    if self._classes == 'all':
      return data
    mask = tf.where(data['gt_classes'] == utils.CLASSES[self._classes])
    data['gt_classes'] = tf.gather_nd(data['gt_classes'], mask)
    data['gt_boxes'] = tf.gather_nd(data['gt_boxes'], mask)
    for k, v in data['gt_attributes'].items():
      data['gt_attributes'][k] = tf.gather_nd(v, mask)
    data['gt_difficulty'] = tf.gather_nd(data['gt_difficulty'], mask)
    # Reset 'bbox/class' to 1 to be a binary classification.
    data['gt_classes'] = tf.ones_like(data['gt_classes'], dtype=tf.int32)
    return data

  def _parse_feature_and_label(
      self, data: Dict[str, Any]
  ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Parse decoded tensors to features and labels.

    Args:
      data: A {name: tensor} dict of decoded tensors.

    Returns:
      features:
        - pillars: A tensor, shape: [N, P, D], type: self._dtype
        - indices: A tensor with shape: [N, 2], type: int32
      labels:
        - cls_targets: A {level_i: [h_i, w_i, L]} dict, type: float32
        - box_targets: A {level_i: [h_i, w_i, L * 4]} dict, type: float32
        - attribute_targets: A {name: {level_i: [h_i, w_i, L * 1]}} dict,
          type: float32
        - cls_weights: A flattened tensor with shape [total_num_anchors],
          total_num_anchors is anchors across all levels, type: float32
        - box_weights: A flattened tensor with shape [total_num_anchors],
          total_num_anchors is anchors across all levels, type: float32
    """
    data = self._filter_non_class_labels(data)

    pillars = data['pillars']
    indices = data['indices']
    classes = data['gt_classes']
    boxes = data['gt_boxes']
    attributes = data['gt_attributes']

    # Label anchors,
    # multi-level labels, {level: [h_l, w_l, ...]}.
    anchor_labeler = anchor.AnchorLabeler(self._match_threshold,
                                          self._unmatched_threshold)
    (cls_targets, box_targets, att_targets, cls_weights,
     box_weights) = anchor_labeler.label_anchors(
         self._anchor_boxes, boxes, tf.expand_dims(classes, axis=1), attributes)

    # Casts input to desired data type.
    pillars = tf.cast(pillars, dtype=self._dtype)

    # Packs features and labels for model_fn outputs.
    features = {
        'pillars': pillars,
        'indices': indices,
    }
    labels = {
        'cls_targets': cls_targets,
        'box_targets': box_targets,
        'attribute_targets': att_targets,
        'cls_weights': cls_weights,
        'box_weights': box_weights,
    }
    return features, labels

  def _parse_train_data(
      self, data: Dict[str, Any]
  ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Parse data for training."""
    # Skip level 2 boxes for training.
    data = self._filter_level_2_labels(data)
    return self._parse_feature_and_label(data)

  def _parse_eval_data(
      self, data: Dict[str, Any]
  ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Parse data for evaluation.

    Args:
      data: A {name: tensor} dict of decoded tensors.

    Returns:
      Other than features and labels for training, evaluation needs groundtruths
      to calculate metrics.
      groundtruths:
        - frame_id: An int64 tensor to identify an example.
        - num_detections: An `int` tensor representing the real number of boxes
          used for computing metrics.
        - classes: A [max_num_detections] int32 tensor
        - boxes: A [max_num_detections, 4] float32 tensor
        - attributes: A {name: [max_num_detections, 1]} float32 dict
        - difficulty: A [max_num_detections] int32 tensor
    """
    features, labels = self._parse_feature_and_label(data)

    # Add for detection generator.
    labels.update({
        'anchor_boxes': self._anchor_boxes,
        'image_shape': tf.convert_to_tensor(self._image_size),
    })

    # Add groundtruth for metric evaluator.
    # The number of boxes to calculate evaluation metrics, will be used to
    # remove padding in evaluator.
    num_detections = tf.minimum(
        tf.shape(data['gt_classes'])[0], self._max_num_detections)
    groundtruths = {
        'frame_id': data['frame_id'],
        'num_detections': num_detections,
        'classes': data['gt_classes'],
        'boxes': data['gt_boxes'],
        'attributes': data['gt_attributes'],
        'difficulty': data['gt_difficulty'],
    }
    # Fix the size for batching
    groundtruths = self._fix_groundtruths_size(groundtruths,
                                               self._max_num_detections)
    labels['groundtruths'] = groundtruths

    return features, labels
