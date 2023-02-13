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

"""Contains definitions of generators to generate the final detections."""
import contextlib
from typing import Any, Dict, List, Optional, Mapping, Sequence, Tuple

# Import libraries

import numpy as np
import tensorflow as tf

from official.vision.modeling.layers import edgetpu
from official.vision.ops import box_ops
from official.vision.ops import nms
from official.vision.ops import preprocess_ops


def _generate_detections_v1(
    boxes: tf.Tensor,
    scores: tf.Tensor,
    attributes: Optional[Mapping[str, tf.Tensor]] = None,
    pre_nms_top_k: int = 5000,
    pre_nms_score_threshold: float = 0.05,
    nms_iou_threshold: float = 0.5,
    max_num_detections: int = 100,
    soft_nms_sigma: Optional[float] = None,
):
  """Generates the final detections given the model outputs.

  The implementation unrolls the batch dimension and process images one by one.
  It required the batch dimension to be statically known and it is TPU
  compatible.

  Args:
    boxes: A `tf.Tensor` with shape `[batch_size, N, num_classes, 4]` or
      `[batch_size, N, 1, 4]` for box predictions on all feature levels. The N
      is the number of total anchors on all levels.
    scores: A `tf.Tensor` with shape `[batch_size, N, num_classes]`, which
      stacks class probability on all feature levels. The N is the number of
      total anchors on all levels. The num_classes is the number of classes
      predicted by the model. Note that the class_outputs here is the raw score.
    attributes: None or a dict of (attribute_name, attributes) pairs. Each
      attributes is a `tf.Tensor` with shape `[batch_size, N, num_classes,
      attribute_size]` or `[batch_size, N, 1, attribute_size]` for attribute
      predictions on all feature levels. The N is the number of total anchors on
      all levels. Can be None if no attribute learning is required.
    pre_nms_top_k: An `int` number of top candidate detections per class before
      NMS.
    pre_nms_score_threshold: A `float` representing the threshold for deciding
      when to remove boxes based on score.
    nms_iou_threshold: A `float` representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    max_num_detections: A scalar representing maximum number of boxes retained
      over all classes.
    soft_nms_sigma: A `float` representing the sigma parameter for Soft NMS.
      When soft_nms_sigma=0.0 (which is default), we fall back to standard NMS.

  Returns:
    nms_boxes: A `float` type `tf.Tensor` of shape
      `[batch_size, max_num_detections, 4]` representing top detected boxes in
      `[y1, x1, y2, x2]`.
    nms_scores: A `float` type `tf.Tensor` of shape
      `[batch_size, max_num_detections]` representing sorted confidence scores
      for detected boxes. The values are between `[0, 1]`.
    nms_classes: An `int` type `tf.Tensor` of shape
      `[batch_size, max_num_detections]` representing classes for detected
      boxes.
    valid_detections: An `int` type `tf.Tensor` of shape `[batch_size]` only the
       top `valid_detections` boxes are valid detections.
    nms_attributes: None or a dict of (attribute_name, attributes). Each
      attribute is a `float` type `tf.Tensor` of shape
      `[batch_size, max_num_detections, attribute_size]` representing attribute
      predictions for detected boxes. Can be an empty dict if no attribute
      learning is required.
  """
  with tf.name_scope('generate_detections'):
    batch_size = scores.get_shape().as_list()[0]
    nmsed_boxes = []
    nmsed_classes = []
    nmsed_scores = []
    valid_detections = []
    if attributes:
      nmsed_attributes = {att_name: [] for att_name in attributes.keys()}
    else:
      nmsed_attributes = {}

    for i in range(batch_size):
      (
          nmsed_boxes_i,
          nmsed_scores_i,
          nmsed_classes_i,
          valid_detections_i,
          nmsed_att_i,
      ) = _generate_detections_per_image(
          boxes[i],
          scores[i],
          attributes={att_name: att[i] for att_name, att in attributes.items()}
          if attributes
          else {},
          pre_nms_top_k=pre_nms_top_k,
          pre_nms_score_threshold=pre_nms_score_threshold,
          nms_iou_threshold=nms_iou_threshold,
          max_num_detections=max_num_detections,
          soft_nms_sigma=soft_nms_sigma,
      )
      nmsed_boxes.append(nmsed_boxes_i)
      nmsed_scores.append(nmsed_scores_i)
      nmsed_classes.append(nmsed_classes_i)
      valid_detections.append(valid_detections_i)
      if attributes:
        for att_name in attributes.keys():
          nmsed_attributes[att_name].append(nmsed_att_i[att_name])

  nmsed_boxes = tf.stack(nmsed_boxes, axis=0)
  nmsed_scores = tf.stack(nmsed_scores, axis=0)
  nmsed_classes = tf.stack(nmsed_classes, axis=0)
  valid_detections = tf.stack(valid_detections, axis=0)
  if attributes:
    for att_name in attributes.keys():
      nmsed_attributes[att_name] = tf.stack(nmsed_attributes[att_name], axis=0)

  return (
      nmsed_boxes,
      nmsed_scores,
      nmsed_classes,
      valid_detections,
      nmsed_attributes,
  )


def _generate_detections_per_image(
    boxes: tf.Tensor,
    scores: tf.Tensor,
    attributes: Optional[Mapping[str, tf.Tensor]] = None,
    pre_nms_top_k: int = 5000,
    pre_nms_score_threshold: float = 0.05,
    nms_iou_threshold: float = 0.5,
    max_num_detections: int = 100,
    soft_nms_sigma: Optional[float] = None,
):
  """Generates the final detections per image given the model outputs.

  Args:
    boxes: A  `tf.Tensor` with shape `[N, num_classes, 4]` or `[N, 1, 4]`, which
      box predictions on all feature levels. The N is the number of total
      anchors on all levels.
    scores: A `tf.Tensor` with shape `[N, num_classes]`, which stacks class
      probability on all feature levels. The N is the number of total anchors on
      all levels. The num_classes is the number of classes predicted by the
      model. Note that the class_outputs here is the raw score.
    attributes: If not None, a dict of `tf.Tensor`. Each value is in shape `[N,
      num_classes, attribute_size]` or `[N, 1, attribute_size]` of attribute
      predictions on all feature levels. The N is the number of total anchors on
      all levels.
    pre_nms_top_k: An `int` number of top candidate detections per class before
      NMS.
    pre_nms_score_threshold: A `float` representing the threshold for deciding
      when to remove boxes based on score.
    nms_iou_threshold: A `float` representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    max_num_detections: A `scalar` representing maximum number of boxes retained
      over all classes.
    soft_nms_sigma: A `float` representing the sigma parameter for Soft NMS.
      When soft_nms_sigma=0.0, we fall back to standard NMS. If set to None,
      `tf.image.non_max_suppression_padded` is called instead.

  Returns:
    nms_boxes: A `float` tf.Tensor of shape `[max_num_detections, 4]`
      representing top detected boxes in `[y1, x1, y2, x2]`.
    nms_scores: A `float` tf.Tensor of shape `[max_num_detections]` representing
      sorted confidence scores for detected boxes. The values are between [0,
      1].
    nms_classes: An `int` tf.Tensor of shape `[max_num_detections]` representing
      classes for detected boxes.
    valid_detections: An `int` tf.Tensor of shape [1] only the top
      `valid_detections` boxes are valid detections.
    nms_attributes: None or a dict. Each value is a `float` tf.Tensor of shape
      `[max_num_detections, attribute_size]` representing attribute predictions
      for detected boxes. Can be an empty dict if `attributes` is None.
  """
  nmsed_boxes = []
  nmsed_scores = []
  nmsed_classes = []
  num_classes_for_box = boxes.get_shape().as_list()[1]
  num_classes = scores.get_shape().as_list()[1]
  if attributes:
    nmsed_attributes = {att_name: [] for att_name in attributes.keys()}
  else:
    nmsed_attributes = {}

  for i in range(num_classes):
    boxes_i = boxes[:, min(num_classes_for_box - 1, i)]
    scores_i = scores[:, i]
    # Obtains pre_nms_top_k before running NMS.
    scores_i, indices = tf.nn.top_k(
        scores_i, k=tf.minimum(tf.shape(scores_i)[-1], pre_nms_top_k)
    )
    boxes_i = tf.gather(boxes_i, indices)

    if soft_nms_sigma is not None:
      (nmsed_indices_i, nmsed_scores_i) = (
          tf.image.non_max_suppression_with_scores(
              tf.cast(boxes_i, tf.float32),
              tf.cast(scores_i, tf.float32),
              max_num_detections,
              iou_threshold=nms_iou_threshold,
              score_threshold=pre_nms_score_threshold,
              soft_nms_sigma=soft_nms_sigma,
              name='nms_detections_' + str(i),
          )
      )
      nmsed_boxes_i = tf.gather(boxes_i, nmsed_indices_i)
      nmsed_boxes_i = preprocess_ops.clip_or_pad_to_fixed_size(
          nmsed_boxes_i, max_num_detections, 0.0
      )
      nmsed_scores_i = preprocess_ops.clip_or_pad_to_fixed_size(
          nmsed_scores_i, max_num_detections, -1.0
      )
    else:
      (nmsed_indices_i, nmsed_num_valid_i) = (
          tf.image.non_max_suppression_padded(
              tf.cast(boxes_i, tf.float32),
              tf.cast(scores_i, tf.float32),
              max_num_detections,
              iou_threshold=nms_iou_threshold,
              score_threshold=pre_nms_score_threshold,
              pad_to_max_output_size=True,
              name='nms_detections_' + str(i),
          )
      )
      nmsed_boxes_i = tf.gather(boxes_i, nmsed_indices_i)
      nmsed_scores_i = tf.gather(scores_i, nmsed_indices_i)
      # Sets scores of invalid boxes to -1.
      nmsed_scores_i = tf.where(
          tf.less(tf.range(max_num_detections), [nmsed_num_valid_i]),
          nmsed_scores_i,
          -tf.ones_like(nmsed_scores_i),
      )

    nmsed_classes_i = tf.fill([max_num_detections], i)
    nmsed_boxes.append(nmsed_boxes_i)
    nmsed_scores.append(nmsed_scores_i)
    nmsed_classes.append(nmsed_classes_i)
    if attributes:
      for att_name, att in attributes.items():
        num_classes_for_attr = att.get_shape().as_list()[1]
        att_i = att[:, min(num_classes_for_attr - 1, i)]
        att_i = tf.gather(att_i, indices)
        nmsed_att_i = tf.gather(att_i, nmsed_indices_i)
        nmsed_att_i = preprocess_ops.clip_or_pad_to_fixed_size(
            nmsed_att_i, max_num_detections, 0.0
        )
        nmsed_attributes[att_name].append(nmsed_att_i)

  # Concats results from all classes and sort them.
  nmsed_boxes = tf.concat(nmsed_boxes, axis=0)
  nmsed_scores = tf.concat(nmsed_scores, axis=0)
  nmsed_classes = tf.concat(nmsed_classes, axis=0)
  nmsed_scores, indices = tf.nn.top_k(
      nmsed_scores, k=max_num_detections, sorted=True
  )
  nmsed_boxes = tf.gather(nmsed_boxes, indices)
  nmsed_classes = tf.gather(nmsed_classes, indices)
  valid_detections = tf.reduce_sum(
      tf.cast(tf.greater(nmsed_scores, -1), tf.int32)
  )
  if attributes:
    for att_name in attributes.keys():
      nmsed_attributes[att_name] = tf.concat(nmsed_attributes[att_name], axis=0)
      nmsed_attributes[att_name] = tf.gather(
          nmsed_attributes[att_name], indices
      )

  return (
      nmsed_boxes,
      nmsed_scores,
      nmsed_classes,
      valid_detections,
      nmsed_attributes,
  )


def _select_top_k_scores(scores_in: tf.Tensor, pre_nms_num_detections: int):
  """Selects top_k scores and indices for each class.

  Args:
    scores_in: A `tf.Tensor` with shape `[batch_size, N, num_classes]`, which
      stacks class logit outputs on all feature levels. The N is the number of
      total anchors on all levels. The num_classes is the number of classes
      predicted by the model.
    pre_nms_num_detections: Number of candidates before NMS.

  Returns:
    scores and indices: A `tf.Tensor` with shape
      `[batch_size, pre_nms_num_detections, num_classes]`.
  """
  batch_size, num_anchors, num_class = scores_in.get_shape().as_list()
  if batch_size is None:
    batch_size = tf.shape(scores_in)[0]
  scores_trans = tf.transpose(scores_in, perm=[0, 2, 1])
  scores_trans = tf.reshape(scores_trans, [-1, num_anchors])

  top_k_scores, top_k_indices = tf.nn.top_k(
      scores_trans, k=pre_nms_num_detections, sorted=True
  )

  top_k_scores = tf.reshape(
      top_k_scores, [batch_size, num_class, pre_nms_num_detections]
  )
  top_k_indices = tf.reshape(
      top_k_indices, [batch_size, num_class, pre_nms_num_detections]
  )

  return tf.transpose(top_k_scores, [0, 2, 1]), tf.transpose(
      top_k_indices, [0, 2, 1]
  )


def _generate_detections_v2_class_agnostic(
    boxes: tf.Tensor,
    scores: tf.Tensor,
    pre_nms_top_k: int = 5000,
    pre_nms_score_threshold: float = 0.05,
    nms_iou_threshold: float = 0.5,
    max_num_detections: int = 100
):
  """Generates the final detections by applying class-agnostic NMS.

  Args:
    boxes: A `tf.Tensor` with shape `[batch_size, N, num_classes, 4]` or
      `[batch_size, N, 1, 4]`, which box predictions on all feature levels. The
      N is the number of total anchors on all levels.
    scores: A `tf.Tensor` with shape `[batch_size, N, num_classes]`, which
      stacks class probability on all feature levels. The N is the number of
      total anchors on all levels. The num_classes is the number of classes
      predicted by the model. Note that the class_outputs here is the raw score.
    pre_nms_top_k: An `int` number of top candidate detections per class before
      NMS.
    pre_nms_score_threshold: A `float` representing the threshold for deciding
      when to remove boxes based on score.
    nms_iou_threshold: A `float` representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    max_num_detections: A `scalar` representing maximum number of boxes retained
      over all classes.

  Returns:
    nms_boxes: A `float` tf.Tensor of shape [batch_size, max_num_detections, 4]
      representing top detected boxes in [y1, x1, y2, x2].
    nms_scores: A `float` tf.Tensor of shape [batch_size, max_num_detections]
      representing sorted confidence scores for detected boxes. The values are
      between [0, 1].
    nms_classes: An `int` tf.Tensor of shape [batch_size, max_num_detections]
      representing classes for detected boxes.
    valid_detections: An `int` tf.Tensor of shape [batch_size] only the top
      `valid_detections` boxes are valid detections.
  """
  with tf.name_scope('generate_detections_class_agnostic'):
    nmsed_boxes = []
    nmsed_classes = []
    nmsed_scores = []
    valid_detections = []
    batch_size, _, num_classes_for_box, _ = boxes.get_shape().as_list()
    if batch_size is None:
      batch_size = tf.shape(boxes)[0]
    _, total_anchors, _ = scores.get_shape().as_list()

    # Keeps only the class with highest score for each predicted box.
    scores_condensed, classes_ids = tf.nn.top_k(
        scores, k=1, sorted=True
    )
    scores_condensed = tf.squeeze(scores_condensed, axis=[2])
    if num_classes_for_box > 1:
      boxes = tf.gather(boxes, classes_ids, axis=2, batch_dims=2)
    boxes_condensed = tf.squeeze(boxes, axis=[2])
    classes_condensed = tf.squeeze(classes_ids, axis=[2])

    # Selects top pre_nms_num scores and indices before NMS.
    num_anchors_filtered = min(total_anchors, pre_nms_top_k)
    scores_filtered, indices_filtered = tf.nn.top_k(
        scores_condensed, k=num_anchors_filtered, sorted=True
    )
    classes_filtered = tf.gather(
        classes_condensed, indices_filtered, axis=1, batch_dims=1
    )
    boxes_filtered = tf.gather(
        boxes_condensed, indices_filtered, axis=1, batch_dims=1
    )

    tf.ensure_shape(boxes_filtered, [None, num_anchors_filtered, 4])
    tf.ensure_shape(classes_filtered, [None, num_anchors_filtered])
    tf.ensure_shape(scores_filtered, [None, num_anchors_filtered])
    boxes_filtered = tf.cast(
        boxes_filtered, tf.float32
    )
    scores_filtered = tf.cast(
        scores_filtered, tf.float32
    )
    # Apply class-agnostic NMS on boxes.
    (nmsed_indices_padded, valid_detections) = (
        tf.image.non_max_suppression_padded(
            boxes=boxes_filtered,
            scores=scores_filtered,
            max_output_size=max_num_detections,
            iou_threshold=nms_iou_threshold,
            pad_to_max_output_size=True,
            score_threshold=pre_nms_score_threshold,
            sorted_input=True,
            name='nms_detections'
        )
    )
    nmsed_boxes = tf.gather(
        boxes_filtered, nmsed_indices_padded, batch_dims=1, axis=1
    )
    nmsed_scores = tf.gather(
        scores_filtered, nmsed_indices_padded, batch_dims=1, axis=1
    )
    nmsed_classes = tf.gather(
        classes_filtered, nmsed_indices_padded, batch_dims=1, axis=1
    )

    # Sets the padded boxes, scores, and classes to 0.
    padding_mask = tf.reshape(
        tf.range(max_num_detections), [1, -1]
    ) < tf.reshape(valid_detections, [-1, 1])
    nmsed_boxes = nmsed_boxes * tf.cast(
        tf.expand_dims(padding_mask, axis=2), nmsed_boxes.dtype
    )
    nmsed_scores = nmsed_scores * tf.cast(padding_mask, nmsed_scores.dtype)
    nmsed_classes = nmsed_classes * tf.cast(padding_mask, nmsed_classes.dtype)

  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


def _generate_detections_v2_class_aware(
    boxes: tf.Tensor,
    scores: tf.Tensor,
    pre_nms_top_k: int = 5000,
    pre_nms_score_threshold: float = 0.05,
    nms_iou_threshold: float = 0.5,
    max_num_detections: int = 100,
):
  """Generates the final detections by using class-aware NMS.

  Args:
    boxes: A `tf.Tensor` with shape `[batch_size, N, num_classes, 4]` or
      `[batch_size, N, 1, 4]`, which box predictions on all feature levels. The
      N is the number of total anchors on all levels.
    scores: A `tf.Tensor` with shape `[batch_size, N, num_classes]`, which
      stacks class probability on all feature levels. The N is the number of
      total anchors on all levels. The num_classes is the number of classes
      predicted by the model. Note that the class_outputs here is the raw score.
    pre_nms_top_k: An `int` number of top candidate detections per class before
      NMS.
    pre_nms_score_threshold: A `float` representing the threshold for deciding
      when to remove boxes based on score.
    nms_iou_threshold: A `float` representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    max_num_detections: A `scalar` representing maximum number of boxes retained
      over all classes.

  Returns:
    nms_boxes: A `float` tf.Tensor of shape [batch_size, max_num_detections, 4]
      representing top detected boxes in [y1, x1, y2, x2].
    nms_scores: A `float` tf.Tensor of shape [batch_size, max_num_detections]
      representing sorted confidence scores for detected boxes. The values are
      between [0, 1].
    nms_classes: An `int` tf.Tensor of shape [batch_size, max_num_detections]
      representing classes for detected boxes.
    valid_detections: An `int` tf.Tensor of shape [batch_size] only the top
      `valid_detections` boxes are valid detections.
  """
  with tf.name_scope('generate_detections'):
    nmsed_boxes = []
    nmsed_classes = []
    nmsed_scores = []
    valid_detections = []
    batch_size, _, num_classes_for_box, _ = boxes.get_shape().as_list()
    if batch_size is None:
      batch_size = tf.shape(boxes)[0]
    _, total_anchors, num_classes = scores.get_shape().as_list()
    # Selects top pre_nms_num scores and indices before NMS.
    scores, indices = _select_top_k_scores(
        scores, min(total_anchors, pre_nms_top_k)
    )
    for i in range(num_classes):
      boxes_i = boxes[:, :, min(num_classes_for_box - 1, i), :]
      scores_i = scores[:, :, i]
      # Obtains pre_nms_top_k before running NMS.
      boxes_i = tf.gather(boxes_i, indices[:, :, i], batch_dims=1, axis=1)

      # Filter out scores.
      boxes_i, scores_i = box_ops.filter_boxes_by_scores(
          boxes_i, scores_i, min_score_threshold=pre_nms_score_threshold
      )

      (nmsed_scores_i, nmsed_boxes_i) = nms.sorted_non_max_suppression_padded(
          tf.cast(scores_i, tf.float32),
          tf.cast(boxes_i, tf.float32),
          max_num_detections,
          iou_threshold=nms_iou_threshold,
      )
      nmsed_classes_i = tf.fill([batch_size, max_num_detections], i)
      nmsed_boxes.append(nmsed_boxes_i)
      nmsed_scores.append(nmsed_scores_i)
      nmsed_classes.append(nmsed_classes_i)
  nmsed_boxes = tf.concat(nmsed_boxes, axis=1)
  nmsed_scores = tf.concat(nmsed_scores, axis=1)
  nmsed_classes = tf.concat(nmsed_classes, axis=1)
  nmsed_scores, indices = tf.nn.top_k(
      nmsed_scores, k=max_num_detections, sorted=True
  )
  nmsed_boxes = tf.gather(nmsed_boxes, indices, batch_dims=1, axis=1)
  nmsed_classes = tf.gather(nmsed_classes, indices, batch_dims=1)
  valid_detections = tf.reduce_sum(
      input_tensor=tf.cast(tf.greater(nmsed_scores, 0.0), tf.int32), axis=1
  )
  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


def _generate_detections_v2(
    boxes: tf.Tensor,
    scores: tf.Tensor,
    pre_nms_top_k: int = 5000,
    pre_nms_score_threshold: float = 0.05,
    nms_iou_threshold: float = 0.5,
    max_num_detections: int = 100,
    use_class_agnostic_nms: Optional[bool] = None,
):
  """Generates the final detections given the model outputs.

  This implementation unrolls classes dimension while using the tf.while_loop
  to implement the batched NMS, so that it can be parallelized at the batch
  dimension. It should give better performance comparing to v1 implementation.
  It is TPU compatible.

  Args:
    boxes: A `tf.Tensor` with shape `[batch_size, N, num_classes, 4]` or
      `[batch_size, N, 1, 4]`, which box predictions on all feature levels. The
      N is the number of total anchors on all levels.
    scores: A `tf.Tensor` with shape `[batch_size, N, num_classes]`, which
      stacks class probability on all feature levels. The N is the number of
      total anchors on all levels. The num_classes is the number of classes
      predicted by the model. Note that the class_outputs here is the raw score.
    pre_nms_top_k: An `int` number of top candidate detections per class before
      NMS.
    pre_nms_score_threshold: A `float` representing the threshold for deciding
      when to remove boxes based on score.
    nms_iou_threshold: A `float` representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    max_num_detections: A `scalar` representing maximum number of boxes retained
      over all classes.
    use_class_agnostic_nms: A `bool` of whether non max suppression is operated
      on all the boxes using max scores across all classes.

  Returns:
    nms_boxes: A `float` tf.Tensor of shape [batch_size, max_num_detections, 4]
      representing top detected boxes in [y1, x1, y2, x2].
    nms_scores: A `float` tf.Tensor of shape [batch_size, max_num_detections]
      representing sorted confidence scores for detected boxes. The values are
      between [0, 1].
    nms_classes: An `int` tf.Tensor of shape [batch_size, max_num_detections]
      representing classes for detected boxes.
    valid_detections: An `int` tf.Tensor of shape [batch_size] only the top
      `valid_detections` boxes are valid detections.
  """
  if use_class_agnostic_nms:
    return _generate_detections_v2_class_agnostic(
        boxes=boxes,
        scores=scores,
        pre_nms_top_k=pre_nms_top_k,
        pre_nms_score_threshold=pre_nms_score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_num_detections=max_num_detections,
    )

  return _generate_detections_v2_class_aware(
      boxes=boxes,
      scores=scores,
      pre_nms_top_k=pre_nms_top_k,
      pre_nms_score_threshold=pre_nms_score_threshold,
      nms_iou_threshold=nms_iou_threshold,
      max_num_detections=max_num_detections,
  )


def _generate_detections_v3(
    boxes: tf.Tensor,
    scores: tf.Tensor,
    pre_nms_score_threshold: float = 0.05,
    nms_iou_threshold: float = 0.5,
    max_num_detections: int = 100,
    refinements: int = 2,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Generates the detections given the model outputs using NMS for EdgeTPU.

  Args:
    boxes: A `tf.Tensor` with shape `[batch_size, num_classes, N, 4]` or
      `[batch_size, 1, N, 4]`, which box predictions on all feature levels. The
      N is the number of total anchors on all levels.
    scores: A `tf.Tensor` with shape `[batch_size, num_classes, N]`, which
      stacks class probability on all feature levels. The N is the number of
      total anchors on all levels. The num_classes is the number of classes
      predicted by the model. Note that the class_outputs here is the raw score.
    pre_nms_score_threshold: A `float` representing the threshold for deciding
      when to remove boxes based on score.
    nms_iou_threshold: A `float` representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    max_num_detections: A `scalar` representing maximum number of boxes retained
      over all classes.
    refinements: Quality parameter for NMS algorithm.

  Returns:
    nms_boxes: A `float` tf.Tensor of shape [batch_size, max_num_detections, 4]
      representing top detected boxes in [y1, x1, y2, x2].
    nms_scores: A `float` tf.Tensor of shape [batch_size, max_num_detections]
      representing sorted confidence scores for detected boxes. The values are
      between [0, 1].
    nms_classes: An `int` tf.Tensor of shape [batch_size, max_num_detections]
      representing classes for detected boxes.
    valid_detections: An `int` tf.Tensor of shape [batch_size] only the top
      `valid_detections` boxes are valid detections.

  Raises:
    ValueError if inputs shapes are not valid.
  """
  one = tf.constant(1, dtype=scores.dtype)
  with tf.name_scope('generate_detections'):
    batch_size, num_box_classes, box_locations, sides = (
        boxes.get_shape().as_list()
    )
    if batch_size is None:
      batch_size = tf.shape(boxes)[0]
    _, num_classes, locations = scores.get_shape().as_list()
    if num_box_classes != 1 and num_box_classes != num_classes:
      raise ValueError('Boxes should have either 1 class or same as scores.')
    if locations != box_locations:
      raise ValueError('Number of locations is different.')
    if sides != 4:
      raise ValueError('Number of sides is incorrect.')
    # Selects pre_nms_score_threshold scores before NMS.
    boxes, scores = box_ops.filter_boxes_by_scores(
        boxes, scores, min_score_threshold=pre_nms_score_threshold
    )

    # EdgeTPU-friendly class-wise NMS, -1 for invalid.
    indices = edgetpu.non_max_suppression_padded(
        boxes,
        scores,
        max_num_detections,
        iou_threshold=nms_iou_threshold,
        refinements=refinements,
    )
    # Gather NMS-ed boxes and scores.
    safe_indices = tf.nn.relu(indices)  # 0 for invalid
    invalid_detections = safe_indices - indices  # 1 for invalid, 0 for valid
    valid_detections = one - invalid_detections  # 0 for invalid, 1 for valid
    safe_indices = tf.cast(safe_indices, tf.int32)
    boxes = tf.gather(boxes, safe_indices, axis=2, batch_dims=2)
    boxes = tf.cast(tf.expand_dims(valid_detections, -1), boxes.dtype) * boxes
    scores = valid_detections * tf.gather(
        scores, safe_indices, axis=2, batch_dims=2
    )
    # Compliment with class numbers.
    classes = tf.constant(np.arange(num_classes), dtype=scores.dtype)
    classes = tf.reshape(classes, [1, num_classes, 1])
    classes = tf.tile(classes, [batch_size, 1, max_num_detections])
    # Flatten classes, locations. Class = -1 for invalid detection
    scores = tf.reshape(scores, [batch_size, num_classes * max_num_detections])
    boxes = tf.reshape(boxes, [batch_size, num_classes * max_num_detections, 4])
    classes = tf.reshape(
        valid_detections * classes - invalid_detections,
        [batch_size, num_classes * max_num_detections],
    )
    # Filter top-k across boxes of all classes
    scores, indices = tf.nn.top_k(scores, k=max_num_detections, sorted=True)
    boxes = tf.gather(boxes, indices, batch_dims=1, axis=1)
    classes = tf.gather(classes, indices, batch_dims=1, axis=1)
    invalid_detections = tf.nn.relu(classes) - classes
    valid_detections = tf.reduce_sum(one - invalid_detections, axis=1)
    return boxes, scores, classes, valid_detections


def _generate_detections_batched(
    boxes: tf.Tensor,
    scores: tf.Tensor,
    pre_nms_score_threshold: float,
    nms_iou_threshold: float,
    max_num_detections: int,
):
  """Generates detected boxes with scores and classes for one-stage detector.

  The function takes output of multi-level ConvNets and anchor boxes and
  generates detected boxes. Note that this used batched nms, which is not
  supported on TPU currently.

  Args:
    boxes: A `tf.Tensor` with shape `[batch_size, N, num_classes, 4]` or
      `[batch_size, N, 1, 4]`, which box predictions on all feature levels. The
      N is the number of total anchors on all levels.
    scores: A `tf.Tensor` with shape `[batch_size, N, num_classes]`, which
      stacks class probability on all feature levels. The N is the number of
      total anchors on all levels. The num_classes is the number of classes
      predicted by the model. Note that the class_outputs here is the raw score.
    pre_nms_score_threshold: A `float` representing the threshold for deciding
      when to remove boxes based on score.
    nms_iou_threshold: A `float` representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    max_num_detections: A `scalar` representing maximum number of boxes retained
      over all classes.

  Returns:
    nms_boxes: A `float` tf.Tensor of shape [batch_size, max_num_detections, 4]
      representing top detected boxes in [y1, x1, y2, x2].
    nms_scores: A `float` tf.Tensor of shape [batch_size, max_num_detections]
      representing sorted confidence scores for detected boxes. The values are
      between [0, 1].
    nms_classes: An `int` tf.Tensor of shape [batch_size, max_num_detections]
      representing classes for detected boxes.
    valid_detections: An `int` tf.Tensor of shape [batch_size] only the top
      `valid_detections` boxes are valid detections.
  """
  with tf.name_scope('generate_detections'):
    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = (
        tf.image.combined_non_max_suppression(
            boxes,
            scores,
            max_output_size_per_class=max_num_detections,
            max_total_size=max_num_detections,
            iou_threshold=nms_iou_threshold,
            score_threshold=pre_nms_score_threshold,
            pad_per_class=False,
            clip_boxes=False,
        )
    )
    nmsed_classes = tf.cast(nmsed_classes, tf.int32)
  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


def _generate_detections_tflite_implements_signature(
    config: Dict[str, Any]
) -> str:
  """Returns `experimental_implements` signature for TFLite's custom NMS op.

  This signature encodes the arguments to correctly initialize TFLite's custom
  post-processing op in the MLIR converter.
  For details on `experimental_implements` see here:
  https://www.tensorflow.org/api_docs/python/tf/function

  Args:
    config: A dictionary of configs defining parameters for TFLite NMS op.

  Returns:
    An `experimental_implements` signature string.
  """
  scale_value = 1.0

  implements_signature = [
      'name: "%s"' % 'TFLite_Detection_PostProcess',
      'attr { key: "max_detections" value { i: %d } }'
      % config['max_detections'],
      'attr { key: "max_classes_per_detection" value { i: %d } }'
      % config['max_classes_per_detection'],
      'attr { key: "use_regular_nms" value { b: %s } }'
      % str(config['use_regular_nms']).lower(),
      'attr { key: "nms_score_threshold" value { f: %f } }'
      % config['nms_score_threshold'],
      'attr { key: "nms_iou_threshold" value { f: %f } }'
      % config['nms_iou_threshold'],
      'attr { key: "y_scale" value { f: %f } }' % scale_value,
      'attr { key: "x_scale" value { f: %f } }' % scale_value,
      'attr { key: "h_scale" value { f: %f } }' % scale_value,
      'attr { key: "w_scale" value { f: %f } }' % scale_value,
      'attr { key: "num_classes" value { i: %d } }' % config['num_classes'],
  ]
  implements_signature = ' '.join(implements_signature)
  return implements_signature


def _generate_detections_tflite(
    raw_boxes: Mapping[str, tf.Tensor],
    raw_scores: Mapping[str, tf.Tensor],
    anchor_boxes: Mapping[str, tf.Tensor],
    config: Dict[str, Any],
) -> Sequence[Any]:
  """Generate detections for conversion to TFLite.

  Mathematically same as class-agnostic NMS, except that the last portion of
  the TF graph constitutes a dummy `tf.function` that contains an annotation
  for conversion to TFLite's custom NMS op. Using this custom op allows
  features like post-training quantization & accelerator support.
  NOTE: This function does NOT return a valid output, and is only meant to
  generate a SavedModel for TFLite conversion via MLIR. The generated SavedModel
  should not be used for inference.
  For TFLite op details, see tensorflow/lite/kernels/detection_postprocess.cc

  Args:
    raw_boxes: A dictionary of tensors for raw boxes. Key is level of features
      and value is a tensor denoting a level of boxes with shape [1, H, W, 4 *
      num_anchors].
    raw_scores: A dictionary of tensors for classes. Key is level of features
      and value is a tensor denoting a level of logits with shape [1, H, W,
      num_class * num_anchors].
    anchor_boxes: A dictionary of tensors for anchor boxes. Key is level of
      features and value is a tensor denoting a level of anchors with shape
      [num_anchors, 4].
    config: A dictionary of configs defining parameters for TFLite NMS op.

  Returns:
    A (dummy) tuple of (boxes, scores, classess, num_detections).

  Raises:
    ValueError: If the last dimension of predicted boxes is not divisible by 4,
      or the last dimension of predicted scores is not divisible by number of
      anchors per location.
  """
  scores, boxes, anchors = [], [], []
  levels = list(raw_scores.keys())
  min_level = int(min(levels))
  max_level = int(max(levels))
  batch_size = tf.shape(raw_scores[str(min_level)])[0]

  num_anchors_per_locations_times_4 = (
      raw_boxes[str(min_level)].get_shape().as_list()[-1]
  )
  if num_anchors_per_locations_times_4 % 4 != 0:
    raise ValueError(
        'The last dimension of predicted boxes should be divisible by 4.'
    )
  num_anchors_per_locations = num_anchors_per_locations_times_4 // 4
  if num_anchors_per_locations_times_4 % 4 != 0:
    raise ValueError(
        'The last dimension of predicted scores should be divisible by'
        f' {num_anchors_per_locations}.'
    )
  num_classes = (
      raw_scores[str(min_level)].get_shape().as_list()[-1]
      // num_anchors_per_locations
  )
  config.update({'num_classes': num_classes})

  for i in range(min_level, max_level + 1):
    scores.append(tf.reshape(raw_scores[str(i)], [batch_size, -1, num_classes]))
    boxes.append(tf.reshape(raw_boxes[str(i)], [batch_size, -1, 4]))
    anchors.append(tf.reshape(anchor_boxes[str(i)], [-1, 4]))
  scores = tf.sigmoid(tf.concat(scores, 1))
  boxes = tf.concat(boxes, 1)
  anchors = tf.concat(anchors, 0)

  ycenter_a = (anchors[..., 0] + anchors[..., 2]) / 2
  xcenter_a = (anchors[..., 1] + anchors[..., 3]) / 2
  ha = anchors[..., 2] - anchors[..., 0]
  wa = anchors[..., 3] - anchors[..., 1]
  anchors = tf.stack([ycenter_a, xcenter_a, ha, wa], axis=-1)

  if config.get('normalize_anchor_coordinates', False):
    # TFLite's object detection APIs require normalized anchors.
    height, width = config['input_image_size']
    normalize_factor = tf.constant(
        [height, width, height, width], dtype=tf.float32
    )
    anchors = anchors / normalize_factor

  # There is no TF equivalent for TFLite's custom post-processing op.
  # So we add an 'empty' composite function here, that is legalized to the
  # custom op with MLIR.
  # For details, see: tensorflow/compiler/mlir/lite/utils/nms_utils.cc
  @tf.function(
      experimental_implements=_generate_detections_tflite_implements_signature(
          config
      )
  )
  # pylint: disable=g-unused-argument,unused-argument
  def dummy_post_processing(input_boxes, input_scores, input_anchors):
    boxes = tf.constant(0.0, dtype=tf.float32, name='boxes')
    scores = tf.constant(0.0, dtype=tf.float32, name='scores')
    classes = tf.constant(0.0, dtype=tf.float32, name='classes')
    num_detections = tf.constant(0.0, dtype=tf.float32, name='num_detections')
    return boxes, classes, scores, num_detections

  return dummy_post_processing(boxes, scores, anchors)[::-1]


@tf.keras.utils.register_keras_serializable(package='Vision')
class DetectionGenerator(tf.keras.layers.Layer):
  """Generates the final detected boxes with scores and classes."""

  def __init__(
      self,
      apply_nms: bool = True,
      pre_nms_top_k: int = 5000,
      pre_nms_score_threshold: float = 0.05,
      nms_iou_threshold: float = 0.5,
      max_num_detections: int = 100,
      nms_version: str = 'v2',
      use_cpu_nms: bool = False,
      soft_nms_sigma: Optional[float] = None,
      use_sigmoid_probability: bool = False,
      **kwargs,
  ):
    """Initializes a detection generator.

    Args:
      apply_nms: A `bool` of whether or not apply non maximum suppression. If
        False, the decoded boxes and their scores are returned.
      pre_nms_top_k: An `int` of the number of top scores proposals to be kept
        before applying NMS.
      pre_nms_score_threshold: A `float` of the score threshold to apply before
        applying  NMS. Proposals whose scores are below this threshold are
        thrown away.
      nms_iou_threshold: A `float` in [0, 1], the NMS IoU threshold.
      max_num_detections: An `int` of the final number of total detections to
        generate.
      nms_version: A string of `batched`, `v1` or `v2` specifies NMS version.
      use_cpu_nms: A `bool` of whether or not enforce NMS to run on CPU.
      soft_nms_sigma: A `float` representing the sigma parameter for Soft NMS.
        When soft_nms_sigma=0.0, we fall back to standard NMS.
      use_sigmoid_probability: A `bool`, if true, use sigmoid to get
        probability, otherwise use softmax.
      **kwargs: Additional keyword arguments passed to Layer.
    """
    self._config_dict = {
        'apply_nms': apply_nms,
        'pre_nms_top_k': pre_nms_top_k,
        'pre_nms_score_threshold': pre_nms_score_threshold,
        'nms_iou_threshold': nms_iou_threshold,
        'max_num_detections': max_num_detections,
        'nms_version': nms_version,
        'use_cpu_nms': use_cpu_nms,
        'soft_nms_sigma': soft_nms_sigma,
        'use_sigmoid_probability': use_sigmoid_probability,
    }
    super(DetectionGenerator, self).__init__(**kwargs)

  def __call__(
      self,
      raw_boxes: tf.Tensor,
      raw_scores: tf.Tensor,
      anchor_boxes: tf.Tensor,
      image_shape: tf.Tensor,
      regression_weights: Optional[List[float]] = None,
      bbox_per_class: bool = True,
  ):
    """Generates final detections.

    Args:
      raw_boxes: A `tf.Tensor` of shape of `[batch_size, K, num_classes * 4]`
        representing the class-specific box coordinates relative to anchors.
      raw_scores: A `tf.Tensor` of shape of `[batch_size, K, num_classes]`
        representing the class logits before applying score activiation.
      anchor_boxes: A `tf.Tensor` of shape of `[batch_size, K, 4]` representing
        the corresponding anchor boxes w.r.t `box_outputs`.
      image_shape: A `tf.Tensor` of shape of `[batch_size, 2]` storing the image
        height and width w.r.t. the scaled image, i.e. the same image space as
        `box_outputs` and `anchor_boxes`.
      regression_weights: A list of four float numbers to scale coordinates.
      bbox_per_class: A `bool`. If True, perform per-class box regression.

    Returns:
      If `apply_nms` = True, the return is a dictionary with keys:
        `detection_boxes`: A `float` tf.Tensor of shape
          [batch, max_num_detections, 4] representing top detected boxes in
          [y1, x1, y2, x2].
        `detection_scores`: A `float` `tf.Tensor` of shape
          [batch, max_num_detections] representing sorted confidence scores for
          detected boxes. The values are between [0, 1].
        `detection_classes`: An `int` tf.Tensor of shape
          [batch, max_num_detections] representing classes for detected boxes.
        `num_detections`: An `int` tf.Tensor of shape [batch] only the first
          `num_detections` boxes are valid detections
      If `apply_nms` = False, the return is a dictionary with keys:
        `decoded_boxes`: A `float` tf.Tensor of shape [batch, num_raw_boxes, 4]
          representing all the decoded boxes.
        `decoded_box_scores`: A `float` tf.Tensor of shape
          [batch, num_raw_boxes] representing socres of all the decoded boxes.
    """
    if self._config_dict['use_sigmoid_probability']:
      box_scores = tf.math.sigmoid(raw_scores)
    else:
      box_scores = tf.nn.softmax(raw_scores, axis=-1)

    # Removes the background class.
    box_scores_shape = tf.shape(box_scores)
    box_scores_shape_list = box_scores.get_shape().as_list()
    batch_size = box_scores_shape[0]
    num_locations = box_scores_shape_list[1]
    num_classes = box_scores_shape_list[-1]

    box_scores = tf.slice(box_scores, [0, 0, 1], [-1, -1, -1])

    if bbox_per_class:
      num_detections = num_locations * (num_classes - 1)
      raw_boxes = tf.reshape(
          raw_boxes, [batch_size, num_locations, num_classes, 4]
      )
      raw_boxes = tf.slice(raw_boxes, [0, 0, 1, 0], [-1, -1, -1, -1])
      anchor_boxes = tf.tile(
          tf.expand_dims(anchor_boxes, axis=2), [1, 1, num_classes - 1, 1]
      )
      raw_boxes = tf.reshape(raw_boxes, [batch_size, num_detections, 4])
      anchor_boxes = tf.reshape(anchor_boxes, [batch_size, num_detections, 4])

    # Box decoding.
    decoded_boxes = box_ops.decode_boxes(
        raw_boxes, anchor_boxes, weights=regression_weights
    )

    # Box clipping
    decoded_boxes = box_ops.clip_boxes(
        decoded_boxes, tf.expand_dims(image_shape, axis=1)
    )

    if bbox_per_class:
      decoded_boxes = tf.reshape(
          decoded_boxes, [batch_size, num_locations, num_classes - 1, 4]
      )
    else:
      decoded_boxes = tf.expand_dims(decoded_boxes, axis=2)

    if not self._config_dict['apply_nms']:
      return {
          'decoded_boxes': decoded_boxes,
          'decoded_box_scores': box_scores,
      }

    # Optionally force the NMS be run on CPU.
    if self._config_dict['use_cpu_nms']:
      nms_context = tf.device('cpu:0')
    else:
      nms_context = contextlib.nullcontext()

    with nms_context:
      if self._config_dict['nms_version'] == 'batched':
        (nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections) = (
            _generate_detections_batched(
                decoded_boxes,
                box_scores,
                self._config_dict['pre_nms_score_threshold'],
                self._config_dict['nms_iou_threshold'],
                self._config_dict['max_num_detections'],
            )
        )
      elif self._config_dict['nms_version'] == 'v1':
        (nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections, _) = (
            _generate_detections_v1(
                decoded_boxes,
                box_scores,
                pre_nms_top_k=self._config_dict['pre_nms_top_k'],
                pre_nms_score_threshold=self._config_dict[
                    'pre_nms_score_threshold'
                ],
                nms_iou_threshold=self._config_dict['nms_iou_threshold'],
                max_num_detections=self._config_dict['max_num_detections'],
                soft_nms_sigma=self._config_dict['soft_nms_sigma'],
            )
        )
      elif self._config_dict['nms_version'] == 'v2':
        (nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections) = (
            _generate_detections_v2(
                decoded_boxes,
                box_scores,
                pre_nms_top_k=self._config_dict['pre_nms_top_k'],
                pre_nms_score_threshold=self._config_dict[
                    'pre_nms_score_threshold'
                ],
                nms_iou_threshold=self._config_dict['nms_iou_threshold'],
                max_num_detections=self._config_dict['max_num_detections'],
            )
        )
      else:
        raise ValueError(
            'NMS version {} not supported.'.format(
                self._config_dict['nms_version']
            )
        )

    # Adds 1 to offset the background class which has index 0.
    nmsed_classes += 1

    return {
        'num_detections': valid_detections,
        'detection_boxes': nmsed_boxes,
        'detection_classes': nmsed_classes,
        'detection_scores': nmsed_scores,
    }

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)


@tf.keras.utils.register_keras_serializable(package='Vision')
class MultilevelDetectionGenerator(tf.keras.layers.Layer):
  """Generates detected boxes with scores and classes for one-stage detector."""

  def __init__(
      self,
      apply_nms: bool = True,
      pre_nms_top_k: int = 5000,
      pre_nms_score_threshold: float = 0.05,
      nms_iou_threshold: float = 0.5,
      max_num_detections: int = 100,
      nms_version: str = 'v1',
      use_cpu_nms: bool = False,
      soft_nms_sigma: Optional[float] = None,
      tflite_post_processing_config: Optional[Dict[str, Any]] = None,
      pre_nms_top_k_sharding_block: Optional[int] = None,
      nms_v3_refinements: Optional[int] = None,
      return_decoded: Optional[bool] = None,
      use_class_agnostic_nms: Optional[bool] = None,
      **kwargs,
  ):
    """Initializes a multi-level detection generator.

    Args:
      apply_nms: A `bool` of whether or not apply non maximum suppression. If
        False, the decoded boxes and their scores are returned.
      pre_nms_top_k: An `int` of the number of top scores proposals to be kept
        before applying NMS.
      pre_nms_score_threshold: A `float` of the score threshold to apply before
        applying NMS. Proposals whose scores are below this threshold are thrown
        away.
      nms_iou_threshold: A `float` in [0, 1], the NMS IoU threshold.
      max_num_detections: An `int` of the final number of total detections to
        generate.
      nms_version: A string of `batched`, `v1` or `v2` specifies NMS version
      use_cpu_nms: A `bool` of whether or not enforce NMS to run on CPU.
      soft_nms_sigma: A `float` representing the sigma parameter for Soft NMS.
        When soft_nms_sigma=0.0, we fall back to standard NMS.
      tflite_post_processing_config: An optional dictionary containing
        post-processing parameters used for TFLite custom NMS op.
      pre_nms_top_k_sharding_block: For v3 (edge tpu friendly) NMS, avoids
        creating long axis for pre_nms_top_k. Will do top_k in shards of size
        [num_classes, pre_nms_top_k_sharding_block * boxes_per_location]
      nms_v3_refinements: For v3 (edge tpu friendly) NMS, sets how close result
        should be to standard NMS. When None, 2 is used. Here is some
        experimental deviations for different refinement values:
        if == 0, AP is reduced 1.0%, AR is reduced 5% on COCO
        if == 1, AP is reduced 0.2%, AR is reduced 2% on COCO
        if == 2, AP is reduced <0.1%, AR is reduced <1% on COCO
      return_decoded: A `bool` of whether to return decoded boxes before NMS
        regardless of whether `apply_nms` is True or not.
      use_class_agnostic_nms: A `bool` of whether non max suppression is
        operated on all the boxes using max scores across all classes.
      **kwargs: Additional keyword arguments passed to Layer.

    Raises:
      ValueError: If `use_class_agnostic_nms` is required by `nms_version` is
      not specified as `v2`.
    """
    if use_class_agnostic_nms and nms_version != 'v2':
      raise ValueError(
          'If not using TFLite custom NMS, `use_class_agnostic_nms` can only be'
          ' enabled for NMS v2 for now, but NMS {} is used! If you are using'
          ' TFLite NMS, please configure TFLite custom NMS for class-agnostic'
          ' NMS.'.format(nms_version)
      )
    self._config_dict = {
        'apply_nms': apply_nms,
        'pre_nms_top_k': pre_nms_top_k,
        'pre_nms_score_threshold': pre_nms_score_threshold,
        'nms_iou_threshold': nms_iou_threshold,
        'max_num_detections': max_num_detections,
        'nms_version': nms_version,
        'use_cpu_nms': use_cpu_nms,
        'soft_nms_sigma': soft_nms_sigma,
        'return_decoded': return_decoded,
        'use_class_agnostic_nms': use_class_agnostic_nms,
    }
    # Don't store if were not defined
    if pre_nms_top_k_sharding_block is not None:
      self._config_dict['pre_nms_top_k_sharding_block'] = (
          pre_nms_top_k_sharding_block
      )
    if nms_v3_refinements is not None:
      self._config_dict['nms_v3_refinements'] = nms_v3_refinements

    if tflite_post_processing_config is not None:
      self._config_dict.update(
          {'tflite_post_processing_config': tflite_post_processing_config}
      )
    super().__init__(**kwargs)

  def _decode_multilevel_outputs(
      self,
      raw_boxes: Mapping[str, tf.Tensor],
      raw_scores: Mapping[str, tf.Tensor],
      anchor_boxes: Mapping[str, tf.Tensor],
      image_shape: tf.Tensor,
      raw_attributes: Optional[Mapping[str, tf.Tensor]] = None,
  ):
    """Collects dict of multilevel boxes, scores, attributes into lists."""
    boxes = []
    scores = []
    if raw_attributes:
      attributes = {att_name: [] for att_name in raw_attributes.keys()}
    else:
      attributes = {}

    levels = list(raw_boxes.keys())
    min_level = int(min(levels))
    max_level = int(max(levels))
    for i in range(min_level, max_level + 1):
      raw_boxes_i = raw_boxes[str(i)]
      raw_scores_i = raw_scores[str(i)]
      batch_size = tf.shape(raw_boxes_i)[0]
      (_, feature_h_i, feature_w_i, num_anchors_per_locations_times_4) = (
          raw_boxes_i.get_shape().as_list()
      )
      num_locations = feature_h_i * feature_w_i
      num_anchors_per_locations = num_anchors_per_locations_times_4 // 4
      num_classes = (
          raw_scores_i.get_shape().as_list()[-1] // num_anchors_per_locations
      )

      # Applies score transformation and remove the implicit background class.
      scores_i = tf.sigmoid(
          tf.reshape(
              raw_scores_i,
              [
                  batch_size,
                  num_locations * num_anchors_per_locations,
                  num_classes,
              ],
          )
      )
      scores_i = tf.slice(scores_i, [0, 0, 1], [-1, -1, -1])

      # Box decoding.
      # The anchor boxes are shared for all data in a batch.
      # One stage detector only supports class agnostic box regression.
      anchor_boxes_i = tf.reshape(
          anchor_boxes[str(i)],
          [batch_size, num_locations * num_anchors_per_locations, 4],
      )
      raw_boxes_i = tf.reshape(
          raw_boxes_i,
          [batch_size, num_locations * num_anchors_per_locations, 4],
      )
      boxes_i = box_ops.decode_boxes(raw_boxes_i, anchor_boxes_i)

      # Box clipping.
      boxes_i = box_ops.clip_boxes(boxes_i, tf.expand_dims(image_shape, axis=1))

      boxes.append(boxes_i)
      scores.append(scores_i)

      if raw_attributes:
        for att_name, raw_att in raw_attributes.items():
          attribute_size = (
              raw_att[str(i)].get_shape().as_list()[-1]
              // num_anchors_per_locations
          )
          att_i = tf.reshape(
              raw_att[str(i)],
              [
                  batch_size,
                  num_locations * num_anchors_per_locations,
                  attribute_size,
              ],
          )
          attributes[att_name].append(att_i)

    boxes = tf.concat(boxes, axis=1)
    boxes = tf.expand_dims(boxes, axis=2)
    scores = tf.concat(scores, axis=1)

    if raw_attributes:
      for att_name in raw_attributes.keys():
        attributes[att_name] = tf.concat(attributes[att_name], axis=1)
        attributes[att_name] = tf.expand_dims(attributes[att_name], axis=2)

    return boxes, scores, attributes

  def _decode_multilevel_outputs_and_pre_nms_top_k(
      self,
      raw_boxes: Mapping[str, tf.Tensor],
      raw_scores: Mapping[str, tf.Tensor],
      anchor_boxes: Mapping[str, tf.Tensor],
      image_shape: tf.Tensor,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Collects dict of multilevel boxes, scores into lists."""
    boxes = None
    scores = None

    pre_nms_top_k = self._config_dict['pre_nms_top_k']
    # TODO(b/258007436): consider removing when compiler be able to handle
    # it on its own.
    pre_nms_top_k_sharding_block = self._config_dict.get(
        'pre_nms_top_k_sharding_block', 128
    )
    levels = list(raw_boxes.keys())
    min_level = int(min(levels))
    max_level = int(max(levels))
    clip_shape = tf.expand_dims(tf.expand_dims(image_shape, axis=1), axis=1)
    for i in range(max_level, min_level - 1, -1):
      (
          batch_size,
          unsharded_h,
          unsharded_w,
          num_anchors_per_locations_times_4,
      ) = (
          raw_boxes[str(i)].get_shape().as_list()
      )
      num_anchors_per_locations = num_anchors_per_locations_times_4 // 4
      if batch_size is None:
        batch_size = tf.shape(raw_boxes[str(i)])[0]
      block = max(1, pre_nms_top_k_sharding_block // unsharded_w)
      boxes_shape = [
          batch_size,
          unsharded_h,
          unsharded_w * num_anchors_per_locations,
          4,
      ]
      decoded_boxes = box_ops.clip_boxes(
          box_ops.decode_boxes(
              tf.reshape(raw_boxes[str(i)], boxes_shape),
              tf.reshape(anchor_boxes[str(i)], boxes_shape),
          ),
          clip_shape,
      )
      for raw_scores_i, decoded_boxes_i in edgetpu.shard_tensors(
          1, block, (raw_scores[str(i)], decoded_boxes)
      ):
        (_, feature_h_i, feature_w_i, _) = raw_scores_i.get_shape().as_list()
        num_locations = feature_h_i * feature_w_i
        num_classes = (
            raw_scores_i.get_shape().as_list()[-1] // num_anchors_per_locations
        )

        # Applies score transformation and remove the implicit background class.
        scores_i = tf.slice(
            tf.transpose(
                tf.reshape(
                    raw_scores_i,
                    [
                        batch_size,
                        num_locations * num_anchors_per_locations,
                        num_classes,
                    ],
                ),
                [0, 2, 1],
            ),
            [0, 1, 0],
            [-1, -1, -1],
        )

        # Box decoding.
        # The anchor boxes are shared for all data in a batch.
        # One stage detector only supports class agnostic box regression.
        boxes_i = tf.tile(
            tf.reshape(
                decoded_boxes_i,
                [batch_size, 1, num_locations * num_anchors_per_locations, 4],
            ),
            [1, num_classes - 1, 1, 1],
        )
        scores, boxes = edgetpu.concat_and_top_k(
            pre_nms_top_k, (scores, scores_i), (boxes, boxes_i)
        )
    boxes: tf.Tensor = boxes  # pytype: disable=annotation-type-mismatch
    return boxes, tf.sigmoid(scores)

  def __call__(
      self,
      raw_boxes: Mapping[str, tf.Tensor],
      raw_scores: Mapping[str, tf.Tensor],
      anchor_boxes: Mapping[str, tf.Tensor],
      image_shape: tf.Tensor,
      raw_attributes: Optional[Mapping[str, tf.Tensor]] = None,
  ) -> Mapping[str, Any]:
    """Generates final detections.

    Args:
      raw_boxes: A `dict` with keys representing FPN levels and values
        representing box tenors of shape `[batch, feature_h, feature_w,
        num_anchors * 4]`.
      raw_scores: A `dict` with keys representing FPN levels and values
        representing logit tensors of shape `[batch, feature_h, feature_w,
        num_anchors * num_classes]`.
      anchor_boxes: A `dict` with keys representing FPN levels and values
        representing anchor tenors of shape `[batch_size, K, 4]` representing
        the corresponding anchor boxes w.r.t `box_outputs`.
      image_shape: A `tf.Tensor` of shape of [batch_size, 2] storing the image
        height and width w.r.t. the scaled image, i.e. the same image space as
        `box_outputs` and `anchor_boxes`.
      raw_attributes: If not None, a `dict` of (attribute_name,
        attribute_prediction) pairs. `attribute_prediction` is a dict that
        contains keys representing FPN levels and values representing tenors of
        shape `[batch, feature_h, feature_w, num_anchors * attribute_size]`.

    Returns:
      If `apply_nms` = True, the return is a dictionary with keys:
        `detection_boxes`: A `float` tf.Tensor of shape
          [batch, max_num_detections, 4] representing top detected boxes in
          [y1, x1, y2, x2].
        `detection_scores`: A `float` tf.Tensor of shape
          [batch, max_num_detections] representing sorted confidence scores for
          detected boxes. The values are between [0, 1].
        `detection_classes`: An `int` tf.Tensor of shape
          [batch, max_num_detections] representing classes for detected boxes.
        `num_detections`: An `int` tf.Tensor of shape [batch] only the first
          `num_detections` boxes are valid detections
        `detection_attributes`: A dict. Values of the dict is a `float`
          tf.Tensor of shape [batch, max_num_detections, attribute_size]
          representing attribute predictions for detected boxes.
      If `apply_nms` = False, the return is a dictionary with following keys. If
      `return_decoded` = True, the following items will also be included even if
      `apply_nms` = True:
        `decoded_boxes`: A `float` tf.Tensor of shape [batch, num_raw_boxes, 4]
          representing all the decoded boxes.
        `decoded_box_scores`: A `float` tf.Tensor of shape
          [batch, num_raw_boxes] representing socres of all the decoded boxes.
        `decoded_box_attributes`: A dict. Values in the dict is a
          `float` tf.Tensor of shape [batch, num_raw_boxes, attribute_size]
          representing attribute predictions of all the decoded boxes.
    """
    if (
        self._config_dict['apply_nms']
        and self._config_dict['nms_version'] == 'tflite'
    ):
      boxes, classes, scores, num_detections = _generate_detections_tflite(
          raw_boxes,
          raw_scores,
          anchor_boxes,
          self.get_config()['tflite_post_processing_config'],
      )
      return {
          'num_detections': num_detections,
          'detection_boxes': boxes,
          'detection_classes': classes,
          'detection_scores': scores,
      }

    if self._config_dict['nms_version'] != 'v3':
      boxes, scores, attributes = self._decode_multilevel_outputs(
          raw_boxes, raw_scores, anchor_boxes, image_shape, raw_attributes
      )
    else:
      attributes = None
      boxes, scores = self._decode_multilevel_outputs_and_pre_nms_top_k(
          raw_boxes, raw_scores, anchor_boxes, image_shape
      )

    decoded_results = {
        'decoded_boxes': boxes,
        'decoded_box_scores': scores,
        'decoded_box_attributes': attributes,
    }

    if not self._config_dict['apply_nms']:
      return decoded_results

    # Optionally force the NMS to run on CPU.
    if self._config_dict['use_cpu_nms']:
      nms_context = tf.device('cpu:0')
    else:
      nms_context = contextlib.nullcontext()

    with nms_context:
      if raw_attributes and (self._config_dict['nms_version'] != 'v1'):
        raise ValueError(
            'Attribute learning is only supported for NMSv1 but NMS {} is used.'
            .format(self._config_dict['nms_version'])
        )
      if self._config_dict['nms_version'] == 'batched':
        (nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections) = (
            _generate_detections_batched(
                boxes,
                scores,
                self._config_dict['pre_nms_score_threshold'],
                self._config_dict['nms_iou_threshold'],
                self._config_dict['max_num_detections'],
            )
        )
        # Set `nmsed_attributes` to None for batched NMS.
        nmsed_attributes = {}
      elif self._config_dict['nms_version'] == 'v1':
        (
            nmsed_boxes,
            nmsed_scores,
            nmsed_classes,
            valid_detections,
            nmsed_attributes,
        ) = _generate_detections_v1(
            boxes,
            scores,
            attributes=attributes if raw_attributes else None,
            pre_nms_top_k=self._config_dict['pre_nms_top_k'],
            pre_nms_score_threshold=self._config_dict[
                'pre_nms_score_threshold'
            ],
            nms_iou_threshold=self._config_dict['nms_iou_threshold'],
            max_num_detections=self._config_dict['max_num_detections'],
            soft_nms_sigma=self._config_dict['soft_nms_sigma'],
        )
      elif self._config_dict['nms_version'] == 'v2':
        (nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections) = (
            _generate_detections_v2(
                boxes,
                scores,
                pre_nms_top_k=self._config_dict['pre_nms_top_k'],
                pre_nms_score_threshold=self._config_dict[
                    'pre_nms_score_threshold'
                ],
                nms_iou_threshold=self._config_dict['nms_iou_threshold'],
                max_num_detections=self._config_dict['max_num_detections'],
                use_class_agnostic_nms=self._config_dict[
                    'use_class_agnostic_nms'
                ],
            )
        )
        # Set `nmsed_attributes` to None for v2.
        nmsed_attributes = {}
      elif self._config_dict['nms_version'] == 'v3':
        (nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections) = (
            _generate_detections_v3(
                boxes,
                scores,
                pre_nms_score_threshold=self._config_dict[
                    'pre_nms_score_threshold'
                ],
                nms_iou_threshold=self._config_dict['nms_iou_threshold'],
                max_num_detections=self._config_dict['max_num_detections'],
                refinements=self._config_dict.get('nms_v3_refinements', 2),
            )
        )
        # Set `nmsed_attributes` to None for v3.
        nmsed_attributes = {}
      else:
        raise ValueError(
            'NMS version {} not supported.'.format(
                self._config_dict['nms_version']
            )
        )

    # Adds 1 to offset the background class which has index 0.
    nmsed_classes += 1

    return {
        **(decoded_results if self._config_dict['return_decoded'] else {}),
        'num_detections': valid_detections,
        'detection_boxes': nmsed_boxes,
        'detection_classes': nmsed_classes,
        'detection_scores': nmsed_scores,
        'detection_attributes': nmsed_attributes,
    }

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)
