# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
from typing import List, Optional, Mapping
# Import libraries
import tensorflow as tf

from official.vision.beta.ops import box_ops
from official.vision.beta.ops import nms


def _generate_detections_v1(boxes: tf.Tensor,
                            scores: tf.Tensor,
                            attributes: Optional[Mapping[str,
                                                         tf.Tensor]] = None,
                            pre_nms_top_k: int = 5000,
                            pre_nms_score_threshold: float = 0.05,
                            nms_iou_threshold: float = 0.5,
                            max_num_detections: int = 100):
  """Generates the final detections given the model outputs.

  The implementation unrolls the batch dimension and process images one by one.
  It required the batch dimension to be statically known and it is TPU
  compatible.

  Args:
    boxes: A `tf.Tensor` with shape `[batch_size, N, num_classes, 4]` or
      `[batch_size, N, 1, 4]` for box predictions on all feature levels. The
      N is the number of total anchors on all levels.
    scores: A `tf.Tensor` with shape `[batch_size, N, num_classes]`, which
      stacks class probability on all feature levels. The N is the number of
      total anchors on all levels. The num_classes is the number of classes
      predicted by the model. Note that the class_outputs here is the raw score.
    attributes: None or a dict of (attribute_name, attributes) pairs. Each
      attributes is a `tf.Tensor` with shape
      `[batch_size, N, num_classes, attribute_size]` or
      `[batch_size, N, 1, attribute_size]` for attribute predictions on all
      feature levels. The N is the number of total anchors on all levels. Can
      be None if no attribute learning is required.
    pre_nms_top_k: An `int` number of top candidate detections per class before
      NMS.
    pre_nms_score_threshold: A `float` representing the threshold for deciding
      when to remove boxes based on score.
    nms_iou_threshold: A `float` representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    max_num_detections: A scalar representing maximum number of boxes retained
      over all classes.

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
      (nmsed_boxes_i, nmsed_scores_i, nmsed_classes_i, valid_detections_i,
       nmsed_att_i) = _generate_detections_per_image(
           boxes[i],
           scores[i],
           attributes={
               att_name: att[i] for att_name, att in attributes.items()
           } if attributes else {},
           pre_nms_top_k=pre_nms_top_k,
           pre_nms_score_threshold=pre_nms_score_threshold,
           nms_iou_threshold=nms_iou_threshold,
           max_num_detections=max_num_detections)
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

  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections, nmsed_attributes


def _generate_detections_per_image(
    boxes: tf.Tensor,
    scores: tf.Tensor,
    attributes: Optional[Mapping[str, tf.Tensor]] = None,
    pre_nms_top_k: int = 5000,
    pre_nms_score_threshold: float = 0.05,
    nms_iou_threshold: float = 0.5,
    max_num_detections: int = 100):
  """Generates the final detections per image given the model outputs.

  Args:
    boxes: A  `tf.Tensor` with shape `[N, num_classes, 4]` or `[N, 1, 4]`, which
      box predictions on all feature levels. The N is the number of total
      anchors on all levels.
    scores: A `tf.Tensor` with shape `[N, num_classes]`, which stacks class
      probability on all feature levels. The N is the number of total anchors on
      all levels. The num_classes is the number of classes predicted by the
      model. Note that the class_outputs here is the raw score.
    attributes: If not None, a dict of `tf.Tensor`. Each value is in shape
      `[N, num_classes, attribute_size]` or `[N, 1, attribute_size]` of
      attribute predictions on all feature levels. The N is the number of total
      anchors on all levels.
    pre_nms_top_k: An `int` number of top candidate detections per class before
      NMS.
    pre_nms_score_threshold: A `float` representing the threshold for deciding
      when to remove boxes based on score.
    nms_iou_threshold: A `float` representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    max_num_detections: A `scalar` representing maximum number of boxes retained
      over all classes.

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
        scores_i, k=tf.minimum(tf.shape(scores_i)[-1], pre_nms_top_k))
    boxes_i = tf.gather(boxes_i, indices)

    (nmsed_indices_i,
     nmsed_num_valid_i) = tf.image.non_max_suppression_padded(
         tf.cast(boxes_i, tf.float32),
         tf.cast(scores_i, tf.float32),
         max_num_detections,
         iou_threshold=nms_iou_threshold,
         score_threshold=pre_nms_score_threshold,
         pad_to_max_output_size=True,
         name='nms_detections_' + str(i))
    nmsed_boxes_i = tf.gather(boxes_i, nmsed_indices_i)
    nmsed_scores_i = tf.gather(scores_i, nmsed_indices_i)
    # Sets scores of invalid boxes to -1.
    nmsed_scores_i = tf.where(
        tf.less(tf.range(max_num_detections), [nmsed_num_valid_i]),
        nmsed_scores_i, -tf.ones_like(nmsed_scores_i))
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
        nmsed_attributes[att_name].append(nmsed_att_i)

  # Concats results from all classes and sort them.
  nmsed_boxes = tf.concat(nmsed_boxes, axis=0)
  nmsed_scores = tf.concat(nmsed_scores, axis=0)
  nmsed_classes = tf.concat(nmsed_classes, axis=0)
  nmsed_scores, indices = tf.nn.top_k(
      nmsed_scores, k=max_num_detections, sorted=True)
  nmsed_boxes = tf.gather(nmsed_boxes, indices)
  nmsed_classes = tf.gather(nmsed_classes, indices)
  valid_detections = tf.reduce_sum(
      tf.cast(tf.greater(nmsed_scores, -1), tf.int32))
  if attributes:
    for att_name in attributes.keys():
      nmsed_attributes[att_name] = tf.concat(nmsed_attributes[att_name], axis=0)
      nmsed_attributes[att_name] = tf.gather(nmsed_attributes[att_name],
                                             indices)

  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections, nmsed_attributes


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
      scores_trans, k=pre_nms_num_detections, sorted=True)

  top_k_scores = tf.reshape(top_k_scores,
                            [batch_size, num_class, pre_nms_num_detections])
  top_k_indices = tf.reshape(top_k_indices,
                             [batch_size, num_class, pre_nms_num_detections])

  return tf.transpose(top_k_scores,
                      [0, 2, 1]), tf.transpose(top_k_indices, [0, 2, 1])


def _generate_detections_v2(boxes: tf.Tensor,
                            scores: tf.Tensor,
                            pre_nms_top_k: int = 5000,
                            pre_nms_score_threshold: float = 0.05,
                            nms_iou_threshold: float = 0.5,
                            max_num_detections: int = 100):
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
        scores, min(total_anchors, pre_nms_top_k))
    for i in range(num_classes):
      boxes_i = boxes[:, :, min(num_classes_for_box - 1, i), :]
      scores_i = scores[:, :, i]
      # Obtains pre_nms_top_k before running NMS.
      boxes_i = tf.gather(boxes_i, indices[:, :, i], batch_dims=1, axis=1)

      # Filter out scores.
      boxes_i, scores_i = box_ops.filter_boxes_by_scores(
          boxes_i, scores_i, min_score_threshold=pre_nms_score_threshold)

      (nmsed_scores_i, nmsed_boxes_i) = nms.sorted_non_max_suppression_padded(
          tf.cast(scores_i, tf.float32),
          tf.cast(boxes_i, tf.float32),
          max_num_detections,
          iou_threshold=nms_iou_threshold)
      nmsed_classes_i = tf.fill([batch_size, max_num_detections], i)
      nmsed_boxes.append(nmsed_boxes_i)
      nmsed_scores.append(nmsed_scores_i)
      nmsed_classes.append(nmsed_classes_i)
  nmsed_boxes = tf.concat(nmsed_boxes, axis=1)
  nmsed_scores = tf.concat(nmsed_scores, axis=1)
  nmsed_classes = tf.concat(nmsed_classes, axis=1)
  nmsed_scores, indices = tf.nn.top_k(
      nmsed_scores, k=max_num_detections, sorted=True)
  nmsed_boxes = tf.gather(nmsed_boxes, indices, batch_dims=1, axis=1)
  nmsed_classes = tf.gather(nmsed_classes, indices, batch_dims=1)
  valid_detections = tf.reduce_sum(
      input_tensor=tf.cast(tf.greater(nmsed_scores, -1), tf.int32), axis=1)
  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


def _generate_detections_batched(boxes: tf.Tensor, scores: tf.Tensor,
                                 pre_nms_score_threshold: float,
                                 nms_iou_threshold: float,
                                 max_num_detections: int):
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
            clip_boxes=False))
    nmsed_classes = tf.cast(nmsed_classes, tf.int32)
  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


@tf.keras.utils.register_keras_serializable(package='Vision')
class DetectionGenerator(tf.keras.layers.Layer):
  """Generates the final detected boxes with scores and classes."""

  def __init__(self,
               apply_nms: bool = True,
               pre_nms_top_k: int = 5000,
               pre_nms_score_threshold: float = 0.05,
               nms_iou_threshold: float = 0.5,
               max_num_detections: int = 100,
               nms_version: str = 'v2',
               use_cpu_nms: bool = False,
               **kwargs):
    """Initializes a detection generator.

    Args:
      apply_nms: A `bool` of whether or not apply non maximum suppression.
        If False, the decoded boxes and their scores are returned.
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
    }
    super(DetectionGenerator, self).__init__(**kwargs)

  def __call__(self,
               raw_boxes: tf.Tensor,
               raw_scores: tf.Tensor,
               anchor_boxes: tf.Tensor,
               image_shape: tf.Tensor,
               regression_weights: Optional[List[float]] = None,
               bbox_per_class: bool = True):
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
      raw_boxes = tf.reshape(raw_boxes,
                             [batch_size, num_locations, num_classes, 4])
      raw_boxes = tf.slice(raw_boxes, [0, 0, 1, 0], [-1, -1, -1, -1])
      anchor_boxes = tf.tile(
          tf.expand_dims(anchor_boxes, axis=2), [1, 1, num_classes - 1, 1])
      raw_boxes = tf.reshape(raw_boxes, [batch_size, num_detections, 4])
      anchor_boxes = tf.reshape(anchor_boxes, [batch_size, num_detections, 4])

    # Box decoding.
    decoded_boxes = box_ops.decode_boxes(
        raw_boxes, anchor_boxes, weights=regression_weights)

    # Box clipping
    decoded_boxes = box_ops.clip_boxes(
        decoded_boxes, tf.expand_dims(image_shape, axis=1))

    if bbox_per_class:
      decoded_boxes = tf.reshape(
          decoded_boxes, [batch_size, num_locations, num_classes - 1, 4])
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
                decoded_boxes, box_scores,
                self._config_dict['pre_nms_score_threshold'],
                self._config_dict['nms_iou_threshold'],
                self._config_dict['max_num_detections']))
      elif self._config_dict['nms_version'] == 'v1':
        (nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections, _) = (
            _generate_detections_v1(
                decoded_boxes,
                box_scores,
                pre_nms_top_k=self._config_dict['pre_nms_top_k'],
                pre_nms_score_threshold=self
                ._config_dict['pre_nms_score_threshold'],
                nms_iou_threshold=self._config_dict['nms_iou_threshold'],
                max_num_detections=self._config_dict['max_num_detections']))
      elif self._config_dict['nms_version'] == 'v2':
        (nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections) = (
            _generate_detections_v2(
                decoded_boxes,
                box_scores,
                pre_nms_top_k=self._config_dict['pre_nms_top_k'],
                pre_nms_score_threshold=self
                ._config_dict['pre_nms_score_threshold'],
                nms_iou_threshold=self._config_dict['nms_iou_threshold'],
                max_num_detections=self._config_dict['max_num_detections']))
      else:
        raise ValueError('NMS version {} not supported.'.format(
            self._config_dict['nms_version']))

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

  def __init__(self,
               apply_nms: bool = True,
               pre_nms_top_k: int = 5000,
               pre_nms_score_threshold: float = 0.05,
               nms_iou_threshold: float = 0.5,
               max_num_detections: int = 100,
               nms_version: str = 'v1',
               use_cpu_nms: bool = False,
               **kwargs):
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
    }
    super(MultilevelDetectionGenerator, self).__init__(**kwargs)

  def _decode_multilevel_outputs(
      self,
      raw_boxes: Mapping[str, tf.Tensor],
      raw_scores: Mapping[str, tf.Tensor],
      anchor_boxes: tf.Tensor,
      image_shape: tf.Tensor,
      raw_attributes: Optional[Mapping[str, tf.Tensor]] = None):
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
      (_, feature_h_i, feature_w_i,
       num_anchors_per_locations_times_4) = raw_boxes_i.get_shape().as_list()
      num_locations = feature_h_i * feature_w_i
      num_anchors_per_locations = num_anchors_per_locations_times_4 // 4
      num_classes = raw_scores_i.get_shape().as_list(
      )[-1] // num_anchors_per_locations

      # Applies score transformation and remove the implicit background class.
      scores_i = tf.sigmoid(
          tf.reshape(raw_scores_i, [
              batch_size, num_locations * num_anchors_per_locations, num_classes
          ]))
      scores_i = tf.slice(scores_i, [0, 0, 1], [-1, -1, -1])

      # Box decoding.
      # The anchor boxes are shared for all data in a batch.
      # One stage detector only supports class agnostic box regression.
      anchor_boxes_i = tf.reshape(
          anchor_boxes[str(i)],
          [batch_size, num_locations * num_anchors_per_locations, 4])
      raw_boxes_i = tf.reshape(
          raw_boxes_i,
          [batch_size, num_locations * num_anchors_per_locations, 4])
      boxes_i = box_ops.decode_boxes(raw_boxes_i, anchor_boxes_i)

      # Box clipping.
      boxes_i = box_ops.clip_boxes(
          boxes_i, tf.expand_dims(image_shape, axis=1))

      boxes.append(boxes_i)
      scores.append(scores_i)

      if raw_attributes:
        for att_name, raw_att in raw_attributes.items():
          attribute_size = raw_att[str(
              i)].get_shape().as_list()[-1] // num_anchors_per_locations
          att_i = tf.reshape(raw_att[str(i)], [
              batch_size, num_locations * num_anchors_per_locations,
              attribute_size
          ])
          attributes[att_name].append(att_i)

    boxes = tf.concat(boxes, axis=1)
    boxes = tf.expand_dims(boxes, axis=2)
    scores = tf.concat(scores, axis=1)

    if raw_attributes:
      for att_name in raw_attributes.keys():
        attributes[att_name] = tf.concat(attributes[att_name], axis=1)
        attributes[att_name] = tf.expand_dims(attributes[att_name], axis=2)

    return boxes, scores, attributes

  def __call__(self,
               raw_boxes: Mapping[str, tf.Tensor],
               raw_scores: Mapping[str, tf.Tensor],
               anchor_boxes: tf.Tensor,
               image_shape: tf.Tensor,
               raw_attributes: Optional[Mapping[str, tf.Tensor]] = None):
    """Generates final detections.

    Args:
      raw_boxes: A `dict` with keys representing FPN levels and values
        representing box tenors of shape `[batch, feature_h, feature_w,
        num_anchors * 4]`.
      raw_scores: A `dict` with keys representing FPN levels and values
        representing logit tensors of shape `[batch, feature_h, feature_w,
        num_anchors]`.
      anchor_boxes: A `tf.Tensor` of shape of [batch_size, K, 4] representing
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
      If `apply_nms` = False, the return is a dictionary with keys:
        `decoded_boxes`: A `float` tf.Tensor of shape [batch, num_raw_boxes, 4]
          representing all the decoded boxes.
        `decoded_box_scores`: A `float` tf.Tensor of shape
          [batch, num_raw_boxes] representing socres of all the decoded boxes.
        `decoded_box_attributes`: A dict. Values in the dict is a
          `float` tf.Tensor of shape [batch, num_raw_boxes, attribute_size]
          representing attribute predictions of all the decoded boxes.
    """
    boxes, scores, attributes = self._decode_multilevel_outputs(
        raw_boxes, raw_scores, anchor_boxes, image_shape, raw_attributes)

    if not self._config_dict['apply_nms']:
      return {
          'decoded_boxes': boxes,
          'decoded_box_scores': scores,
          'decoded_box_attributes': attributes,
      }

    # Optionally force the NMS to run on CPU.
    if self._config_dict['use_cpu_nms']:
      nms_context = tf.device('cpu:0')
    else:
      nms_context = contextlib.nullcontext()

    with nms_context:
      if raw_attributes and (self._config_dict['nms_version'] != 'v1'):
        raise ValueError(
            'Attribute learning is only supported for NMSv1 but NMS {} is used.'
            .format(self._config_dict['nms_version']))
      if self._config_dict['nms_version'] == 'batched':
        (nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections) = (
            _generate_detections_batched(
                boxes, scores, self._config_dict['pre_nms_score_threshold'],
                self._config_dict['nms_iou_threshold'],
                self._config_dict['max_num_detections']))
        # Set `nmsed_attributes` to None for batched NMS.
        nmsed_attributes = {}
      elif self._config_dict['nms_version'] == 'v1':
        (nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections,
         nmsed_attributes) = (
             _generate_detections_v1(
                 boxes,
                 scores,
                 attributes=attributes if raw_attributes else None,
                 pre_nms_top_k=self._config_dict['pre_nms_top_k'],
                 pre_nms_score_threshold=self
                 ._config_dict['pre_nms_score_threshold'],
                 nms_iou_threshold=self._config_dict['nms_iou_threshold'],
                 max_num_detections=self._config_dict['max_num_detections']))
      elif self._config_dict['nms_version'] == 'v2':
        (nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections) = (
            _generate_detections_v2(
                boxes,
                scores,
                pre_nms_top_k=self._config_dict['pre_nms_top_k'],
                pre_nms_score_threshold=self
                ._config_dict['pre_nms_score_threshold'],
                nms_iou_threshold=self._config_dict['nms_iou_threshold'],
                max_num_detections=self._config_dict['max_num_detections']))
        # Set `nmsed_attributes` to None for v2.
        nmsed_attributes = {}
      else:
        raise ValueError('NMS version {} not supported.'.format(
            self._config_dict['nms_version']))

    # Adds 1 to offset the background class which has index 0.
    nmsed_classes += 1

    return {
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
