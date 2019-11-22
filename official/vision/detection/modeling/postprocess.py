# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Post-processing model outputs to generate detection."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import functools

import tensorflow.compat.v2 as tf

from official.vision.detection.ops import nms
from official.vision.detection.utils import box_utils


def generate_detections_factory(params):
  """Factory to select function to generate detection."""
  if params.use_batched_nms:
    func = functools.partial(
        _generate_detections_batched,
        max_total_size=params.max_total_size,
        nms_iou_threshold=params.nms_iou_threshold,
        score_threshold=params.score_threshold)
  else:
    func = functools.partial(
        _generate_detections,
        max_total_size=params.max_total_size,
        nms_iou_threshold=params.nms_iou_threshold,
        score_threshold=params.score_threshold)
  return func


def _generate_detections(boxes,
                         scores,
                         max_total_size=100,
                         nms_iou_threshold=0.3,
                         score_threshold=0.05,
                         pre_nms_num_boxes=5000):
  """Generate the final detections given the model outputs.

  This uses classes unrolling with while loop based NMS, could be parralled at batch dimension.

  Args:
    boxes: a tensor with shape [batch_size, N, num_classes, 4] or [batch_size,
      N, 1, 4], which box predictions on all feature levels. The N is the number
      of total anchors on all levels.
    scores: a tensor with shape [batch_size, N, num_classes], which stacks class
      probability on all feature levels. The N is the number of total anchors on
      all levels. The num_classes is the number of classes predicted by the
      model. Note that the class_outputs here is the raw score.
    max_total_size: a scalar representing maximum number of boxes retained over
      all classes.
    nms_iou_threshold: a float representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    score_threshold: a float representing the threshold for deciding when to
      remove boxes based on score.
    pre_nms_num_boxes: an int number of top candidate detections per class
      before NMS.

  Returns:
    nms_boxes: `float` Tensor of shape [batch_size, max_total_size, 4]
      representing top detected boxes in [y1, x1, y2, x2].
    nms_scores: `float` Tensor of shape [batch_size, max_total_size]
      representing sorted confidence scores for detected boxes. The values are
      between [0, 1].
    nms_classes: `int` Tensor of shape [batch_size, max_total_size] representing
      classes for detected boxes.
    valid_detections: `int` Tensor of shape [batch_size] only the top
      `valid_detections` boxes are valid detections.
  """
  with tf.name_scope('generate_detections'):
    nmsed_boxes = []
    nmsed_classes = []
    nmsed_scores = []
    valid_detections = []
    batch_size, _, num_classes_for_box, _ = boxes.get_shape().as_list()
    num_classes = scores.get_shape().as_list()[2]
    for i in range(num_classes):
      boxes_i = boxes[:, :, min(num_classes_for_box - 1, i), :]
      scores_i = scores[:, :, i]
      # Obtains pre_nms_num_boxes before running NMS.
      scores_i, indices = tf.nn.top_k(
          scores_i,
          k=tf.minimum(tf.shape(input=scores_i)[-1], pre_nms_num_boxes))
      boxes_i = tf.gather(boxes_i, indices, batch_dims=1, axis=1)

      # Filter out scores.
      boxes_i, scores_i = box_utils.filter_boxes_by_scores(
          boxes_i, scores_i, min_score_threshold=score_threshold)

      (nmsed_scores_i, nmsed_boxes_i) = nms.sorted_non_max_suppression_padded(
          tf.cast(scores_i, tf.float32),
          tf.cast(boxes_i, tf.float32),
          max_total_size,
          iou_threshold=nms_iou_threshold)
      nmsed_classes_i = tf.fill([batch_size, max_total_size], i)
      nmsed_boxes.append(nmsed_boxes_i)
      nmsed_scores.append(nmsed_scores_i)
      nmsed_classes.append(nmsed_classes_i)
  nmsed_boxes = tf.concat(nmsed_boxes, axis=1)
  nmsed_scores = tf.concat(nmsed_scores, axis=1)
  nmsed_classes = tf.concat(nmsed_classes, axis=1)
  nmsed_scores, indices = tf.nn.top_k(
      nmsed_scores, k=max_total_size, sorted=True)
  nmsed_boxes = tf.gather(nmsed_boxes, indices, batch_dims=1, axis=1)
  nmsed_classes = tf.gather(nmsed_classes, indices, batch_dims=1)
  valid_detections = tf.reduce_sum(
      input_tensor=tf.cast(tf.greater(nmsed_scores, -1), tf.int32), axis=1)
  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


def _generate_detections_per_image(boxes,
                                   scores,
                                   max_total_size=100,
                                   nms_iou_threshold=0.3,
                                   score_threshold=0.05,
                                   pre_nms_num_boxes=5000):
  """Generate the final detections per image given the model outputs.

  Args:
    boxes: a tensor with shape [N, num_classes, 4] or [N, 1, 4], which box
      predictions on all feature levels. The N is the number of total anchors on
      all levels.
    scores: a tensor with shape [N, num_classes], which stacks class probability
      on all feature levels. The N is the number of total anchors on all levels.
      The num_classes is the number of classes predicted by the model. Note that
      the class_outputs here is the raw score.
    max_total_size: a scalar representing maximum number of boxes retained over
      all classes.
    nms_iou_threshold: a float representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    score_threshold: a float representing the threshold for deciding when to
      remove boxes based on score.
    pre_nms_num_boxes: an int number of top candidate detections per class
      before NMS.

  Returns:
    nms_boxes: `float` Tensor of shape [max_total_size, 4] representing top
      detected boxes in [y1, x1, y2, x2].
    nms_scores: `float` Tensor of shape [max_total_size] representing sorted
      confidence scores for detected boxes. The values are between [0, 1].
    nms_classes: `int` Tensor of shape [max_total_size] representing classes for
      detected boxes.
    valid_detections: `int` Tensor of shape [1] only the top `valid_detections`
      boxes are valid detections.
  """
  nmsed_boxes = []
  nmsed_scores = []
  nmsed_classes = []
  num_classes_for_box = boxes.get_shape().as_list()[1]
  num_classes = scores.get_shape().as_list()[1]
  for i in range(num_classes):
    boxes_i = boxes[:, min(num_classes_for_box-1, i)]
    scores_i = scores[:, i]

    # Obtains pre_nms_num_boxes before running NMS.
    scores_i, indices = tf.nn.top_k(
        scores_i, k=tf.minimum(tf.shape(input=scores_i)[-1], pre_nms_num_boxes))
    boxes_i = tf.gather(boxes_i, indices)

    (nmsed_indices_i,
     nmsed_num_valid_i) = tf.image.non_max_suppression_padded(
         tf.cast(boxes_i, tf.float32),
         tf.cast(scores_i, tf.float32),
         max_total_size,
         iou_threshold=nms_iou_threshold,
         score_threshold=score_threshold,
         pad_to_max_output_size=True,
         name='nms_detections_' + str(i))
    nmsed_boxes_i = tf.gather(boxes_i, nmsed_indices_i)
    nmsed_scores_i = tf.gather(scores_i, nmsed_indices_i)
    # Sets scores of invalid boxes to -1.
    nmsed_scores_i = tf.where(
        tf.less(tf.range(max_total_size), [nmsed_num_valid_i]), nmsed_scores_i,
        -tf.ones_like(nmsed_scores_i))
    nmsed_classes_i = tf.fill([max_total_size], i)
    nmsed_boxes.append(nmsed_boxes_i)
    nmsed_scores.append(nmsed_scores_i)
    nmsed_classes.append(nmsed_classes_i)
  # Concats results from all classes and sort them.
  nmsed_boxes = tf.concat(nmsed_boxes, axis=0)
  nmsed_scores = tf.concat(nmsed_scores, axis=0)
  nmsed_classes = tf.concat(nmsed_classes, axis=0)
  nmsed_scores, indices = tf.nn.top_k(
      nmsed_scores,
      k=max_total_size,
      sorted=True)
  nmsed_boxes = tf.gather(nmsed_boxes, indices)
  nmsed_classes = tf.gather(nmsed_classes, indices)
  valid_detections = tf.reduce_sum(
      input_tensor=tf.cast(tf.greater(nmsed_scores, -1), tf.int32))
  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


def _generate_detections_batched(boxes,
                                 scores,
                                 max_total_size,
                                 nms_iou_threshold,
                                 score_threshold):
  """Generates detected boxes with scores and classes for one-stage detector.

  The function takes output of multi-level ConvNets and anchor boxes and
  generates detected boxes. Note that this used batched nms, which is not
  supported on TPU currently.

  Args:
    boxes: a tensor with shape [batch_size, N, num_classes, 4] or
      [batch_size, N, 1, 4], which box predictions on all feature levels. The N
      is the number of total anchors on all levels.
    scores: a tensor with shape [batch_size, N, num_classes], which
      stacks class probability on all feature levels. The N is the number of
      total anchors on all levels. The num_classes is the number of classes
      predicted by the model. Note that the class_outputs here is the raw score.
    max_total_size: a scalar representing maximum number of boxes retained over
      all classes.
    nms_iou_threshold: a float representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    score_threshold: a float representing the threshold for deciding when to
      remove boxes based on score.
  Returns:
    nms_boxes: `float` Tensor of shape [batch_size, max_total_size, 4]
      representing top detected boxes in [y1, x1, y2, x2].
    nms_scores: `float` Tensor of shape [batch_size, max_total_size]
      representing sorted confidence scores for detected boxes. The values are
      between [0, 1].
    nms_classes: `int` Tensor of shape [batch_size, max_total_size] representing
      classes for detected boxes.
    valid_detections: `int` Tensor of shape [batch_size] only the top
      `valid_detections` boxes are valid detections.
  """
  with tf.name_scope('generate_detections'):
    # TODO(tsungyi): Removes normalization/denomalization once the
    # tf.image.combined_non_max_suppression is coordinate system agnostic.
    # Normalizes maximum box cooridinates to 1.
    normalizer = tf.reduce_max(input_tensor=boxes)
    boxes /= normalizer
    (nmsed_boxes, nmsed_scores, nmsed_classes,
     valid_detections) = tf.image.combined_non_max_suppression(
         boxes,
         scores,
         max_output_size_per_class=max_total_size,
         max_total_size=max_total_size,
         iou_threshold=nms_iou_threshold,
         score_threshold=score_threshold,
         pad_per_class=False,)
    # De-normalizes box cooridinates.
    nmsed_boxes *= normalizer
  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


def _apply_score_activation(logits, num_classes, activation):
  """Applies activation to logits and removes the background class.

  Note that it is assumed that the background class has index 0, which is
  sliced away after the score transformation.

  Args:
    logits: the raw logit tensor.
    num_classes: the total number of classes including one background class.
    activation: the score activation type, one of 'SIGMOID', 'SOFTMAX' and
      'IDENTITY'.

  Returns:
    scores: the tensor after applying score transformation and background
      class removal.
  """
  batch_size = tf.shape(input=logits)[0]
  logits = tf.reshape(logits, [batch_size, -1, num_classes])
  if activation == 'SIGMOID':
    scores = tf.sigmoid(logits)
  elif activation == 'SOFTMAX':
    scores = tf.softmax(logits)
  elif activation == 'IDENTITY':
    pass
  else:
    raise ValueError(
        'The score activation should be SIGMOID, SOFTMAX or IDENTITY')
  scores = scores[..., 1:]
  return scores


class GenerateOneStageDetections(tf.keras.layers.Layer):
  """Generates detected boxes with scores and classes for one-stage detector."""

  def __init__(self, params, **kwargs):
    super(GenerateOneStageDetections, self).__init__(**kwargs)

    self._generate_detections = generate_detections_factory(params)
    self._min_level = params.min_level
    self._max_level = params.max_level
    self._num_classes = params.num_classes
    self._score_activation = 'SIGMOID'

  def call(self, inputs):
    box_outputs, class_outputs, anchor_boxes, image_shape = inputs
    # Collects outputs from all levels into a list.
    boxes = []
    scores = []
    for i in range(self._min_level, self._max_level + 1):
      batch_size = tf.shape(input=class_outputs[i])[0]

      # Applies score transformation and remove the implicit background class.
      scores_i = _apply_score_activation(
          class_outputs[i], self._num_classes, self._score_activation)

      # Box decoding.
      # The anchor boxes are shared for all data in a batch.
      # One stage detector only supports class agnostic box regression.
      anchor_boxes_i = tf.reshape(anchor_boxes[i], [batch_size, -1, 4])
      box_outputs_i = tf.reshape(box_outputs[i], [batch_size, -1, 4])
      boxes_i = box_utils.decode_boxes(box_outputs_i, anchor_boxes_i)

      # Box clipping.
      boxes_i = box_utils.clip_boxes(boxes_i, image_shape)

      boxes.append(boxes_i)
      scores.append(scores_i)
    boxes = tf.concat(boxes, axis=1)
    scores = tf.concat(scores, axis=1)
    boxes = tf.expand_dims(boxes, axis=2)

    (nmsed_boxes, nmsed_scores, nmsed_classes,
     valid_detections) = self._generate_detections(
         tf.cast(boxes, tf.float32), tf.cast(scores, tf.float32))
    # Adds 1 to offset the background class which has index 0.
    nmsed_classes += 1
    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections
