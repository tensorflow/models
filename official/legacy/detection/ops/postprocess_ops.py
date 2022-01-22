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

"""Post-processing model outputs to generate detection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

from official.legacy.detection.ops import nms
from official.legacy.detection.utils import box_utils


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
        score_threshold=params.score_threshold,
        pre_nms_num_boxes=params.pre_nms_num_boxes)
  return func


def _select_top_k_scores(scores_in, pre_nms_num_detections):
  """Select top_k scores and indices for each class.

  Args:
    scores_in: a Tensor with shape [batch_size, N, num_classes], which stacks
      class logit outputs on all feature levels. The N is the number of total
      anchors on all levels. The num_classes is the number of classes predicted
      by the model.
    pre_nms_num_detections: Number of candidates before NMS.

  Returns:
    scores and indices: Tensors with shape [batch_size, pre_nms_num_detections,
      num_classes].
  """
  batch_size, num_anchors, num_class = scores_in.get_shape().as_list()
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


def _generate_detections(boxes,
                         scores,
                         max_total_size=100,
                         nms_iou_threshold=0.3,
                         score_threshold=0.05,
                         pre_nms_num_boxes=5000):
  """Generate the final detections given the model outputs.

  This uses classes unrolling with while loop based NMS, could be parralled
  at batch dimension.

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
    _, total_anchors, num_classes = scores.get_shape().as_list()
    # Selects top pre_nms_num scores and indices before NMS.
    scores, indices = _select_top_k_scores(
        scores, min(total_anchors, pre_nms_num_boxes))
    for i in range(num_classes):
      boxes_i = boxes[:, :, min(num_classes_for_box - 1, i), :]
      scores_i = scores[:, :, i]
      # Obtains pre_nms_num_boxes before running NMS.
      boxes_i = tf.gather(boxes_i, indices[:, :, i], batch_dims=1, axis=1)

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
    boxes_i = boxes[:, min(num_classes_for_box - 1, i)]
    scores_i = scores[:, i]

    # Obtains pre_nms_num_boxes before running NMS.
    scores_i, indices = tf.nn.top_k(
        scores_i, k=tf.minimum(tf.shape(input=scores_i)[-1], pre_nms_num_boxes))
    boxes_i = tf.gather(boxes_i, indices)

    (nmsed_indices_i, nmsed_num_valid_i) = tf.image.non_max_suppression_padded(
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
      nmsed_scores, k=max_total_size, sorted=True)
  nmsed_boxes = tf.gather(nmsed_boxes, indices)
  nmsed_classes = tf.gather(nmsed_classes, indices)
  valid_detections = tf.reduce_sum(
      input_tensor=tf.cast(tf.greater(nmsed_scores, -1), tf.int32))
  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


def _generate_detections_batched(boxes, scores, max_total_size,
                                 nms_iou_threshold, score_threshold):
  """Generates detected boxes with scores and classes for one-stage detector.

  The function takes output of multi-level ConvNets and anchor boxes and
  generates detected boxes. Note that this used batched nms, which is not
  supported on TPU currently.

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
    normalizer = tf.reduce_max(boxes)
    boxes /= normalizer
    (nmsed_boxes, nmsed_scores, nmsed_classes,
     valid_detections) = tf.image.combined_non_max_suppression(
         boxes,
         scores,
         max_output_size_per_class=max_total_size,
         max_total_size=max_total_size,
         iou_threshold=nms_iou_threshold,
         score_threshold=score_threshold,
         pad_per_class=False,
     )
    # De-normalizes box cooridinates.
    nmsed_boxes *= normalizer
  nmsed_classes = tf.cast(nmsed_classes, tf.int32)
  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


class MultilevelDetectionGenerator(tf.keras.layers.Layer):
  """Generates detected boxes with scores and classes for one-stage detector."""

  def __init__(self, min_level, max_level, params):
    self._min_level = min_level
    self._max_level = max_level
    self._generate_detections = generate_detections_factory(params)
    super(MultilevelDetectionGenerator, self).__init__(autocast=False)

  def call(self, box_outputs, class_outputs, anchor_boxes, image_shape):
    # Collects outputs from all levels into a list.
    boxes = []
    scores = []
    for i in range(self._min_level, self._max_level + 1):
      box_outputs_i_shape = tf.shape(box_outputs[i])
      batch_size = box_outputs_i_shape[0]
      num_anchors_per_locations = box_outputs_i_shape[-1] // 4
      num_classes = tf.shape(class_outputs[i])[-1] // num_anchors_per_locations

      # Applies score transformation and remove the implicit background class.
      scores_i = tf.sigmoid(
          tf.reshape(class_outputs[i], [batch_size, -1, num_classes]))
      scores_i = tf.slice(scores_i, [0, 0, 1], [-1, -1, -1])

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

    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = (
        self._generate_detections(tf.expand_dims(boxes, axis=2), scores))

    # Adds 1 to offset the background class which has index 0.
    nmsed_classes += 1
    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


class GenericDetectionGenerator(tf.keras.layers.Layer):
  """Generates the final detected boxes with scores and classes."""

  def __init__(self, params):
    super(GenericDetectionGenerator, self).__init__(autocast=False)
    self._generate_detections = generate_detections_factory(params)

  def call(self, box_outputs, class_outputs, anchor_boxes, image_shape):
    """Generate final detections.

    Args:
      box_outputs: a tensor of shape of [batch_size, K, num_classes * 4]
        representing the class-specific box coordinates relative to anchors.
      class_outputs: a tensor of shape of [batch_size, K, num_classes]
        representing the class logits before applying score activiation.
      anchor_boxes: a tensor of shape of [batch_size, K, 4] representing the
        corresponding anchor boxes w.r.t `box_outputs`.
      image_shape: a tensor of shape of [batch_size, 2] storing the image height
        and width w.r.t. the scaled image, i.e. the same image space as
        `box_outputs` and `anchor_boxes`.

    Returns:
      nms_boxes: `float` Tensor of shape [batch_size, max_total_size, 4]
        representing top detected boxes in [y1, x1, y2, x2].
      nms_scores: `float` Tensor of shape [batch_size, max_total_size]
        representing sorted confidence scores for detected boxes. The values are
        between [0, 1].
      nms_classes: `int` Tensor of shape [batch_size, max_total_size]
        representing classes for detected boxes.
      valid_detections: `int` Tensor of shape [batch_size] only the top
        `valid_detections` boxes are valid detections.
    """
    class_outputs = tf.nn.softmax(class_outputs, axis=-1)

    # Removes the background class.
    class_outputs_shape = tf.shape(class_outputs)
    batch_size = class_outputs_shape[0]
    num_locations = class_outputs_shape[1]
    num_classes = class_outputs_shape[-1]
    num_detections = num_locations * (num_classes - 1)

    class_outputs = tf.slice(class_outputs, [0, 0, 1], [-1, -1, -1])
    box_outputs = tf.reshape(
        box_outputs,
        tf.stack([batch_size, num_locations, num_classes, 4], axis=-1))
    box_outputs = tf.slice(box_outputs, [0, 0, 1, 0], [-1, -1, -1, -1])
    anchor_boxes = tf.tile(
        tf.expand_dims(anchor_boxes, axis=2), [1, 1, num_classes - 1, 1])
    box_outputs = tf.reshape(box_outputs,
                             tf.stack([batch_size, num_detections, 4], axis=-1))
    anchor_boxes = tf.reshape(
        anchor_boxes, tf.stack([batch_size, num_detections, 4], axis=-1))

    # Box decoding.
    decoded_boxes = box_utils.decode_boxes(
        box_outputs, anchor_boxes, weights=[10.0, 10.0, 5.0, 5.0])

    # Box clipping
    decoded_boxes = box_utils.clip_boxes(decoded_boxes, image_shape)

    decoded_boxes = tf.reshape(
        decoded_boxes,
        tf.stack([batch_size, num_locations, num_classes - 1, 4], axis=-1))

    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = (
        self._generate_detections(decoded_boxes, class_outputs))

    # Adds 1 to offset the background class which has index 0.
    nmsed_classes += 1

    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


class OlnDetectionGenerator(GenericDetectionGenerator):
  """Generates the final detected boxes with scores and classes."""

  def __call__(self, box_outputs, class_outputs, anchor_boxes, image_shape,
               is_single_fg_score=False, keep_nms=True):
    """Generate final detections for Object Localization Network (OLN).

    Args:
      box_outputs: a tensor of shape of [batch_size, K, num_classes * 4]
        representing the class-specific box coordinates relative to anchors.
      class_outputs: a tensor of shape of [batch_size, K, num_classes]
        representing the class logits before applying score activiation.
      anchor_boxes: a tensor of shape of [batch_size, K, 4] representing the
        corresponding anchor boxes w.r.t `box_outputs`.
      image_shape: a tensor of shape of [batch_size, 2] storing the image height
        and width w.r.t. the scaled image, i.e. the same image space as
        `box_outputs` and `anchor_boxes`.
      is_single_fg_score: a Bool indicator of whether class_outputs includes the
        background scores concatenated or not. By default, class_outputs is a
        concatenation of both scores for the foreground and background. That is,
        scores_without_bg=False.
      keep_nms: a Bool indicator of whether to perform NMS or not.

    Returns:
      nms_boxes: `float` Tensor of shape [batch_size, max_total_size, 4]
        representing top detected boxes in [y1, x1, y2, x2].
      nms_scores: `float` Tensor of shape [batch_size, max_total_size]
        representing sorted confidence scores for detected boxes. The values are
        between [0, 1].
      nms_classes: `int` Tensor of shape [batch_size, max_total_size]
        representing classes for detected boxes.
      valid_detections: `int` Tensor of shape [batch_size] only the top
        `valid_detections` boxes are valid detections.
    """
    if is_single_fg_score:
      # Concatenates dummy background scores.
      dummy_bg_scores = tf.zeros_like(class_outputs)
      class_outputs = tf.stack([dummy_bg_scores, class_outputs], -1)
    else:
      class_outputs = tf.nn.softmax(class_outputs, axis=-1)

    # Removes the background class.
    class_outputs_shape = tf.shape(class_outputs)
    batch_size = class_outputs_shape[0]
    num_locations = class_outputs_shape[1]
    num_classes = class_outputs_shape[-1]
    num_detections = num_locations * (num_classes - 1)

    class_outputs = tf.slice(class_outputs, [0, 0, 1], [-1, -1, -1])
    box_outputs = tf.reshape(
        box_outputs,
        tf.stack([batch_size, num_locations, num_classes, 4], axis=-1))
    box_outputs = tf.slice(box_outputs, [0, 0, 1, 0], [-1, -1, -1, -1])
    anchor_boxes = tf.tile(
        tf.expand_dims(anchor_boxes, axis=2), [1, 1, num_classes - 1, 1])
    box_outputs = tf.reshape(box_outputs,
                             tf.stack([batch_size, num_detections, 4], axis=-1))
    anchor_boxes = tf.reshape(
        anchor_boxes, tf.stack([batch_size, num_detections, 4], axis=-1))

    # Box decoding. For RPN outputs, box_outputs are all zeros.
    decoded_boxes = box_utils.decode_boxes(
        box_outputs, anchor_boxes, weights=[10.0, 10.0, 5.0, 5.0])

    # Box clipping
    decoded_boxes = box_utils.clip_boxes(decoded_boxes, image_shape)

    decoded_boxes = tf.reshape(
        decoded_boxes,
        tf.stack([batch_size, num_locations, num_classes - 1, 4], axis=-1))

    if keep_nms:
      nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = (
          self._generate_detections(decoded_boxes, class_outputs))
      # Adds 1 to offset the background class which has index 0.
      nmsed_classes += 1
    else:
      nmsed_boxes = decoded_boxes[:, :, 0, :]
      nmsed_scores = class_outputs[:, :, 0]
      nmsed_classes = tf.cast(tf.ones_like(nmsed_scores), tf.int32)
      valid_detections = tf.cast(
          tf.reduce_sum(tf.ones_like(nmsed_scores), axis=-1), tf.int32)

    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections
