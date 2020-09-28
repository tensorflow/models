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
"""Generators to generate the final detections."""

# Import libraries

import tensorflow as tf

from official.vision.beta.ops import box_ops
from official.vision.beta.ops import nms


def _generate_detections_v1(boxes,
                            scores,
                            pre_nms_top_k=5000,
                            pre_nms_score_threshold=0.05,
                            nms_iou_threshold=0.5,
                            max_num_detections=100):
  """Generate the final detections given the model outputs.

  The implementation unrolls the batch dimension and process images one by one.
  It required the batch dimension to be statically known and it is TPU
  compatible.

  Args:
    boxes: a tensor with shape [batch_size, N, num_classes, 4] or
      [batch_size, N, 1, 4], which box predictions on all feature levels. The N
      is the number of total anchors on all levels.
    scores: a tensor with shape [batch_size, N, num_classes], which
      stacks class probability on all feature levels. The N is the number of
      total anchors on all levels. The num_classes is the number of classes
      predicted by the model. Note that the class_outputs here is the raw score.
    pre_nms_top_k: an int number of top candidate detections per class
      before NMS.
    pre_nms_score_threshold: a float representing the threshold for deciding
      when to remove boxes based on score.
    nms_iou_threshold: a float representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    max_num_detections: a scalar representing maximum number of boxes retained
      over all classes.

  Returns:
    nms_boxes: `float` Tensor of shape [batch_size, max_num_detections, 4]
      representing top detected boxes in [y1, x1, y2, x2].
    nms_scores: `float` Tensor of shape [batch_size, max_num_detections]
      representing sorted confidence scores for detected boxes. The values are
      between [0, 1].
    nms_classes: `int` Tensor of shape [batch_size, max_num_detections]
      representing classes for detected boxes.
    valid_detections: `int` Tensor of shape [batch_size] only the top
      `valid_detections` boxes are valid detections.
  """
  with tf.name_scope('generate_detections'):
    batch_size = scores.get_shape().as_list()[0]
    nmsed_boxes = []
    nmsed_classes = []
    nmsed_scores = []
    valid_detections = []
    for i in range(batch_size):
      (nmsed_boxes_i, nmsed_scores_i, nmsed_classes_i,
       valid_detections_i) = _generate_detections_per_image(
           boxes[i],
           scores[i],
           max_num_detections,
           nms_iou_threshold,
           pre_nms_score_threshold,
           pre_nms_top_k)
      nmsed_boxes.append(nmsed_boxes_i)
      nmsed_scores.append(nmsed_scores_i)
      nmsed_classes.append(nmsed_classes_i)
      valid_detections.append(valid_detections_i)
  nmsed_boxes = tf.stack(nmsed_boxes, axis=0)
  nmsed_scores = tf.stack(nmsed_scores, axis=0)
  nmsed_classes = tf.stack(nmsed_classes, axis=0)
  valid_detections = tf.stack(valid_detections, axis=0)
  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


def _generate_detections_per_image(boxes,
                                   scores,
                                   pre_nms_top_k=5000,
                                   pre_nms_score_threshold=0.05,
                                   nms_iou_threshold=0.5,
                                   max_num_detections=100):
  """Generate the final detections per image given the model outputs.

  Args:
    boxes: a tensor with shape [N, num_classes, 4] or [N, 1, 4], which box
      predictions on all feature levels. The N is the number of total anchors on
      all levels.
    scores: a tensor with shape [N, num_classes], which stacks class probability
      on all feature levels. The N is the number of total anchors on all levels.
      The num_classes is the number of classes predicted by the model. Note that
      the class_outputs here is the raw score.
    pre_nms_top_k: an int number of top candidate detections per class
      before NMS.
    pre_nms_score_threshold: a float representing the threshold for deciding
      when to remove boxes based on score.
    nms_iou_threshold: a float representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    max_num_detections: a scalar representing maximum number of boxes retained
      over all classes.

  Returns:
    nms_boxes: `float` Tensor of shape [max_num_detections, 4] representing top
      detected boxes in [y1, x1, y2, x2].
    nms_scores: `float` Tensor of shape [max_num_detections] representing sorted
      confidence scores for detected boxes. The values are between [0, 1].
    nms_classes: `int` Tensor of shape [max_num_detections] representing classes
      for detected boxes.
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
  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


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


def _generate_detections_v2(boxes,
                            scores,
                            pre_nms_top_k=5000,
                            pre_nms_score_threshold=0.05,
                            nms_iou_threshold=0.5,
                            max_num_detections=100):
  """Generate the final detections given the model outputs.

  This implementation unrolls classes dimension while using the tf.while_loop
  to implement the batched NMS, so that it can be parallelized at the batch
  dimension. It should give better performance comparing to v1 implementation.
  It is TPU compatible.

  Args:
    boxes: a tensor with shape [batch_size, N, num_classes, 4] or [batch_size,
      N, 1, 4], which box predictions on all feature levels. The N is the number
      of total anchors on all levels.
    scores: a tensor with shape [batch_size, N, num_classes], which stacks class
      probability on all feature levels. The N is the number of total anchors on
      all levels. The num_classes is the number of classes predicted by the
      model. Note that the class_outputs here is the raw score.
    pre_nms_top_k: an int number of top candidate detections per class
      before NMS.
    pre_nms_score_threshold: a float representing the threshold for deciding
      when to remove boxes based on score.
    nms_iou_threshold: a float representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    max_num_detections: a scalar representing maximum number of boxes retained
      over all classes.

  Returns:
    nms_boxes: `float` Tensor of shape [batch_size, max_num_detections, 4]
      representing top detected boxes in [y1, x1, y2, x2].
    nms_scores: `float` Tensor of shape [batch_size, max_num_detections]
      representing sorted confidence scores for detected boxes. The values are
      between [0, 1].
    nms_classes: `int` Tensor of shape [batch_size, max_num_detections]
      representing classes for detected boxes.
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


def _generate_detections_batched(boxes,
                                 scores,
                                 pre_nms_score_threshold,
                                 nms_iou_threshold,
                                 max_num_detections):
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
    pre_nms_score_threshold: a float representing the threshold for deciding
      when to remove boxes based on score.
    nms_iou_threshold: a float representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    max_num_detections: a scalar representing maximum number of boxes retained
      over all classes.

  Returns:
    nms_boxes: `float` Tensor of shape [batch_size, max_num_detections, 4]
      representing top detected boxes in [y1, x1, y2, x2].
    nms_scores: `float` Tensor of shape [batch_size, max_num_detections]
      representing sorted confidence scores for detected boxes. The values are
      between [0, 1].
    nms_classes: `int` Tensor of shape [batch_size, max_num_detections]
      representing classes for detected boxes.
    valid_detections: `int` Tensor of shape [batch_size] only the top
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
  return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


@tf.keras.utils.register_keras_serializable(package='Vision')
class DetectionGenerator(tf.keras.layers.Layer):
  """Generates the final detected boxes with scores and classes."""

  def __init__(self,
               apply_nms=True,
               pre_nms_top_k=5000,
               pre_nms_score_threshold=0.05,
               nms_iou_threshold=0.5,
               max_num_detections=100,
               use_batched_nms=False,
               **kwargs):
    """Initializes a detection generator.

    Args:
      apply_nms: bool, whether or not apply non maximum suppression. If False,
        the decoded boxes and their scores are returned.
      pre_nms_top_k: int, the number of top scores proposals to be kept before
        applying NMS.
      pre_nms_score_threshold: float, the score threshold to apply before
        applying  NMS. Proposals whose scores are below this threshold are
        thrown away.
      nms_iou_threshold: float in [0, 1], the NMS IoU threshold.
      max_num_detections: int, the final number of total detections to generate.
      use_batched_nms: bool, whether or not use
        `tf.image.combined_non_max_suppression`.
      **kwargs: other key word arguments passed to Layer.
    """
    self._config_dict = {
        'apply_nms': apply_nms,
        'pre_nms_top_k': pre_nms_top_k,
        'pre_nms_score_threshold': pre_nms_score_threshold,
        'nms_iou_threshold': nms_iou_threshold,
        'max_num_detections': max_num_detections,
        'use_batched_nms': use_batched_nms,
    }
    super(DetectionGenerator, self).__init__(**kwargs)

  def __call__(self,
               raw_boxes,
               raw_scores,
               anchor_boxes,
               image_shape):
    """Generate final detections.

    Args:
      raw_boxes: a tensor of shape of [batch_size, K, num_classes * 4]
        representing the class-specific box coordinates relative to anchors.
      raw_scores: a tensor of shape of [batch_size, K, num_classes]
        representing the class logits before applying score activiation.
      anchor_boxes: a tensor of shape of [batch_size, K, 4] representing the
        corresponding anchor boxes w.r.t `box_outputs`.
      image_shape: a tensor of shape of [batch_size, 2] storing the image height
        and width w.r.t. the scaled image, i.e. the same image space as
        `box_outputs` and `anchor_boxes`.

    Returns:
      If `apply_nms` = True, the return is a dictionary with keys:
        `detection_boxes`: float Tensor of shape [batch, max_num_detections, 4]
          representing top detected boxes in [y1, x1, y2, x2].
        `detection_scores`: float Tensor of shape [batch, max_num_detections]
          representing sorted confidence scores for detected boxes. The values
          are between [0, 1].
        `detection_classes`: int Tensor of shape [batch, max_num_detections]
          representing classes for detected boxes.
        `num_detections`: int Tensor of shape [batch] only the first
          `num_detections` boxes are valid detections
      If `apply_nms` = False, the return is a dictionary with keys:
        `decoded_boxes`: float Tensor of shape [batch, num_raw_boxes, 4]
          representing all the decoded boxes.
        `decoded_box_scores`: float Tensor of shape [batch, num_raw_boxes]
          representing socres of all the decoded boxes.
    """
    box_scores = tf.nn.softmax(raw_scores, axis=-1)

    # Removes the background class.
    box_scores_shape = tf.shape(box_scores)
    batch_size = box_scores_shape[0]
    num_locations = box_scores_shape[1]
    num_classes = box_scores_shape[-1]
    num_detections = num_locations * (num_classes - 1)

    box_scores = tf.slice(box_scores, [0, 0, 1], [-1, -1, -1])
    raw_boxes = tf.reshape(
        raw_boxes,
        tf.stack([batch_size, num_locations, num_classes, 4], axis=-1))
    raw_boxes = tf.slice(
        raw_boxes, [0, 0, 1, 0], [-1, -1, -1, -1])
    anchor_boxes = tf.tile(
        tf.expand_dims(anchor_boxes, axis=2), [1, 1, num_classes - 1, 1])
    raw_boxes = tf.reshape(
        raw_boxes,
        tf.stack([batch_size, num_detections, 4], axis=-1))
    anchor_boxes = tf.reshape(
        anchor_boxes,
        tf.stack([batch_size, num_detections, 4], axis=-1))

    # Box decoding.
    decoded_boxes = box_ops.decode_boxes(
        raw_boxes, anchor_boxes, weights=[10.0, 10.0, 5.0, 5.0])

    # Box clipping
    decoded_boxes = box_ops.clip_boxes(
        decoded_boxes, tf.expand_dims(image_shape, axis=1))

    decoded_boxes = tf.reshape(
        decoded_boxes,
        tf.stack([batch_size, num_locations, num_classes - 1, 4], axis=-1))

    if not self._config_dict['apply_nms']:
      return {
          'decoded_boxes': decoded_boxes,
          'decoded_box_scores': box_scores,
      }

    if self._config_dict['use_batched_nms']:
      nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = (
          _generate_detections_batched(
              decoded_boxes,
              box_scores,
              self._config_dict['pre_nms_score_threshold'],
              self._config_dict['nms_iou_threshold'],
              self._config_dict['max_num_detections']))
    else:
      nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = (
          _generate_detections_v2(
              decoded_boxes,
              box_scores,
              self._config_dict['pre_nms_top_k'],
              self._config_dict['pre_nms_score_threshold'],
              self._config_dict['nms_iou_threshold'],
              self._config_dict['max_num_detections']))

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
               apply_nms=True,
               pre_nms_top_k=5000,
               pre_nms_score_threshold=0.05,
               nms_iou_threshold=0.5,
               max_num_detections=100,
               use_batched_nms=False,
               **kwargs):
    """Initializes a detection generator.

    Args:
      apply_nms: bool, whether or not apply non maximum suppression. If False,
        the decoded boxes and their scores are returned.
      pre_nms_top_k: int, the number of top scores proposals to be kept before
        applying NMS.
      pre_nms_score_threshold: float, the score threshold to apply before
        applying  NMS. Proposals whose scores are below this threshold are
        thrown away.
      nms_iou_threshold: float in [0, 1], the NMS IoU threshold.
      max_num_detections: int, the final number of total detections to generate.
      use_batched_nms: bool, whether or not use
        `tf.image.combined_non_max_suppression`.
      **kwargs: other key word arguments passed to Layer.
    """
    self._config_dict = {
        'apply_nms': apply_nms,
        'pre_nms_top_k': pre_nms_top_k,
        'pre_nms_score_threshold': pre_nms_score_threshold,
        'nms_iou_threshold': nms_iou_threshold,
        'max_num_detections': max_num_detections,
        'use_batched_nms': use_batched_nms,
    }
    super(MultilevelDetectionGenerator, self).__init__(**kwargs)

  def __call__(self,
               raw_boxes,
               raw_scores,
               anchor_boxes,
               image_shape):
    """Generate final detections.

    Args:
      raw_boxes: a dict with keys representing FPN levels and values
        representing box tenors of shape
        [batch, feature_h, feature_w, num_anchors * 4].
      raw_scores: a dict with keys representing FPN levels and values
        representing logit tensors of shape
        [batch, feature_h, feature_w, num_anchors].
      anchor_boxes: a tensor of shape of [batch_size, K, 4] representing the
        corresponding anchor boxes w.r.t `box_outputs`.
      image_shape: a tensor of shape of [batch_size, 2] storing the image height
        and width w.r.t. the scaled image, i.e. the same image space as
        `box_outputs` and `anchor_boxes`.

    Returns:
      If `apply_nms` = True, the return is a dictionary with keys:
        `detection_boxes`: float Tensor of shape [batch, max_num_detections, 4]
          representing top detected boxes in [y1, x1, y2, x2].
        `detection_scores`: float Tensor of shape [batch, max_num_detections]
          representing sorted confidence scores for detected boxes. The values
          are between [0, 1].
        `detection_classes`: int Tensor of shape [batch, max_num_detections]
          representing classes for detected boxes.
        `num_detections`: int Tensor of shape [batch] only the first
          `num_detections` boxes are valid detections
      If `apply_nms` = False, the return is a dictionary with keys:
        `decoded_boxes`: float Tensor of shape [batch, num_raw_boxes, 4]
          representing all the decoded boxes.
        `decoded_box_scores`: float Tensor of shape [batch, num_raw_boxes]
          representing socres of all the decoded boxes.
    """
    # Collects outputs from all levels into a list.
    boxes = []
    scores = []
    levels = list(raw_boxes.keys())
    min_level = int(min(levels))
    max_level = int(max(levels))
    for i in range(min_level, max_level + 1):
      raw_boxes_i_shape = tf.shape(raw_boxes[str(i)])
      batch_size = raw_boxes_i_shape[0]
      num_anchors_per_locations = raw_boxes_i_shape[-1] // 4
      num_classes = tf.shape(
          raw_scores[str(i)])[-1] // num_anchors_per_locations

      # Applies score transformation and remove the implicit background class.
      scores_i = tf.sigmoid(
          tf.reshape(raw_scores[str(i)], [batch_size, -1, num_classes]))
      scores_i = tf.slice(scores_i, [0, 0, 1], [-1, -1, -1])

      # Box decoding.
      # The anchor boxes are shared for all data in a batch.
      # One stage detector only supports class agnostic box regression.
      anchor_boxes_i = tf.reshape(anchor_boxes[str(i)], [batch_size, -1, 4])
      raw_boxes_i = tf.reshape(raw_boxes[str(i)], [batch_size, -1, 4])
      boxes_i = box_ops.decode_boxes(raw_boxes_i, anchor_boxes_i)

      # Box clipping.
      boxes_i = box_ops.clip_boxes(
          boxes_i, tf.expand_dims(image_shape, axis=1))

      boxes.append(boxes_i)
      scores.append(scores_i)
    boxes = tf.concat(boxes, axis=1)
    boxes = tf.expand_dims(boxes, axis=2)
    scores = tf.concat(scores, axis=1)

    if not self._config_dict['apply_nms']:
      return {
          'decoded_boxes': boxes,
          'decoded_box_scores': scores,
      }

    if self._config_dict['use_batched_nms']:
      nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = (
          _generate_detections_batched(
              boxes,
              scores,
              self._config_dict['pre_nms_score_threshold'],
              self._config_dict['nms_iou_threshold'],
              self._config_dict['max_num_detections']))
    else:
      nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = (
          _generate_detections_v2(
              boxes,
              scores,
              self._config_dict['pre_nms_top_k'],
              self._config_dict['pre_nms_score_threshold'],
              self._config_dict['nms_iou_threshold'],
              self._config_dict['max_num_detections']))

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
