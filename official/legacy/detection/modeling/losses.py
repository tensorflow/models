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

"""Losses used for detection models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow as tf


def focal_loss(logits, targets, alpha, gamma, normalizer):
  """Compute the focal loss between `logits` and the golden `target` values.

  Focal loss = -(1-pt)^gamma * log(pt)
  where pt is the probability of being classified to the true class.

  Args:
    logits: A float32 tensor of size
      [batch, height_in, width_in, num_predictions].
    targets: A float32 tensor of size
      [batch, height_in, width_in, num_predictions].
    alpha: A float32 scalar multiplying alpha to the loss from positive examples
      and (1-alpha) to the loss from negative examples.
    gamma: A float32 scalar modulating loss from hard and easy examples.
    normalizer: A float32 scalar normalizes the total loss from all examples.

  Returns:
    loss: A float32 Tensor of size [batch, height_in, width_in, num_predictions]
      representing normalized loss on the prediction map.
  """
  with tf.name_scope('focal_loss'):
    positive_label_mask = tf.math.equal(targets, 1.0)
    cross_entropy = (
        tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
    # Below are comments/derivations for computing modulator.
    # For brevity, let x = logits,  z = targets, r = gamma, and p_t = sigmod(x)
    # for positive samples and 1 - sigmoid(x) for negative examples.
    #
    # The modulator, defined as (1 - P_t)^r, is a critical part in focal loss
    # computation. For r > 0, it puts more weights on hard examples, and less
    # weights on easier ones. However if it is directly computed as (1 - P_t)^r,
    # its back-propagation is not stable when r < 1. The implementation here
    # resolves the issue.
    #
    # For positive samples (labels being 1),
    #    (1 - p_t)^r
    #  = (1 - sigmoid(x))^r
    #  = (1 - (1 / (1 + exp(-x))))^r
    #  = (exp(-x) / (1 + exp(-x)))^r
    #  = exp(log((exp(-x) / (1 + exp(-x)))^r))
    #  = exp(r * log(exp(-x)) - r * log(1 + exp(-x)))
    #  = exp(- r * x - r * log(1 + exp(-x)))
    #
    # For negative samples (labels being 0),
    #    (1 - p_t)^r
    #  = (sigmoid(x))^r
    #  = (1 / (1 + exp(-x)))^r
    #  = exp(log((1 / (1 + exp(-x)))^r))
    #  = exp(-r * log(1 + exp(-x)))
    #
    # Therefore one unified form for positive (z = 1) and negative (z = 0)
    # samples is:
    #      (1 - p_t)^r = exp(-r * z * x - r * log(1 + exp(-x))).
    neg_logits = -1.0 * logits
    modulator = tf.math.exp(gamma * targets * neg_logits -
                            gamma * tf.math.log1p(tf.math.exp(neg_logits)))
    loss = modulator * cross_entropy
    weighted_loss = tf.where(positive_label_mask, alpha * loss,
                             (1.0 - alpha) * loss)
    weighted_loss /= normalizer
  return weighted_loss


class RpnScoreLoss(object):
  """Region Proposal Network score loss function."""

  def __init__(self, params):
    self._rpn_batch_size_per_im = params.rpn_batch_size_per_im
    self._binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM, from_logits=True)

  def __call__(self, score_outputs, labels):
    """Computes total RPN detection loss.

    Computes total RPN detection loss including box and score from all levels.

    Args:
      score_outputs: an OrderDict with keys representing levels and values
        representing scores in [batch_size, height, width, num_anchors].
      labels: the dictionary that returned from dataloader that includes
        groundturth targets.

    Returns:
      rpn_score_loss: a scalar tensor representing total score loss.
    """
    with tf.name_scope('rpn_loss'):
      levels = sorted(score_outputs.keys())

      score_losses = []
      for level in levels:
        score_losses.append(
            self._rpn_score_loss(
                score_outputs[level],
                labels[level],
                normalizer=tf.cast(
                    tf.shape(score_outputs[level])[0] *
                    self._rpn_batch_size_per_im, dtype=tf.float32)))

      # Sums per level losses to total loss.
      return tf.math.add_n(score_losses)

  def _rpn_score_loss(self, score_outputs, score_targets, normalizer=1.0):
    """Computes score loss."""
    # score_targets has three values:
    # (1) score_targets[i]=1, the anchor is a positive sample.
    # (2) score_targets[i]=0, negative.
    # (3) score_targets[i]=-1, the anchor is don't care (ignore).
    with tf.name_scope('rpn_score_loss'):
      mask = tf.math.logical_or(tf.math.equal(score_targets, 1),
                                tf.math.equal(score_targets, 0))

      score_targets = tf.math.maximum(score_targets,
                                      tf.zeros_like(score_targets))

      score_targets = tf.expand_dims(score_targets, axis=-1)
      score_outputs = tf.expand_dims(score_outputs, axis=-1)
      score_loss = self._binary_crossentropy(
          score_targets, score_outputs, sample_weight=mask)

      score_loss /= normalizer
      return score_loss


class RpnBoxLoss(object):
  """Region Proposal Network box regression loss function."""

  def __init__(self, params):
    logging.info('RpnBoxLoss huber_loss_delta %s', params.huber_loss_delta)
    # The delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].
    self._huber_loss = tf.keras.losses.Huber(
        delta=params.huber_loss_delta, reduction=tf.keras.losses.Reduction.SUM)

  def __call__(self, box_outputs, labels):
    """Computes total RPN detection loss.

    Computes total RPN detection loss including box and score from all levels.

    Args:
      box_outputs: an OrderDict with keys representing levels and values
        representing box regression targets in
        [batch_size, height, width, num_anchors * 4].
      labels: the dictionary that returned from dataloader that includes
        groundturth targets.

    Returns:
      rpn_box_loss: a scalar tensor representing total box regression loss.
    """
    with tf.name_scope('rpn_loss'):
      levels = sorted(box_outputs.keys())

      box_losses = []
      for level in levels:
        box_losses.append(self._rpn_box_loss(box_outputs[level], labels[level]))

      # Sum per level losses to total loss.
      return tf.add_n(box_losses)

  def _rpn_box_loss(self, box_outputs, box_targets, normalizer=1.0):
    """Computes box regression loss."""
    with tf.name_scope('rpn_box_loss'):
      mask = tf.cast(tf.not_equal(box_targets, 0.0), dtype=tf.float32)
      box_targets = tf.expand_dims(box_targets, axis=-1)
      box_outputs = tf.expand_dims(box_outputs, axis=-1)
      box_loss = self._huber_loss(box_targets, box_outputs, sample_weight=mask)
      # The loss is normalized by the sum of non-zero weights and additional
      # normalizer provided by the function caller. Using + 0.01 here to avoid
      # division by zero.
      box_loss /= normalizer * (tf.reduce_sum(mask) + 0.01)
      return box_loss


class OlnRpnCenterLoss(object):
  """Object Localization Network RPN centerness regression loss function."""

  def __init__(self):
    self._l1_loss = tf.keras.losses.MeanAbsoluteError(
        reduction=tf.keras.losses.Reduction.SUM)

  def __call__(self, center_outputs, labels):
    """Computes total RPN centerness regression loss.

    Computes total RPN centerness score regression loss from all levels.

    Args:
      center_outputs: an OrderDict with keys representing levels and values
        representing anchor centerness regression targets in
        [batch_size, height, width, num_anchors * 4].
      labels: the dictionary that returned from dataloader that includes
        groundturth targets.

    Returns:
      rpn_center_loss: a scalar tensor representing total centerness regression
        loss.
    """
    with tf.name_scope('rpn_loss'):
      # Normalizer.
      levels = sorted(center_outputs.keys())
      num_valid = 0
      # 0<pos<1, neg=0, ign=-1
      for level in levels:
        num_valid += tf.reduce_sum(tf.cast(
            tf.greater(labels[level], -1.0), tf.float32))  # in and out of box
      num_valid += 1e-12

      # Centerness loss over multi levels.
      center_losses = []
      for level in levels:
        center_losses.append(
            self._rpn_center_l1_loss(
                center_outputs[level], labels[level],
                normalizer=num_valid))

      # Sum per level losses to total loss.
      return tf.add_n(center_losses)

  def _rpn_center_l1_loss(self, center_outputs, center_targets,
                          normalizer=1.0):
    """Computes centerness regression loss."""
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].
    with tf.name_scope('rpn_center_loss'):

      # mask = tf.greater(center_targets, 0.0)  # inside box only.
      mask = tf.greater(center_targets, -1.0)  # in and out of box.
      center_targets = tf.maximum(center_targets, tf.zeros_like(center_targets))
      center_outputs = tf.sigmoid(center_outputs)
      center_targets = tf.expand_dims(center_targets, -1)
      center_outputs = tf.expand_dims(center_outputs, -1)
      mask = tf.cast(mask, dtype=tf.float32)
      center_loss = self._l1_loss(center_targets, center_outputs,
                                  sample_weight=mask)
      center_loss /= normalizer
      return center_loss


class OlnRpnIoULoss(object):
  """Object Localization Network RPN box-lrtb regression iou loss function."""

  def __call__(self, box_outputs, labels, center_targets):
    """Computes total RPN detection loss.

    Computes total RPN box regression loss from all levels.

    Args:
      box_outputs: an OrderDict with keys representing levels and values
        representing box regression targets in
        [batch_size, height, width, num_anchors * 4].
        last channel: (left, right, top, bottom).
      labels: the dictionary that returned from dataloader that includes
        groundturth targets (left, right, top, bottom).
      center_targets: valid_target mask.

    Returns:
      rpn_iou_loss: a scalar tensor representing total box regression loss.
    """
    with tf.name_scope('rpn_loss'):
      # Normalizer.
      levels = sorted(box_outputs.keys())
      normalizer = 0.
      for level in levels:
        # center_targets pos>0, neg=0, ign=-1.
        mask_ = tf.cast(tf.logical_and(
            tf.greater(center_targets[level][..., 0], 0.0),
            tf.greater(tf.reduce_min(labels[level], -1), 0.0)), tf.float32)
        normalizer += tf.reduce_sum(mask_)
      normalizer += 1e-8
      # iou_loss over multi levels.
      iou_losses = []
      for level in levels:
        iou_losses.append(
            self._rpn_iou_loss(
                box_outputs[level], labels[level],
                center_weight=center_targets[level][..., 0],
                normalizer=normalizer))
      # Sum per level losses to total loss.
      return tf.add_n(iou_losses)

  def _rpn_iou_loss(self, box_outputs, box_targets,
                    center_weight=None, normalizer=1.0):
    """Computes box regression loss."""
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].
    with tf.name_scope('rpn_iou_loss'):
      mask = tf.logical_and(
          tf.greater(center_weight, 0.0),
          tf.greater(tf.reduce_min(box_targets, -1), 0.0))

      pred_left = box_outputs[..., 0]
      pred_right = box_outputs[..., 1]
      pred_top = box_outputs[..., 2]
      pred_bottom = box_outputs[..., 3]

      gt_left = box_targets[..., 0]
      gt_right = box_targets[..., 1]
      gt_top = box_targets[..., 2]
      gt_bottom = box_targets[..., 3]

      inter_width = (tf.minimum(pred_left, gt_left) +
                     tf.minimum(pred_right, gt_right))
      inter_height = (tf.minimum(pred_top, gt_top) +
                      tf.minimum(pred_bottom, gt_bottom))
      inter_area = inter_width * inter_height
      union_area = ((pred_left + pred_right) * (pred_top + pred_bottom) +
                    (gt_left + gt_right) * (gt_top + gt_bottom) -
                    inter_area)
      iou = inter_area / (union_area + 1e-8)
      mask_ = tf.cast(mask, tf.float32)
      iou = tf.clip_by_value(iou, clip_value_min=1e-8, clip_value_max=1.0)
      neg_log_iou = -tf.math.log(iou)
      iou_loss = tf.reduce_sum(neg_log_iou * mask_)
      iou_loss /= normalizer
      return iou_loss


class FastrcnnClassLoss(object):
  """Fast R-CNN classification loss function."""

  def __init__(self):
    self._categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM, from_logits=True)

  def __call__(self, class_outputs, class_targets):
    """Computes the class loss (Fast-RCNN branch) of Mask-RCNN.

    This function implements the classification loss of the Fast-RCNN.

    The classification loss is softmax on all RoIs.
    Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/fast_rcnn_heads.py  # pylint: disable=line-too-long

    Args:
      class_outputs: a float tensor representing the class prediction for each box
        with a shape of [batch_size, num_boxes, num_classes].
      class_targets: a float tensor representing the class label for each box
        with a shape of [batch_size, num_boxes].

    Returns:
      a scalar tensor representing total class loss.
    """
    with tf.name_scope('fast_rcnn_loss'):
      batch_size, num_boxes, num_classes = class_outputs.get_shape().as_list()
      class_targets = tf.cast(class_targets, dtype=tf.int32)
      class_targets_one_hot = tf.one_hot(class_targets, num_classes)
      return self._fast_rcnn_class_loss(class_outputs, class_targets_one_hot,
                                        normalizer=batch_size * num_boxes / 2.0)

  def _fast_rcnn_class_loss(self, class_outputs, class_targets_one_hot,
                            normalizer):
    """Computes classification loss."""
    with tf.name_scope('fast_rcnn_class_loss'):
      class_loss = self._categorical_crossentropy(class_targets_one_hot,
                                                  class_outputs)

      class_loss /= normalizer
      return class_loss


class FastrcnnBoxLoss(object):
  """Fast R-CNN box regression loss function."""

  def __init__(self, params):
    logging.info('FastrcnnBoxLoss huber_loss_delta %s', params.huber_loss_delta)
    # The delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].
    self._huber_loss = tf.keras.losses.Huber(
        delta=params.huber_loss_delta, reduction=tf.keras.losses.Reduction.SUM)

  def __call__(self, box_outputs, class_targets, box_targets):
    """Computes the box loss (Fast-RCNN branch) of Mask-RCNN.

    This function implements the box regression loss of the Fast-RCNN. As the
    `box_outputs` produces `num_classes` boxes for each RoI, the reference model
    expands `box_targets` to match the shape of `box_outputs` and selects only
    the target that the RoI has a maximum overlap. (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py)  # pylint: disable=line-too-long
    Instead, this function selects the `box_outputs` by the `class_targets` so
    that it doesn't expand `box_targets`.

    The box loss is smooth L1-loss on only positive samples of RoIs.
    Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/fast_rcnn_heads.py  # pylint: disable=line-too-long

    Args:
      box_outputs: a float tensor representing the box prediction for each box
        with a shape of [batch_size, num_boxes, num_classes * 4].
      class_targets: a float tensor representing the class label for each box
        with a shape of [batch_size, num_boxes].
      box_targets: a float tensor representing the box label for each box
        with a shape of [batch_size, num_boxes, 4].

    Returns:
      box_loss: a scalar tensor representing total box regression loss.
    """
    with tf.name_scope('fast_rcnn_loss'):
      class_targets = tf.cast(class_targets, dtype=tf.int32)

      # Selects the box from `box_outputs` based on `class_targets`, with which
      # the box has the maximum overlap.
      (batch_size, num_rois,
       num_class_specific_boxes) = box_outputs.get_shape().as_list()
      num_classes = num_class_specific_boxes // 4
      box_outputs = tf.reshape(box_outputs,
                               [batch_size, num_rois, num_classes, 4])

      box_indices = tf.reshape(
          class_targets + tf.tile(
              tf.expand_dims(
                  tf.range(batch_size) * num_rois * num_classes, 1),
              [1, num_rois]) + tf.tile(
                  tf.expand_dims(tf.range(num_rois) * num_classes, 0),
                  [batch_size, 1]), [-1])

      box_outputs = tf.matmul(
          tf.one_hot(
              box_indices,
              batch_size * num_rois * num_classes,
              dtype=box_outputs.dtype), tf.reshape(box_outputs, [-1, 4]))
      box_outputs = tf.reshape(box_outputs, [batch_size, -1, 4])

      return self._fast_rcnn_box_loss(box_outputs, box_targets, class_targets)

  def _fast_rcnn_box_loss(self, box_outputs, box_targets, class_targets,
                          normalizer=1.0):
    """Computes box regression loss."""
    with tf.name_scope('fast_rcnn_box_loss'):
      mask = tf.tile(tf.expand_dims(tf.greater(class_targets, 0), axis=2),
                     [1, 1, 4])
      mask = tf.cast(mask, dtype=tf.float32)
      box_targets = tf.expand_dims(box_targets, axis=-1)
      box_outputs = tf.expand_dims(box_outputs, axis=-1)
      box_loss = self._huber_loss(box_targets, box_outputs, sample_weight=mask)
      # The loss is normalized by the number of ones in mask,
      # additianal normalizer provided by the user and using 0.01 here to avoid
      # division by 0.
      box_loss /= normalizer * (tf.reduce_sum(mask) + 0.01)
      return box_loss


class OlnBoxScoreLoss(object):
  """Object Localization Network Box-Iou scoring function."""

  def __init__(self, params):
    self._ignore_threshold = params.ignore_threshold
    self._l1_loss = tf.keras.losses.MeanAbsoluteError(
        reduction=tf.keras.losses.Reduction.SUM)

  def __call__(self, score_outputs, score_targets):
    """Computes the class loss (Fast-RCNN branch) of Mask-RCNN.

    This function implements the classification loss of the Fast-RCNN.

    The classification loss is softmax on all RoIs.
    Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/fast_rcnn_heads.py  # pylint: disable=line-too-long

    Args:
      score_outputs: a float tensor representing the class prediction for each box
        with a shape of [batch_size, num_boxes, num_classes].
      score_targets: a float tensor representing the class label for each box
        with a shape of [batch_size, num_boxes].

    Returns:
      a scalar tensor representing total score loss.
    """
    with tf.name_scope('fast_rcnn_loss'):
      score_outputs = tf.squeeze(score_outputs, -1)

      mask = tf.greater(score_targets, self._ignore_threshold)
      num_valid = tf.reduce_sum(tf.cast(mask, tf.float32))
      score_targets = tf.maximum(score_targets, tf.zeros_like(score_targets))
      score_outputs = tf.sigmoid(score_outputs)
      score_targets = tf.expand_dims(score_targets, -1)
      score_outputs = tf.expand_dims(score_outputs, -1)
      mask = tf.cast(mask, dtype=tf.float32)
      score_loss = self._l1_loss(score_targets, score_outputs,
                                 sample_weight=mask)
      score_loss /= (num_valid + 1e-10)
      return score_loss


class MaskrcnnLoss(object):
  """Mask R-CNN instance segmentation mask loss function."""

  def __init__(self):
    self._binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM, from_logits=True)

  def __call__(self, mask_outputs, mask_targets, select_class_targets):
    """Computes the mask loss of Mask-RCNN.

    This function implements the mask loss of Mask-RCNN. As the `mask_outputs`
    produces `num_classes` masks for each RoI, the reference model expands
    `mask_targets` to match the shape of `mask_outputs` and selects only the
    target that the RoI has a maximum overlap. (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/mask_rcnn.py)  # pylint: disable=line-too-long
    Instead, this implementation selects the `mask_outputs` by the `class_targets`
    so that it doesn't expand `mask_targets`. Note that the selection logic is
    done in the post-processing of mask_rcnn_fn in mask_rcnn_architecture.py.

    Args:
      mask_outputs: a float tensor representing the prediction for each mask,
        with a shape of
        [batch_size, num_masks, mask_height, mask_width].
      mask_targets: a float tensor representing the binary mask of ground truth
        labels for each mask with a shape of
        [batch_size, num_masks, mask_height, mask_width].
      select_class_targets: a tensor with a shape of [batch_size, num_masks],
        representing the foreground mask targets.

    Returns:
      mask_loss: a float tensor representing total mask loss.
    """
    with tf.name_scope('mask_rcnn_loss'):
      (batch_size, num_masks, mask_height,
       mask_width) = mask_outputs.get_shape().as_list()

      weights = tf.tile(
          tf.reshape(tf.greater(select_class_targets, 0),
                     [batch_size, num_masks, 1, 1]),
          [1, 1, mask_height, mask_width])
      weights = tf.cast(weights, dtype=tf.float32)

      mask_targets = tf.expand_dims(mask_targets, axis=-1)
      mask_outputs = tf.expand_dims(mask_outputs, axis=-1)
      mask_loss = self._binary_crossentropy(mask_targets, mask_outputs,
                                            sample_weight=weights)

      # The loss is normalized by the number of 1's in weights and
      # + 0.01 is used to avoid division by zero.
      return mask_loss / (tf.reduce_sum(weights) + 0.01)


class RetinanetClassLoss(object):
  """RetinaNet class loss."""

  def __init__(self, params, num_classes):
    self._num_classes = num_classes
    self._focal_loss_alpha = params.focal_loss_alpha
    self._focal_loss_gamma = params.focal_loss_gamma

  def __call__(self, cls_outputs, labels, num_positives):
    """Computes total detection loss.

    Computes total detection loss including box and class loss from all levels.

    Args:
      cls_outputs: an OrderDict with keys representing levels and values
        representing logits in [batch_size, height, width,
        num_anchors * num_classes].
      labels: the dictionary that returned from dataloader that includes
        class groundturth targets.
      num_positives: number of positive examples in the minibatch.

    Returns:
      an integar tensor representing total class loss.
    """
    # Sums all positives in a batch for normalization and avoids zero
    # num_positives_sum, which would lead to inf loss during training
    num_positives_sum = tf.reduce_sum(input_tensor=num_positives) + 1.0

    cls_losses = []
    for level in cls_outputs.keys():
      cls_losses.append(self.class_loss(
          cls_outputs[level], labels[level], num_positives_sum))
    # Sums per level losses to total loss.
    return tf.add_n(cls_losses)

  def class_loss(self, cls_outputs, cls_targets, num_positives,
                 ignore_label=-2):
    """Computes RetinaNet classification loss."""
    # Onehot encoding for classification labels.
    cls_targets_one_hot = tf.one_hot(cls_targets, self._num_classes)
    bs, height, width, _, _ = cls_targets_one_hot.get_shape().as_list()
    cls_targets_one_hot = tf.reshape(cls_targets_one_hot,
                                     [bs, height, width, -1])
    loss = focal_loss(tf.cast(cls_outputs, dtype=tf.float32),
                      tf.cast(cls_targets_one_hot, dtype=tf.float32),
                      self._focal_loss_alpha,
                      self._focal_loss_gamma,
                      num_positives)

    ignore_loss = tf.where(
        tf.equal(cls_targets, ignore_label),
        tf.zeros_like(cls_targets, dtype=tf.float32),
        tf.ones_like(cls_targets, dtype=tf.float32),
    )
    ignore_loss = tf.expand_dims(ignore_loss, -1)
    ignore_loss = tf.tile(ignore_loss, [1, 1, 1, 1, self._num_classes])
    ignore_loss = tf.reshape(ignore_loss, tf.shape(input=loss))
    return tf.reduce_sum(input_tensor=ignore_loss * loss)


class RetinanetBoxLoss(object):
  """RetinaNet box loss."""

  def __init__(self, params):
    self._huber_loss = tf.keras.losses.Huber(
        delta=params.huber_loss_delta, reduction=tf.keras.losses.Reduction.SUM)

  def __call__(self, box_outputs, labels, num_positives):
    """Computes box detection loss.

    Computes total detection loss including box and class loss from all levels.

    Args:
      box_outputs: an OrderDict with keys representing levels and values
        representing box regression targets in [batch_size, height, width,
        num_anchors * 4].
      labels: the dictionary that returned from dataloader that includes
        box groundturth targets.
      num_positives: number of positive examples in the minibatch.

    Returns:
      an integer tensor representing total box regression loss.
    """
    # Sums all positives in a batch for normalization and avoids zero
    # num_positives_sum, which would lead to inf loss during training
    num_positives_sum = tf.reduce_sum(input_tensor=num_positives) + 1.0

    box_losses = []
    for level in box_outputs.keys():
      box_targets_l = labels[level]
      box_losses.append(
          self.box_loss(box_outputs[level], box_targets_l, num_positives_sum))
    # Sums per level losses to total loss.
    return tf.add_n(box_losses)

  def box_loss(self, box_outputs, box_targets, num_positives):
    """Computes RetinaNet box regression loss."""
    # The delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
    normalizer = num_positives * 4.0
    mask = tf.cast(tf.not_equal(box_targets, 0.0), dtype=tf.float32)
    box_targets = tf.expand_dims(box_targets, axis=-1)
    box_outputs = tf.expand_dims(box_outputs, axis=-1)
    box_loss = self._huber_loss(box_targets, box_outputs, sample_weight=mask)
    box_loss /= normalizer
    return box_loss


class ShapemaskMseLoss(object):
  """ShapeMask mask Mean Squared Error loss function wrapper."""

  def __call__(self, probs, labels, valid_mask):
    """Compute instance segmentation loss.

    Args:
      probs: A Tensor of shape [batch_size * num_points, height, width,
        num_classes]. The logits are not necessarily between 0 and 1.
      labels: A float32/float16 Tensor of shape [batch_size, num_instances,
          mask_size, mask_size], where mask_size =
          mask_crop_size * gt_upsample_scale for fine mask, or mask_crop_size
          for coarse masks and shape priors.
      valid_mask: a binary mask indicating valid training masks.

    Returns:
      loss: an float tensor representing total mask classification loss.
    """
    with tf.name_scope('shapemask_prior_loss'):
      batch_size, num_instances = valid_mask.get_shape().as_list()[:2]
      diff = (tf.cast(labels, dtype=tf.float32) -
              tf.cast(probs, dtype=tf.float32))
      diff *= tf.cast(
          tf.reshape(valid_mask, [batch_size, num_instances, 1, 1]),
          tf.float32)
      # Adding 0.001 in the denominator to avoid division by zero.
      loss = tf.nn.l2_loss(diff) / (tf.reduce_sum(labels) + 0.001)
    return loss


class ShapemaskLoss(object):
  """ShapeMask mask loss function wrapper."""

  def __init__(self):
    self._binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM, from_logits=True)

  def __call__(self, logits, labels, valid_mask):
    """ShapeMask mask cross entropy loss function wrapper.

    Args:
      logits: A Tensor of shape [batch_size * num_instances, height, width,
        num_classes]. The logits are not necessarily between 0 and 1.
      labels: A float16/float32 Tensor of shape [batch_size, num_instances,
        mask_size, mask_size], where mask_size =
        mask_crop_size * gt_upsample_scale for fine mask, or mask_crop_size
        for coarse masks and shape priors.
      valid_mask: a binary mask of shape [batch_size, num_instances]
        indicating valid training masks.
    Returns:
      loss: an float tensor representing total mask classification loss.
    """
    with tf.name_scope('shapemask_loss'):
      batch_size, num_instances = valid_mask.get_shape().as_list()[:2]
      labels = tf.cast(labels, tf.float32)
      logits = tf.cast(logits, tf.float32)
      loss = self._binary_crossentropy(labels, logits)
      loss *= tf.cast(tf.reshape(
          valid_mask, [batch_size, num_instances, 1, 1]), loss.dtype)
      # Adding 0.001 in the denominator to avoid division by zero.
      loss = tf.reduce_sum(loss) / (tf.reduce_sum(labels) + 0.001)
    return loss
