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
"""Losses used for Mask-RCNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf


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
    positive_label_mask = tf.equal(targets, 1.0)
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
    modulator = tf.exp(gamma * targets * neg_logits -
                       gamma * tf.math.log1p(tf.exp(neg_logits)))
    loss = modulator * cross_entropy
    weighted_loss = tf.where(positive_label_mask, alpha * loss,
                             (1.0 - alpha) * loss)
    weighted_loss /= normalizer
  return weighted_loss


class RpnScoreLoss(object):
  """Region Proposal Network score loss function."""

  def __init__(self, params):
    raise ValueError('Not TF 2.0 ready.')
    self._batch_size = params.batch_size
    self._rpn_batch_size_per_im = params.rpn_batch_size_per_im

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
        score_targets_l = labels['score_targets_%d' % level]
        score_losses.append(
            self._rpn_score_loss(
                score_outputs[level],
                score_targets_l,
                normalizer=tf.cast(
                    self._batch_size * self._rpn_batch_size_per_im,
                    dtype=tf.float32)))

      # Sums per level losses to total loss.
      return tf.add_n(score_losses)

  def _rpn_score_loss(self, score_outputs, score_targets, normalizer=1.0):
    """Computes score loss."""
    # score_targets has three values:
    # (1) score_targets[i]=1, the anchor is a positive sample.
    # (2) score_targets[i]=0, negative.
    # (3) score_targets[i]=-1, the anchor is don't care (ignore).
    with tf.name_scope('rpn_score_loss'):
      mask = tf.logical_or(tf.equal(score_targets, 1),
                           tf.equal(score_targets, 0))
      score_targets = tf.maximum(score_targets, tf.zeros_like(score_targets))
      # RPN score loss is sum over all except ignored samples.
      score_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
          score_targets,
          score_outputs,
          weights=mask,
          reduction=tf.compat.v1.losses.Reduction.SUM)
      score_loss /= normalizer
      return score_loss


class RpnBoxLoss(object):
  """Region Proposal Network box regression loss function."""

  def __init__(self, params):
    raise ValueError('Not TF 2.0 ready.')
    self._delta = params.huber_loss_delta

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
    with tf.compat.v1.name_scope('rpn_loss'):
      levels = sorted(box_outputs.keys())

      box_losses = []
      for level in levels:
        box_targets_l = labels['box_targets_%d' % level]
        box_losses.append(
            self._rpn_box_loss(
                box_outputs[level], box_targets_l, delta=self._delta))

      # Sum per level losses to total loss.
      return tf.add_n(box_losses)

  def _rpn_box_loss(self, box_outputs, box_targets, normalizer=1.0, delta=1./9):
    """Computes box regression loss."""
    # The delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].
    with tf.compat.v1.name_scope('rpn_box_loss'):
      mask = tf.not_equal(box_targets, 0.0)
      # The loss is normalized by the sum of non-zero weights before additional
      # normalizer provided by the function caller.
      box_loss = tf.compat.v1.losses.huber_loss(
          box_targets,
          box_outputs,
          weights=mask,
          delta=delta,
          reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
      box_loss /= normalizer
      return box_loss


class FastrcnnClassLoss(object):
  """Fast R-CNN classification loss function."""

  def __init__(self):
    raise ValueError('Not TF 2.0 ready.')

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
    with tf.compat.v1.name_scope('fast_rcnn_loss'):
      _, _, _, num_classes = class_outputs.get_shape().as_list()
      class_targets = tf.cast(class_targets, dtype=tf.int32)
      class_targets_one_hot = tf.one_hot(class_targets, num_classes)
      return self._fast_rcnn_class_loss(class_outputs, class_targets_one_hot)

  def _fast_rcnn_class_loss(self, class_outputs, class_targets_one_hot,
                            normalizer=1.0):
    """Computes classification loss."""
    with tf.compat.v1.name_scope('fast_rcnn_class_loss'):
      # The loss is normalized by the sum of non-zero weights before additional
      # normalizer provided by the function caller.
      class_loss = tf.compat.v1.losses.softmax_cross_entropy(
          class_targets_one_hot,
          class_outputs,
          reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
      class_loss /= normalizer
      return class_loss


class FastrcnnBoxLoss(object):
  """Fast R-CNN box regression loss function."""

  def __init__(self, params):
    raise ValueError('Not TF 2.0 ready.')
    self._delta = params.huber_loss_delta

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
    with tf.compat.v1.name_scope('fast_rcnn_loss'):
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

      return self._fast_rcnn_box_loss(box_outputs, box_targets, class_targets,
                                      delta=self._delta)

  def _fast_rcnn_box_loss(self, box_outputs, box_targets, class_targets,
                          normalizer=1.0, delta=1.):
    """Computes box regression loss."""
    # The delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].
    with tf.compat.v1.name_scope('fast_rcnn_box_loss'):
      mask = tf.tile(tf.expand_dims(tf.greater(class_targets, 0), axis=2),
                     [1, 1, 4])
      # The loss is normalized by the sum of non-zero weights before additional
      # normalizer provided by the function caller.
      box_loss = tf.compat.v1.losses.huber_loss(
          box_targets,
          box_outputs,
          weights=mask,
          delta=delta,
          reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
      box_loss /= normalizer
      return box_loss


class MaskrcnnLoss(object):
  """Mask R-CNN instance segmentation mask loss function."""

  def __init__(self):
    raise ValueError('Not TF 2.0 ready.')

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
    with tf.compat.v1.name_scope('mask_loss'):
      (batch_size, num_masks, mask_height,
       mask_width) = mask_outputs.get_shape().as_list()

      weights = tf.tile(
          tf.reshape(tf.greater(select_class_targets, 0),
                     [batch_size, num_masks, 1, 1]),
          [1, 1, mask_height, mask_width])
      return tf.compat.v1.losses.sigmoid_cross_entropy(
          mask_targets,
          mask_outputs,
          weights=weights,
          reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)


class RetinanetClassLoss(object):
  """RetinaNet class loss."""

  def __init__(self, params):
    self._num_classes = params.num_classes
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
    loss = focal_loss(cls_outputs, cls_targets_one_hot,
                      self._focal_loss_alpha, self._focal_loss_gamma,
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
      an integar tensor representing total box regression loss.
    """
    # Sums all positives in a batch for normalization and avoids zero
    # num_positives_sum, which would lead to inf loss during training
    num_positives_sum = tf.reduce_sum(input_tensor=num_positives) + 1.0

    box_losses = []
    for level in box_outputs.keys():
      # Onehot encoding for classification labels.
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
    mask = tf.not_equal(box_targets, 0.0)
    box_loss = self._huber_loss(box_targets, box_outputs, sample_weight=mask)
    box_loss /= normalizer
    return box_loss


class ShapeMaskLoss(object):
  """ShapeMask mask loss function wrapper."""

  def __init__(self):
    raise ValueError('Not TF 2.0 ready.')

  def __call__(self, logits, scaled_labels, classes,
               category_loss=True, mse_loss=False):
    """Compute instance segmentation loss.

    Args:
      logits: A Tensor of shape [batch_size * num_points, height, width,
        num_classes]. The logits are not necessarily between 0 and 1.
      scaled_labels: A float16 Tensor of shape [batch_size, num_instances,
          mask_size, mask_size], where mask_size =
          mask_crop_size * gt_upsample_scale for fine mask, or mask_crop_size
          for coarse masks and shape priors.
      classes: A int tensor of shape [batch_size, num_instances].
      category_loss: use class specific mask prediction or not.
      mse_loss: use mean square error for mask loss or not

    Returns:
      mask_loss: an float tensor representing total mask classification loss.
      iou: a float tensor representing the IoU between target and prediction.
    """
    classes = tf.reshape(classes, [-1])
    _, _, height, width = scaled_labels.get_shape().as_list()
    scaled_labels = tf.reshape(scaled_labels, [-1, height, width])

    if not category_loss:
      logits = logits[:, :, :, 0]
    else:
      logits = tf.transpose(a=logits, perm=(0, 3, 1, 2))
      gather_idx = tf.stack([tf.range(tf.size(input=classes)), classes - 1],
                            axis=1)
      logits = tf.gather_nd(logits, gather_idx)

    # Ignore loss on empty mask targets.
    valid_labels = tf.reduce_any(
        input_tensor=tf.greater(scaled_labels, 0), axis=[1, 2])
    if mse_loss:
      # Logits are probabilities in the case of shape prior prediction.
      logits *= tf.reshape(
          tf.cast(valid_labels, logits.dtype), [-1, 1, 1])
      weighted_loss = tf.nn.l2_loss(scaled_labels - logits)
      probs = logits
    else:
      weighted_loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=scaled_labels, logits=logits)
      probs = tf.sigmoid(logits)
      weighted_loss *= tf.reshape(
          tf.cast(valid_labels, weighted_loss.dtype), [-1, 1, 1])

    iou = tf.reduce_sum(
        input_tensor=tf.minimum(scaled_labels, probs)) / tf.reduce_sum(
            input_tensor=tf.maximum(scaled_labels, probs))
    mask_loss = tf.reduce_sum(input_tensor=weighted_loss) / tf.reduce_sum(
        input_tensor=scaled_labels)
    return tf.cast(mask_loss, tf.float32), tf.cast(iou, tf.float32)
