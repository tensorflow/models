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

"""Losses for maskrcn model."""

# Import libraries
import tensorflow as tf


class RpnScoreLoss(object):
  """Region Proposal Network score loss function."""

  def __init__(self, rpn_batch_size_per_im):
    self._rpn_batch_size_per_im = rpn_batch_size_per_im
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
                    self._rpn_batch_size_per_im,
                    dtype=tf.float32)))

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

  def __init__(self, huber_loss_delta: float):
    # The delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].
    self._huber_loss = tf.keras.losses.Huber(
        delta=huber_loss_delta, reduction=tf.keras.losses.Reduction.SUM)

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


class FastrcnnClassLoss(object):
  """Fast R-CNN classification loss function."""

  def __init__(self, use_binary_cross_entropy=False):
    """Initializes loss computation.

    Args:
      use_binary_cross_entropy: If true, uses binary cross entropy loss,
        otherwise uses categorical cross entropy loss.
    """
    self._use_binary_cross_entropy = use_binary_cross_entropy

  def __call__(self, class_outputs, class_targets):
    """Computes the class loss (Fast-RCNN branch) of Mask-RCNN.

    This function implements the classification loss of the Fast-RCNN.

    The classification loss is categorical (or binary) cross entropy on all
    RoIs.
    Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/fast_rcnn_heads.py  # pylint: disable=line-too-long

    Args:
      class_outputs: a float tensor representing the class prediction for each
        box with a shape of [batch_size, num_boxes, num_classes].
      class_targets: a float tensor representing the class label for each box
        with a shape of [batch_size, num_boxes].

    Returns:
      a scalar tensor representing total class loss.
    """
    with tf.name_scope('fast_rcnn_loss'):
      num_classes = class_outputs.get_shape().as_list()[-1]
      class_targets_one_hot = tf.one_hot(
          tf.cast(class_targets, dtype=tf.int32), num_classes)
      if self._use_binary_cross_entropy:
        cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            class_targets_one_hot, class_outputs)
        return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, axis=-1))
      else:
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(class_targets_one_hot,
                                                    class_outputs))


class FastrcnnBoxLoss(object):
  """Fast R-CNN box regression loss function."""

  def __init__(self,
               huber_loss_delta: float,
               class_agnostic_bbox_pred: bool = False):
    """Initiate Faster RCNN box loss.

    Args:
      huber_loss_delta: the delta is typically around the mean value of
        regression target. for instances, the regression targets of 512x512
        input with 6 anchors on P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].
      class_agnostic_bbox_pred: if True, class agnostic bounding box prediction
        is performed.
    """
    self._huber_loss = tf.keras.losses.Huber(
        delta=huber_loss_delta, reduction=tf.keras.losses.Reduction.SUM)
    self._class_agnostic_bbox_pred = class_agnostic_bbox_pred

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
      if not self._class_agnostic_bbox_pred:
        box_outputs = self._assign_class_targets(box_outputs, class_targets)

      return self._fast_rcnn_box_loss(box_outputs, box_targets, class_targets)

  def _assign_class_targets(self, box_outputs, class_targets):
    """Selects the box from `box_outputs` based on `class_targets`, with which the box has the maximum overlap."""
    (batch_size, num_rois,
     num_class_specific_boxes) = box_outputs.get_shape().as_list()
    num_classes = num_class_specific_boxes // 4
    box_outputs = tf.reshape(box_outputs,
                             [batch_size, num_rois, num_classes, 4])
    class_targets_ont_hot = tf.one_hot(
        class_targets, num_classes, dtype=box_outputs.dtype)
    return tf.einsum('bnij,bni->bnj', box_outputs, class_targets_ont_hot)

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
