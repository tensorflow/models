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

"""Losses for maskconver model."""

import functools
import tensorflow as tf, tf_keras

LARGE_NUM = 1e9


class PenaltyReducedLogisticFocalLoss(object):
  """Penalty-reduced pixelwise logistic regression with focal loss."""

  def __init__(self, alpha=2.0, beta=4.0, sigmoid_clip_value=1e-4):
    """Constructor.

    The loss is defined in Equation (1) of the Objects as Points[1] paper.
    Although the loss is defined per-pixel in the output space, this class
    assumes that each pixel is an anchor to be compatible with the base class.

    [1]: https://arxiv.org/abs/1904.07850

    Args:
      alpha: Focussing parameter of the focal loss. Increasing this will
        decrease the loss contribution of the well classified examples.
      beta: The local penalty reduction factor. Increasing this will decrease
        the contribution of loss due to negative pixels near the keypoint.
      sigmoid_clip_value: The sigmoid operation used internally will be clipped
        between [sigmoid_clip_value, 1 - sigmoid_clip_value)
    """
    self._alpha = alpha
    self._beta = beta
    self._sigmoid_clip_value = sigmoid_clip_value
    super().__init__()

  def __call__(self, prediction_tensor, target_tensor, weights=1.0):
    """Compute loss function.

    In all input tensors, `num_anchors` is the total number of pixels in the
    the output space.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted unscaled logits for each class.
        The function will compute sigmoid on this tensor internally.
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing a tensor with the 'splatted' keypoints,
        possibly using a gaussian kernel. This function assumes that
        the target is bounded between [0, 1].
      weights: a float tensor of shape, either [batch_size, num_anchors,
        num_classes] or [batch_size, num_anchors, 1]. If the shape is
        [batch_size, num_anchors, 1], all the classses are equally weighted.

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors, num_classes]
        representing the value of the loss function.
    """
    with tf.name_scope('prlf_loss'):
      ignore = tf.cast(tf.math.equal(target_tensor, -1.0), tf.float32)
      is_present_tensor = tf.math.equal(target_tensor, 1.0)
      prediction_tensor = tf.clip_by_value(
          tf.sigmoid(prediction_tensor),
          self._sigmoid_clip_value, 1 - self._sigmoid_clip_value)

      positive_loss = (tf.math.pow((1 - prediction_tensor), self._alpha) *
                       tf.math.log(prediction_tensor))
      negative_loss = (tf.math.pow((1 - target_tensor), self._beta) *
                       tf.math.pow(prediction_tensor, self._alpha) *
                       tf.math.log(1 - prediction_tensor))

      loss = -tf.where(is_present_tensor, positive_loss, negative_loss)
      return loss * weights * (1 - ignore)


class EmbedLoss:
  """Embedding loss class."""

  def __init__(
      self, projection_norm: bool = True,
      temperature: float = 0.1,
      max_num_detections: int = 100,
      num_classes: int = 91,
      class_agnostic: bool = False):
    """Initializes `ContrastiveLoss`."""
    self._projection_norm = projection_norm
    self._temperature = temperature
    self._max_num_detections = max_num_detections
    self._num_classes = num_classes
    self._class_agnostic = class_agnostic

  def __call__(self, embed_outputs, matched_gt_indices, matched_gt_classes):
    """Computes the embedding loss."""
    with tf.name_scope('feature_nms_loss'):
      # Get (normalized) hidden1 and hidden2.
      if self._projection_norm:
        embed_outputs = tf.math.l2_normalize(embed_outputs, -1)
      batch_size = tf.shape(embed_outputs)[0]
      num_boxes = tf.shape(embed_outputs)[1]
      matched_gt_indices = tf.cast(matched_gt_indices, tf.int32)
      if not self._class_agnostic:
        matched_gt_classes = tf.cast(matched_gt_classes, tf.int32)

      # labels = [batch_size, num_boxes, max_num_detections]
      labels = tf.one_hot(matched_gt_indices, self._max_num_detections)
      ta_matmul = functools.partial(tf.matmul, transpose_a=True)
      # num_boxes_per_index = [batch_size, max_num_detections]
      num_boxes_per_index = tf.reduce_sum(labels, axis=1)

      # mean_emb = [batch_size, max_num_detections, embed_dim]
      mean_emb = tf.stop_gradient(tf.math.divide_no_nan(
          ta_matmul(labels, embed_outputs),
          tf.expand_dims(num_boxes_per_index, -1)))

      if self._projection_norm:
        mean_emb = tf.math.l2_normalize(mean_emb, -1)

      tb_matmul = functools.partial(tf.matmul, transpose_b=True)
      # logits = [batch_size, num_boxes, max_num_detections]
      logits = tb_matmul(embed_outputs, mean_emb) / self._temperature

      mask = tf.ones([batch_size, num_boxes, self._max_num_detections],
                     tf.int32)
      if not self._class_agnostic:
        # Force value "-1"(negative sample)->"0" to be used for indices.
        # These wrong indices will be ignored since there is no proposals with
        # gt_classes = 0 (background)
        classes = tf.one_hot(matched_gt_classes, self._num_classes)
        matched_gt_classes = tf.nn.relu(matched_gt_classes)
        matched_gt_indices = tf.nn.relu(matched_gt_indices)
        batch_indices = tf.ones([batch_size, num_boxes],
                                tf.int32) * tf.expand_dims(
                                    range(batch_size), -1)
        indices = tf.stack(
            [batch_indices, matched_gt_classes, matched_gt_indices], -1)
        # gt_class_indices = [batch_size, num_classes, max_num_detections]
        gt_class_indices = tf.scatter_nd(
            indices, tf.ones([batch_size, num_boxes]),
            [batch_size, self._num_classes, self._max_num_detections])
        # A class which has multiple objects in an image is updated more than
        # once.
        gt_class_indices = tf.clip_by_value(
            gt_class_indices, clip_value_min=0, clip_value_max=1)
        # mask = [batch_size, num_boxes, max_num_detections]
        mask = tf.cast(tf.matmul(classes, gt_class_indices), tf.int32)

      valid_gt_indices = tf.math.count_nonzero(
          labels, axis=1, dtype=tf.int32)
      mask *= tf.expand_dims(valid_gt_indices, 1)

      mask = tf.where(tf.equal(mask, 0), LARGE_NUM, 0.)

      loss_mask = tf.math.count_nonzero(labels, axis=2)
      loss_mask = tf.where(tf.equal(loss_mask, 0), 0., 1.)

      logits = logits - mask

      loss = tf.nn.softmax_cross_entropy_with_logits(
          labels,
          logits)
      loss = tf.reduce_sum(loss * loss_mask) / (tf.reduce_sum(loss_mask) + 1.0)

      return loss
