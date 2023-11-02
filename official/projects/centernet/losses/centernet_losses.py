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

"""Losses for centernet model."""


import tensorflow as tf, tf_keras


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
    super(PenaltyReducedLogisticFocalLoss, self).__init__()

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
      is_present_tensor = tf.math.equal(target_tensor, 1.0)
      prediction_tensor = tf.clip_by_value(tf.sigmoid(prediction_tensor),
                                           self._sigmoid_clip_value,
                                           1 - self._sigmoid_clip_value)

      positive_loss = (tf.math.pow((1 - prediction_tensor), self._alpha) *
                       tf.math.log(prediction_tensor))
      negative_loss = (tf.math.pow((1 - target_tensor), self._beta) *
                       tf.math.pow(prediction_tensor, self._alpha) *
                       tf.math.log(1 - prediction_tensor))

      loss = -tf.where(is_present_tensor, positive_loss, negative_loss)
      return loss * weights


class L1LocalizationLoss(object):
  """L1 loss or absolute difference."""

  def __call__(self, prediction_tensor, target_tensor, weights=1.0):
    """Compute loss function.

    When used in a per-pixel manner, each pixel should be given as an anchor.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors]
        representing the (encoded) predicted locations of objects.
      target_tensor: A float tensor of shape [batch_size, num_anchors]
        representing the regression targets
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors] tensor
        representing the value of the loss function.
    """
    with tf.name_scope('l1l_loss'):
      return tf.compat.v1.losses.absolute_difference(
          labels=target_tensor,
          predictions=prediction_tensor,
          weights=weights,
          reduction=tf.losses.Reduction.NONE
      )
