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

"""Losses for centernet model."""

# Import libraries
import tensorflow as tf

from tensorflow.python.keras.utils import losses_utils


def absolute_difference(
    labels,
    predictions,
    sample_weight=None,
    reduction=tf.keras.losses.Reduction.NONE):
  """Adds an Absolute Difference loss to the training procedure.
  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a `Tensor` of
  shape `[batch_size]`, then the total loss for each sample of the batch is
  rescaled by the corresponding element in the `weights` vector. If the shape
  of `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.
  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    sample_weight: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions
      must be either `1`, or the same as the corresponding `losses`
      dimension).
    reduction: Type of reduction to apply to loss.
  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.
  Raises:
    ValueError: If the shape of `predictions` doesn't match that of
      `labels` or if the shape of `weights` is invalid or if `labels`
      or `predictions` is None.
  """
  if labels is None:
    raise ValueError("labels must not be None.")
  if predictions is None:
    raise ValueError("predictions must not be None.")
  with tf.name_scope("absolute_difference"):
    predictions = tf.cast(predictions, dtype=tf.float32)
    labels = tf.cast(labels, dtype=tf.float32)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    losses = tf.abs(tf.subtract(predictions, labels))
    return losses_utils.compute_weighted_loss(
        losses=losses,
        sample_weight=sample_weight,
        reduction=reduction)


class PenaltyReducedLogisticFocalLoss(tf.keras.losses.Loss):
  """Penalty-reduced pixelwise logistic regression with focal loss.
  The loss is defined in Equation (1) of the Objects as Points[1] paper.
  Although the loss is defined per-pixel in the output space, this class
  assumes that each pixel is an anchor to be compatible with the base class.
  [1]: https://arxiv.org/abs/1904.07850
  """
  
  def __init__(self,
               alpha=2.0,
               beta=4.0,
               sigmoid_clip_value=1e-4,
               reduction=tf.keras.losses.Reduction.AUTO,
               name=None):
    """Constructor.
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
    super(PenaltyReducedLogisticFocalLoss, self).__init__(reduction=reduction,
                                                          name=name)
  
  def call(self, y_true, y_pred):
    """Compute loss function.
    In all input tensors, `num_anchors` is the total number of pixels in the
    the output space.
    Args:
      y_true: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted unscaled logits for each class.
        The function will compute sigmoid on this tensor internally.
      y_pred: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing a tensor with the 'splatted' keypoints,
        possibly using a gaussian kernel. This function assumes that
        the target is bounded between [0, 1].
    Returns:
      loss: a float tensor of shape [batch_size, num_anchors, num_classes]
        representing the value of the loss function.
    """
    
    target_tensor = y_true
    
    is_present_tensor = tf.math.equal(target_tensor, 1.0)
    prediction_tensor = tf.clip_by_value(tf.sigmoid(y_pred),
                                         self._sigmoid_clip_value,
                                         1 - self._sigmoid_clip_value)
    positive_loss = (tf.math.pow((1 - prediction_tensor), self._alpha) *
                     tf.math.log(prediction_tensor))
    negative_loss = (tf.math.pow((1 - target_tensor), self._beta) *
                     tf.math.pow(prediction_tensor, self._alpha) *
                     tf.math.log(1 - prediction_tensor))
    
    loss = -tf.where(is_present_tensor, positive_loss, negative_loss)
    return loss
  
  def get_config(self):
    """Returns the config dictionary for a `Loss` instance."""
    return {
        'alpha': self._alpha,
        'beta': self._beta,
        'sigmoid_clip_value': self._sigmoid_clip_value,
        **super(PenaltyReducedLogisticFocalLoss, self).get_config()
    }


class L1LocalizationLoss(tf.keras.losses.Loss):
  """L1 loss or absolute difference.
  When used in a per-pixel manner, each pixel should be given as an anchor.
  """
  
  def __call__(self, y_true, y_pred, sample_weight=None):
    """Compute loss function.
    Args:
      y_true: A float tensor of shape [batch_size, num_anchors]
        representing the regression targets
      y_pred: A float tensor of shape [batch_size, num_anchors]
        representing the (encoded) predicted locations of objects.
      sample_weight: a float tensor of shape [batch_size, num_anchors]
    Returns:
      loss: a float tensor of shape [batch_size, num_anchors] tensor
        representing the value of the loss function.
    """
    return absolute_difference(
        labels=y_true,
        predictions=y_pred,
        sample_weight=sample_weight,
        reduction=self._get_reduction()
    )
  
  call = __call__
