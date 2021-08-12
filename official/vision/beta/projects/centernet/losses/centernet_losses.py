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
import abc

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


class Loss(abc.ABC):
  """Abstract base class for loss functions."""
  
  def __call__(self,
               prediction_tensor,
               target_tensor,
               ignore_nan_targets=False,
               losses_mask=None,
               scope=None,
               **params):
    """Call the loss function.

    Args:
      prediction_tensor: an N-d tensor of shape [batch, anchors, ...]
        representing predicted quantities.
      target_tensor: an N-d tensor of shape [batch, anchors, ...] representing
        regression or classification targets.
      ignore_nan_targets: whether to ignore nan targets in the loss computation.
        E.g. can be used if the target tensor is missing groundtruth data that
        shouldn't be factored into the loss.
      losses_mask: A [batch] boolean tensor that indicates whether losses should
        be applied to individual images in the batch. For elements that
        are False, corresponding prediction, target, and weight tensors will not
        contribute to loss computation. If None, no filtering will take place
        prior to loss computation.
      scope: Op scope name. Defaults to 'Loss' if None.
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: a tensor representing the value of the loss function.
    """
    scope = self.__class__.__name__ if scope is None else scope
    with tf.name_scope(scope):
      if ignore_nan_targets:
        target_tensor = tf.where(tf.math.is_nan(target_tensor),
                                 prediction_tensor,
                                 target_tensor)
      if losses_mask is not None:
        tensor_multiplier = self._get_loss_multiplier_for_tensor(
            prediction_tensor,
            losses_mask)
        prediction_tensor *= tensor_multiplier
        target_tensor *= tensor_multiplier
        
        if 'weights' in params:
          params['weights'] = tf.convert_to_tensor(params['weights'])
          weights_multiplier = self._get_loss_multiplier_for_tensor(
              params['weights'],
              losses_mask)
          params['weights'] *= weights_multiplier
      return self._compute_loss(prediction_tensor, target_tensor, **params)
  
  def _get_loss_multiplier_for_tensor(self, tensor, losses_mask):
    loss_multiplier_shape = tf.stack([-1] + [1] * (len(tensor.shape) - 1))
    return tf.cast(tf.reshape(losses_mask, loss_multiplier_shape), tf.float32)
  
  @abc.abstractmethod
  def _compute_loss(self, prediction_tensor, target_tensor, **params):
    """Method to be overridden by implementations.

    Args:
      prediction_tensor: a tensor representing predicted quantities
      target_tensor: a tensor representing regression or classification targets
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: an N-d tensor of shape [batch, anchors, ...] containing the loss per
        anchor
    """
    pass


class PenaltyReducedLogisticFocalLoss(Loss):
  """Penalty-reduced pixelwise logistic regression with focal loss.

  The loss is defined in Equation (1) of the Objects as Points[1] paper.
  Although the loss is defined per-pixel in the output space, this class
  assumes that each pixel is an anchor to be compatible with the base class.

  [1]: https://arxiv.org/abs/1904.07850
  """
  
  def __init__(self, alpha=2.0, beta=4.0, sigmoid_clip_value=1e-4):
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
    super(PenaltyReducedLogisticFocalLoss, self).__init__()
  
  def _compute_loss(self, prediction_tensor, target_tensor, weights):
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


class L1LocalizationLoss(Loss):
  """L1 loss or absolute difference.
  When used in a per-pixel manner, each pixel should be given as an anchor.
  """
  
  def _compute_loss(self, prediction_tensor, target_tensor, weights=1.0):
    """Compute loss function.

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
    return absolute_difference(
        labels=target_tensor,
        predictions=prediction_tensor,
        sample_weight=weights,
        reduction=tf.losses.Reduction.NONE
    )
