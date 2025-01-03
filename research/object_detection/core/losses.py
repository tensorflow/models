# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Classification and regression loss functions for object detection.

Localization losses:
 * WeightedL2LocalizationLoss
 * WeightedSmoothL1LocalizationLoss
 * WeightedIOULocalizationLoss

Classification losses:
 * WeightedSigmoidClassificationLoss
 * WeightedSoftmaxClassificationLoss
 * WeightedSoftmaxClassificationAgainstLogitsLoss
 * BootstrappedSigmoidClassificationLoss
 * WeightedDiceClassificationLoss
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import tensorflow.compat.v1 as tf
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.utils import ops
from object_detection.utils import shape_utils


class Loss(six.with_metaclass(abc.ABCMeta, object)):
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
    with tf.name_scope(scope, 'Loss',
                       [prediction_tensor, target_tensor, params]) as scope:
      if ignore_nan_targets:
        target_tensor = tf.where(tf.is_nan(target_tensor),
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


class WeightedL2LocalizationLoss(Loss):
  """L2 localization loss function with anchorwise output support.

  Loss[b,a] = .5 * ||weights[b,a] * (prediction[b,a,:] - target[b,a,:])||^2
  """

  def _compute_loss(self, prediction_tensor, target_tensor, weights):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the (encoded) predicted locations of objects.
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the regression targets
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors] tensor
        representing the value of the loss function.
    """
    weighted_diff = (prediction_tensor - target_tensor) * tf.expand_dims(
        weights, 2)
    square_diff = 0.5 * tf.square(weighted_diff)
    return tf.reduce_sum(square_diff, 2)


class WeightedSmoothL1LocalizationLoss(Loss):
  """Smooth L1 localization loss function aka Huber Loss..

  The smooth L1_loss is defined elementwise as .5 x^2 if |x| <= delta and
  delta * (|x|- 0.5*delta) otherwise, where x is the difference between
  predictions and target.

  See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
  """

  def __init__(self, delta=1.0):
    """Constructor.

    Args:
      delta: delta for smooth L1 loss.
    """
    super(WeightedSmoothL1LocalizationLoss, self).__init__()
    self._delta = delta

  def _compute_loss(self, prediction_tensor, target_tensor, weights):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the (encoded) predicted locations of objects.
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the regression targets
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors] tensor
        representing the value of the loss function.
    """
    return tf.reduce_sum(tf.losses.huber_loss(
        target_tensor,
        prediction_tensor,
        delta=self._delta,
        weights=tf.expand_dims(weights, axis=2),
        loss_collection=None,
        reduction=tf.losses.Reduction.NONE
    ), axis=2)


class WeightedIOULocalizationLoss(Loss):
  """IOU localization loss function.

  Sums the IOU for corresponding pairs of predicted/groundtruth boxes
  and for each pair assign a loss of 1 - IOU.  We then compute a weighted
  sum over all pairs which is returned as the total loss.
  """

  def _compute_loss(self, prediction_tensor, target_tensor, weights):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors, 4]
        representing the decoded predicted boxes
      target_tensor: A float tensor of shape [batch_size, num_anchors, 4]
        representing the decoded target boxes
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors] tensor
        representing the value of the loss function.
    """
    predicted_boxes = box_list.BoxList(tf.reshape(prediction_tensor, [-1, 4]))
    target_boxes = box_list.BoxList(tf.reshape(target_tensor, [-1, 4]))
    per_anchor_iou_loss = 1.0 - box_list_ops.matched_iou(predicted_boxes,
                                                         target_boxes)
    return tf.reshape(weights, [-1]) * per_anchor_iou_loss


class WeightedGIOULocalizationLoss(Loss):
  """GIOU localization loss function.

  Sums the GIOU loss for corresponding pairs of predicted/groundtruth boxes
  and for each pair assign a loss of 1 - GIOU.  We then compute a weighted
  sum over all pairs which is returned as the total loss.
  """

  def _compute_loss(self, prediction_tensor, target_tensor, weights):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors, 4]
        representing the decoded predicted boxes
      target_tensor: A float tensor of shape [batch_size, num_anchors, 4]
        representing the decoded target boxes
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors] tensor
        representing the value of the loss function.
    """
    batch_size, num_anchors, _ = shape_utils.combined_static_and_dynamic_shape(
        prediction_tensor)
    predicted_boxes = tf.reshape(prediction_tensor, [-1, 4])
    target_boxes = tf.reshape(target_tensor, [-1, 4])

    per_anchor_iou_loss = 1 - ops.giou(predicted_boxes, target_boxes)
    return tf.reshape(tf.reshape(weights, [-1]) * per_anchor_iou_loss,
                      [batch_size, num_anchors])


class WeightedSigmoidClassificationLoss(Loss):
  """Sigmoid cross entropy classification loss function."""

  def _compute_loss(self,
                    prediction_tensor,
                    target_tensor,
                    weights,
                    class_indices=None):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape, either [batch_size, num_anchors,
        num_classes] or [batch_size, num_anchors, 1]. If the shape is
        [batch_size, num_anchors, 1], all the classses are equally weighted.
      class_indices: (Optional) A 1-D integer tensor of class indices.
        If provided, computes loss only for the specified class indices.

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors, num_classes]
        representing the value of the loss function.
    """
    if class_indices is not None:
      weights *= tf.reshape(
          ops.indices_to_dense_vector(class_indices,
                                      tf.shape(prediction_tensor)[2]),
          [1, 1, -1])
    per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target_tensor, logits=prediction_tensor))
    return per_entry_cross_ent * weights


class WeightedDiceClassificationLoss(Loss):
  """Dice loss for classification [1][2].

  [1]: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
  [2]: https://arxiv.org/abs/1606.04797

  """

  def __init__(self, squared_normalization, is_prediction_probability=False):
    """Initializes the loss object.

    Args:
      squared_normalization: boolean, if set, we square the probabilities in the
        denominator term used for normalization.
      is_prediction_probability: boolean, whether or not the input
        prediction_tensor represents a probability. If false, it is
        first converted to a probability by applying sigmoid.
    """

    self._squared_normalization = squared_normalization
    self.is_prediction_probability = is_prediction_probability
    super(WeightedDiceClassificationLoss, self).__init__()

  def _compute_loss(self,
                    prediction_tensor,
                    target_tensor,
                    weights,
                    class_indices=None):
    """Computes the loss value.

    Dice loss uses the area of the ground truth and prediction tensors for
    normalization. We compute area by summing along the anchors (2nd) dimension.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_pixels,
        num_classes] representing the predicted logits for each class.
        num_pixels denotes the total number of pixels in the spatial dimensions
        of the mask after flattening.
      target_tensor: A float tensor of shape [batch_size, num_pixels,
        num_classes] representing one-hot encoded classification targets.
        num_pixels denotes the total number of pixels in the spatial dimensions
        of the mask after flattening.
      weights: a float tensor of shape, either [batch_size, num_anchors,
        num_classes] or [batch_size, num_anchors, 1]. If the shape is
        [batch_size, num_anchors, 1], all the classses are equally weighted.
      class_indices: (Optional) A 1-D integer tensor of class indices.
        If provided, computes loss only for the specified class indices.

    Returns:
      loss: a float tensor of shape [batch_size, num_classes]
        representing the value of the loss function.
    """
    if class_indices is not None:
      weights *= tf.reshape(
          ops.indices_to_dense_vector(class_indices,
                                      tf.shape(prediction_tensor)[2]),
          [1, 1, -1])

    if self.is_prediction_probability:
      prob_tensor = prediction_tensor
    else:
      prob_tensor = tf.nn.sigmoid(prediction_tensor)

    if self._squared_normalization:
      prob_tensor = tf.pow(prob_tensor, 2)
      target_tensor = tf.pow(target_tensor, 2)

    prob_tensor *= weights
    target_tensor *= weights

    prediction_area = tf.reduce_sum(prob_tensor, axis=1)
    gt_area = tf.reduce_sum(target_tensor, axis=1)

    intersection = tf.reduce_sum(prob_tensor * target_tensor, axis=1)
    dice_coeff = 2 * intersection / tf.maximum(gt_area + prediction_area, 1.0)
    dice_loss = 1 - dice_coeff

    return dice_loss


class SigmoidFocalClassificationLoss(Loss):
  """Sigmoid focal cross entropy loss.

  Focal loss down-weights well classified examples and focusses on the hard
  examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
  """

  def __init__(self, gamma=2.0, alpha=0.25):
    """Constructor.

    Args:
      gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
      alpha: optional alpha weighting factor to balance positives vs negatives.
    """
    super(SigmoidFocalClassificationLoss, self).__init__()
    self._alpha = alpha
    self._gamma = gamma

  def _compute_loss(self,
                    prediction_tensor,
                    target_tensor,
                    weights,
                    class_indices=None):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape, either [batch_size, num_anchors,
        num_classes] or [batch_size, num_anchors, 1]. If the shape is
        [batch_size, num_anchors, 1], all the classses are equally weighted.
      class_indices: (Optional) A 1-D integer tensor of class indices.
        If provided, computes loss only for the specified class indices.

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors, num_classes]
        representing the value of the loss function.
    """
    if class_indices is not None:
      weights *= tf.reshape(
          ops.indices_to_dense_vector(class_indices,
                                      tf.shape(prediction_tensor)[2]),
          [1, 1, -1])
    per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target_tensor, logits=prediction_tensor))
    prediction_probabilities = tf.sigmoid(prediction_tensor)
    p_t = ((target_tensor * prediction_probabilities) +
           ((1 - target_tensor) * (1 - prediction_probabilities)))
    modulating_factor = 1.0
    if self._gamma:
      modulating_factor = tf.pow(1.0 - p_t, self._gamma)
    alpha_weight_factor = 1.0
    if self._alpha is not None:
      alpha_weight_factor = (target_tensor * self._alpha +
                             (1 - target_tensor) * (1 - self._alpha))
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                per_entry_cross_ent)
    return focal_cross_entropy_loss * weights


class WeightedSoftmaxClassificationLoss(Loss):
  """Softmax loss function."""

  def __init__(self, logit_scale=1.0):
    """Constructor.

    Args:
      logit_scale: When this value is high, the prediction is "diffused" and
                   when this value is low, the prediction is made peakier.
                   (default 1.0)

    """
    super(WeightedSoftmaxClassificationLoss, self).__init__()
    self._logit_scale = logit_scale

  def _compute_loss(self, prediction_tensor, target_tensor, weights):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape, either [batch_size, num_anchors,
        num_classes] or [batch_size, num_anchors, 1]. If the shape is
        [batch_size, num_anchors, 1], all the classses are equally weighted.

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors]
        representing the value of the loss function.
    """
    weights = tf.reduce_mean(weights, axis=2)
    num_classes = prediction_tensor.get_shape().as_list()[-1]
    prediction_tensor = tf.divide(
        prediction_tensor, self._logit_scale, name='scale_logit')
    per_row_cross_ent = (tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(target_tensor, [-1, num_classes]),
        logits=tf.reshape(prediction_tensor, [-1, num_classes])))
    return tf.reshape(per_row_cross_ent, tf.shape(weights)) * weights


class WeightedSoftmaxClassificationAgainstLogitsLoss(Loss):
  """Softmax loss function against logits.

   Targets are expected to be provided in logits space instead of "one hot" or
   "probability distribution" space.
  """

  def __init__(self, logit_scale=1.0):
    """Constructor.

    Args:
      logit_scale: When this value is high, the target is "diffused" and
                   when this value is low, the target is made peakier.
                   (default 1.0)

    """
    super(WeightedSoftmaxClassificationAgainstLogitsLoss, self).__init__()
    self._logit_scale = logit_scale

  def _scale_and_softmax_logits(self, logits):
    """Scale logits then apply softmax."""
    scaled_logits = tf.divide(logits, self._logit_scale, name='scale_logits')
    return tf.nn.softmax(scaled_logits, name='convert_scores')

  def _compute_loss(self, prediction_tensor, target_tensor, weights):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing logit classification targets
      weights: a float tensor of shape, either [batch_size, num_anchors,
        num_classes] or [batch_size, num_anchors, 1]. If the shape is
        [batch_size, num_anchors, 1], all the classses are equally weighted.

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors]
        representing the value of the loss function.
    """
    weights = tf.reduce_mean(weights, axis=2)
    num_classes = prediction_tensor.get_shape().as_list()[-1]
    target_tensor = self._scale_and_softmax_logits(target_tensor)
    prediction_tensor = tf.divide(prediction_tensor, self._logit_scale,
                                  name='scale_logits')

    per_row_cross_ent = (tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(target_tensor, [-1, num_classes]),
        logits=tf.reshape(prediction_tensor, [-1, num_classes])))
    return tf.reshape(per_row_cross_ent, tf.shape(weights)) * weights


class BootstrappedSigmoidClassificationLoss(Loss):
  """Bootstrapped sigmoid cross entropy classification loss function.

  This loss uses a convex combination of training labels and the current model's
  predictions as training targets in the classification loss. The idea is that
  as the model improves over time, its predictions can be trusted more and we
  can use these predictions to mitigate the damage of noisy/incorrect labels,
  because incorrect labels are likely to be eventually highly inconsistent with
  other stimuli predicted to have the same label by the model.

  In "soft" bootstrapping, we use all predicted class probabilities, whereas in
  "hard" bootstrapping, we use the single class favored by the model.

  See also Training Deep Neural Networks On Noisy Labels with Bootstrapping by
  Reed et al. (ICLR 2015).
  """

  def __init__(self, alpha, bootstrap_type='soft'):
    """Constructor.

    Args:
      alpha: a float32 scalar tensor between 0 and 1 representing interpolation
        weight
      bootstrap_type: set to either 'hard' or 'soft' (default)

    Raises:
      ValueError: if bootstrap_type is not either 'hard' or 'soft'
    """
    super(BootstrappedSigmoidClassificationLoss, self).__init__()
    if bootstrap_type != 'hard' and bootstrap_type != 'soft':
      raise ValueError('Unrecognized bootstrap_type: must be one of '
                       '\'hard\' or \'soft.\'')
    self._alpha = alpha
    self._bootstrap_type = bootstrap_type

  def _compute_loss(self, prediction_tensor, target_tensor, weights):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape, either [batch_size, num_anchors,
        num_classes] or [batch_size, num_anchors, 1]. If the shape is
        [batch_size, num_anchors, 1], all the classses are equally weighted.

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors, num_classes]
        representing the value of the loss function.
    """
    if self._bootstrap_type == 'soft':
      bootstrap_target_tensor = self._alpha * target_tensor + (
          1.0 - self._alpha) * tf.sigmoid(prediction_tensor)
    else:
      bootstrap_target_tensor = self._alpha * target_tensor + (
          1.0 - self._alpha) * tf.cast(
              tf.sigmoid(prediction_tensor) > 0.5, tf.float32)
    per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
        labels=bootstrap_target_tensor, logits=prediction_tensor))
    return per_entry_cross_ent * weights


class HardExampleMiner(object):
  """Hard example mining for regions in a list of images.

  Implements hard example mining to select a subset of regions to be
  back-propagated. For each image, selects the regions with highest losses,
  subject to the condition that a newly selected region cannot have
  an IOU > iou_threshold with any of the previously selected regions.
  This can be achieved by re-using a greedy non-maximum suppression algorithm.
  A constraint on the number of negatives mined per positive region can also be
  enforced.

  Reference papers: "Training Region-based Object Detectors with Online
  Hard Example Mining" (CVPR 2016) by Srivastava et al., and
  "SSD: Single Shot MultiBox Detector" (ECCV 2016) by Liu et al.
  """

  def __init__(self,
               num_hard_examples=64,
               iou_threshold=0.7,
               loss_type='both',
               cls_loss_weight=0.05,
               loc_loss_weight=0.06,
               max_negatives_per_positive=None,
               min_negatives_per_image=0):
    """Constructor.

    The hard example mining implemented by this class can replicate the behavior
    in the two aforementioned papers (Srivastava et al., and Liu et al).
    To replicate the A2 paper (Srivastava et al), num_hard_examples is set
    to a fixed parameter (64 by default) and iou_threshold is set to .7 for
    running non-max-suppression the predicted boxes prior to hard mining.
    In order to replicate the SSD paper (Liu et al), num_hard_examples should
    be set to None, max_negatives_per_positive should be 3 and iou_threshold
    should be 1.0 (in order to effectively turn off NMS).

    Args:
      num_hard_examples: maximum number of hard examples to be
        selected per image (prior to enforcing max negative to positive ratio
        constraint).  If set to None, all examples obtained after NMS are
        considered.
      iou_threshold: minimum intersection over union for an example
        to be discarded during NMS.
      loss_type: use only classification losses ('cls', default),
        localization losses ('loc') or both losses ('both').
        In the last case, cls_loss_weight and loc_loss_weight are used to
        compute weighted sum of the two losses.
      cls_loss_weight: weight for classification loss.
      loc_loss_weight: weight for location loss.
      max_negatives_per_positive: maximum number of negatives to retain for
        each positive anchor. By default, num_negatives_per_positive is None,
        which means that we do not enforce a prespecified negative:positive
        ratio.  Note also that num_negatives_per_positives can be a float
        (and will be converted to be a float even if it is passed in otherwise).
      min_negatives_per_image: minimum number of negative anchors to sample for
        a given image. Setting this to a positive number allows sampling
        negatives in an image without any positive anchors and thus not biased
        towards at least one detection per image.
    """
    self._num_hard_examples = num_hard_examples
    self._iou_threshold = iou_threshold
    self._loss_type = loss_type
    self._cls_loss_weight = cls_loss_weight
    self._loc_loss_weight = loc_loss_weight
    self._max_negatives_per_positive = max_negatives_per_positive
    self._min_negatives_per_image = min_negatives_per_image
    if self._max_negatives_per_positive is not None:
      self._max_negatives_per_positive = float(self._max_negatives_per_positive)
    self._num_positives_list = None
    self._num_negatives_list = None

  def __call__(self,
               location_losses,
               cls_losses,
               decoded_boxlist_list,
               match_list=None):
    """Computes localization and classification losses after hard mining.

    Args:
      location_losses: a float tensor of shape [num_images, num_anchors]
        representing anchorwise localization losses.
      cls_losses: a float tensor of shape [num_images, num_anchors]
        representing anchorwise classification losses.
      decoded_boxlist_list: a list of decoded BoxList representing location
        predictions for each image.
      match_list: an optional list of matcher.Match objects encoding the match
        between anchors and groundtruth boxes for each image of the batch,
        with rows of the Match objects corresponding to groundtruth boxes
        and columns corresponding to anchors.  Match objects in match_list are
        used to reference which anchors are positive, negative or ignored.  If
        self._max_negatives_per_positive exists, these are then used to enforce
        a prespecified negative to positive ratio.

    Returns:
      mined_location_loss: a float scalar with sum of localization losses from
        selected hard examples.
      mined_cls_loss: a float scalar with sum of classification losses from
        selected hard examples.
    Raises:
      ValueError: if location_losses, cls_losses and decoded_boxlist_list do
        not have compatible shapes (i.e., they must correspond to the same
        number of images).
      ValueError: if match_list is specified but its length does not match
        len(decoded_boxlist_list).
    """
    mined_location_losses = []
    mined_cls_losses = []
    location_losses = tf.unstack(location_losses)
    cls_losses = tf.unstack(cls_losses)
    num_images = len(decoded_boxlist_list)
    if not match_list:
      match_list = num_images * [None]
    if not len(location_losses) == len(decoded_boxlist_list) == len(cls_losses):
      raise ValueError('location_losses, cls_losses and decoded_boxlist_list '
                       'do not have compatible shapes.')
    if not isinstance(match_list, list):
      raise ValueError('match_list must be a list.')
    if len(match_list) != len(decoded_boxlist_list):
      raise ValueError('match_list must either be None or have '
                       'length=len(decoded_boxlist_list).')
    num_positives_list = []
    num_negatives_list = []
    for ind, detection_boxlist in enumerate(decoded_boxlist_list):
      box_locations = detection_boxlist.get()
      match = match_list[ind]
      image_losses = cls_losses[ind]
      if self._loss_type == 'loc':
        image_losses = location_losses[ind]
      elif self._loss_type == 'both':
        image_losses *= self._cls_loss_weight
        image_losses += location_losses[ind] * self._loc_loss_weight
      if self._num_hard_examples is not None:
        num_hard_examples = self._num_hard_examples
      else:
        num_hard_examples = detection_boxlist.num_boxes()
      selected_indices = tf.image.non_max_suppression(
          box_locations, image_losses, num_hard_examples, self._iou_threshold)
      if self._max_negatives_per_positive is not None and match:
        (selected_indices, num_positives,
         num_negatives) = self._subsample_selection_to_desired_neg_pos_ratio(
             selected_indices, match, self._max_negatives_per_positive,
             self._min_negatives_per_image)
        num_positives_list.append(num_positives)
        num_negatives_list.append(num_negatives)
      mined_location_losses.append(
          tf.reduce_sum(tf.gather(location_losses[ind], selected_indices)))
      mined_cls_losses.append(
          tf.reduce_sum(tf.gather(cls_losses[ind], selected_indices)))
    location_loss = tf.reduce_sum(tf.stack(mined_location_losses))
    cls_loss = tf.reduce_sum(tf.stack(mined_cls_losses))
    if match and self._max_negatives_per_positive:
      self._num_positives_list = num_positives_list
      self._num_negatives_list = num_negatives_list
    return (location_loss, cls_loss)

  def summarize(self):
    """Summarize the number of positives and negatives after mining."""
    if self._num_positives_list and self._num_negatives_list:
      avg_num_positives = tf.reduce_mean(
          tf.cast(self._num_positives_list, dtype=tf.float32))
      avg_num_negatives = tf.reduce_mean(
          tf.cast(self._num_negatives_list, dtype=tf.float32))
      tf.summary.scalar('HardExampleMiner/NumPositives', avg_num_positives)
      tf.summary.scalar('HardExampleMiner/NumNegatives', avg_num_negatives)

  def _subsample_selection_to_desired_neg_pos_ratio(self,
                                                    indices,
                                                    match,
                                                    max_negatives_per_positive,
                                                    min_negatives_per_image=0):
    """Subsample a collection of selected indices to a desired neg:pos ratio.

    This function takes a subset of M indices (indexing into a large anchor
    collection of N anchors where M<N) which are labeled as positive/negative
    via a Match object (matched indices are positive, unmatched indices
    are negative).  It returns a subset of the provided indices retaining all
    positives as well as up to the first K negatives, where:
      K=floor(num_negative_per_positive * num_positives).

    For example, if indices=[2, 4, 5, 7, 9, 10] (indexing into 12 anchors),
    with positives=[2, 5] and negatives=[4, 7, 9, 10] and
    num_negatives_per_positive=1, then the returned subset of indices
    is [2, 4, 5, 7].

    Args:
      indices: An integer tensor of shape [M] representing a collection
        of selected anchor indices
      match: A matcher.Match object encoding the match between anchors and
        groundtruth boxes for a given image, with rows of the Match objects
        corresponding to groundtruth boxes and columns corresponding to anchors.
      max_negatives_per_positive: (float) maximum number of negatives for
        each positive anchor.
      min_negatives_per_image: minimum number of negative anchors for a given
        image. Allow sampling negatives in image without any positive anchors.

    Returns:
      selected_indices: An integer tensor of shape [M'] representing a
        collection of selected anchor indices with M' <= M.
      num_positives: An integer tensor representing the number of positive
        examples in selected set of indices.
      num_negatives: An integer tensor representing the number of negative
        examples in selected set of indices.
    """
    positives_indicator = tf.gather(match.matched_column_indicator(), indices)
    negatives_indicator = tf.gather(match.unmatched_column_indicator(), indices)
    num_positives = tf.reduce_sum(tf.cast(positives_indicator, dtype=tf.int32))
    max_negatives = tf.maximum(
        min_negatives_per_image,
        tf.cast(max_negatives_per_positive *
                tf.cast(num_positives, dtype=tf.float32), dtype=tf.int32))
    topk_negatives_indicator = tf.less_equal(
        tf.cumsum(tf.cast(negatives_indicator, dtype=tf.int32)), max_negatives)
    subsampled_selection_indices = tf.where(
        tf.logical_or(positives_indicator, topk_negatives_indicator))
    num_negatives = tf.size(subsampled_selection_indices) - num_positives
    return (tf.reshape(tf.gather(indices, subsampled_selection_indices), [-1]),
            num_positives, num_negatives)


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

    positive_loss = (tf.math.pow((1 - prediction_tensor), self._alpha)*
                     tf.math.log(prediction_tensor))
    negative_loss = (tf.math.pow((1 - target_tensor), self._beta)*
                     tf.math.pow(prediction_tensor, self._alpha)*
                     tf.math.log(1 - prediction_tensor))

    loss = -tf.where(is_present_tensor, positive_loss, negative_loss)
    return loss * weights


class L1LocalizationLoss(Loss):
  """L1 loss or absolute difference.

  When used in a per-pixel manner, each pixel should be given as an anchor.
  """

  def _compute_loss(self, prediction_tensor, target_tensor, weights):
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
    return tf.losses.absolute_difference(
        target_tensor,
        prediction_tensor,
        weights=weights,
        loss_collection=None,
        reduction=tf.losses.Reduction.NONE
    )
