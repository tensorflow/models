# Lint as: python2, python3
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

"""Mask Head.

Contains Mask prediction head classes for different meta architectures.
All the mask prediction heads have a predict function that receives the
`features` as the first argument and returns `mask_predictions`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from six.moves import range
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

from object_detection.predictors.heads import head
from object_detection.utils import ops

slim = contrib_slim


class MaskRCNNMaskHead(head.Head):
  """Mask RCNN mask prediction head.

  Please refer to Mask RCNN paper:
  https://arxiv.org/abs/1703.06870
  """

  def __init__(self,
               num_classes,
               conv_hyperparams_fn=None,
               mask_height=14,
               mask_width=14,
               mask_prediction_num_conv_layers=2,
               mask_prediction_conv_depth=256,
               masks_are_class_agnostic=False,
               convolve_then_upsample=False):
    """Constructor.

    Args:
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      conv_hyperparams_fn: A function to generate tf-slim arg_scope with
        hyperparameters for convolution ops.
      mask_height: Desired output mask height. The default value is 14.
      mask_width: Desired output mask width. The default value is 14.
      mask_prediction_num_conv_layers: Number of convolution layers applied to
        the image_features in mask prediction branch.
      mask_prediction_conv_depth: The depth for the first conv2d_transpose op
        applied to the image_features in the mask prediction branch. If set
        to 0, the depth of the convolution layers will be automatically chosen
        based on the number of object classes and the number of channels in the
        image features.
      masks_are_class_agnostic: Boolean determining if the mask-head is
        class-agnostic or not.
      convolve_then_upsample: Whether to apply convolutions on mask features
        before upsampling using nearest neighbor resizing. Otherwise, mask
        features are resized to [`mask_height`, `mask_width`] using bilinear
        resizing before applying convolutions.

    Raises:
      ValueError: conv_hyperparams_fn is None.
    """
    super(MaskRCNNMaskHead, self).__init__()
    self._num_classes = num_classes
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._mask_height = mask_height
    self._mask_width = mask_width
    self._mask_prediction_num_conv_layers = mask_prediction_num_conv_layers
    self._mask_prediction_conv_depth = mask_prediction_conv_depth
    self._masks_are_class_agnostic = masks_are_class_agnostic
    self._convolve_then_upsample = convolve_then_upsample
    if conv_hyperparams_fn is None:
      raise ValueError('conv_hyperparams_fn is None.')

  def _get_mask_predictor_conv_depth(self,
                                     num_feature_channels,
                                     num_classes,
                                     class_weight=3.0,
                                     feature_weight=2.0):
    """Computes the depth of the mask predictor convolutions.

    Computes the depth of the mask predictor convolutions given feature channels
    and number of classes by performing a weighted average of the two in
    log space to compute the number of convolution channels. The weights that
    are used for computing the weighted average do not need to sum to 1.

    Args:
      num_feature_channels: An integer containing the number of feature
        channels.
      num_classes: An integer containing the number of classes.
      class_weight: Class weight used in computing the weighted average.
      feature_weight: Feature weight used in computing the weighted average.

    Returns:
      An integer containing the number of convolution channels used by mask
        predictor.
    """
    num_feature_channels_log = math.log(float(num_feature_channels), 2.0)
    num_classes_log = math.log(float(num_classes), 2.0)
    weighted_num_feature_channels_log = (
        num_feature_channels_log * feature_weight)
    weighted_num_classes_log = num_classes_log * class_weight
    total_weight = feature_weight + class_weight
    num_conv_channels_log = round(
        (weighted_num_feature_channels_log + weighted_num_classes_log) /
        total_weight)
    return int(math.pow(2.0, num_conv_channels_log))

  def predict(self, features, num_predictions_per_location=1):
    """Performs mask prediction.

    Args:
      features: A float tensor of shape [batch_size, height, width, channels]
        containing features for a batch of images.
      num_predictions_per_location: Int containing number of predictions per
        location.

    Returns:
      instance_masks: A float tensor of shape
          [batch_size, 1, num_classes, mask_height, mask_width].

    Raises:
      ValueError: If num_predictions_per_location is not 1.
    """
    if num_predictions_per_location != 1:
      raise ValueError('Only num_predictions_per_location=1 is supported')
    num_conv_channels = self._mask_prediction_conv_depth
    if num_conv_channels == 0:
      num_feature_channels = features.get_shape().as_list()[3]
      num_conv_channels = self._get_mask_predictor_conv_depth(
          num_feature_channels, self._num_classes)
    with slim.arg_scope(self._conv_hyperparams_fn()):
      if not self._convolve_then_upsample:
        features = tf.image.resize_bilinear(
            features, [self._mask_height, self._mask_width],
            align_corners=True)
      for _ in range(self._mask_prediction_num_conv_layers - 1):
        features = slim.conv2d(
            features,
            num_outputs=num_conv_channels,
            kernel_size=[3, 3])
      if self._convolve_then_upsample:
        # Replace Transposed Convolution with a Nearest Neighbor upsampling step
        # followed by 3x3 convolution.
        height_scale = self._mask_height // features.shape[1].value
        width_scale = self._mask_width // features.shape[2].value
        features = ops.nearest_neighbor_upsampling(
            features, height_scale=height_scale, width_scale=width_scale)
        features = slim.conv2d(
            features,
            num_outputs=num_conv_channels,
            kernel_size=[3, 3])

      num_masks = 1 if self._masks_are_class_agnostic else self._num_classes
      mask_predictions = slim.conv2d(
          features,
          num_outputs=num_masks,
          activation_fn=None,
          normalizer_fn=None,
          kernel_size=[3, 3])
      return tf.expand_dims(
          tf.transpose(mask_predictions, perm=[0, 3, 1, 2]),
          axis=1,
          name='MaskPredictor')


class ConvolutionalMaskHead(head.Head):
  """Convolutional class prediction head."""

  def __init__(self,
               is_training,
               num_classes,
               use_dropout,
               dropout_keep_prob,
               kernel_size,
               use_depthwise=False,
               mask_height=7,
               mask_width=7,
               masks_are_class_agnostic=False):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: Number of classes.
      use_dropout: Option to use dropout or not.  Note that a single dropout
        op is applied here prior to both box and class predictions, which stands
        in contrast to the ConvolutionalBoxPredictor below.
      dropout_keep_prob: Keep probability for dropout.
        This is only used if use_dropout is True.
      kernel_size: Size of final convolution kernel.  If the
        spatial resolution of the feature map is smaller than the kernel size,
        then the kernel size is automatically set to be
        min(feature_width, feature_height).
      use_depthwise: Whether to use depthwise convolutions for prediction
        steps. Default is False.
      mask_height: Desired output mask height. The default value is 7.
      mask_width: Desired output mask width. The default value is 7.
      masks_are_class_agnostic: Boolean determining if the mask-head is
        class-agnostic or not.

    Raises:
      ValueError: if min_depth > max_depth.
    """
    super(ConvolutionalMaskHead, self).__init__()
    self._is_training = is_training
    self._num_classes = num_classes
    self._use_dropout = use_dropout
    self._dropout_keep_prob = dropout_keep_prob
    self._kernel_size = kernel_size
    self._use_depthwise = use_depthwise
    self._mask_height = mask_height
    self._mask_width = mask_width
    self._masks_are_class_agnostic = masks_are_class_agnostic

  def predict(self, features, num_predictions_per_location):
    """Predicts boxes.

    Args:
      features: A float tensor of shape [batch_size, height, width, channels]
        containing image features.
      num_predictions_per_location: Number of box predictions to be made per
        spatial location.

    Returns:
      mask_predictions: A float tensors of shape
        [batch_size, num_anchors, num_masks, mask_height, mask_width]
        representing the mask predictions for the proposals.
    """
    image_feature = features
    # Add a slot for the background class.
    if self._masks_are_class_agnostic:
      num_masks = 1
    else:
      num_masks = self._num_classes
    num_mask_channels = num_masks * self._mask_height * self._mask_width
    net = image_feature
    if self._use_dropout:
      net = slim.dropout(net, keep_prob=self._dropout_keep_prob)
    if self._use_depthwise:
      mask_predictions = slim.separable_conv2d(
          net, None, [self._kernel_size, self._kernel_size],
          padding='SAME', depth_multiplier=1, stride=1,
          rate=1, scope='MaskPredictor_depthwise')
      mask_predictions = slim.conv2d(
          mask_predictions,
          num_predictions_per_location * num_mask_channels,
          [1, 1],
          activation_fn=None,
          normalizer_fn=None,
          normalizer_params=None,
          scope='MaskPredictor')
    else:
      mask_predictions = slim.conv2d(
          net,
          num_predictions_per_location * num_mask_channels,
          [self._kernel_size, self._kernel_size],
          activation_fn=None,
          normalizer_fn=None,
          normalizer_params=None,
          scope='MaskPredictor')
    batch_size = features.get_shape().as_list()[0]
    if batch_size is None:
      batch_size = tf.shape(features)[0]
    mask_predictions = tf.reshape(
        mask_predictions,
        [batch_size, -1, num_masks, self._mask_height, self._mask_width])
    return mask_predictions


# TODO(alirezafathi): See if possible to unify Weight Shared with regular
# convolutional mask head.
class WeightSharedConvolutionalMaskHead(head.Head):
  """Weight shared convolutional mask prediction head."""

  def __init__(self,
               num_classes,
               kernel_size=3,
               use_dropout=False,
               dropout_keep_prob=0.8,
               mask_height=7,
               mask_width=7,
               masks_are_class_agnostic=False):
    """Constructor.

    Args:
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      kernel_size: Size of final convolution kernel.
      use_dropout: Whether to apply dropout to class prediction head.
      dropout_keep_prob: Probability of keeping activiations.
      mask_height: Desired output mask height. The default value is 7.
      mask_width: Desired output mask width. The default value is 7.
      masks_are_class_agnostic: Boolean determining if the mask-head is
        class-agnostic or not.
    """
    super(WeightSharedConvolutionalMaskHead, self).__init__()
    self._num_classes = num_classes
    self._kernel_size = kernel_size
    self._use_dropout = use_dropout
    self._dropout_keep_prob = dropout_keep_prob
    self._mask_height = mask_height
    self._mask_width = mask_width
    self._masks_are_class_agnostic = masks_are_class_agnostic

  def predict(self, features, num_predictions_per_location):
    """Predicts boxes.

    Args:
      features: A float tensor of shape [batch_size, height, width, channels]
        containing image features.
      num_predictions_per_location: Number of box predictions to be made per
        spatial location.

    Returns:
      mask_predictions: A tensor of shape
        [batch_size, num_anchors, num_classes, mask_height, mask_width]
        representing the mask predictions for the proposals.
    """
    mask_predictions_net = features
    if self._masks_are_class_agnostic:
      num_masks = 1
    else:
      num_masks = self._num_classes
    num_mask_channels = num_masks * self._mask_height * self._mask_width
    if self._use_dropout:
      mask_predictions_net = slim.dropout(
          mask_predictions_net, keep_prob=self._dropout_keep_prob)
    mask_predictions = slim.conv2d(
        mask_predictions_net,
        num_predictions_per_location * num_mask_channels,
        [self._kernel_size, self._kernel_size],
        activation_fn=None, stride=1, padding='SAME',
        normalizer_fn=None,
        scope='MaskPredictor')
    batch_size = features.get_shape().as_list()[0]
    if batch_size is None:
      batch_size = tf.shape(features)[0]
    mask_predictions = tf.reshape(
        mask_predictions,
        [batch_size, -1, num_masks, self._mask_height, self._mask_width])
    return mask_predictions
