# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""A wrapper around the Keras InceptionResnetV2 models for object detection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from object_detection.core import freezable_batch_norm


class _LayersOverride(object):
  """Alternative Keras layers interface for the Keras InceptionResNetV2."""

  def __init__(self,
               batchnorm_training,
               output_stride=16,
               align_feature_maps=False,
               batchnorm_scale=False,
               default_batchnorm_momentum=0.999,
               default_batchnorm_epsilon=1e-3,
               weight_decay=0.00004):
    """Alternative tf.keras.layers interface, for use by InceptionResNetV2.

    It is used by the Keras applications kwargs injection API to
    modify the Inception Resnet V2 Keras application with changes required by
    the Object Detection API.

    These injected interfaces make the following changes to the network:

    - Supports freezing batch norm layers
    - Adds support for feature map alignment (like in the Slim model)
    - Adds support for changing the output stride (like in the Slim model)
    - Adds support for overriding various batch norm hyperparameters

    Because the Keras inception resnet v2 application does not assign explicit
    names to most individual layers, the injection of output stride support
    works by identifying convolution layers according to their filter counts
    and pre-feature-map-alignment padding arguments.

    Args:
      batchnorm_training: Bool. Assigned to Batch norm layer `training` param
        when constructing `freezable_batch_norm.FreezableBatchNorm` layers.
      output_stride: A scalar that specifies the requested ratio of input to
        output spatial resolution. Only supports 8 and 16.
      align_feature_maps: When true, changes all the VALID paddings in the
        network to SAME padding so that the feature maps are aligned.
      batchnorm_scale: If True, uses an explicit `gamma` multiplier to scale the
        activations in the batch normalization layer.
      default_batchnorm_momentum: Float. Batch norm layers will be constructed
        using this value as the momentum.
      default_batchnorm_epsilon: small float added to variance to avoid
        dividing by zero.
      weight_decay: the l2 regularization weight decay for weights variables.
        (gets multiplied by 0.5 to map from slim l2 regularization weight to
        Keras l2 regularization weight).
    """
    self._use_atrous = output_stride == 8
    self._align_feature_maps = align_feature_maps
    self._batchnorm_training = batchnorm_training
    self._batchnorm_scale = batchnorm_scale
    self._default_batchnorm_momentum = default_batchnorm_momentum
    self._default_batchnorm_epsilon = default_batchnorm_epsilon
    self.regularizer = tf.keras.regularizers.l2(weight_decay * 0.5)

  def Conv2D(self, filters, kernel_size, **kwargs):
    """Builds a Conv2D layer according to the current Object Detection config.

    Overrides the Keras InceptionResnetV2 application's convolutions with ones
    that follow the spec specified by the Object Detection hyperparameters.

    If feature map alignment is enabled, the padding will be forced to 'same'.
    If output_stride is 8, some conv2d layers will be matched according to
    their name or filter counts or pre-alignment padding parameters, and will
    have the correct 'dilation rate' or 'strides' set.

    Args:
      filters: The number of filters to use for the convolution.
      kernel_size: The kernel size to specify the height and width of the 2D
        convolution window.
      **kwargs: Keyword args specified by the Keras application for
        constructing the convolution.

    Returns:
      A Keras Conv2D layer specified by the Object Detection hyperparameter
      configurations.
    """
    kwargs['kernel_regularizer'] = self.regularizer
    kwargs['bias_regularizer'] = self.regularizer

    # Because the Keras application does not set explicit names for most layers,
    # (instead allowing names to auto-increment), we must match individual
    # layers in the model according to their filter count, name, or
    # pre-alignment mapping. This means we can only align the feature maps
    # after we have applied our updates in cases where output_stride=8.
    if self._use_atrous and (filters == 384):
      kwargs['strides'] = 1

    name = kwargs.get('name')
    if self._use_atrous and (
        (name and 'block17' in name) or
        (filters == 128 or filters == 160 or
         (filters == 192 and kwargs.get('padding', '').lower() != 'valid'))):
      kwargs['dilation_rate'] = 2

    if self._align_feature_maps:
      kwargs['padding'] = 'same'

    return tf.keras.layers.Conv2D(filters, kernel_size, **kwargs)

  def MaxPooling2D(self, pool_size, strides, **kwargs):
    """Builds a pooling layer according to the current Object Detection config.

    Overrides the Keras InceptionResnetV2 application's MaxPooling2D layers with
    ones that follow the spec specified by the Object Detection hyperparameters.

    If feature map alignment is enabled, the padding will be forced to 'same'.
    If output_stride is 8, some pooling layers will be matched according to
    their pre-alignment padding parameters, and will have their 'strides'
    argument overridden.

    Args:
      pool_size: The pool size specified by the Keras application.
      strides: The strides specified by the unwrapped Keras application.
      **kwargs: Keyword args specified by the Keras application for
        constructing the max pooling layer.

    Returns:
      A MaxPool2D layer specified by the Object Detection hyperparameter
      configurations.
    """
    if self._use_atrous and kwargs.get('padding', '').lower() == 'valid':
      strides = 1

    if self._align_feature_maps:
      kwargs['padding'] = 'same'

    return tf.keras.layers.MaxPool2D(pool_size, strides=strides, **kwargs)

  # We alias MaxPool2D because Keras has that alias
  MaxPool2D = MaxPooling2D  # pylint: disable=invalid-name

  def BatchNormalization(self, **kwargs):
    """Builds a normalization layer.

    Overrides the Keras application batch norm with the norm specified by the
    Object Detection configuration.

    Args:
      **kwargs: Keyword arguments from the `layers.BatchNormalization` calls in
        the Keras application.

    Returns:
      A normalization layer specified by the Object Detection hyperparameter
      configurations.
    """
    kwargs['scale'] = self._batchnorm_scale
    return freezable_batch_norm.FreezableBatchNorm(
        training=self._batchnorm_training,
        epsilon=self._default_batchnorm_epsilon,
        momentum=self._default_batchnorm_momentum,
        **kwargs)

  # Forward all non-overridden methods to the keras layers
  def __getattr__(self, item):
    return getattr(tf.keras.layers, item)


# pylint: disable=invalid-name
def inception_resnet_v2(
    batchnorm_training,
    output_stride=16,
    align_feature_maps=False,
    batchnorm_scale=False,
    weight_decay=0.00004,
    default_batchnorm_momentum=0.9997,
    default_batchnorm_epsilon=0.001,
    **kwargs):
  """Instantiates the InceptionResnetV2 architecture.

  (Modified for object detection)

  This wraps the InceptionResnetV2 tensorflow Keras application, but uses the
  Keras application's kwargs-based monkey-patching API to override the Keras
  architecture with the following changes:

  - Supports freezing batch norm layers with FreezableBatchNorms
  - Adds support for feature map alignment (like in the Slim model)
  - Adds support for changing the output stride (like in the Slim model)
  - Changes the default batchnorm momentum to 0.9997
  - Adds support for overriding various batchnorm hyperparameters

  Args:
      batchnorm_training: Bool. Assigned to Batch norm layer `training` param
        when constructing `freezable_batch_norm.FreezableBatchNorm` layers.
      output_stride: A scalar that specifies the requested ratio of input to
        output spatial resolution. Only supports 8 and 16.
      align_feature_maps: When true, changes all the VALID paddings in the
        network to SAME padding so that the feature maps are aligned.
      batchnorm_scale: If True, uses an explicit `gamma` multiplier to scale the
        activations in the batch normalization layer.
      weight_decay: the l2 regularization weight decay for weights variables.
        (gets multiplied by 0.5 to map from slim l2 regularization weight to
        Keras l2 regularization weight).
      default_batchnorm_momentum: Float. Batch norm layers will be constructed
        using this value as the momentum.
      default_batchnorm_epsilon: small float added to variance to avoid
        dividing by zero.
      **kwargs: Keyword arguments forwarded directly to the
        `tf.keras.applications.InceptionResNetV2` method that constructs the
        Keras model.

  Returns:
      A Keras model instance.
  """
  if output_stride != 8 and output_stride != 16:
    raise ValueError('output_stride must be 8 or 16.')

  layers_override = _LayersOverride(
      batchnorm_training,
      output_stride,
      align_feature_maps=align_feature_maps,
      batchnorm_scale=batchnorm_scale,
      default_batchnorm_momentum=default_batchnorm_momentum,
      default_batchnorm_epsilon=default_batchnorm_epsilon,
      weight_decay=weight_decay)
  return tf.keras.applications.InceptionResNetV2(
      layers=layers_override, **kwargs)
# pylint: enable=invalid-name
