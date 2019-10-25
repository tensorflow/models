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

"""A wrapper around the MobileNet v2 models for Keras, for object detection."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from object_detection.core import freezable_batch_norm
from object_detection.models.keras_models import model_utils
from object_detection.utils import ops


# pylint: disable=invalid-name
# This method copied from the slim mobilenet base network code (same license)
def _make_divisible(v, divisor, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v


class _LayersOverride(object):
  """Alternative Keras layers interface for the Keras MobileNetV2."""

  def __init__(self,
               batchnorm_training,
               default_batchnorm_momentum=0.999,
               conv_hyperparams=None,
               use_explicit_padding=False,
               alpha=1.0,
               min_depth=None,
               conv_defs=None):
    """Alternative tf.keras.layers interface, for use by the Keras MobileNetV2.

    It is used by the Keras applications kwargs injection API to
    modify the Mobilenet v2 Keras application with changes required by
    the Object Detection API.

    These injected interfaces make the following changes to the network:

    - Applies the Object Detection hyperparameter configuration
    - Supports FreezableBatchNorms
    - Adds support for a min number of filters for each layer
    - Makes the `alpha` parameter affect the final convolution block even if it
        is less than 1.0
    - Adds support for explicit padding of convolutions

    Args:
      batchnorm_training: Bool. Assigned to Batch norm layer `training` param
        when constructing `freezable_batch_norm.FreezableBatchNorm` layers.
      default_batchnorm_momentum: Float. When 'conv_hyperparams' is None,
        batch norm layers will be constructed using this value as the momentum.
      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
        containing hyperparameters for convolution ops. Optionally set to `None`
        to use default mobilenet_v2 layer builders.
      use_explicit_padding: If True, use 'valid' padding for convolutions,
        but explicitly pre-pads inputs so that the output dimensions are the
        same as if 'same' padding were used. Off by default.
      alpha: The width multiplier referenced in the MobileNetV2 paper. It
        modifies the number of filters in each convolutional layer.
      min_depth: Minimum number of filters in the convolutional layers.
      conv_defs: Network layout to specify the mobilenet_v2 body. Default is
        `None` to use the default mobilenet_v2 network layout.
    """
    self._alpha = alpha
    self._batchnorm_training = batchnorm_training
    self._default_batchnorm_momentum = default_batchnorm_momentum
    self._conv_hyperparams = conv_hyperparams
    self._use_explicit_padding = use_explicit_padding
    self._min_depth = min_depth
    self._conv_defs = conv_defs
    self.regularizer = tf.keras.regularizers.l2(0.00004 * 0.5)
    self.initializer = tf.truncated_normal_initializer(stddev=0.09)

  def _FixedPaddingLayer(self, kernel_size):
    return tf.keras.layers.Lambda(lambda x: ops.fixed_padding(x, kernel_size))

  def Conv2D(self, filters, **kwargs):
    """Builds a Conv2D layer according to the current Object Detection config.

    Overrides the Keras MobileNetV2 application's convolutions with ones that
    follow the spec specified by the Object Detection hyperparameters.

    Args:
      filters: The number of filters to use for the convolution.
      **kwargs: Keyword args specified by the Keras application for
        constructing the convolution.

    Returns:
      A one-arg callable that will either directly apply a Keras Conv2D layer to
      the input argument, or that will first pad the input then apply a Conv2D
      layer.
    """
    # Make sure 'alpha' is always applied to the last convolution block's size
    # (This overrides the Keras application's functionality)
    layer_name = kwargs.get('name')
    if layer_name == 'Conv_1':
      if self._conv_defs:
        filters = model_utils.get_conv_def(self._conv_defs, 'Conv_1')
      else:
        filters = 1280
      if self._alpha < 1.0:
        filters = _make_divisible(filters * self._alpha, 8)

    # Apply the minimum depth to the convolution layers
    if (self._min_depth and (filters < self._min_depth)
        and not kwargs.get('name').endswith('expand')):
      filters = self._min_depth

    if self._conv_hyperparams:
      kwargs = self._conv_hyperparams.params(**kwargs)
    else:
      kwargs['kernel_regularizer'] = self.regularizer
      kwargs['kernel_initializer'] = self.initializer

    kwargs['padding'] = 'same'
    kernel_size = kwargs.get('kernel_size')
    if self._use_explicit_padding and kernel_size > 1:
      kwargs['padding'] = 'valid'
      def padded_conv(features):
        padded_features = self._FixedPaddingLayer(kernel_size)(features)
        return tf.keras.layers.Conv2D(filters, **kwargs)(padded_features)

      return padded_conv
    else:
      return tf.keras.layers.Conv2D(filters, **kwargs)

  def DepthwiseConv2D(self, **kwargs):
    """Builds a DepthwiseConv2D according to the Object Detection config.

    Overrides the Keras MobileNetV2 application's convolutions with ones that
    follow the spec specified by the Object Detection hyperparameters.

    Args:
      **kwargs: Keyword args specified by the Keras application for
        constructing the convolution.

    Returns:
      A one-arg callable that will either directly apply a Keras DepthwiseConv2D
      layer to the input argument, or that will first pad the input then apply
      the depthwise convolution.
    """
    if self._conv_hyperparams:
      kwargs = self._conv_hyperparams.params(**kwargs)
      # Both the regularizer and initializer apply to the depthwise layer in
      # MobilenetV1, so we remap the kernel_* to depthwise_* here.
      kwargs['depthwise_regularizer'] = kwargs['kernel_regularizer']
      kwargs['depthwise_initializer'] = kwargs['kernel_initializer']
    else:
      kwargs['depthwise_regularizer'] = self.regularizer
      kwargs['depthwise_initializer'] = self.initializer

    kwargs['padding'] = 'same'
    kernel_size = kwargs.get('kernel_size')
    if self._use_explicit_padding and kernel_size > 1:
      kwargs['padding'] = 'valid'
      def padded_depthwise_conv(features):
        padded_features = self._FixedPaddingLayer(kernel_size)(features)
        return tf.keras.layers.DepthwiseConv2D(**kwargs)(padded_features)

      return padded_depthwise_conv
    else:
      return tf.keras.layers.DepthwiseConv2D(**kwargs)

  def BatchNormalization(self, **kwargs):
    """Builds a normalization layer.

    Overrides the Keras application batch norm with the norm specified by the
    Object Detection configuration.

    Args:
      **kwargs: Only the name is used, all other params ignored.
        Required for matching `layers.BatchNormalization` calls in the Keras
        application.

    Returns:
      A normalization layer specified by the Object Detection hyperparameter
      configurations.
    """
    name = kwargs.get('name')
    if self._conv_hyperparams:
      return self._conv_hyperparams.build_batch_norm(
          training=self._batchnorm_training,
          name=name)
    else:
      return freezable_batch_norm.FreezableBatchNorm(
          training=self._batchnorm_training,
          epsilon=1e-3,
          momentum=self._default_batchnorm_momentum,
          name=name)

  def Input(self, shape):
    """Builds an Input layer.

    Overrides the Keras application Input layer with one that uses a
    tf.placeholder_with_default instead of a tf.placeholder. This is necessary
    to ensure the application works when run on a TPU.

    Args:
      shape: The shape for the input layer to use. (Does not include a dimension
        for the batch size).
    Returns:
      An input layer for the specified shape that internally uses a
      placeholder_with_default.
    """
    default_size = 224
    default_batch_size = 1
    shape = list(shape)
    default_shape = [default_size if dim is None else dim for dim in shape]

    input_tensor = tf.constant(0.0, shape=[default_batch_size] + default_shape)

    placeholder_with_default = tf.placeholder_with_default(
        input=input_tensor, shape=[None] + shape)
    if tf.executing_eagerly():
      return tf.keras.layers.Input(shape=shape)
    else:
      return tf.keras.layers.Input(tensor=placeholder_with_default)

  # pylint: disable=unused-argument
  def ReLU(self, *args, **kwargs):
    """Builds an activation layer.

    Overrides the Keras application ReLU with the activation specified by the
    Object Detection configuration.

    Args:
      *args: Ignored, required to match the `tf.keras.ReLU` interface
      **kwargs: Only the name is used,
        required to match `tf.keras.ReLU` interface

    Returns:
      An activation layer specified by the Object Detection hyperparameter
      configurations.
    """
    name = kwargs.get('name')
    if self._conv_hyperparams:
      return self._conv_hyperparams.build_activation_layer(name=name)
    else:
      return tf.keras.layers.Lambda(tf.nn.relu6, name=name)
  # pylint: enable=unused-argument

  # pylint: disable=unused-argument
  def ZeroPadding2D(self, **kwargs):
    """Replaces explicit padding in the Keras application with a no-op.

    Args:
      **kwargs: Ignored, required to match the Keras applications usage.

    Returns:
      A no-op identity lambda.
    """
    return lambda x: x
  # pylint: enable=unused-argument

  # Forward all non-overridden methods to the keras layers
  def __getattr__(self, item):
    return getattr(tf.keras.layers, item)


def mobilenet_v2(batchnorm_training,
                 default_batchnorm_momentum=0.9997,
                 conv_hyperparams=None,
                 use_explicit_padding=False,
                 alpha=1.0,
                 min_depth=None,
                 conv_defs=None,
                 **kwargs):
  """Instantiates the MobileNetV2 architecture, modified for object detection.

  This wraps the MobileNetV2 tensorflow Keras application, but uses the
  Keras application's kwargs-based monkey-patching API to override the Keras
  architecture with the following changes:

  - Changes the default batchnorm momentum to 0.9997
  - Applies the Object Detection hyperparameter configuration
  - Supports FreezableBatchNorms
  - Adds support for a min number of filters for each layer
  - Makes the `alpha` parameter affect the final convolution block even if it
      is less than 1.0
  - Adds support for explicit padding of convolutions
  - Makes the Input layer use a tf.placeholder_with_default instead of a
      tf.placeholder, to work on TPUs.

  Args:
      batchnorm_training: Bool. Assigned to Batch norm layer `training` param
        when constructing `freezable_batch_norm.FreezableBatchNorm` layers.
      default_batchnorm_momentum: Float. When 'conv_hyperparams' is None,
        batch norm layers will be constructed using this value as the momentum.
      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
        containing hyperparameters for convolution ops. Optionally set to `None`
        to use default mobilenet_v2 layer builders.
      use_explicit_padding: If True, use 'valid' padding for convolutions,
        but explicitly pre-pads inputs so that the output dimensions are the
        same as if 'same' padding were used. Off by default.
      alpha: The width multiplier referenced in the MobileNetV2 paper. It
        modifies the number of filters in each convolutional layer.
      min_depth: Minimum number of filters in the convolutional layers.
      conv_defs: Network layout to specify the mobilenet_v2 body. Default is
        `None` to use the default mobilenet_v2 network layout.
      **kwargs: Keyword arguments forwarded directly to the
        `tf.keras.applications.MobilenetV2` method that constructs the Keras
        model.

  Returns:
      A Keras model instance.
  """
  layers_override = _LayersOverride(
      batchnorm_training,
      default_batchnorm_momentum=default_batchnorm_momentum,
      conv_hyperparams=conv_hyperparams,
      use_explicit_padding=use_explicit_padding,
      min_depth=min_depth,
      alpha=alpha,
      conv_defs=conv_defs)
  return tf.keras.applications.MobileNetV2(alpha=alpha,
                                           layers=layers_override,
                                           **kwargs)
# pylint: enable=invalid-name
