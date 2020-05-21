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

"""A wrapper around the Keras MobilenetV1 models for object detection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from object_detection.core import freezable_batch_norm
from object_detection.models.keras_models import model_utils


def _fixed_padding(inputs, kernel_size, rate=1):  # pylint: disable=invalid-name
  """Pads the input along the spatial dimensions independently of input size.

  Pads the input such that if it was used in a convolution with 'VALID' padding,
  the output would have the same dimensions as if the unpadded input was used
  in a convolution with 'SAME' padding.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
    rate: An integer, rate for atrous convolution.

  Returns:
    output: A tensor of size [batch, height_out, width_out, channels] with the
      input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                           kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
  pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
  pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
  pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]],
                                  [pad_beg[1], pad_end[1]], [0, 0]])
  return padded_inputs


class _LayersOverride(object):
  """Alternative Keras layers interface for the Keras MobileNetV1."""

  def __init__(self,
               batchnorm_training,
               default_batchnorm_momentum=0.999,
               conv_hyperparams=None,
               use_explicit_padding=False,
               alpha=1.0,
               min_depth=None,
               conv_defs=None):
    """Alternative tf.keras.layers interface, for use by the Keras MobileNetV1.

    It is used by the Keras applications kwargs injection API to
    modify the MobilenetV1 Keras application with changes required by
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
        to use default mobilenet_v1 layer builders.
      use_explicit_padding: If True, use 'valid' padding for convolutions,
        but explicitly pre-pads inputs so that the output dimensions are the
        same as if 'same' padding were used. Off by default.
      alpha: The width multiplier referenced in the MobileNetV1 paper. It
        modifies the number of filters in each convolutional layer. It's called
        depth multiplier in Keras application MobilenetV1.
      min_depth: Minimum number of filters in the convolutional layers.
      conv_defs: Network layout to specify the mobilenet_v1 body. Default is
        `None` to use the default mobilenet_v1 network layout.
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

  def _FixedPaddingLayer(self, kernel_size, rate=1):
    return tf.keras.layers.Lambda(
        lambda x: _fixed_padding(x, kernel_size, rate))

  def Conv2D(self, filters, kernel_size, **kwargs):
    """Builds a Conv2D layer according to the current Object Detection config.

    Overrides the Keras MobileNetV1 application's convolutions with ones that
    follow the spec specified by the Object Detection hyperparameters.

    Args:
      filters: The number of filters to use for the convolution.
      kernel_size: The kernel size to specify the height and width of the 2D
        convolution window. In this function, the kernel size is expected to
        be pair of numbers and the numbers must be equal for this function.
      **kwargs: Keyword args specified by the Keras application for
        constructing the convolution.

    Returns:
      A one-arg callable that will either directly apply a Keras Conv2D layer to
      the input argument, or that will first pad the input then apply a Conv2D
      layer.

    Raises:
      ValueError: if kernel size is not a pair of equal
        integers (representing a square kernel).
    """
    if not isinstance(kernel_size, tuple):
      raise ValueError('kernel is expected to be a tuple.')
    if len(kernel_size) != 2:
      raise ValueError('kernel is expected to be length two.')
    if kernel_size[0] != kernel_size[1]:
      raise ValueError('kernel is expected to be square.')
    layer_name = kwargs['name']
    if self._conv_defs:
      conv_filters = model_utils.get_conv_def(self._conv_defs, layer_name)
      if conv_filters:
        filters = conv_filters
    # Apply the width multiplier and the minimum depth to the convolution layers
    filters = int(filters * self._alpha)
    if self._min_depth and filters < self._min_depth:
      filters = self._min_depth

    if self._conv_hyperparams:
      kwargs = self._conv_hyperparams.params(**kwargs)
    else:
      kwargs['kernel_regularizer'] = self.regularizer
      kwargs['kernel_initializer'] = self.initializer

    kwargs['padding'] = 'same'
    if self._use_explicit_padding and kernel_size[0] > 1:
      kwargs['padding'] = 'valid'
      def padded_conv(features):  # pylint: disable=invalid-name
        padded_features = self._FixedPaddingLayer(kernel_size)(features)
        return tf.keras.layers.Conv2D(
            filters, kernel_size, **kwargs)(padded_features)
      return padded_conv
    else:
      return tf.keras.layers.Conv2D(filters, kernel_size, **kwargs)

  def DepthwiseConv2D(self, kernel_size, **kwargs):
    """Builds a DepthwiseConv2D according to the Object Detection config.

    Overrides the Keras MobileNetV2 application's convolutions with ones that
    follow the spec specified by the Object Detection hyperparameters.

    Args:
      kernel_size: The kernel size to specify the height and width of the 2D
        convolution window.
      **kwargs: Keyword args specified by the Keras application for
        constructing the convolution.

    Returns:
      A one-arg callable that will either directly apply a Keras DepthwiseConv2D
      layer to the input argument, or that will first pad the input then apply
      the depthwise convolution.
    """
    if self._conv_hyperparams:
      kwargs = self._conv_hyperparams.params(**kwargs)
      # Both regularizer and initializaer also applies to depthwise layer in
      # MobilenetV1, so we remap the kernel_* to depthwise_* here.
      kwargs['depthwise_regularizer'] = kwargs['kernel_regularizer']
      kwargs['depthwise_initializer'] = kwargs['kernel_initializer']
    else:
      kwargs['depthwise_regularizer'] = self.regularizer
      kwargs['depthwise_initializer'] = self.initializer

    kwargs['padding'] = 'same'
    if self._use_explicit_padding:
      kwargs['padding'] = 'valid'
      def padded_depthwise_conv(features):  # pylint: disable=invalid-name
        padded_features = self._FixedPaddingLayer(kernel_size)(features)
        return tf.keras.layers.DepthwiseConv2D(
            kernel_size, **kwargs)(padded_features)
      return padded_depthwise_conv
    else:
      return tf.keras.layers.DepthwiseConv2D(kernel_size, **kwargs)

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
    return model_utils.input_layer(shape, placeholder_with_default)

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
  def ZeroPadding2D(self, padding, **kwargs):
    """Replaces explicit padding in the Keras application with a no-op.

    Args:
      padding: The padding values for image height and width.
      **kwargs: Ignored, required to match the Keras applications usage.

    Returns:
      A no-op identity lambda.
    """
    return lambda x: x
  # pylint: enable=unused-argument

  # Forward all non-overridden methods to the keras layers
  def __getattr__(self, item):
    return getattr(tf.keras.layers, item)


# pylint: disable=invalid-name
def mobilenet_v1(batchnorm_training,
                 default_batchnorm_momentum=0.9997,
                 conv_hyperparams=None,
                 use_explicit_padding=False,
                 alpha=1.0,
                 min_depth=None,
                 conv_defs=None,
                 **kwargs):
  """Instantiates the MobileNetV1 architecture, modified for object detection.

  This wraps the MobileNetV1 tensorflow Keras application, but uses the
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
        to use default mobilenet_v1 layer builders.
      use_explicit_padding: If True, use 'valid' padding for convolutions,
        but explicitly pre-pads inputs so that the output dimensions are the
        same as if 'same' padding were used. Off by default.
      alpha: The width multiplier referenced in the MobileNetV1 paper. It
        modifies the number of filters in each convolutional layer.
      min_depth: Minimum number of filters in the convolutional layers.
      conv_defs: Network layout to specify the mobilenet_v1 body. Default is
        `None` to use the default mobilenet_v1 network layout.
      **kwargs: Keyword arguments forwarded directly to the
        `tf.keras.applications.Mobilenet` method that constructs the Keras
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
  return tf.keras.applications.MobileNet(
      alpha=alpha, layers=layers_override, **kwargs)
# pylint: enable=invalid-name
