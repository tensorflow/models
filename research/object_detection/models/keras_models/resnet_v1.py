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

"""A wrapper around the Keras Resnet V1 models for object detection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from object_detection.core import freezable_batch_norm
from object_detection.models.keras_models import model_utils

try:
  from keras.applications import resnet  # pylint:disable=g-import-not-at-top
except ImportError:
  from tensorflow.python.keras.applications import resnet  # pylint:disable=g-import-not-at-top


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
  kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
  pad_total = kernel_size_effective - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  padded_inputs = tf.pad(
      inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
  return padded_inputs


class _LayersOverride(object):
  """Alternative Keras layers interface for the Keras Resnet V1."""

  def __init__(self,
               batchnorm_training,
               batchnorm_scale=True,
               default_batchnorm_momentum=0.997,
               default_batchnorm_epsilon=1e-5,
               weight_decay=0.0001,
               conv_hyperparams=None,
               min_depth=8,
               depth_multiplier=1):
    """Alternative tf.keras.layers interface, for use by the Keras Resnet V1.

    The class is used by the Keras applications kwargs injection API to
    modify the Resnet V1 Keras application with changes required by
    the Object Detection API.

    Args:
      batchnorm_training: Bool. Assigned to Batch norm layer `training` param
        when constructing `freezable_batch_norm.FreezableBatchNorm` layers.
      batchnorm_scale: If True, uses an explicit `gamma` multiplier to scale
        the activations in the batch normalization layer.
      default_batchnorm_momentum: Float. When 'conv_hyperparams' is None,
        batch norm layers will be constructed using this value as the momentum.
      default_batchnorm_epsilon: Float. When 'conv_hyperparams' is None,
        batch norm layers will be constructed using this value as the epsilon.
      weight_decay: The weight decay to use for regularizing the model.
      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
        containing hyperparameters for convolution ops. Optionally set to `None`
        to use default resnet_v1 layer builders.
      min_depth: Minimum number of filters in the convolutional layers.
      depth_multiplier: The depth multiplier to modify the number of filters
        in the convolutional layers.
    """
    self._batchnorm_training = batchnorm_training
    self._batchnorm_scale = batchnorm_scale
    self._default_batchnorm_momentum = default_batchnorm_momentum
    self._default_batchnorm_epsilon = default_batchnorm_epsilon
    self._conv_hyperparams = conv_hyperparams
    self._min_depth = min_depth
    self._depth_multiplier = depth_multiplier
    self.regularizer = tf.keras.regularizers.l2(weight_decay)
    self.initializer = tf.variance_scaling_initializer()

  def _FixedPaddingLayer(self, kernel_size, rate=1):  # pylint: disable=invalid-name
    return tf.keras.layers.Lambda(
        lambda x: _fixed_padding(x, kernel_size, rate))

  def Conv2D(self, filters, kernel_size, **kwargs):  # pylint: disable=invalid-name
    """Builds a Conv2D layer according to the current Object Detection config.

    Overrides the Keras Resnet application's convolutions with ones that
    follow the spec specified by the Object Detection hyperparameters.

    Args:
      filters: The number of filters to use for the convolution.
      kernel_size: The kernel size to specify the height and width of the 2D
        convolution window.
      **kwargs: Keyword args specified by the Keras application for
        constructing the convolution.

    Returns:
      A one-arg callable that will either directly apply a Keras Conv2D layer to
      the input argument, or that will first pad the input then apply a Conv2D
      layer.
    """
    # Apply the minimum depth to the convolution layers.
    filters = max(int(filters * self._depth_multiplier), self._min_depth)

    if self._conv_hyperparams:
      kwargs = self._conv_hyperparams.params(**kwargs)
    else:
      kwargs['kernel_regularizer'] = self.regularizer
      kwargs['kernel_initializer'] = self.initializer

    # Set use_bias as false to keep it consistent with Slim Resnet model.
    kwargs['use_bias'] = False

    kwargs['padding'] = 'same'
    stride = kwargs.get('strides')
    if stride and kernel_size and stride > 1 and kernel_size > 1:
      kwargs['padding'] = 'valid'
      def padded_conv(features):  # pylint: disable=invalid-name
        padded_features = self._FixedPaddingLayer(kernel_size)(features)
        return tf.keras.layers.Conv2D(
            filters, kernel_size, **kwargs)(padded_features)
      return padded_conv
    else:
      return tf.keras.layers.Conv2D(filters, kernel_size, **kwargs)

  def Activation(self, *args, **kwargs):  # pylint: disable=unused-argument,invalid-name
    """Builds an activation layer.

    Overrides the Keras application Activation layer specified by the
    Object Detection configuration.

    Args:
      *args: Ignored,
        required to match the `tf.keras.layers.Activation` interface.
      **kwargs: Only the name is used,
        required to match `tf.keras.layers.Activation` interface.

    Returns:
      An activation layer specified by the Object Detection hyperparameter
      configurations.
    """
    name = kwargs.get('name')
    if self._conv_hyperparams:
      return self._conv_hyperparams.build_activation_layer(name=name)
    else:
      return tf.keras.layers.Lambda(tf.nn.relu, name=name)

  def BatchNormalization(self, **kwargs):  # pylint: disable=invalid-name
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
      kwargs['scale'] = self._batchnorm_scale
      kwargs['epsilon'] = self._default_batchnorm_epsilon
      return freezable_batch_norm.FreezableBatchNorm(
          training=self._batchnorm_training,
          momentum=self._default_batchnorm_momentum,
          **kwargs)

  def Input(self, shape):  # pylint: disable=invalid-name
    """Builds an Input layer.

    Overrides the Keras application Input layer with one that uses a
    tf.placeholder_with_default instead of a tf.placeholder. This is necessary
    to ensure the application works when run on a TPU.

    Args:
      shape: A tuple of integers representing the shape of the input, which
        includes both spatial share and channels, but not the batch size.
        Elements of this tuple can be None; 'None' elements represent dimensions
        where the shape is not known.

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

  def MaxPooling2D(self, pool_size, **kwargs):  # pylint: disable=invalid-name
    """Builds a MaxPooling2D layer with default padding as 'SAME'.

    This is specified by the default resnet arg_scope in slim.

    Args:
      pool_size: The pool size specified by the Keras application.
      **kwargs: Ignored, required to match the Keras applications usage.

    Returns:
      A MaxPooling2D layer with default padding as 'SAME'.
    """
    kwargs['padding'] = 'same'
    return tf.keras.layers.MaxPooling2D(pool_size, **kwargs)

  # Add alias as Keras also has it.
  MaxPool2D = MaxPooling2D  # pylint: disable=invalid-name

  def ZeroPadding2D(self, padding, **kwargs):  # pylint: disable=unused-argument,invalid-name
    """Replaces explicit padding in the Keras application with a no-op.

    Args:
      padding: The padding values for image height and width.
      **kwargs: Ignored, required to match the Keras applications usage.

    Returns:
      A no-op identity lambda.
    """
    return lambda x: x

  # Forward all non-overridden methods to the keras layers
  def __getattr__(self, item):
    return getattr(tf.keras.layers, item)


# pylint: disable=invalid-name
def resnet_v1_50(batchnorm_training,
                 batchnorm_scale=True,
                 default_batchnorm_momentum=0.997,
                 default_batchnorm_epsilon=1e-5,
                 weight_decay=0.0001,
                 conv_hyperparams=None,
                 min_depth=8,
                 depth_multiplier=1,
                 **kwargs):
  """Instantiates the Resnet50 architecture, modified for object detection.

  Args:
    batchnorm_training: Bool. Assigned to Batch norm layer `training` param
      when constructing `freezable_batch_norm.FreezableBatchNorm` layers.
    batchnorm_scale: If True, uses an explicit `gamma` multiplier to scale
      the activations in the batch normalization layer.
    default_batchnorm_momentum: Float. When 'conv_hyperparams' is None,
      batch norm layers will be constructed using this value as the momentum.
    default_batchnorm_epsilon: Float. When 'conv_hyperparams' is None,
      batch norm layers will be constructed using this value as the epsilon.
    weight_decay: The weight decay to use for regularizing the model.
    conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
      containing hyperparameters for convolution ops. Optionally set to `None`
      to use default resnet_v1 layer builders.
    min_depth: Minimum number of filters in the convolutional layers.
    depth_multiplier: The depth multiplier to modify the number of filters
      in the convolutional layers.
    **kwargs: Keyword arguments forwarded directly to the
      `tf.keras.applications.Mobilenet` method that constructs the Keras
      model.

  Returns:
    A Keras ResnetV1-50 model instance.
  """
  layers_override = _LayersOverride(
      batchnorm_training,
      batchnorm_scale=batchnorm_scale,
      default_batchnorm_momentum=default_batchnorm_momentum,
      default_batchnorm_epsilon=default_batchnorm_epsilon,
      conv_hyperparams=conv_hyperparams,
      weight_decay=weight_decay,
      min_depth=min_depth,
      depth_multiplier=depth_multiplier)
  return tf.keras.applications.resnet.ResNet50(
      layers=layers_override, **kwargs)


def resnet_v1_101(batchnorm_training,
                  batchnorm_scale=True,
                  default_batchnorm_momentum=0.997,
                  default_batchnorm_epsilon=1e-5,
                  weight_decay=0.0001,
                  conv_hyperparams=None,
                  min_depth=8,
                  depth_multiplier=1,
                  **kwargs):
  """Instantiates the Resnet50 architecture, modified for object detection.

  Args:
    batchnorm_training: Bool. Assigned to Batch norm layer `training` param
      when constructing `freezable_batch_norm.FreezableBatchNorm` layers.
    batchnorm_scale: If True, uses an explicit `gamma` multiplier to scale
      the activations in the batch normalization layer.
    default_batchnorm_momentum: Float. When 'conv_hyperparams' is None,
      batch norm layers will be constructed using this value as the momentum.
    default_batchnorm_epsilon: Float. When 'conv_hyperparams' is None,
      batch norm layers will be constructed using this value as the epsilon.
    weight_decay: The weight decay to use for regularizing the model.
    conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
      containing hyperparameters for convolution ops. Optionally set to `None`
      to use default resnet_v1 layer builders.
    min_depth: Minimum number of filters in the convolutional layers.
    depth_multiplier: The depth multiplier to modify the number of filters
      in the convolutional layers.
    **kwargs: Keyword arguments forwarded directly to the
      `tf.keras.applications.Mobilenet` method that constructs the Keras
      model.

  Returns:
    A Keras ResnetV1-101 model instance.
  """
  layers_override = _LayersOverride(
      batchnorm_training,
      batchnorm_scale=batchnorm_scale,
      default_batchnorm_momentum=default_batchnorm_momentum,
      default_batchnorm_epsilon=default_batchnorm_epsilon,
      conv_hyperparams=conv_hyperparams,
      weight_decay=weight_decay,
      min_depth=min_depth,
      depth_multiplier=depth_multiplier)
  return tf.keras.applications.resnet.ResNet101(
      layers=layers_override, **kwargs)


def resnet_v1_152(batchnorm_training,
                  batchnorm_scale=True,
                  default_batchnorm_momentum=0.997,
                  default_batchnorm_epsilon=1e-5,
                  weight_decay=0.0001,
                  conv_hyperparams=None,
                  min_depth=8,
                  depth_multiplier=1,
                  **kwargs):
  """Instantiates the Resnet50 architecture, modified for object detection.

  Args:
    batchnorm_training: Bool. Assigned to Batch norm layer `training` param
      when constructing `freezable_batch_norm.FreezableBatchNorm` layers.
    batchnorm_scale: If True, uses an explicit `gamma` multiplier to scale
      the activations in the batch normalization layer.
    default_batchnorm_momentum: Float. When 'conv_hyperparams' is None,
      batch norm layers will be constructed using this value as the momentum.
    default_batchnorm_epsilon: Float. When 'conv_hyperparams' is None,
      batch norm layers will be constructed using this value as the epsilon.
    weight_decay: The weight decay to use for regularizing the model.
    conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
      containing hyperparameters for convolution ops. Optionally set to `None`
      to use default resnet_v1 layer builders.
    min_depth: Minimum number of filters in the convolutional layers.
    depth_multiplier: The depth multiplier to modify the number of filters
      in the convolutional layers.
    **kwargs: Keyword arguments forwarded directly to the
      `tf.keras.applications.Mobilenet` method that constructs the Keras
      model.

  Returns:
    A Keras ResnetV1-152 model instance.
  """
  layers_override = _LayersOverride(
      batchnorm_training,
      batchnorm_scale=batchnorm_scale,
      default_batchnorm_momentum=default_batchnorm_momentum,
      default_batchnorm_epsilon=default_batchnorm_epsilon,
      conv_hyperparams=conv_hyperparams,
      weight_decay=weight_decay,
      min_depth=min_depth,
      depth_multiplier=depth_multiplier)
  return tf.keras.applications.resnet.ResNet152(
      layers=layers_override, **kwargs)
# pylint: enable=invalid-name


# The following codes are based on the existing keras ResNet model pattern:
# google3/third_party/py/keras/applications/resnet.py
def block_basic(x,
                filters,
                kernel_size=3,
                stride=1,
                conv_shortcut=False,
                name=None):
  """A residual block for ResNet18/34.

  Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default False, use convolution shortcut if True, otherwise
        identity shortcut.
      name: string, block label.

  Returns:
    Output tensor for the residual block.
  """
  layers = tf.keras.layers
  bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

  preact = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_bn')(
          x)
  preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

  if conv_shortcut:
    shortcut = layers.Conv2D(
        filters, 1, strides=1, name=name + '_0_conv')(
            preact)
  else:
    shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

  x = layers.ZeroPadding2D(
      padding=((1, 1), (1, 1)), name=name + '_1_pad')(
          preact)
  x = layers.Conv2D(
      filters, kernel_size, strides=1, use_bias=False, name=name + '_1_conv')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(
          x)
  x = layers.Activation('relu', name=name + '_1_relu')(x)

  x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
  x = layers.Conv2D(
      filters,
      kernel_size,
      strides=stride,
      use_bias=False,
      name=name + '_2_conv')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(
          x)
  x = layers.Activation('relu', name=name + '_2_relu')(x)
  x = layers.Add(name=name + '_out')([shortcut, x])
  return x


def stack_basic(x, filters, blocks, stride1=2, name=None):
  """A set of stacked residual blocks for ResNet18/34.

  Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.

  Returns:
      Output tensor for the stacked blocks.
  """
  x = block_basic(x, filters, conv_shortcut=True, name=name + '_block1')
  for i in range(2, blocks):
    x = block_basic(x, filters, name=name + '_block' + str(i))
  x = block_basic(
      x, filters, stride=stride1, name=name + '_block' + str(blocks))
  return x


def resnet_v1_18(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 classifier_activation='softmax'):
  """Instantiates the ResNet18 architecture."""

  def stack_fn(x):
    x = stack_basic(x, 64, 2, stride1=1, name='conv2')
    x = stack_basic(x, 128, 2, name='conv3')
    x = stack_basic(x, 256, 2, name='conv4')
    return stack_basic(x, 512, 2, name='conv5')

  return resnet.ResNet(
      stack_fn,
      True,
      True,
      'resnet18',
      include_top,
      weights,
      input_tensor,
      input_shape,
      pooling,
      classes,
      classifier_activation=classifier_activation)


def resnet_v1_34(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 classifier_activation='softmax'):
  """Instantiates the ResNet34 architecture."""

  def stack_fn(x):
    x = stack_basic(x, 64, 3, stride1=1, name='conv2')
    x = stack_basic(x, 128, 4, name='conv3')
    x = stack_basic(x, 256, 6, name='conv4')
    return stack_basic(x, 512, 3, name='conv5')

  return resnet.ResNet(
      stack_fn,
      True,
      True,
      'resnet34',
      include_top,
      weights,
      input_tensor,
      input_shape,
      pooling,
      classes,
      classifier_activation=classifier_activation)
