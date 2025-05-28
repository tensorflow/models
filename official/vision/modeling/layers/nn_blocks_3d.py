# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Contains common building blocks for 3D networks."""
import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.vision.modeling.layers import nn_layers


@tf_keras.utils.register_keras_serializable(package='Vision')
class SelfGating(tf_keras.layers.Layer):
  """Feature gating as used in S3D-G.

  This implements the S3D-G network from:
  Saining Xie, Chen Sun, Jonathan Huang, Zhuowen Tu, Kevin Murphy.
  Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video
  Classification.
  (https://arxiv.org/pdf/1712.04851.pdf)
  """

  def __init__(self, filters, **kwargs):
    """Initializes a self-gating layer.

    Args:
      filters: An `int` number of filters for the convolutional layer.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(SelfGating, self).__init__(**kwargs)
    self._filters = filters

  def build(self, input_shape):
    self._spatial_temporal_average = tf_keras.layers.GlobalAveragePooling3D()

    # No BN and activation after conv.
    self._transformer_w = tf_keras.layers.Conv3D(
        filters=self._filters,
        kernel_size=[1, 1, 1],
        use_bias=True,
        kernel_initializer=tf_keras.initializers.TruncatedNormal(
            mean=0.0, stddev=0.01))

    super(SelfGating, self).build(input_shape)

  def call(self, inputs):
    x = self._spatial_temporal_average(inputs)

    x = tf.expand_dims(x, 1)
    x = tf.expand_dims(x, 2)
    x = tf.expand_dims(x, 3)

    x = self._transformer_w(x)
    x = tf.nn.sigmoid(x)

    return tf.math.multiply(x, inputs)


@tf_keras.utils.register_keras_serializable(package='Vision')
class BottleneckBlock3D(tf_keras.layers.Layer):
  """Creates a 3D bottleneck block."""

  def __init__(self,
               filters,
               temporal_kernel_size,
               temporal_strides,
               spatial_strides,
               stochastic_depth_drop_rate=0.0,
               se_ratio=None,
               use_self_gating=False,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               **kwargs):
    """Initializes a 3D bottleneck block with BN after convolutions.

    Args:
      filters: An `int` number of filters for the first two convolutions. Note
        that the third and final convolution will use 4 times as many filters.
      temporal_kernel_size: An `int` of kernel size for the temporal
        convolutional layer.
      temporal_strides: An `int` of ftemporal stride for the temporal
        convolutional layer.
      spatial_strides: An `int` of spatial stride for the spatial convolutional
        layer.
      stochastic_depth_drop_rate: A `float` or None. If not None, drop rate for
        the stochastic depth layer.
      se_ratio: A `float` or None. Ratio of the Squeeze-and-Excitation layer.
      use_self_gating: A `bool` of whether to apply self-gating module or not.
      kernel_initializer: A `str` of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf_keras.regularizers.Regularizer` object for Conv2d.
        Default to None.
      activation: A `str` name of the activation function.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(BottleneckBlock3D, self).__init__(**kwargs)

    self._filters = filters
    self._temporal_kernel_size = temporal_kernel_size
    self._spatial_strides = spatial_strides
    self._temporal_strides = temporal_strides
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._use_self_gating = use_self_gating
    self._se_ratio = se_ratio
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._norm = tf_keras.layers.BatchNormalization

    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)

  def build(self, input_shape):
    self._shortcut_maxpool = tf_keras.layers.MaxPool3D(
        pool_size=[1, 1, 1],
        strides=[
            self._temporal_strides, self._spatial_strides, self._spatial_strides
        ])

    self._shortcut_conv = tf_keras.layers.Conv3D(
        filters=4 * self._filters,
        kernel_size=1,
        strides=[
            self._temporal_strides, self._spatial_strides, self._spatial_strides
        ],
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm0 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        synchronized=self._use_sync_bn)

    self._temporal_conv = tf_keras.layers.Conv3D(
        filters=self._filters,
        kernel_size=[self._temporal_kernel_size, 1, 1],
        strides=[self._temporal_strides, 1, 1],
        padding='same',
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        synchronized=self._use_sync_bn)

    self._spatial_conv = tf_keras.layers.Conv3D(
        filters=self._filters,
        kernel_size=[1, 3, 3],
        strides=[1, self._spatial_strides, self._spatial_strides],
        padding='same',
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        synchronized=self._use_sync_bn)

    self._expand_conv = tf_keras.layers.Conv3D(
        filters=4 * self._filters,
        kernel_size=[1, 1, 1],
        strides=[1, 1, 1],
        padding='same',
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm3 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        synchronized=self._use_sync_bn)

    if self._se_ratio and self._se_ratio > 0 and self._se_ratio <= 1:
      self._squeeze_excitation = nn_layers.SqueezeExcitation(
          in_filters=self._filters * 4,
          out_filters=self._filters * 4,
          se_ratio=self._se_ratio,
          use_3d_input=True,
          kernel_initializer=tf_utils.clone_initializer(
              self._kernel_initializer),
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)
    else:
      self._squeeze_excitation = None

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = None

    if self._use_self_gating:
      self._self_gating = SelfGating(filters=4 * self._filters)
    else:
      self._self_gating = None

    super(BottleneckBlock3D, self).build(input_shape)

  def get_config(self):
    config = {
        'filters': self._filters,
        'temporal_kernel_size': self._temporal_kernel_size,
        'temporal_strides': self._temporal_strides,
        'spatial_strides': self._spatial_strides,
        'use_self_gating': self._use_self_gating,
        'se_ratio': self._se_ratio,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon
    }
    base_config = super(BottleneckBlock3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, training=None):
    in_filters = inputs.shape.as_list()[-1]
    if in_filters == 4 * self._filters:
      if self._temporal_strides == 1 and self._spatial_strides == 1:
        shortcut = inputs
      else:
        shortcut = self._shortcut_maxpool(inputs)
    else:
      shortcut = self._shortcut_conv(inputs)
      shortcut = self._norm0(shortcut)

    x = self._temporal_conv(inputs)
    x = self._norm1(x)
    x = self._activation_fn(x)

    x = self._spatial_conv(x)
    x = self._norm2(x)
    x = self._activation_fn(x)

    x = self._expand_conv(x)
    x = self._norm3(x)

    # Apply self-gating, SE, stochastic depth.
    if self._self_gating:
      x = self._self_gating(x)
    if self._squeeze_excitation:
      x = self._squeeze_excitation(x)
    if self._stochastic_depth:
      x = self._stochastic_depth(x, training=training)

    # Apply activation before additional modules.
    x = self._activation_fn(x + shortcut)

    return x
