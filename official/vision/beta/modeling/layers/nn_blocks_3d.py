# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Contains common building blocks for 3D networks."""
# Import libraries
import tensorflow as tf

from official.modeling import tf_utils


@tf.keras.utils.register_keras_serializable(package='Vision')
class SelfGating(tf.keras.layers.Layer):
  """Feature gating as used in S3D-G (https://arxiv.org/pdf/1712.04851.pdf)."""

  def __init__(self, filters, **kwargs):
    """Constructor.

    Args:
      filters: `int` number of filters for the convolutional layer.
      **kwargs: keyword arguments to be passed.
    """
    super(SelfGating, self).__init__(**kwargs)
    self._filters = filters

  def build(self, input_shape):
    self._spatial_temporal_average = tf.keras.layers.GlobalAveragePooling3D()

    # No BN and activation after conv.
    self._transformer_w = tf.keras.layers.Conv3D(
        filters=self._filters,
        kernel_size=[1, 1, 1],
        use_bias=True,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(
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


@tf.keras.utils.register_keras_serializable(package='Vision')
class BottleneckBlock3D(tf.keras.layers.Layer):
  """A 3D bottleneck block."""

  def __init__(self,
               filters,
               temporal_kernel_size,
               temporal_strides,
               spatial_strides,
               use_self_gating=False,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               **kwargs):
    """A 3D bottleneck block with BN after convolutions.

    Args:
      filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
      temporal_kernel_size: `int` kernel size for the temporal convolutional
        layer.
      temporal_strides: `int` temporal stride for the temporal convolutional
        layer.
      spatial_strides: `int` spatial stride for the spatial convolutional layer.
      use_self_gating: `bool` apply self-gating module or not.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
        Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
        Default to None.
      activation: `str` name of the activation function.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      **kwargs: keyword arguments to be passed.
    """
    super(BottleneckBlock3D, self).__init__(**kwargs)

    self._filters = filters
    self._temporal_kernel_size = temporal_kernel_size
    self._spatial_strides = spatial_strides
    self._temporal_strides = temporal_strides
    self._use_self_gating = use_self_gating
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    if use_sync_bn:
      self._norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf.keras.layers.BatchNormalization
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)

  def build(self, input_shape):
    self._shortcut_maxpool = tf.keras.layers.MaxPool3D(
        pool_size=[1, 1, 1],
        strides=[
            self._temporal_strides, self._spatial_strides, self._spatial_strides
        ])

    self._shortcut_conv = tf.keras.layers.Conv3D(
        filters=4 * self._filters,
        kernel_size=1,
        strides=[
            self._temporal_strides, self._spatial_strides, self._spatial_strides
        ],
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm0 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)

    self._temporal_conv = tf.keras.layers.Conv3D(
        filters=self._filters,
        kernel_size=[self._temporal_kernel_size, 1, 1],
        strides=[self._temporal_strides, 1, 1],
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)

    self._spatial_conv = tf.keras.layers.Conv3D(
        filters=self._filters,
        kernel_size=[1, 3, 3],
        strides=[1, self._spatial_strides, self._spatial_strides],
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)

    self._expand_conv = tf.keras.layers.Conv3D(
        filters=4 * self._filters,
        kernel_size=[1, 1, 1],
        strides=[1, 1, 1],
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm3 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)

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
        'use_projection': self._use_projection,
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
    # Apply activation before additional modules.
    x = self._activation_fn(x + shortcut)

    if self._self_gating:
      x = self._self_gating(x)

    return x
