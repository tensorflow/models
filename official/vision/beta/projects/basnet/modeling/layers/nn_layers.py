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
"""Contains common building blocks for neural networks."""

# Import libraries
import tensorflow as tf

from official.modeling import tf_utils

@tf.keras.utils.register_keras_serializable(package='Vision')
class ConvBNReLU(tf.keras.layers.Layer):
  """ 'Conv + BN + ReLU' layer in BASNet. """

  def __init__(self,
               filters,
               kernel_size,
               strides,
               dilation,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               **kwargs):
    """Implementation for squeeze and excitation.

    Args:
      filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
      strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
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
 
    super(ConvBNReLU, self).__init__(**kwargs)

    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._dilation = dilation
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._activation = activation
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon

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
    self._conv1 = tf.keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=self._kernel_size,
        strides=self._strides,
        padding='same',
        dilation_rate=self._dilation,
        use_bias=True,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
 
    self._bn1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon        
        )
    super(ConvBNReLU, self).build(input_shape)

  def get_config(self):
    config = {
        'filters': self._filters,
        'kernel_size': self._kernel_size,
        'strides': self._strides,
        'dilation': self._dilation,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon
    }
    base_config = super(ConvBNReLU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


  def call(self, inputs):
    x = self._conv1(inputs)
    """
    x = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon        
        )(x)
    """
    x = self._bn1(x)
    x = self._activation_fn(x)

    return x



@tf.keras.utils.register_keras_serializable(package='Vision')
class SqueezeExcitation(tf.keras.layers.Layer):
  """Squeeze and excitation layer."""

  def __init__(self,
               in_filters,
               se_ratio,
               expand_ratio,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               **kwargs):
    """Implementation for squeeze and excitation.

    Args:
      in_filters: `int` number of filters of the input tensor.
      se_ratio: `float` or None. If not None, se ratio for the squeeze and
        excitation layer.
      expand_ratio: `int` expand_ratio for a MBConv block.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
        Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
        Default to None.
      activation: `str` name of the activation function.
      **kwargs: keyword arguments to be passed.
    """
    super(SqueezeExcitation, self).__init__(**kwargs)

    self._in_filters = in_filters
    self._se_ratio = se_ratio
    self._expand_ratio = expand_ratio
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._spatial_axis = [1, 2]
    else:
      self._spatial_axis = [2, 3]
    self._activation_fn = tf_utils.get_activation(activation)

  def build(self, input_shape):
    num_reduced_filters = max(1, int(self._in_filters * self._se_ratio))

    self._se_reduce = tf.keras.layers.Conv2D(
        filters=num_reduced_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)

    self._se_expand = tf.keras.layers.Conv2D(
        filters=self._in_filters * self._expand_ratio,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)

    super(SqueezeExcitation, self).build(input_shape)

  def get_config(self):
    config = {
        'in_filters': self._in_filters,
        'se_ratio': self._se_ratio,
        'expand_ratio': self._expand_ratio,
        'strides': self._strides,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
    }
    base_config = super(SqueezeExcitation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    x = tf.reduce_mean(inputs, self._spatial_axis, keepdims=True)
    x = self._se_expand(self._activation_fn(self._se_reduce(x)))

    return tf.sigmoid(x) * inputs


def get_stochastic_depth_rate(init_rate, i, n):
  """Get drop connect rate for the ith block.

  Args:
    init_rate: `float` initial drop rate.
    i: `int` order of the current block.
    n: `int` total number of blocks.

  Returns:
    Drop rate of the ith block.
  """
  if init_rate is not None:
    if init_rate < 0 or init_rate > 1:
      raise ValueError('Initial drop rate must be within 0 and 1.')
    rate = init_rate * float(i) / n
  else:
    rate = None
  return rate


@tf.keras.utils.register_keras_serializable(package='Vision')
class StochasticDepth(tf.keras.layers.Layer):
  """Stochastic depth layer."""

  def __init__(self, stochastic_depth_drop_rate, **kwargs):
    """Initialize stochastic depth.

    Args:
      stochastic_depth_drop_rate: `float` drop rate.
      **kwargs: keyword arguments to be passed.

    Returns:
      A output tensor, which should have the same shape as input.
    """
    super(StochasticDepth, self).__init__(**kwargs)
    self._drop_rate = stochastic_depth_drop_rate

  def get_config(self):
    config = {'drop_rate': self._drop_rate}
    base_config = super(StochasticDepth, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, training=None):
    if training is None:
      is_training = tf.keras.backend.learning_phase()
    if not is_training or self._drop_rate is None or self._drop_rate == 0:
      return inputs

    keep_prob = 1.0 - self._drop_rate
    batch_size = tf.shape(inputs)[0]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(
        [batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output


@tf.keras.utils.register_keras_serializable(package='Vision')
def pyramid_feature_fusion(inputs, target_level):
  """Fuse all feature maps in the feature pyramid at the target level.

  Args:
    inputs: a dictionary containing the feature pyramid. The size of the input
      tensor needs to be fixed.
    target_level: `int` the target feature level for feature fusion.

  Returns:
    A float Tensor of shape [batch_size, feature_height, feature_width,
      feature_channel].
  """
  # Convert keys to int.
  pyramid_feats = {int(k): v for k, v in inputs.items()}
  min_level = min(pyramid_feats.keys())
  max_level = max(pyramid_feats.keys())
  resampled_feats = []

  for l in range(min_level, max_level + 1):
    if l == target_level:
      resampled_feats.append(pyramid_feats[l])
    else:
      feat = pyramid_feats[l]
      target_size = list(feat.shape[1:3])
      target_size[0] *= 2**(l - target_level)
      target_size[1] *= 2**(l - target_level)
      # Casts feat to float32 so the resize op can be run on TPU.
      feat = tf.cast(feat, tf.float32)
      feat = tf.image.resize(
          feat, size=target_size, method=tf.image.ResizeMethod.BILINEAR)
      # Casts it back to be compatible with the rest opetations.
      feat = tf.cast(feat, pyramid_feats[l].dtype)
      resampled_feats.append(feat)

  return tf.math.add_n(resampled_feats)
