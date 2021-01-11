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

from typing import Optional

# Import libraries

from absl import logging
import tensorflow as tf

from official.modeling import tf_utils


def make_divisible(value: float,
                   divisor: int,
                   min_value: Optional[float] = None
                   ) -> int:
  """This is to ensure that all layers have channels that are divisible by 8.

  Args:
    value: `float` original value.
    divisor: `int` the divisor that need to be checked upon.
    min_value: `float` minimum value threshold.

  Returns:
    The adjusted value in `int` that divisible against divisor.
  """
  if min_value is None:
    min_value = divisor
  new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_value < 0.9 * value:
    new_value += divisor
  return new_value


def round_filters(filters: int,
                  multiplier: float,
                  divisor: int = 8,
                  min_depth: Optional[int] = None,
                  skip: bool = False):
  """Round number of filters based on width multiplier."""
  orig_f = filters
  if skip or not multiplier:
    return filters

  new_filters = make_divisible(value=filters * multiplier,
                               divisor=divisor,
                               min_value=min_depth)

  logging.info('round_filter input=%s output=%s', orig_f, new_filters)
  return int(new_filters)


@tf.keras.utils.register_keras_serializable(package='Vision')
class SqueezeExcitation(tf.keras.layers.Layer):
  """Squeeze and excitation layer."""

  def __init__(self,
               in_filters,
               out_filters,
               se_ratio,
               divisible_by=1,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               gating_activation='sigmoid',
               **kwargs):
    """Implementation for squeeze and excitation.

    Args:
      in_filters: `int` number of filters of the input tensor.
      out_filters: `int` number of filters of the output tensor.
      se_ratio: `float` or None. If not None, se ratio for the squeeze and
        excitation layer.
      divisible_by: `int` ensures all inner dimensions are divisible by this
        number.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
        Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
        Default to None.
      activation: `str` name of the activation function.
      gating_activation: `str` name of the activation function for final gating
        function.
      **kwargs: keyword arguments to be passed.
    """
    super(SqueezeExcitation, self).__init__(**kwargs)

    self._in_filters = in_filters
    self._out_filters = out_filters
    self._se_ratio = se_ratio
    self._divisible_by = divisible_by
    self._activation = activation
    self._gating_activation = gating_activation
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._spatial_axis = [1, 2]
    else:
      self._spatial_axis = [2, 3]
    self._activation_fn = tf_utils.get_activation(activation)
    self._gating_activation_fn = tf_utils.get_activation(gating_activation)

  def build(self, input_shape):
    num_reduced_filters = make_divisible(
        max(1, int(self._in_filters * self._se_ratio)),
        divisor=self._divisible_by)

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
        filters=self._out_filters,
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
        'out_filters': self._out_filters,
        'se_ratio': self._se_ratio,
        'divisible_by': self._divisible_by,
        'strides': self._strides,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'gating_activation': self._gating_activation,
    }
    base_config = super(SqueezeExcitation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    x = tf.reduce_mean(inputs, self._spatial_axis, keepdims=True)
    x = self._activation_fn(self._se_reduce(x))
    x = self._gating_activation_fn(self._se_expand(x))
    return x * inputs


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
      training = tf.keras.backend.learning_phase()
    if not training or self._drop_rate is None or self._drop_rate == 0:
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
