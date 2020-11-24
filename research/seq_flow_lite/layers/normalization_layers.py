# Copyright 2020 The TensorFlow Authors All Rights Reserved.
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
# Lint as: python3
"""Layers for normalization."""
import tensorflow as tf

from layers import base_layers # import seq_flow_lite module
from layers import quantization_layers # import seq_flow_lite module
from tf_ops import tf_custom_ops_py # import seq_flow_lite module


class BatchNormalization(base_layers.BaseLayer):
  """A class that applies batch normalization to the input tensor."""

  def __init__(self, ema_decay=0.999, **kwargs):
    self.ema_decay = ema_decay
    super(BatchNormalization, self).__init__(**kwargs)

  def build(self, input_shapes):
    self.reduce_dims = list(range(len(input_shapes) - 1))
    shape = [input_shapes[-1]]
    self.offset = self.add_weight(
        "offset",
        shape=shape,
        initializer=tf.keras.initializers.Zeros(),
        trainable=True)
    self.scale = self.add_weight(
        "scale",
        shape=shape,
        initializer=tf.keras.initializers.Ones(),
        trainable=True)
    self.mva_mean = self.add_weight(
        "mva_mean",
        shape=shape,
        initializer=tf.keras.initializers.Zeros(),
        trainable=False)
    self.mva_var = self.add_weight(
        "mva_variance",
        shape=shape,
        initializer=tf.keras.initializers.Ones(),
        trainable=False)

  def call(self, inputs):
    mean_mom, var_mom = None, None
    if self.parameters.mode == base_layers.TRAIN:
      mean_mom, var_mom = tf.nn.moments(inputs, self.reduce_dims)
    return self._batch_norm(inputs, mean_mom, var_mom)

  def _batch_norm(self, inputs, mean_mom, var_mom):
    if self.parameters.mode == base_layers.TRAIN:
      # During training compute summay stats, update them to moving average
      # variables and use the summary stas for batch normalization.
      with tf.control_dependencies([
          self.assign_moving_average(self.mva_mean, mean_mom, self.ema_decay),
          self.assign_moving_average(self.mva_var, var_mom, self.ema_decay)
      ]):
        tensor = tf.nn.batch_normalization(inputs, mean_mom, var_mom,
                                           self.offset, self.scale, 1e-9)
    else:
      # During eval/inference use the moving average variable for batch
      # normalization. The variables would be frozen to constants before
      # saving graph.
      tensor = tf.nn.batch_normalization(inputs, self.mva_mean, self.mva_var,
                                         self.offset, self.scale, 1e-9)
    return tensor


class VarLenBatchNormalization(BatchNormalization):
  """A class that applies batch normalization to the input tensor."""

  def __init__(self, rank=2, **kwargs):
    self.rank = rank
    assert rank == 2 or rank == 4
    super(VarLenBatchNormalization, self).__init__(**kwargs)

  def _reduce(self, tensor, multiplier):
    return tf.reduce_sum(tensor, axis=self.reduce_dims) * multiplier

  def call(self, inputs, mask, inverse_normalizer):
    if self.parameters.mode == base_layers.TRAIN:
      self._assert_rank_and_type(inputs, self.rank)
      self._assert_rank_and_type(mask, self.rank)
      inputs = mask * inputs
      mean_mom = self._reduce(inputs, inverse_normalizer)
      var_mom = self._reduce(inputs * inputs, inverse_normalizer)
      return mask * self._batch_norm(inputs, mean_mom, var_mom)
    elif self.parameters.mode == base_layers.EVAL:
      return mask * self._batch_norm(inputs, None, None)
    return self._batch_norm(inputs, None, None)


class LayerNormalization(base_layers.BaseLayer):
  """A class that applies layer normalization to the input tensor."""

  def __init__(self, axes=None, **kwargs):
    self.axes = axes or [-1]
    self.qactivation = quantization_layers.ActivationQuantization(**kwargs)
    super(LayerNormalization, self).__init__(**kwargs)

  def build(self, input_shape):
    self.rank = len(input_shape)
    for i, axis in enumerate(self.axes):
      if axis < 0:
        self.axes[i] += self.rank
      assert (self.axes[i] > 0 and self.axes[i] < self.rank)
    self.offset = self.add_weight(
        "offset",
        shape=[1],
        initializer=tf.keras.initializers.Zeros(),
        trainable=True)
    self.scale = self.add_weight(
        "scale",
        shape=[1],
        initializer=tf.keras.initializers.Ones(),
        trainable=True)

  def call(self, tensor):
    tensor = self.qactivation(tensor)
    if self.parameters.mode != base_layers.TFLITE:
      mean, variance = tf.nn.moments(tensor, self.axes, keepdims=True)
      # If all the values in the tensor are same, variance will be 0. Adding a
      # small epsilon to variance ensures that we get 0 as the normalized result
      # instead of NaN in the resulting tensor.
      tensor = (tensor - mean) / tf.sqrt(variance + 1e-6)
      return tensor * self.scale + self.offset
    else:
      return tf_custom_ops_py.layer_norm(
          tensor, self.scale, self.offset, axes=self.axes)
