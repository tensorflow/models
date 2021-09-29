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
"""Layers for quantization."""

import tensorflow as tf

from layers import base_layers # import seq_flow_lite module


class ActivationQuantization(base_layers.BaseLayer):
  """A class that applies quantization to a activation tensor."""

  def __init__(self, ema_decay=0.99, num_bits=8, **kwargs):
    self.ema_decay = ema_decay
    self.num_bits = num_bits
    super(ActivationQuantization, self).__init__(**kwargs)

  def build(self, input_shapes):
    if self.parameters.quantize:
      self.min_var = self.add_weight(
          "min", initializer=tf.keras.initializers.Zeros(), trainable=False)
      self.max_var = self.add_weight(
          "max", initializer=tf.keras.initializers.Ones(), trainable=False)

  def call(self, inputs):
    if self.parameters.quantize:
      if self.parameters.mode == base_layers.TRAIN:
        # Toco expects 0.0 to be part of the quantization range.
        batch_min = tf.minimum(tf.reduce_min(inputs), 0.0)
        min_var = self.assign_moving_average(self.min_var, batch_min,
                                             self.ema_decay)

        batch_max = tf.maximum(tf.reduce_max(inputs), 0.0)
        max_var = self.assign_moving_average(self.max_var, batch_max,
                                             self.ema_decay)
        with tf.control_dependencies([min_var, max_var]):
          return tf.quantization.fake_quant_with_min_max_vars(
              inputs, batch_min, batch_max, num_bits=self.num_bits)
      else:
        return tf.quantization.fake_quant_with_min_max_vars(
            inputs, self.min_var, self.max_var, num_bits=self.num_bits)
    return inputs

  def quantize_using_range(self, inputs):
    # This method can only be called after a call to "call" method in this class
    if self.parameters.quantize:
      return tf.quantization.fake_quant_with_min_max_vars(
          inputs, self.min_var, self.max_var, num_bits=self.num_bits)
    return inputs


class ConcatQuantization(ActivationQuantization):
  """A class that applies quantization to a activation tensor."""

  def __init__(self, axis=2, **kwargs):
    self.axis = axis
    super(ConcatQuantization, self).__init__(**kwargs)

  def _reduce_list(self, tensor_list, functor):
    reduce_result = [functor(tensor) for tensor in tensor_list]
    # Toco expects 0.0 to be part of the quantization range.
    reduce_result.append(tf.constant(0.0))
    return functor(tf.stack(reduce_result))

  def call(self, tensors):
    # Ignore empty invocations done to build the keras layer.
    if tensors is None:
      return
    if self.parameters.quantize:
      if self.parameters.mode == base_layers.TRAIN:
        # Toco expects 0.0 to be part of the quantization range.
        batch_min = self._reduce_list(tensors, tf.reduce_min)
        min_var = self.assign_moving_average(self.min_var, batch_min,
                                             self.ema_decay)

        batch_max = self._reduce_list(tensors, tf.reduce_max)
        max_var = self.assign_moving_average(self.max_var, batch_max,
                                             self.ema_decay)
      else:
        min_var, max_var = self.min_var, self.max_var

      tensors = [
          tf.quantization.fake_quant_with_min_max_vars(
              tensor, min_var, max_var, num_bits=self.num_bits)
          for tensor in tensors
      ]
      tensor = tf.concat(tensors, axis=self.axis)
      return tf.quantization.fake_quant_with_min_max_vars(
          tensor, min_var, max_var, num_bits=self.num_bits)
    return tf.concat(tensors, axis=self.axis)
