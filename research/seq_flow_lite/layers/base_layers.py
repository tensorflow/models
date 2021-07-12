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
"""Base layer for building models trained with quantization."""

import tensorflow as tf

TRAIN = "train"
EVAL = "eval"
PREDICT = "infer"
TFLITE = "tflite"
_MODE = [TRAIN, EVAL, PREDICT, TFLITE]


class Parameters:
  """A class that encapsulates parameters."""

  def __init__(self,
               mode,
               quantize=True,
               regularizer_scale=0.0,
               invalid_logit=-1e6,
               initializer=None):
    assert isinstance(quantize, bool)
    self.quantize = quantize
    assert mode in _MODE
    self.mode = mode
    self.regularizer_scale = regularizer_scale
    self.invalid_logit = invalid_logit
    self.initializer = initializer


class BaseLayer(tf.keras.layers.Layer):
  """Base class for encoders."""

  def __init__(self, parameters, **kwargs):
    assert isinstance(parameters, Parameters)
    self.parameters = parameters
    super(BaseLayer, self).__init__(**kwargs)

  def _assert_rank_and_type(self, tensor, rank, dtype=tf.float32):
    assert len(tensor.get_shape().as_list()) == rank
    assert tensor.dtype == dtype

  def add_qweight(self, shape, num_bits=8):
    """Return a quantized weight variable for the given shape."""
    if self.parameters.initializer is not None:
      initializer = self.parameters.initializer
    else:
      initializer = tf.keras.initializers.GlorotUniform()
    weight = self.add_weight(
        "weight", shape, initializer=initializer, trainable=True)
    self.add_reg_loss(weight)
    return self._weight_quantization(weight, num_bits=num_bits)

  def _weight_quantization(self, tensor, num_bits=8):
    """Quantize weights when enabled."""
    # For infer mode, toco computes the min/max from the weights offline to
    # quantize it. During train/eval this is computed from the current value
    # in the session by the graph itself.
    if self.parameters.quantize and self.parameters.mode in [TRAIN, EVAL]:
      # Toco expects 0.0 to be part of the quantization range.
      batch_min = tf.minimum(tf.reduce_min(tensor), 0.0)
      batch_max = tf.maximum(tf.reduce_max(tensor), 0.0)

      return tf.quantization.fake_quant_with_min_max_vars(
          tensor, batch_min, batch_max, num_bits=num_bits)
    else:
      return tensor

  def add_bias(self, shape):
    weight = self.add_weight(
        "bias",
        shape,
        initializer=tf.keras.initializers.Zeros(),
        trainable=True)
    self.add_reg_loss(weight)
    return weight

  def add_reg_loss(self, weight):
    if self.parameters.regularizer_scale > 0.0:
      reg_scale = tf.convert_to_tensor(self.parameters.regularizer_scale)
      reg_loss = tf.nn.l2_loss(weight) * reg_scale
      self.add_loss(reg_loss)

  def assign_moving_average(self, var, update, ema_decay):
    return var.assign(var.read_value() * (1 - ema_decay) + (ema_decay) * update)

  def qrange_sigmoid(self, tensor):
    if self.parameters.quantize:
      return tf.quantization.fake_quant_with_min_max_args(tensor, 0.0, 1.0)
    return tensor

  def qrange_tanh(self, tensor):
    if self.parameters.quantize:
      return tf.quantization.fake_quant_with_min_max_args(tensor, -1.0, 1.0)
    return tensor

  def quantized_tanh(self, tensor):
    return self.qrange_tanh(tf.tanh(tensor))

  def quantized_sigmoid(self, tensor):
    return self.qrange_sigmoid(tf.sigmoid(tensor))

  def get_batch_dimension(self, tensor):
    return tensor.get_shape().as_list()[0] or tf.shape(tensor)[0]
