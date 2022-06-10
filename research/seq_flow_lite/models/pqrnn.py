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
"""Implementation of pQRNN model."""

from absl import logging
import tensorflow as tf

from layers import base_layers # import seq_flow_lite module
from layers import dense_layers # import seq_flow_lite module
from layers import misc_layers # import seq_flow_lite module
from layers import projection_layers # import seq_flow_lite module
from layers import qrnn_layers # import seq_flow_lite module
from layers import quantization_layers # import seq_flow_lite module


class Encoder(tf.keras.layers.Layer):
  """A pQRNN keras model."""

  def __init__(self, config, mode, **kwargs):
    super(Encoder, self).__init__(**kwargs)

    def _get_params(varname, default_value=None):
      value = config[varname] if varname in config else default_value
      default = "" if varname in config else " (default)"
      logging.info("%s = %s%s", varname, value, default)
      setattr(self, varname, value)

    _get_params("projection_bottleneck_size")
    _get_params("qrnn_state_size")
    _get_params("qrnn_kernel_width", 3)
    _get_params("qrnn_zoneout_probability")
    _get_params("number_qrnn_layers")
    _get_params("labels")
    _get_params("regularizer_scale")
    _get_params("quantize")

    self.num_classes = len(self.labels)
    self.parameters = base_layers.Parameters(
        mode, quantize=self.quantize, regularizer_scale=self.regularizer_scale)

    self.bottleneck_layer = dense_layers.BaseQDenseVarLen(
        units=self.projection_bottleneck_size,
        rank=3,
        parameters=self.parameters)

    self.qrnn_stack = qrnn_layers.QRNNBidirectionalStack(
        parameters=self.parameters,
        zoneout_probability=self.qrnn_zoneout_probability,
        kwidth=self.qrnn_kernel_width,
        state_size=self.qrnn_state_size,
        num_layers=self.number_qrnn_layers)

    self.attention_pool = misc_layers.AttentionPooling(
        parameters=self.parameters)

    self.final_fc = dense_layers.BaseQDense(
        units=self.num_classes,
        rank=2,
        parameters=self.parameters,
        activation=None)

  def call(self, projection, seq_length):
    mask = tf.sequence_mask(
        seq_length, tf.shape(projection)[1], dtype=tf.float32)
    inverse_normalizer = tf.math.reciprocal(tf.reduce_sum(mask))
    maskr3 = tf.expand_dims(mask, axis=2)
    if self.parameters.mode in [base_layers.TRAIN, base_layers.EVAL]:
      projection = projection * maskr3
    bottleneck = self.bottleneck_layer(projection, maskr3, inverse_normalizer)
    outputs = self.qrnn_stack(bottleneck, maskr3, inverse_normalizer)
    pre_logits = self.attention_pool(outputs, maskr3, inverse_normalizer)
    return self.final_fc(pre_logits)

class Model(Encoder):

  def __init__(self, config, mode, **kwargs):
    super(Model, self).__init__(config, mode, **kwargs)
    self.projection = projection_layers.ProjectionLayer(config, mode)

  def call(self, inputs):
    projection, seq_length = self.projection(inputs)
    return super(Model, self).call(projection, seq_length)
