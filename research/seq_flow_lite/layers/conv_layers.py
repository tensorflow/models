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
"""Base layer for convolution."""
import copy
import tensorflow as tf

from layers import base_layers # import seq_flow_lite module
from layers import normalization_layers # import seq_flow_lite module
from layers import quantization_layers # import seq_flow_lite module


class EncoderQConvolution(base_layers.BaseLayer):
  """Quantized encoder convolution layers."""

  def __init__(self,
               filters,
               ksize,
               stride=1,
               padding="SAME",
               dilations=None,
               activation=tf.keras.layers.ReLU(),
               bias=True,
               rank=4,
               normalization_fn=None,
               **kwargs):
    self.out_filters = filters
    assert rank >= 3 and rank <= 4
    self.rank = rank
    self.ksize = self._unpack(ksize)
    self.strides = self._unpack(stride)
    self.dilations = [1] + self._unpack(dilations) + [1] if dilations else None
    self.activation = activation
    self.bias = bias
    self.padding = padding
    self.qoutput = quantization_layers.ActivationQuantization(**kwargs)
    self._create_normalizer(normalization_fn=normalization_fn, **kwargs)
    super(EncoderQConvolution, self).__init__(**kwargs)

  def _unpack(self, value):
    if not isinstance(value, list):
      assert isinstance(value, int)
      return [1 if self.rank == 3 else value, value]
    else:
      assert len(value) == 2 and self.rank == 4
      assert isinstance(value[0], int) and isinstance(value[1], int)
      return value

  def build(self, input_shapes):
    assert len(input_shapes) == self.rank
    self.in_filters = input_shapes[-1]
    shape = self.ksize + [self.in_filters, self.out_filters]
    self.filters = self.add_weight_wrapper(shape=shape)
    if self.bias:
      self.b = self.add_bias(shape=[self.out_filters])

  def _create_normalizer(self, normalization_fn, **kwargs):
    if normalization_fn is None:
      self.normalization = normalization_layers.BatchNormalization(**kwargs)
    else:
      self.normalization = copy.deepcopy(normalization_fn)

  def _conv_r4(self, inputs, normalize_method):
    outputs = tf.nn.conv2d(
        inputs,
        self.quantize_parameter(self.filters),
        strides=self.strides,
        padding=self.padding,
        dilations=self.dilations)
    if self.bias:
      outputs = tf.nn.bias_add(outputs, self.b)
    outputs = normalize_method(outputs)
    if self.activation:
      outputs = self.activation(outputs)
    return self.qoutput(outputs)

  def _conv_r3(self, inputs, normalize_method):
    bsz = self.get_batch_dimension(inputs)
    inputs_r4 = tf.reshape(inputs, [bsz, 1, -1, self.in_filters])
    outputs = self._conv_r4(inputs_r4, normalize_method)
    return tf.reshape(outputs, [bsz, -1, self.out_filters])

  def call(self, inputs):

    def normalize_method(tensor):
      return self.normalization(tensor)

    return self._do_call(inputs, normalize_method)

  def _do_call(self, inputs, normalize_method):
    if self.rank == 3:
      return self._conv_r3(inputs, normalize_method)
    return self._conv_r4(inputs, normalize_method)

  def quantize_using_output_range(self, tensor):
    return self.qoutput.quantize_using_range(tensor)


class EncoderQConvolutionVarLen(EncoderQConvolution):
  """Convolution on variable length sequence."""

  def _create_normalizer(self, normalization_fn, **kwargs):
    if normalization_fn is None:
      self.normalization = normalization_layers.VarLenBatchNormalization(
          rank=4, **kwargs)
    else:
      self.normalization = copy.deepcopy(normalization_fn)

  def call(self, inputs, mask, inverse_normalizer):

    def normalize_method(tensor):
      return self.normalization(tensor, mask, inverse_normalizer)

    return self._do_call(inputs, normalize_method)
