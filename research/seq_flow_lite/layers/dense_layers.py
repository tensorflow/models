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
"""Basic dense layers."""
import copy
import tensorflow as tf

from layers import base_layers # import seq_flow_lite module
from layers import normalization_layers # import seq_flow_lite module
from layers import quantization_layers # import seq_flow_lite module


class BaseQDense(base_layers.BaseLayer):
  """Quantized encoder dense layers."""

  def __init__(self,
               units,
               activation=tf.keras.layers.ReLU(),
               bias=True,
               rank=2,
               normalize=True,
               quantize_output=True,
               normalization_fn=None,
               **kwargs):
    self.units = units
    self.rank = rank
    assert rank >= 2 and rank <= 4
    self.activation = activation
    self.bias = bias
    self.normalize = normalize
    self.quantize_output = quantize_output
    if quantize_output:
      self.qoutput = quantization_layers.ActivationQuantization(**kwargs)
    self._create_normalizer(normalization_fn=normalization_fn, **kwargs)
    super(BaseQDense, self).__init__(**kwargs)

  def build(self, input_shapes):
    assert len(input_shapes) == self.rank
    if self.rank == 4:
      assert input_shapes[1] == 1 or input_shapes[2] == 1
    self.in_units = input_shapes[-1]
    shape = [self.in_units, self.units]
    self.w = self.add_weight_wrapper(shape=shape)
    if self.bias:
      self.b = self.add_bias(shape=[self.units])

  def _create_normalizer(self, normalization_fn, **kwargs):
    if normalization_fn is None:
      self.normalization = normalization_layers.BatchNormalization(**kwargs)
    else:
      self.normalization = copy.deepcopy(normalization_fn)

  def _dense_r2(self, inputs, normalize_method):
    outputs = tf.matmul(inputs, self.quantize_parameter(self.w))
    if self.bias:
      outputs = tf.nn.bias_add(outputs, self.b)
    if self.normalize:
      outputs = normalize_method(outputs)
    if self.activation:
      outputs = self.activation(outputs)
    if self.quantize_output:
      return self.qoutput(outputs)
    else:
      return outputs

  def _dense_r34(self, inputs, normalize_method):
    bsz = self.get_batch_dimension(inputs)
    outputs = tf.reshape(inputs, [-1, self.in_units])
    outputs = self._dense_r2(outputs, normalize_method)
    if self.rank == 3:
      return tf.reshape(outputs, [bsz, -1, self.units])
    elif inputs.get_shape().as_list()[1] == 1:
      return tf.reshape(outputs, [bsz, 1, -1, self.units])
    else:
      return tf.reshape(outputs, [bsz, -1, 1, self.units])

  def call(self, inputs):

    def normalize_method(tensor):
      return self.normalization(tensor)

    return self._do_call(inputs, normalize_method)

  def _do_call(self, inputs, normalize_method):
    if self.rank == 2:
      return self._dense_r2(inputs, normalize_method)
    return self._dense_r34(inputs, normalize_method)

  def quantize_using_output_range(self, tensor):
    return self.qoutput.quantize_using_range(tensor)


class BaseQDenseVarLen(BaseQDense):
  """Dense on variable length sequence."""

  def _create_normalizer(self, normalization_fn, **kwargs):
    if normalization_fn is None:
      self.normalization = normalization_layers.VarLenBatchNormalization(
          rank=2, **kwargs)
    else:
      self.normalization = copy.deepcopy(normalization_fn)

  def call(self, inputs, mask, inverse_normalizer=None):
    if inverse_normalizer is None:
      inverse_normalizer = self.inverse_normalizer(mask)

    def normalize_method(tensor):
      maskr2 = tf.reshape(mask, [-1, 1])
      return self.normalization(tensor, maskr2, inverse_normalizer)

    return self._do_call(inputs, normalize_method)
