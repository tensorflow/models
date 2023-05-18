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
"""Layers for QRNN."""
import tensorflow as tf

from layers import base_layers # import seq_flow_lite module
from layers import conv_layers # import seq_flow_lite module
from layers import dense_layers # import seq_flow_lite module
from layers import quantization_layers # import seq_flow_lite module
from tf_ops import tf_custom_ops_py # import seq_flow_lite module

QUASI_RNN_POOLING_F = "f"
QUASI_RNN_POOLING_FO = "fo"
QUASI_RNN_POOLING_IFO = "ifo"
_QUASI_RNN_POOLING_TO_NUMBER_OF_GATES_MAP = {
    QUASI_RNN_POOLING_F: 2,
    QUASI_RNN_POOLING_FO: 3,
    QUASI_RNN_POOLING_IFO: 4,
}


class QRNNUnidirectionalPoolingCore(base_layers.BaseLayer):
  """Create a unidirectional QRNN pooling inner loop."""

  def __init__(self, forward=True, **kwargs):
    self.forward = forward
    super(QRNNUnidirectionalPoolingCore, self).__init__(**kwargs)

  def call(self, multiplier, constant):
    if self.parameters.mode != base_layers.TFLITE:
      return self._qrnn_pooling(multiplier, constant)
    else:
      return tf_custom_ops_py.pooling_op(multiplier, constant,
                                         [1.0 if self.forward else 0.0])

  def _qrnn_pooling(self, multipler, constant):
    """Pooling step computes the internal states for all timesteps."""
    assert multipler.get_shape().as_list() == constant.get_shape().as_list()

    gate_static_shape = multipler.get_shape().as_list()
    gate_shape = tf.shape(multipler)

    feature_size = gate_static_shape[2]
    assert feature_size is not None
    batch_size = gate_static_shape[0] or gate_shape[0]
    max_timestep = gate_static_shape[1] or gate_shape[1]

    dynamic_loop = gate_static_shape[1] is None

    # Get multiplier/constant in [timestep, batch, feature_size] format
    multiplier_transposed = tf.transpose(multipler, [1, 0, 2])
    constant_transposed = tf.transpose(constant, [1, 0, 2])

    # Start state
    state = tf.zeros((batch_size, feature_size), tf.float32)
    if dynamic_loop:

      # One pooling step
      def _step(index, state, states):
        m = multiplier_transposed[index, :, :]
        c = constant_transposed[index, :, :]
        new_state = state * m + c
        next_index = index + 1 if self.forward else index - 1
        return next_index, new_state, states.write(index, new_state)

      # Termination condition
      def _termination(index, state, states):
        del state, states
        return (index < max_timestep) if self.forward else (index >= 0)

      states = tf.TensorArray(tf.float32, size=max_timestep)
      index = 0 if self.forward else max_timestep - 1

      # Dynamic pooling loop
      _, state, states = tf.while_loop(_termination, _step,
                                       [index, state, states])
      states = states.stack()
    else:
      # Unstack them to process one timestep at a time
      multiplier_list = tf.unstack(multiplier_transposed)
      constant_list = tf.unstack(constant_transposed)
      states = []

      # Unroll either forward or backward based on the flag `forward`
      timesteps = list(range(max_timestep)) if self.forward else reversed(
          list(range(max_timestep)))

      # Static pooling loop
      for time in timesteps:
        state = state * multiplier_list[time] + constant_list[time]
        states.append(state)

      # Stack them back in the right order
      states = tf.stack(states if self.forward else list(reversed(states)))

    # Change to [batch, timestep, feature_size]
    return tf.transpose(states, [1, 0, 2])


class QRNNUnidirectionalPooling(base_layers.BaseLayer):
  """Create a unidirectional QRNN pooling."""

  def __init__(self,
               zoneout_probability=0.0,
               forward=True,
               pooling=QUASI_RNN_POOLING_FO,
               output_quantized=True,
               **kwargs):
    self.zoneout_probability = zoneout_probability
    self.pooling = pooling
    self.forward = forward
    self.output_quantized = output_quantized
    if output_quantized and self.pooling == QUASI_RNN_POOLING_IFO:
      self.qoutputs = quantization_layers.ActivationQuantization()
    self.num_gates = _QUASI_RNN_POOLING_TO_NUMBER_OF_GATES_MAP[pooling]
    assert pooling in _QUASI_RNN_POOLING_TO_NUMBER_OF_GATES_MAP.keys()
    self.pooling_core = QRNNUnidirectionalPoolingCore(forward=forward, **kwargs)
    super(QRNNUnidirectionalPooling, self).__init__(**kwargs)

  def call(self, gates, mask):
    return self._create_qrnn_pooling_unidirectional(gates, mask)

  def _qrnn_preprocess(self, gates):
    """Preprocess the gate inputs to the pooling layer."""
    assert self.num_gates == len(gates)
    dim = lambda tensor, index: tensor.get_shape().as_list()[index]

    for tensor in gates:
      assert len(tensor.get_shape().as_list()) == 3
      for idx in range(3):
        assert dim(gates[0], idx) == dim(tensor, idx)

    if self.pooling == QUASI_RNN_POOLING_F:
      z = self.quantized_tanh(gates[0], tf_only=True)
      f = self.quantized_sigmoid(gates[1], tf_only=True)
      return f, self.qrange_tanh(self.qrange_sigmoid(1 - f) * z), 1
    elif self.pooling == QUASI_RNN_POOLING_FO:
      z = self.quantized_tanh(gates[0], tf_only=True)
      f = self.quantized_sigmoid(gates[1], tf_only=True)
      o = self.quantized_sigmoid(gates[2], tf_only=True)
      return f, self.qrange_tanh(self.qrange_sigmoid(1 - f) * z), o
    else:  # self.pooling == QUASI_RNN_POOLING_IFO:
      z = self.quantized_tanh(gates[0], tf_only=True)
      i = self.quantized_sigmoid(gates[1], tf_only=True)
      f = self.quantized_sigmoid(gates[2], tf_only=True)
      o = self.quantized_sigmoid(gates[3], tf_only=True)
      return f, self.qrange_tanh(i * z), o

  def _qrnn_postprocess(self, states, multiplier):
    """Postprocess the states and return the output tensors."""
    if self.pooling == QUASI_RNN_POOLING_F:
      return states
    elif self.pooling == QUASI_RNN_POOLING_FO:
      return self.qrange_tanh(states) * multiplier
    else:  # self.pooling == QUASI_RNN_POOLING_IFO
      return self.qoutputs(states) * multiplier

  def _qrnn_zoneout(self, multipler, constant):
    """Zoneout regularization for Quasi RNN."""
    enable_zoneout = self.zoneout_probability > 0.0
    if enable_zoneout and self.parameters.mode == base_layers.TRAIN:
      # zoneout_mask is 1.0 with self.zoneout_probability and 0.0 with
      # probability (1 - self.zoneout_probability)
      zoneout_mask = tf.random.uniform(tf.shape(multipler), maxval=1.0)
      zoneout_mask = tf.floor(zoneout_mask + self.zoneout_probability)

      # When zoneout_mask is 1.0, do not update the state, retain the old state.
      # This is achieved by making the multiplier 1.0 and constant 0.0.
      # When zoneout_mask is 0.0 the multiplier and constant are unaffected.
      # multipler is expected to be in the range [0.0, 1.0]. This is true since
      # it is the result of a sigmoid.
      multipler = tf.maximum(zoneout_mask, multipler)
      constant *= (1 - zoneout_mask)
    return multipler, constant

  def _create_qrnn_pooling_unidirectional(self, gates, mask):
    """Create QRNN Pooling in either forward or backward direction."""
    m1, c1, outgate = self._qrnn_preprocess(gates)

    # For inference zero padding will not be used. Hence sequence length is
    # not necessary.
    if self.parameters.mode not in [base_layers.PREDICT, base_layers.TFLITE]:
      m1 = m1 * mask + (1 - mask) * tf.ones_like(m1)
      c1 *= mask

    m1, c1 = self._qrnn_zoneout(m1, c1)

    states = self.pooling_core(m1, c1)

    outputs = self._qrnn_postprocess(states, outgate)

    # For inference zero padding will not be used. Hence sequence length is
    # not necessary.
    if self.parameters.mode not in [base_layers.PREDICT, base_layers.TFLITE]:
      outputs *= mask

    if self.output_quantized:
      if self.pooling in [QUASI_RNN_POOLING_FO, QUASI_RNN_POOLING_F]:
        outputs = self.qrange_tanh(outputs)
      else:
        outputs = self.qoutputs.quantize_using_range(outputs)

    return outputs


class QRNNUnidirectional(base_layers.BaseLayer):
  """Create a unidirectional QRNN encoder."""

  def __init__(self,
               kwidth,
               state_size,
               zoneout_probability=0.0,
               forward=True,
               pooling=QUASI_RNN_POOLING_FO,
               output_quantized=True,
               normalization_fn=None,
               **kwargs):
    self.forward = forward
    self.kwidth = kwidth
    self.pooling = pooling
    self.state_size = state_size
    assert pooling in _QUASI_RNN_POOLING_TO_NUMBER_OF_GATES_MAP.keys()
    self.num_gates = _QUASI_RNN_POOLING_TO_NUMBER_OF_GATES_MAP[pooling]
    self.gate_layers = []
    for _ in range(self.num_gates):
      self.gate_layers.append(
          conv_layers.EncoderQConvolutionVarLen(
              filters=state_size,
              ksize=kwidth,
              rank=3,
              padding="VALID",
              activation=None,
              normalization_fn=normalization_fn,
              **kwargs))
    padding = [kwidth - 1, 0] if forward else [0, kwidth - 1]
    self.zero_pad = tf.keras.layers.ZeroPadding1D(padding=padding)
    self.qrnn_pooling = QRNNUnidirectionalPooling(
        forward=forward,
        zoneout_probability=zoneout_probability,
        output_quantized=output_quantized,
        pooling=pooling,
        **kwargs)
    super(QRNNUnidirectional, self).__init__(**kwargs)

  def call(self, inputs, mask, inverse_normalizer=None):
    if inverse_normalizer is None:
      inverse_normalizer = tf.math.reciprocal(tf.reduce_sum(mask))
    self._assert_rank_and_type(inputs, 3)
    self._assert_rank_and_type(mask, 3)
    maskr4 = tf.expand_dims(mask, axis=1)
    padded_inputs = self.zero_pad(inputs)
    gates = [
        layer(padded_inputs, maskr4, inverse_normalizer)
        for layer in self.gate_layers
    ]
    return self.qrnn_pooling(gates, mask)


class QRNNUnidirectionalWithBottleneck(base_layers.BaseLayer):
  """Create a unidirectional QRNN encoder with bottlenecks."""

  def __init__(self,
               kwidth,
               state_size,
               bottleneck_size,
               zoneout_probability=0.0,
               forward=True,
               pooling=QUASI_RNN_POOLING_FO,
               output_quantized=True,
               **kwargs):
    self.bottleneck_size = bottleneck_size
    self.state_size = state_size
    self.forward = forward
    self.kwidth = kwidth
    self.pooling = pooling
    self.state_size = state_size
    assert pooling in _QUASI_RNN_POOLING_TO_NUMBER_OF_GATES_MAP.keys()
    self.num_gates = _QUASI_RNN_POOLING_TO_NUMBER_OF_GATES_MAP[pooling]
    self.qrnn_pooling = QRNNUnidirectionalPooling(
        forward=forward,
        zoneout_probability=zoneout_probability,
        output_quantized=output_quantized,
        pooling=pooling,
        **kwargs)
    self.pre_conv_layers = []
    self.gate_layers = []
    self.post_conv_layers = []
    for _ in range(self.num_gates):
      self.pre_conv_layers.append(
          dense_layers.BaseQDense(bottleneck_size, rank=3, **kwargs))
      self.gate_layers.append(
          conv_layers.EncoderQConvolution(
              filters=bottleneck_size,
              ksize=kwidth,
              rank=3,
              padding="SAME",
              normalization_fn=None,
              **kwargs))
      self.post_conv_layers.append(
          dense_layers.BaseQDense(
              state_size, rank=3, activation=None, **kwargs))
    super(QRNNUnidirectionalWithBottleneck, self).__init__(**kwargs)

  def call(self, inputs, mask, inverse_normalizer=None):
    if inverse_normalizer is None:
      inverse_normalizer = tf.math.reciprocal(tf.reduce_sum(mask))
    self._assert_rank_and_type(inputs, 3)
    self._assert_rank_and_type(mask, 3)
    pre_conv_out = [layer(inputs) for layer in self.pre_conv_layers]
    gates = [layer(pre_conv_out[i]) for i, layer in enumerate(self.gate_layers)]
    post_conv_out = [
        layer(gates[i]) for i, layer in enumerate(self.post_conv_layers)
    ]
    return self.qrnn_pooling(post_conv_out, mask)


class QRNNBidirectional(base_layers.BaseLayer):
  """Create a bidirectional QRNN encoder."""

  def __init__(self,
               kwidth,
               state_size,
               zoneout_probability=0.0,
               pooling=QUASI_RNN_POOLING_FO,
               bottleneck_size=None,
               normalization_fn=None,
               **kwargs):
    self.pooling = pooling
    if bottleneck_size is None:
      self.forward = QRNNUnidirectional(
          kwidth=kwidth,
          state_size=state_size,
          forward=True,
          output_quantized=False,
          zoneout_probability=zoneout_probability,
          pooling=pooling,
          normalization_fn=normalization_fn,
          **kwargs)
      self.backward = QRNNUnidirectional(
          kwidth=kwidth,
          state_size=state_size,
          forward=False,
          output_quantized=False,
          zoneout_probability=zoneout_probability,
          pooling=pooling,
          normalization_fn=normalization_fn,
          **kwargs)
    else:
      assert normalization_fn is None, (
          "normalization_fn will not take an effect")
      self.forward = QRNNUnidirectionalWithBottleneck(
          kwidth=kwidth,
          state_size=state_size,
          bottleneck_size=bottleneck_size,
          forward=True,
          output_quantized=False,
          zoneout_probability=zoneout_probability,
          pooling=pooling,
          **kwargs)
      self.backward = QRNNUnidirectionalWithBottleneck(
          kwidth=kwidth,
          state_size=state_size,
          bottleneck_size=bottleneck_size,
          forward=False,
          output_quantized=False,
          zoneout_probability=zoneout_probability,
          pooling=pooling,
          **kwargs)

    self.qconcat = quantization_layers.ConcatQuantization(axis=2, **kwargs)
    super(QRNNBidirectional, self).__init__(**kwargs)

  def call(self, inputs, mask, inverse_normalizer=None):
    if inverse_normalizer is None:
      inverse_normalizer = tf.math.reciprocal(tf.reduce_sum(mask))
    fwd_outputs = self.forward(inputs, mask, inverse_normalizer)
    bwd_outputs = self.backward(inputs, mask, inverse_normalizer)

    if self.pooling in [QUASI_RNN_POOLING_FO, QUASI_RNN_POOLING_F]:
      outputs = [self.qrange_tanh(fwd_outputs), self.qrange_tanh(bwd_outputs)]
      outputs = self.qrange_tanh(tf.concat(outputs, axis=2))
    else:
      outputs = self.qconcat([fwd_outputs, bwd_outputs])

    return outputs


class QRNNBidirectionalStack(base_layers.BaseLayer):
  """Create a stack of bidirectional QRNN encoder."""

  def __init__(self,
               num_layers,
               kwidth,
               state_size,
               zoneout_probability=0.0,
               layerwise_decaying_zoneout=True,
               pooling=QUASI_RNN_POOLING_FO,
               bottleneck_size=None,
               normalization_fn=None,
               **kwargs):
    self.layers = []
    zp = zoneout_probability
    for idx in range(num_layers):
      if layerwise_decaying_zoneout:
        zp = (zoneout_probability**(idx + 1))
      self.layers.append(
          QRNNBidirectional(
              kwidth=kwidth,
              state_size=state_size,
              zoneout_probability=zp,
              pooling=pooling,
              bottleneck_size=bottleneck_size,
              normalization_fn=normalization_fn,
              **kwargs))
    super(QRNNBidirectionalStack, self).__init__(**kwargs)

  def call(self, inputs, maskr3, inverse_normalizer):
    return self._apply_qrnn_stack(inputs, maskr3, inverse_normalizer)

  def _apply_qrnn_stack(self, inputs, mask3, inverse_normalizer):
    if self.parameters.mode not in [base_layers.PREDICT, base_layers.TFLITE]:
      inputs = inputs * mask3
    for layer in self.layers:
      outputs = layer(inputs, mask3, inverse_normalizer)
      inputs = outputs
    return outputs


class QRNNBidirectionalStackWithSeqLength(QRNNBidirectionalStack):

  def call(self, inputs, sequence_length):
    mask = tf.sequence_mask(
        sequence_length, tf.shape(inputs)[1], dtype=tf.float32)
    inverse_normalizer = tf.math.reciprocal(tf.reduce_sum(mask))
    maskr3 = tf.expand_dims(mask, 2)
    return self._apply_qrnn_stack(inputs, maskr3, inverse_normalizer)
