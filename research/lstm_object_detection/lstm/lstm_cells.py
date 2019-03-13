# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""BottleneckConvLSTMCell implementation."""

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import variables

slim = tf.contrib.slim

_batch_norm = tf.contrib.layers.batch_norm


class BottleneckConvLSTMCell(tf.contrib.rnn.RNNCell):
  """Basic LSTM recurrent network cell using separable convolutions.

  The implementation is based on:
  Mobile Video Object Detection with Temporally-Aware Feature Maps
  https://arxiv.org/abs/1711.06368.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  This LSTM first projects inputs to the size of the output before doing gate
  computations. This saves params unless the input is less than a third of the
  state size channel-wise.
  """

  def __init__(self,
               filter_size,
               output_size,
               num_units,
               forget_bias=1.0,
               activation=tf.tanh,
               flattened_state=False,
               clip_state=False,
               output_bottleneck=False,
               visualize_gates=True):
    """Initializes the basic LSTM cell.

    Args:
      filter_size: collection, conv filter size.
      output_size: collection, the width/height dimensions of the cell/output.
      num_units: int, The number of channels in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      activation: Activation function of the inner states.
      flattened_state: if True, state tensor will be flattened and stored as
        a 2-d tensor. Use for exporting the model to tfmini.
      clip_state: if True, clip state between [-6, 6].
      output_bottleneck: if True, the cell bottleneck will be concatenated
        to the cell output.
      visualize_gates: if True, add histogram summaries of all gates
        and outputs to tensorboard.
    """
    self._filter_size = list(filter_size)
    self._output_size = list(output_size)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._activation = activation
    self._viz_gates = visualize_gates
    self._flattened_state = flattened_state
    self._clip_state = clip_state
    self._output_bottleneck = output_bottleneck
    self._param_count = self._num_units
    for dim in self._output_size:
      self._param_count *= dim

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self._output_size + [self._num_units],
                                         self._output_size + [self._num_units])

  @property
  def state_size_flat(self):
    return tf.contrib.rnn.LSTMStateTuple([self._param_count],
                                         [self._param_count])

  @property
  def output_size(self):
    return self._output_size + [self._num_units]

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM) with bottlenecking.

    Args:
      inputs: Input tensor at the current timestep.
      state: Tuple of tensors, the state and output at the previous timestep.
      scope: Optional scope.
    Returns:
      A tuple where the first element is the LSTM output and the second is
      a LSTMStateTuple of the state at the current timestep.
    """
    scope = scope or 'conv_lstm_cell'
    with tf.variable_scope(scope):
      c, h = state

      # unflatten state if necessary
      if self._flattened_state:
        c = tf.reshape(c, [-1] + self.output_size)
        h = tf.reshape(h, [-1] + self.output_size)

      # summary of input passed into cell
      if self._viz_gates:
        slim.summaries.add_histogram_summary(inputs, 'cell_input')

      bottleneck = tf.contrib.layers.separable_conv2d(
          tf.concat([inputs, h], 3),
          self._num_units,
          self._filter_size,
          depth_multiplier=1,
          activation_fn=self._activation,
          normalizer_fn=None,
          scope='bottleneck')

      if self._viz_gates:
        slim.summaries.add_histogram_summary(bottleneck, 'bottleneck')

      concat = tf.contrib.layers.separable_conv2d(
          bottleneck,
          4 * self._num_units,
          self._filter_size,
          depth_multiplier=1,
          activation_fn=None,
          normalizer_fn=None,
          scope='gates')

      i, j, f, o = tf.split(concat, 4, 3)

      new_c = (
          c * tf.sigmoid(f + self._forget_bias) +
          tf.sigmoid(i) * self._activation(j))
      if self._clip_state:
        new_c = tf.clip_by_value(new_c, -6, 6)
      new_h = self._activation(new_c) * tf.sigmoid(o)
      # summary of cell output and new state
      if self._viz_gates:
        slim.summaries.add_histogram_summary(new_h, 'cell_output')
        slim.summaries.add_histogram_summary(new_c, 'cell_state')

      output = new_h
      if self._output_bottleneck:
        output = tf.concat([new_h, bottleneck], axis=3)

      # reflatten state to store it
      if self._flattened_state:
        new_c = tf.reshape(new_c, [-1, self._param_count])
        new_h = tf.reshape(new_h, [-1, self._param_count])

      return output, tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

  def init_state(self, state_name, batch_size, dtype, learned_state=False):
    """Creates an initial state compatible with this cell.

    Args:
      state_name: name of the state tensor
      batch_size: model batch size
      dtype: dtype for the tensor values i.e. tf.float32
      learned_state: whether the initial state should be learnable. If false,
        the initial state is set to all 0's

    Returns:
      The created initial state.
    """
    state_size = (
        self.state_size_flat if self._flattened_state else self.state_size)
    # list of 2 zero tensors or variables tensors, depending on if
    # learned_state is true
    ret_flat = [(variables.model_variable(
        state_name + str(i),
        shape=s,
        dtype=dtype,
        initializer=tf.truncated_normal_initializer(stddev=0.03))
                 if learned_state else tf.zeros(
                     [batch_size] + s, dtype=dtype, name=state_name))
                for i, s in enumerate(state_size)]

    # duplicates initial state across the batch axis if it's learned
    if learned_state:
      ret_flat = [
          tf.stack([tensor
                    for i in range(int(batch_size))])
          for tensor in ret_flat
      ]
    for s, r in zip(state_size, ret_flat):
      r.set_shape([None] + s)
    return tf.contrib.framework.nest.pack_sequence_as(
        structure=[1, 1], flat_sequence=ret_flat)
