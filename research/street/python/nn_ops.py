# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Ops and utilities for neural networks.

For now, just an LSTM layer.
"""
import shapes
import tensorflow as tf
from tensorflow.python.framework import ops
rnn = tf.load_op_library("../cc/rnn_ops.so")


def rnn_helper(inp,
               length,
               cell_type=None,
               direction="forward",
               name=None,
               *args,
               **kwargs):
  """Adds ops for a recurrent neural network layer.

  This function calls an actual implementation of a recurrent neural network
  based on `cell_type`.

  There are three modes depending on the value of `direction`:

    forward: Adds a forward RNN.
    backward: Adds a backward RNN.
    bidirectional: Adds both forward and backward RNNs and creates a
                   bidirectional RNN.

  Args:
    inp: A 3-D tensor of shape [`batch_size`, `max_length`, `feature_dim`].
    length: A 1-D tensor of shape [`batch_size`] and type int64. Each element
            represents the length of the corresponding sequence in `inp`.
    cell_type: Cell type of RNN. Currently can only be "lstm".
    direction: One of "forward", "backward", "bidirectional".
    name: Name of the op.
    *args: Other arguments to the layer.
    **kwargs: Keyword arugments to the layer.

  Returns:
    A 3-D tensor of shape [`batch_size`, `max_length`, `num_nodes`].
  """

  assert cell_type is not None
  rnn_func = None
  if cell_type == "lstm":
    rnn_func = lstm_layer
  assert rnn_func is not None
  assert direction in ["forward", "backward", "bidirectional"]

  with tf.variable_scope(name):
    if direction in ["forward", "bidirectional"]:
      forward = rnn_func(
          inp=inp,
          length=length,
          backward=False,
          name="forward",
          *args,
          **kwargs)
      if isinstance(forward, tuple):
        # lstm_layer returns a tuple (output, memory). We only need the first
        # element.
        forward = forward[0]
    if direction in ["backward", "bidirectional"]:
      backward = rnn_func(
          inp=inp,
          length=length,
          backward=True,
          name="backward",
          *args,
          **kwargs)
      if isinstance(backward, tuple):
        # lstm_layer returns a tuple (output, memory). We only need the first
        # element.
        backward = backward[0]
    if direction == "forward":
      out = forward
    elif direction == "backward":
      out = backward
    else:
      out = tf.concat(axis=2, values=[forward, backward])
  return out


@ops.RegisterShape("VariableLSTM")
def _variable_lstm_shape(op):
  """Shape function for the VariableLSTM op."""
  input_shape = op.inputs[0].get_shape().with_rank(4)
  state_shape = op.inputs[1].get_shape().with_rank(2)
  memory_shape = op.inputs[2].get_shape().with_rank(2)
  w_m_m_shape = op.inputs[3].get_shape().with_rank(3)
  batch_size = input_shape[0].merge_with(state_shape[0])
  batch_size = input_shape[0].merge_with(memory_shape[0])
  seq_len = input_shape[1]
  gate_num = input_shape[2].merge_with(w_m_m_shape[1])
  output_dim = input_shape[3].merge_with(state_shape[1])
  output_dim = output_dim.merge_with(memory_shape[1])
  output_dim = output_dim.merge_with(w_m_m_shape[0])
  output_dim = output_dim.merge_with(w_m_m_shape[2])
  return [[batch_size, seq_len, output_dim],
          [batch_size, seq_len, gate_num, output_dim],
          [batch_size, seq_len, output_dim]]


@ops.RegisterGradient("VariableLSTM")
def _variable_lstm_grad(op, act_grad, gate_grad, mem_grad):
  """Gradient function for the VariableLSTM op."""
  initial_state = op.inputs[1]
  initial_memory = op.inputs[2]
  w_m_m = op.inputs[3]
  act = op.outputs[0]
  gate_raw_act = op.outputs[1]
  memory = op.outputs[2]
  return rnn.variable_lstm_grad(initial_state, initial_memory, w_m_m, act,
                                gate_raw_act, memory, act_grad, gate_grad,
                                mem_grad)


def lstm_layer(inp,
               length=None,
               state=None,
               memory=None,
               num_nodes=None,
               backward=False,
               clip=50.0,
               reg_func=tf.nn.l2_loss,
               weight_reg=False,
               weight_collection="LSTMWeights",
               bias_reg=False,
               stddev=None,
               seed=None,
               decode=False,
               use_native_weights=False,
               name=None):
  """Adds ops for an LSTM layer.

  This adds ops for the following operations:

    input => (forward-LSTM|backward-LSTM) => output

  The direction of the LSTM is determined by `backward`. If it is false, the
  forward LSTM is used, the backward one otherwise.

  Args:
    inp: A 3-D tensor of shape [`batch_size`, `max_length`, `feature_dim`].
    length: A 1-D tensor of shape [`batch_size`] and type int64. Each element
            represents the length of the corresponding sequence in `inp`.
    state: If specified, uses it as the initial state.
    memory: If specified, uses it as the initial memory.
    num_nodes: The number of LSTM cells.
    backward: If true, reverses the `inp` before adding the ops. The output is
              also reversed so that the direction is the same as `inp`.
    clip: Value used to clip the cell values.
    reg_func: Function used for the weight regularization such as
              `tf.nn.l2_loss`.
    weight_reg: If true, regularize the filter weights with `reg_func`.
    weight_collection: Collection to add the weights to for regularization.
    bias_reg: If true, regularize the bias vector with `reg_func`.
    stddev: Standard deviation used to initialize the variables.
    seed: Seed used to initialize the variables.
    decode: If true, does not add ops which are not used for inference.
    use_native_weights: If true, uses weights in the same format as the native
                        implementations.
    name: Name of the op.

  Returns:
    A 3-D tensor of shape [`batch_size`, `max_length`, `num_nodes`].
  """
  with tf.variable_scope(name):
    if backward:
      if length is None:
        inp = tf.reverse(inp, [1])
      else:
        inp = tf.reverse_sequence(inp, length, 1, 0)

    num_prev = inp.get_shape()[2]
    if stddev:
      initializer = tf.truncated_normal_initializer(stddev=stddev, seed=seed)
    else:
      initializer = tf.uniform_unit_scaling_initializer(seed=seed)

    if use_native_weights:
      with tf.variable_scope("LSTMCell"):
        w = tf.get_variable(
            "W_0",
            shape=[num_prev + num_nodes, 4 * num_nodes],
            initializer=initializer,
            dtype=tf.float32)
        w_i_m = tf.slice(w, [0, 0], [num_prev, 4 * num_nodes], name="w_i_m")
        w_m_m = tf.reshape(
            tf.slice(w, [num_prev, 0], [num_nodes, 4 * num_nodes]),
            [num_nodes, 4, num_nodes],
            name="w_m_m")
    else:
      w_i_m = tf.get_variable("w_i_m", [num_prev, 4 * num_nodes],
                              initializer=initializer)
      w_m_m = tf.get_variable("w_m_m", [num_nodes, 4, num_nodes],
                              initializer=initializer)

    if not decode and weight_reg:
      tf.add_to_collection(weight_collection, reg_func(w_i_m, name="w_i_m_reg"))
      tf.add_to_collection(weight_collection, reg_func(w_m_m, name="w_m_m_reg"))

    batch_size = shapes.tensor_dim(inp, dim=0)
    num_frames = shapes.tensor_dim(inp, dim=1)
    prev = tf.reshape(inp, tf.stack([batch_size * num_frames, num_prev]))

    if use_native_weights:
      with tf.variable_scope("LSTMCell"):
        b = tf.get_variable(
            "B",
            shape=[4 * num_nodes],
            initializer=tf.zeros_initializer(),
            dtype=tf.float32)
      biases = tf.identity(b, name="biases")
    else:
      biases = tf.get_variable(
          "biases", [4 * num_nodes], initializer=tf.constant_initializer(0.0))
    if not decode and bias_reg:
      tf.add_to_collection(
          weight_collection, reg_func(
              biases, name="biases_reg"))
    prev = tf.nn.xw_plus_b(prev, w_i_m, biases)

    prev = tf.reshape(prev, tf.stack([batch_size, num_frames, 4, num_nodes]))
    if state is None:
      state = tf.fill(tf.stack([batch_size, num_nodes]), 0.0)
    if memory is None:
      memory = tf.fill(tf.stack([batch_size, num_nodes]), 0.0)

    out, _, mem = rnn.variable_lstm(prev, state, memory, w_m_m, clip=clip)

    if backward:
      if length is None:
        out = tf.reverse(out, [1])
      else:
        out = tf.reverse_sequence(out, length, 1, 0)

  return out, mem
