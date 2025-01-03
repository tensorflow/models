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

"""Custom RNN decoder."""

import tensorflow.compat.v1 as tf
import lstm_object_detection.lstm.utils as lstm_utils


class _NoVariableScope(object):

  def __enter__(self):
    return

  def __exit__(self, exc_type, exc_value, traceback):
    return False


def rnn_decoder(decoder_inputs,
                initial_state,
                cell,
                loop_function=None,
                scope=None):
  """RNN decoder for the LSTM-SSD model.

  This decoder returns a list of all states, rather than only the final state.
  Args:
    decoder_inputs: A list of 4D Tensors with shape [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    scope: optional VariableScope for the created subgraph.
  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 4D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      states: A list of the same length as decoder_inputs of the state of each
        cell at each time-step. It is a 2D Tensor of shape
        [batch_size x cell.state_size].
  """
  with tf.variable_scope(scope) if scope else _NoVariableScope():
    state_tuple = initial_state
    outputs = []
    states = []
    prev = None
    for local_step, decoder_input in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with tf.variable_scope('loop_function', reuse=True):
          decoder_input = loop_function(prev, local_step)
      output, state_tuple = cell(decoder_input, state_tuple)
      outputs.append(output)
      states.append(state_tuple)
      if loop_function is not None:
        prev = output
  return outputs, states

def multi_input_rnn_decoder(decoder_inputs,
                            initial_state,
                            cell,
                            sequence_step,
                            selection_strategy='RANDOM',
                            is_training=None,
                            is_quantized=False,
                            preprocess_fn_list=None,
                            pre_bottleneck=False,
                            flatten_state=False,
                            scope=None):
  """RNN decoder for the Interleaved LSTM-SSD model.

  This decoder takes multiple sequences of inputs and selects the input to feed
  to the rnn at each timestep using its selection_strategy, which can be random,
  learned, or deterministic.
  This decoder returns a list of all states, rather than only the final state.
  Args:
    decoder_inputs: A list of lists of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    sequence_step: Tensor [batch_size] of the step number of the first elements
      in the sequence.
    selection_strategy: Method for picking the decoder_input to use at each
      timestep. Must be 'RANDOM', 'SKIPX' for integer X,  where X is the number
      of times to use the second input before using the first.
    is_training: boolean, whether the network is training. When using learned
      selection, attempts exploration if training.
    is_quantized: flag to enable/disable quantization mode.
    preprocess_fn_list: List of functions accepting two tensor arguments: one
      timestep of decoder_inputs and the lstm state. If not None,
      decoder_inputs[i] will be updated with preprocess_fn[i] at the start of
      each timestep.
    pre_bottleneck: if True, use separate bottleneck weights for each sequence.
      Useful when input sequences have differing numbers of channels. Final
      bottlenecks will have the same dimension.
    flatten_state: Whether the LSTM state is flattened.
    scope: optional VariableScope for the created subgraph.
  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      states: A list of the same length as decoder_inputs of the state of each
        cell at each time-step. It is a 2D Tensor of shape
        [batch_size x cell.state_size].
  Raises:
    ValueError: If selection_strategy is not recognized or unexpected unroll
      length.
  """
  if flatten_state and len(decoder_inputs[0]) > 1:
    raise ValueError('In export mode, unroll length should not be more than 1')
  with tf.variable_scope(scope) if scope else _NoVariableScope():
    state_tuple = initial_state
    outputs = []
    states = []
    batch_size = decoder_inputs[0][0].shape[0].value
    num_sequences = len(decoder_inputs)
    sequence_length = len(decoder_inputs[0])

    for local_step in range(sequence_length):
      for sequence_index in range(num_sequences):
        if preprocess_fn_list is not None:
          decoder_inputs[sequence_index][local_step] = (
              preprocess_fn_list[sequence_index](
                  decoder_inputs[sequence_index][local_step], state_tuple[0]))
        if pre_bottleneck:
          decoder_inputs[sequence_index][local_step] = cell.pre_bottleneck(
              inputs=decoder_inputs[sequence_index][local_step],
              state=state_tuple[1],
              input_index=sequence_index)

      action = generate_action(selection_strategy, local_step, sequence_step,
                               [batch_size, 1, 1, 1])
      inputs, _ = (
          select_inputs(decoder_inputs, action, local_step, is_training,
                        is_quantized))
      # Mark base network endpoints under raw_inputs/
      with tf.name_scope(None):
        inputs = tf.identity(inputs, 'raw_inputs/base_endpoint')
      output, state_tuple_out = cell(inputs, state_tuple)
      state_tuple = select_state(state_tuple, state_tuple_out, action)

      outputs.append(output)
      states.append(state_tuple)
  return outputs, states


def generate_action(selection_strategy, local_step, sequence_step,
                    action_shape):
  """Generate current (binary) action based on selection strategy.

  Args:
    selection_strategy: Method for picking the decoder_input to use at each
      timestep. Must be 'RANDOM', 'SKIPX' for integer X,  where X is the number
      of times to use the second input before using the first.
    local_step: Tensor [batch_size] of the step number within the current
      unrolled batch.
    sequence_step: Tensor [batch_size] of the step number of the first elements
      in the sequence.
    action_shape: The shape of action tensor to be generated.

  Returns:
    A tensor of shape action_shape, each element is an individual action.

  Raises:
    ValueError: if selection_strategy is not supported or if 'SKIP' is not
      followed by numerics.
  """
  if selection_strategy.startswith('RANDOM'):
    action = tf.random.uniform(action_shape, maxval=2, dtype=tf.int32)
    action = tf.minimum(action, 1)

    # First step always runs large network.
    if local_step == 0 and sequence_step is not None:
      action *= tf.minimum(
          tf.reshape(tf.cast(sequence_step, tf.int32), action_shape), 1)
  elif selection_strategy.startswith('SKIP'):
    inter_count = int(selection_strategy[4:])
    if local_step % (inter_count + 1) == 0:
      action = tf.zeros(action_shape)
    else:
      action = tf.ones(action_shape)
  else:
    raise ValueError('Selection strategy %s not recognized' %
                     selection_strategy)
  return tf.cast(action, tf.int32)


def select_inputs(decoder_inputs, action, local_step, is_training, is_quantized,
                  get_alt_inputs=False):
  """Selects sequence from decoder_inputs based on 1D actions.

  Given multiple input batches, creates a single output batch by
  selecting from the action[i]-ith input for the i-th batch element.

  Args:
    decoder_inputs: A 2-D list of tensor inputs.
    action: A tensor of shape [batch_size]. Each element corresponds to an index
      of decoder_inputs to choose.
    local_step: The current timestep.
    is_training: boolean, whether the network is training. When using learned
      selection, attempts exploration if training.
    is_quantized: flag to enable/disable quantization mode.
    get_alt_inputs: Whether the non-chosen inputs should also be returned.

  Returns:
    The constructed output. Also outputs the elements that were not chosen
    if get_alt_inputs is True, otherwise None.

  Raises:
    ValueError: if the decoder inputs contains other than two sequences.
  """
  num_seqs = len(decoder_inputs)
  if not num_seqs == 2:
    raise ValueError('Currently only supports two sets of inputs.')
  stacked_inputs = tf.stack(
      [decoder_inputs[seq_index][local_step] for seq_index in range(num_seqs)],
      axis=-1)
  action_index = tf.one_hot(action, num_seqs)
  selected_inputs = (
      lstm_utils.quantize_op(stacked_inputs * action_index, is_training,
                             is_quantized, scope='quant_selected_inputs'))
  inputs = tf.reduce_sum(selected_inputs, axis=-1)
  inputs_alt = None
  # Only works for 2 models.
  if get_alt_inputs:
    # Reverse of action_index.
    action_index_alt = tf.one_hot(action, num_seqs, on_value=0.0, off_value=1.0)
    selected_inputs = (
        lstm_utils.quantize_op(stacked_inputs * action_index_alt, is_training,
                               is_quantized, scope='quant_selected_inputs_alt'))
    inputs_alt = tf.reduce_sum(selected_inputs, axis=-1)
  return inputs, inputs_alt

def select_state(previous_state, new_state, action):
  """Select state given action.

  Currently only supports binary action. If action is 0, it means the state is
  generated from the large model, and thus we will update the state. Otherwise,
  if the action is 1, it means the state is generated from the small model, and
  in interleaved model, we skip this state update.

  Args:
    previous_state: A state tuple representing state from previous step.
    new_state: A state tuple representing newly computed state.
    action: A tensor the same shape as state.

  Returns:
    A state tuple selected based on the given action.
  """
  action = tf.cast(action, tf.float32)
  state_c = previous_state[0] * action + new_state[0] * (1 - action)
  state_h = previous_state[1] * action + new_state[1] * (1 - action)
  return (state_c, state_h)
