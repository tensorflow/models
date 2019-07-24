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
import lstm_object_detection.lstm.utils as lstm_utils

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
               flatten_state=False,
               clip_state=False,
               output_bottleneck=False,
               pre_bottleneck=False,
               visualize_gates=False):
    """Initializes the basic LSTM cell.

    Args:
      filter_size: collection, conv filter size.
      output_size: collection, the width/height dimensions of the cell/output.
      num_units: int, The number of channels in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      activation: Activation function of the inner states.
      flatten_state: if True, state tensor will be flattened and stored as
        a 2-d tensor. Use for exporting the model to tfmini.
      clip_state: if True, clip state between [-6, 6].
      output_bottleneck: if True, the cell bottleneck will be concatenated
        to the cell output.
      pre_bottleneck: if True, cell assumes that bottlenecking was performing
        before the function was called.
      visualize_gates: if True, add histogram summaries of all gates
        and outputs to tensorboard.
    """
    self._filter_size = list(filter_size)
    self._output_size = list(output_size)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._activation = activation
    self._viz_gates = visualize_gates
    self._flatten_state = flatten_state
    self._clip_state = clip_state
    self._output_bottleneck = output_bottleneck
    self._pre_bottleneck = pre_bottleneck
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
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      c, h = state

      # unflatten state if necessary
      if self._flatten_state:
        c = tf.reshape(c, [-1] + self.output_size)
        h = tf.reshape(h, [-1] + self.output_size)

      # summary of input passed into cell
      if self._viz_gates:
        slim.summaries.add_histogram_summary(inputs, 'cell_input')
      if self._pre_bottleneck:
        bottleneck = inputs
      else:
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
      if self._flatten_state:
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
        self.state_size_flat if self._flatten_state else self.state_size)
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

  def pre_bottleneck(self, inputs, state, input_index):
    """Apply pre-bottleneck projection to inputs.

    Pre-bottleneck operation maps features of different channels into the same
    dimension. The purpose of this op is to share the features from both large
    and small models in the same LSTM cell.

    Args:
      inputs: 4D Tensor with shape [batch_size x width x height x input_size].
      state: 4D Tensor with shape [batch_size x width x height x state_size].
      input_index: integer index indicating which base features the inputs
        correspoding to.
    Returns:
      inputs: pre-bottlenecked inputs.
    Raises:
      ValueError: If pre_bottleneck is not set or inputs is not rank 4.
    """
    # Sometimes state is a tuple, in which case it cannot be modified, e.g.
    # during training, tf.contrib.training.SequenceQueueingStateSaver
    # returns the state as a tuple. This should not be an issue since we
    # only need to modify state[1] during export, when state should be a
    # list.
    if not len(inputs.shape) == 4:
      raise ValueError('Expect rank 4 feature tensor.')
    if not self._flatten_state and not len(state.shape) == 4:
      raise ValueError('Expect rank 4 state tensor.')
    if self._flatten_state and not len(state.shape) == 2:
      raise ValueError('Expect rank 2 state tensor when flatten_state is set.')
    with tf.name_scope(None):
      state = tf.identity(state, name='raw_inputs/init_lstm_h')
    if self._flatten_state:
      batch_size = inputs.shape[0]
      height = inputs.shape[1]
      width = inputs.shape[2]
      state = tf.reshape(state, [batch_size, height, width, -1])
    with tf.variable_scope('conv_lstm_cell', reuse=tf.AUTO_REUSE):
      scope_name = 'bottleneck_%d' % input_index
      inputs = tf.contrib.layers.separable_conv2d(
          tf.concat([inputs, state], 3),
          self.output_size[-1],
          self._filter_size,
          depth_multiplier=1,
          activation_fn=tf.nn.relu6,
          normalizer_fn=None,
          scope=scope_name)
      # For exporting inference graph, we only mark the first timestep.
      with tf.name_scope(None):
        inputs = tf.identity(
            inputs, name='raw_outputs/base_endpoint_%d' % (input_index + 1))
    return inputs


class GroupedConvLSTMCell(tf.contrib.rnn.RNNCell):
  """Basic LSTM recurrent network cell using separable convolutions.

  The implementation is based on: https://arxiv.org/abs/1903.10172.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  This LSTM first projects inputs to the size of the output before doing gate
  computations. This saves params unless the input is less than a third of the
  state size channel-wise. Computation of bottlenecks and gates is divided
  into independent groups for further savings.
  """

  def __init__(self,
               filter_size,
               output_size,
               num_units,
               is_training,
               forget_bias=1.0,
               activation=tf.tanh,
               use_batch_norm=False,
               flatten_state=False,
               groups=4,
               clip_state=False,
               scale_state=False,
               output_bottleneck=False,
               pre_bottleneck=False,
               is_quantized=False,
               visualize_gates=False):
    """Initialize the basic LSTM cell.

    Args:
      filter_size: collection, conv filter size
      output_size: collection, the width/height dimensions of the cell/output
      num_units: int, The number of channels in the LSTM cell.
      is_training: Whether the LSTM is in training mode.
      forget_bias: float, The bias added to forget gates (see above).
      activation: Activation function of the inner states.
      use_batch_norm: if True, use batch norm after convolution
      flatten_state: if True, state tensor will be flattened and stored as
        a 2-d tensor. Use for exporting the model to tfmini
      groups: Number of groups to split the state into. Must evenly divide
        num_units.
      clip_state: if True, clips state between [-6, 6].
      scale_state: if True, scales state so that all values are under 6 at all
        times.
      output_bottleneck: if True, the cell bottleneck will be concatenated
        to the cell output.
      pre_bottleneck: if True, cell assumes that bottlenecking was performing
        before the function was called.
      is_quantized: if True, the model is in quantize mode, which requires
        quantization friendly concat and separable_conv2d ops.
      visualize_gates: if True, add histogram summaries of all gates
        and outputs to tensorboard

    Raises:
      ValueError: when both clip_state and scale_state are enabled.
    """
    if clip_state and scale_state:
      raise ValueError('clip_state and scale_state cannot both be enabled.')

    self._filter_size = list(filter_size)
    self._output_size = list(output_size)
    self._num_units = num_units
    self._is_training = is_training
    self._forget_bias = forget_bias
    self._activation = activation
    self._use_batch_norm = use_batch_norm
    self._viz_gates = visualize_gates
    self._flatten_state = flatten_state
    self._param_count = self._num_units
    self._groups = groups
    self._scale_state = scale_state
    self._clip_state = clip_state
    self._output_bottleneck = output_bottleneck
    self._pre_bottleneck = pre_bottleneck
    self._is_quantized = is_quantized
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

  @property
  def filter_size(self):
    return self._filter_size

  @property
  def num_groups(self):
    return self._groups

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM) with bottlenecking.

    Includes logic for quantization-aware training. Note that all concats and
    activations use fixed ranges unless stated otherwise.

    Args:
      inputs: Input tensor at the current timestep.
      state: Tuple of tensors, the state at the previous timestep.
      scope: Optional scope.
    Returns:
      A tuple where the first element is the LSTM output and the second is
      a LSTMStateTuple of the state at the current timestep.
    """
    scope = scope or 'conv_lstm_cell'
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      c, h = state

      # Set nodes to be under raw_inputs/ name scope for tfmini export.
      with tf.name_scope(None):
        c = tf.identity(c, name='raw_inputs/init_lstm_c')
        # When pre_bottleneck is enabled, input h handle is in rnn_decoder.py
        if not self._pre_bottleneck:
          h = tf.identity(h, name='raw_inputs/init_lstm_h')

      # unflatten state if necessary
      if self._flatten_state:
        c = tf.reshape(c, [-1] + self.output_size)
        h = tf.reshape(h, [-1] + self.output_size)

      c_list = tf.split(c, self._groups, axis=3)
      if self._pre_bottleneck:
        inputs_list = tf.split(inputs, self._groups, axis=3)
      else:
        h_list = tf.split(h, self._groups, axis=3)
      out_bottleneck = []
      out_c = []
      out_h = []
      # summary of input passed into cell
      if self._viz_gates:
        slim.summaries.add_histogram_summary(inputs, 'cell_input')

      for k in range(self._groups):
        if self._pre_bottleneck:
          bottleneck = inputs_list[k]
        else:
          if self._use_batch_norm:
            b_x = lstm_utils.quantizable_separable_conv2d(
                inputs,
                self._num_units / self._groups,
                self._filter_size,
                is_quantized=self._is_quantized,
                depth_multiplier=1,
                activation_fn=None,
                normalizer_fn=None,
                scope='bottleneck_%d_x' % k)
            b_h = lstm_utils.quantizable_separable_conv2d(
                h_list[k],
                self._num_units / self._groups,
                self._filter_size,
                is_quantized=self._is_quantized,
                depth_multiplier=1,
                activation_fn=None,
                normalizer_fn=None,
                scope='bottleneck_%d_h' % k)
            b_x = slim.batch_norm(
                b_x, scale=True, is_training=self._is_training,
                scope='BatchNorm_%d_X' % k)
            b_h = slim.batch_norm(
                b_h, scale=True, is_training=self._is_training,
                scope='BatchNorm_%d_H' % k)
            bottleneck = b_x + b_h
          else:
            # All concats use fixed quantization ranges to prevent rescaling
            # at inference. Both |inputs| and |h_list| are tensors resulting
            # from Relu6 operations so we fix the ranges to [0, 6].
            bottleneck_concat = lstm_utils.quantizable_concat(
                [inputs, h_list[k]],
                axis=3,
                is_training=self._is_training,
                is_quantized=self._is_quantized,
                scope='bottleneck_%d/quantized_concat' % k)

            bottleneck = lstm_utils.quantizable_separable_conv2d(
                bottleneck_concat,
                self._num_units / self._groups,
                self._filter_size,
                is_quantized=self._is_quantized,
                depth_multiplier=1,
                activation_fn=self._activation,
                normalizer_fn=None,
                scope='bottleneck_%d' % k)

        concat = lstm_utils.quantizable_separable_conv2d(
            bottleneck,
            4 * self._num_units / self._groups,
            self._filter_size,
            is_quantized=self._is_quantized,
            depth_multiplier=1,
            activation_fn=None,
            normalizer_fn=None,
            scope='concat_conv_%d' % k)

        # Since there is no activation in the previous separable conv, we
        # quantize here. A starting range of [-6, 6] is used because the
        # tensors are input to a Sigmoid function that saturates at these
        # ranges.
        concat = lstm_utils.quantize_op(
            concat,
            is_training=self._is_training,
            default_min=-6,
            default_max=6,
            is_quantized=self._is_quantized,
            scope='gates_%d/act_quant' % k)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(concat, 4, 3)

        f_add = f + self._forget_bias
        f_add = lstm_utils.quantize_op(
            f_add,
            is_training=self._is_training,
            default_min=-6,
            default_max=6,
            is_quantized=self._is_quantized,
            scope='forget_gate_%d/add_quant' % k)
        f_act = tf.sigmoid(f_add)
        # The quantization range is fixed for the sigmoid to ensure that zero
        # is exactly representable.
        f_act = lstm_utils.quantize_op(
            f_act,
            is_training=False,
            default_min=0,
            default_max=1,
            is_quantized=self._is_quantized,
            scope='forget_gate_%d/act_quant' % k)

        a = c_list[k] * f_act
        a = lstm_utils.quantize_op(
            a,
            is_training=self._is_training,
            is_quantized=self._is_quantized,
            scope='forget_gate_%d/mul_quant' % k)

        i_act = tf.sigmoid(i)
        # The quantization range is fixed for the sigmoid to ensure that zero
        # is exactly representable.
        i_act = lstm_utils.quantize_op(
            i_act,
            is_training=False,
            default_min=0,
            default_max=1,
            is_quantized=self._is_quantized,
            scope='input_gate_%d/act_quant' % k)

        j_act = self._activation(j)
        # The quantization range is fixed for the relu6 to ensure that zero
        # is exactly representable.
        j_act = lstm_utils.quantize_op(
            j_act,
            is_training=False,
            default_min=0,
            default_max=6,
            is_quantized=self._is_quantized,
            scope='new_input_%d/act_quant' % k)

        b = i_act * j_act
        b = lstm_utils.quantize_op(
            b,
            is_training=self._is_training,
            is_quantized=self._is_quantized,
            scope='input_gate_%d/mul_quant' % k)

        new_c = a + b
        # The quantization range is fixed to [0, 6] due to an optimization in
        # TFLite. The order of operations is as fllows:
        #     Add -> FakeQuant -> Relu6 -> FakeQuant -> Concat.
        # The fakequant ranges to the concat must be fixed to ensure all inputs
        # to the concat have the same range, removing the need for rescaling.
        # The quantization ranges input to the relu6 are propagated to its
        # output. Any mismatch between these two ranges will cause an error.
        new_c = lstm_utils.quantize_op(
            new_c,
            is_training=False,
            default_min=0,
            default_max=6,
            is_quantized=self._is_quantized,
            scope='new_c_%d/add_quant' % k)

        if not self._is_quantized:
          if self._scale_state:
            normalizer = tf.maximum(1.0,
                                    tf.reduce_max(new_c, axis=(1, 2, 3)) / 6)
            new_c /= tf.reshape(normalizer, [tf.shape(new_c)[0], 1, 1, 1])
          elif self._clip_state:
            new_c = tf.clip_by_value(new_c, -6, 6)

        new_c_act = self._activation(new_c)
        # The quantization range is fixed for the relu6 to ensure that zero
        # is exactly representable.
        new_c_act = lstm_utils.quantize_op(
            new_c_act,
            is_training=False,
            default_min=0,
            default_max=6,
            is_quantized=self._is_quantized,
            scope='new_c_%d/act_quant' % k)

        o_act = tf.sigmoid(o)
        # The quantization range is fixed for the sigmoid to ensure that zero
        # is exactly representable.
        o_act = lstm_utils.quantize_op(
            o_act,
            is_training=False,
            default_min=0,
            default_max=1,
            is_quantized=self._is_quantized,
            scope='output_%d/act_quant' % k)

        new_h = new_c_act * o_act
        # The quantization range is fixed since it is input to a concat.
        # A range of [0, 6] is used since |new_h| is a product of ranges [0, 6]
        # and [0, 1].
        new_h_act = lstm_utils.quantize_op(
            new_h,
            is_training=False,
            default_min=0,
            default_max=6,
            is_quantized=self._is_quantized,
            scope='new_h_%d/act_quant' % k)

        out_bottleneck.append(bottleneck)
        out_c.append(new_c_act)
        out_h.append(new_h_act)

      # Since all inputs to the below concats are already quantized, we can use
      # a regular concat operation.
      new_c = tf.concat(out_c, axis=3)
      new_h = tf.concat(out_h, axis=3)

      # |bottleneck| is input to a concat with |new_h|. We must use
      # quantizable_concat() with a fixed range that matches |new_h|.
      bottleneck = lstm_utils.quantizable_concat(
          out_bottleneck,
          axis=3,
          is_training=False,
          is_quantized=self._is_quantized,
          scope='out_bottleneck/quantized_concat')

      # summary of cell output and new state
      if self._viz_gates:
        slim.summaries.add_histogram_summary(new_h, 'cell_output')
        slim.summaries.add_histogram_summary(new_c, 'cell_state')

      output = new_h
      if self._output_bottleneck:
        output = lstm_utils.quantizable_concat(
            [new_h, bottleneck],
            axis=3,
            is_training=False,
            is_quantized=self._is_quantized,
            scope='new_output/quantized_concat')

      # reflatten state to store it
      if self._flatten_state:
        new_c = tf.reshape(new_c, [-1, self._param_count], name='lstm_c')
        new_h = tf.reshape(new_h, [-1, self._param_count], name='lstm_h')

      # Set nodes to be under raw_outputs/ name scope for tfmini export.
      with tf.name_scope(None):
        new_c = tf.identity(new_c, name='raw_outputs/lstm_c')
        new_h = tf.identity(new_h, name='raw_outputs/lstm_h')
      states_and_output = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

      return output, states_and_output

  def init_state(self, state_name, batch_size, dtype, learned_state=False):
    """Creates an initial state compatible with this cell.

    Args:
      state_name: name of the state tensor
      batch_size: model batch size
      dtype: dtype for the tensor values i.e. tf.float32
      learned_state: whether the initial state should be learnable. If false,
        the initial state is set to all 0's

    Returns:
      ret: the created initial state
    """
    state_size = (self.state_size_flat if self._flatten_state
                  else self.state_size)
    # list of 2 zero tensors or variables tensors,
    # depending on if learned_state is true
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
      ret_flat = [tf.stack([tensor for i in range(int(batch_size))])
                  for tensor in ret_flat]
    for s, r in zip(state_size, ret_flat):
      r = tf.reshape(r, [-1] + s)
    ret = tf.contrib.framework.nest.pack_sequence_as(
        structure=[1, 1], flat_sequence=ret_flat)
    return ret

  def pre_bottleneck(self, inputs, state, input_index):
    """Apply pre-bottleneck projection to inputs.

    Pre-bottleneck operation maps features of different channels into the same
    dimension. The purpose of this op is to share the features from both large
    and small models in the same LSTM cell.

    Args:
      inputs: 4D Tensor with shape [batch_size x width x height x input_size].
      state: 4D Tensor with shape [batch_size x width x height x state_size].
      input_index: integer index indicating which base features the inputs
        correspoding to.
    Returns:
      inputs: pre-bottlenecked inputs.
    Raises:
      ValueError: If pre_bottleneck is not set or inputs is not rank 4.
    """
    # Sometimes state is a tuple, in which case it cannot be modified, e.g.
    # during training, tf.contrib.training.SequenceQueueingStateSaver
    # returns the state as a tuple. This should not be an issue since we
    # only need to modify state[1] during export, when state should be a
    # list.
    if not self._pre_bottleneck:
      raise ValueError('Only applied when pre_bottleneck is set to true.')
    if not len(inputs.shape) == 4:
      raise ValueError('Expect a rank 4 feature tensor.')
    if not self._flatten_state and not len(state.shape) == 4:
      raise ValueError('Expect rank 4 state tensor.')
    if self._flatten_state and not len(state.shape) == 2:
      raise ValueError('Expect rank 2 state tensor when flatten_state is set.')
    with tf.name_scope(None):
      state = tf.identity(state, name='raw_inputs/init_lstm_h')
    if self._flatten_state:
      batch_size = inputs.shape[0]
      height = inputs.shape[1]
      width = inputs.shape[2]
      state = tf.reshape(state, [batch_size, height, width, -1])
    with tf.variable_scope('conv_lstm_cell', reuse=tf.AUTO_REUSE):
      state_split = tf.split(state, self._groups, axis=3)
      with tf.variable_scope('bottleneck_%d' % input_index):
        bottleneck_out = []
        for k in range(self._groups):
          with tf.variable_scope('group_%d' % k):
            bottleneck_out.append(
                lstm_utils.quantizable_separable_conv2d(
                    lstm_utils.quantizable_concat(
                        [inputs, state_split[k]],
                        axis=3,
                        is_training=self._is_training,
                        is_quantized=self._is_quantized,
                        scope='quantized_concat'),
                    self.output_size[-1] / self._groups,
                    self._filter_size,
                    is_quantized=self._is_quantized,
                    depth_multiplier=1,
                    activation_fn=tf.nn.relu6,
                    normalizer_fn=None,
                    scope='project'))
        inputs = lstm_utils.quantizable_concat(
            bottleneck_out,
            axis=3,
            is_training=self._is_training,
            is_quantized=self._is_quantized,
            scope='bottleneck_out/quantized_concat')
      # For exporting inference graph, we only mark the first timestep.
      with tf.name_scope(None):
        inputs = tf.identity(
            inputs,
            name='raw_outputs/base_endpoint_%d' % (input_index + 1))
    return inputs
