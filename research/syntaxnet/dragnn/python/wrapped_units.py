# Copyright 2017 Google Inc. All Rights Reserved.
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

"""Network units wrapping TensorFlows' tf.contrib.rnn cells.

Please put all wrapping logic for tf.contrib.rnn in this module; this will help
collect common subroutines that prove useful.
"""

import abc

import tensorflow as tf

from dragnn.python import network_units as dragnn
from syntaxnet.util import check


def capture_variables(function, scope_name):
  """Captures and returns variables created by a function.

  Runs |function| in a scope of name |scope_name| and returns the list of
  variables created by |function|.

  Args:
    function: Function whose variables should be captured.  The function should
        take one argument, its enclosing variable scope.
    scope_name: Variable scope in which the |function| is evaluated.

  Returns:
    List of created variables.
  """
  # Use a dict to dedupe captured variables.
  created_vars = {}

  def _custom_getter(getter, *args, **kwargs):
    """Calls the real getter and captures its result in |created_vars|."""
    real_variable = getter(*args, **kwargs)
    created_vars[real_variable.name] = real_variable
    return real_variable

  with tf.variable_scope(
      scope_name, reuse=None, custom_getter=_custom_getter) as scope:
    function(scope)
  return created_vars.values()


def apply_with_captured_variables(function, scope_name, component):
  """Applies a function using previously-captured variables.

  The counterpart to capture_variables(); invokes |function| in a scope of name
  |scope_name|, extracting captured variables from the |component|.

  Args:
    function: Function to apply using captured variables.  The function should
        take one argument, its enclosing variable scope.
    scope_name: Variable scope in which the |function| is evaluated.  Must match
        the scope passed to capture_variables().
    component: Component from which to extract captured variables.

  Returns:
    Results of function application.
  """
  def _custom_getter(getter, *args, **kwargs):
    """Retrieves the normal or moving-average variables."""
    return component.get_variable(var_params=getter(*args, **kwargs))

  with tf.variable_scope(
      scope_name, reuse=True, custom_getter=_custom_getter) as scope:
    return function(scope)


class BaseLSTMNetwork(dragnn.NetworkUnitInterface):
  """Base class for wrapped LSTM networks.

  This LSTM network unit supports multiple layers with layer normalization.
  Because it is imported from tf.contrib.rnn, we need to capture the created
  variables during initialization time.

  Layers:
    ...subclass-specific layers...
    last_layer: Alias for the activations of the last hidden layer.
    logits: Logits associated with component actions.
  """

  def __init__(self, component, additional_attr_defaults=None):
    """Initializes the LSTM base class.

    Parameters used:
      hidden_layer_sizes: Comma-delimited number of hidden units for each layer.
      input_dropout_rate (-1.0): Input dropout rate for each layer.  If < 0.0,
          use the global |dropout_rate| hyperparameter.
      recurrent_dropout_rate (0.8): Recurrent dropout rate.  If < 0.0, use the
          global |recurrent_dropout_rate| hyperparameter.
      layer_norm (True): Whether or not to use layer norm.

    Hyperparameters used:
      dropout_rate: Input dropout rate.
      recurrent_dropout_rate: Recurrent dropout rate.

    Args:
      component: parent ComponentBuilderBase object.
      additional_attr_defaults: Additional attributes for use by derived class.
    """
    attr_defaults = additional_attr_defaults or {}
    attr_defaults.update({
        'layer_norm': True,
        'input_dropout_rate': -1.0,
        'recurrent_dropout_rate': 0.8,
        'hidden_layer_sizes': '256',
    })
    self._attrs = dragnn.get_attrs_with_defaults(
        component.spec.network_unit.parameters,
        defaults=attr_defaults)

    self._hidden_layer_sizes = map(int,
                                   self._attrs['hidden_layer_sizes'].split(','))

    self._input_dropout_rate = self._attrs['input_dropout_rate']
    if self._input_dropout_rate < 0.0:
      self._input_dropout_rate = component.master.hyperparams.dropout_rate

    self._recurrent_dropout_rate = self._attrs['recurrent_dropout_rate']
    if self._recurrent_dropout_rate < 0.0:
      self._recurrent_dropout_rate = (
          component.master.hyperparams.recurrent_dropout_rate)
    if self._recurrent_dropout_rate < 0.0:
      self._recurrent_dropout_rate = component.master.hyperparams.dropout_rate

    tf.logging.info('[%s] input_dropout_rate=%s recurrent_dropout_rate=%s',
                    component.name, self._input_dropout_rate,
                    self._recurrent_dropout_rate)

    layers, context_layers = self.create_hidden_layers(component,
                                                       self._hidden_layer_sizes)
    last_layer_dim = layers[-1].dim
    layers.append(
        dragnn.Layer(component, name='last_layer', dim=last_layer_dim))
    layers.append(
        dragnn.Layer(component, name='logits', dim=component.num_actions))

    # Provide initial layers and context layers, so the base class constructor
    # can safely use accessors like get_layer_size().
    super(BaseLSTMNetwork, self).__init__(
        component, init_layers=layers, init_context_layers=context_layers)

    # Allocate parameters for the softmax.
    self._params.append(
        tf.get_variable(
            'weights_softmax', [last_layer_dim, component.num_actions],
            initializer=tf.random_normal_initializer(stddev=1e-4)))
    self._params.append(
        tf.get_variable(
            'bias_softmax', [component.num_actions],
            initializer=tf.zeros_initializer()))

  def get_logits(self, network_tensors):
    """Returns the logits for prediction."""
    return network_tensors[self.get_layer_index('logits')]

  @abc.abstractmethod
  def create_hidden_layers(self, component, hidden_layer_sizes):
    """Creates hidden network layers.

    Args:
      component: Parent ComponentBuilderBase object.
      hidden_layer_sizes: List of requested hidden layer activation sizes.

    Returns:
      layers: List of layers created by this network.
      context_layers: List of context layers created by this network.
    """
    pass

  def _append_base_layers(self, hidden_layers):
    """Appends layers defined by the base class to the |hidden_layers|."""
    last_layer = hidden_layers[-1]

    logits = tf.nn.xw_plus_b(last_layer,
                             self._component.get_variable('weights_softmax'),
                             self._component.get_variable('bias_softmax'))
    return hidden_layers + [last_layer, logits]

  def _create_cell(self, num_units, during_training):
    """Creates a single LSTM cell, possibly with dropout.

    Requires that BaseLSTMNetwork.__init__() was called.

    Args:
      num_units: Number of hidden units in the cell.
      during_training: Whether to create a cell for training (vs inference).

    Returns:
      A RNNCell of the requested size, possibly with dropout.
    """
    # No dropout in inference mode.
    if not during_training:
      return tf.contrib.rnn.LayerNormBasicLSTMCell(
          num_units, layer_norm=self._attrs['layer_norm'], reuse=True)

    # Otherwise, apply dropout to inputs and recurrences.
    cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
        num_units,
        dropout_keep_prob=self._recurrent_dropout_rate,
        layer_norm=self._attrs['layer_norm'])
    cell = tf.contrib.rnn.DropoutWrapper(
        cell, input_keep_prob=self._input_dropout_rate)
    return cell

  def _create_train_cells(self):
    """Creates a list of LSTM cells for training."""
    return [
        self._create_cell(num_units, during_training=True)
        for num_units in self._hidden_layer_sizes
    ]

  def _create_inference_cells(self):
    """Creates a list of LSTM cells for inference."""
    return [
        self._create_cell(num_units, during_training=False)
        for num_units in self._hidden_layer_sizes
    ]

  def _capture_variables_as_params(self, function):
    """Captures variables created by a function in |self._params|."""
    self._params.extend(capture_variables(function, 'cell'))

  def _apply_with_captured_variables(self, function):
    """Applies a function using previously-captured variables."""
    return apply_with_captured_variables(function, 'cell', self._component)


class LayerNormBasicLSTMNetwork(BaseLSTMNetwork):
  """Wrapper around tf.contrib.rnn.LayerNormBasicLSTMCell.

  Features:
    All inputs are concatenated.

  Subclass-specific layers:
    state_c_<n>: Cell states for the <n>'th LSTM layer (0-origin).
    state_h_<n>: Hidden states for the <n>'th LSTM layer (0-origin).
  """

  def __init__(self, component):
    """Sets up context and output layers, as well as a final softmax."""
    super(LayerNormBasicLSTMNetwork, self).__init__(component)

    # Wrap lists of training and inference sub-cells into multi-layer RNN cells.
    # Note that a |MultiRNNCell| state is a tuple of per-layer sub-states.
    self._train_cell = tf.contrib.rnn.MultiRNNCell(self._create_train_cells())
    self._inference_cell = tf.contrib.rnn.MultiRNNCell(
        self._create_inference_cells())

    def _cell_closure(scope):
      """Applies the LSTM cell to placeholder inputs and state."""
      placeholder_inputs = tf.placeholder(
          dtype=tf.float32, shape=(1, self._concatenated_input_dim))

      placeholder_substates = []
      for num_units in self._hidden_layer_sizes:
        placeholder_substate = tf.contrib.rnn.LSTMStateTuple(
            tf.placeholder(dtype=tf.float32, shape=(1, num_units)),
            tf.placeholder(dtype=tf.float32, shape=(1, num_units)))
        placeholder_substates.append(placeholder_substate)
      placeholder_state = tuple(placeholder_substates)

      self._train_cell(
          inputs=placeholder_inputs, state=placeholder_state, scope=scope)

    self._capture_variables_as_params(_cell_closure)

  def create_hidden_layers(self, component, hidden_layer_sizes):
    """See base class."""
    # Construct the layer meta info for the DRAGNN builder. Note that the order
    # of h and c are reversed compared to the vanilla DRAGNN LSTM cell, as
    # this is the standard in tf.contrib.rnn.
    #
    # NB: The h activations of the last LSTM must be the last layer, in order
    # for _append_base_layers() to work.
    layers = []
    for index, num_units in enumerate(hidden_layer_sizes):
      layers.append(
          dragnn.Layer(component, name='state_c_%d' % index, dim=num_units))
      layers.append(
          dragnn.Layer(component, name='state_h_%d' % index, dim=num_units))
    context_layers = list(layers)  # copy |layers|, don't alias it
    return layers, context_layers

  def create(self,
             fixed_embeddings,
             linked_embeddings,
             context_tensor_arrays,
             attention_tensor,
             during_training,
             stride=None):
    """See base class."""
    # NB: This cell pulls the lstm's h and c vectors from context_tensor_arrays
    # instead of through linked features.
    check.Eq(
        len(context_tensor_arrays), 2 * len(self._hidden_layer_sizes),
        'require two context tensors per hidden layer')

    # Rearrange the context tensors into a tuple of LSTM sub-states.
    length = context_tensor_arrays[0].size()
    substates = []
    for index, num_units in enumerate(self._hidden_layer_sizes):
      state_c = context_tensor_arrays[2 * index].read(length - 1)
      state_h = context_tensor_arrays[2 * index + 1].read(length - 1)

      # Fix shapes that for some reason are not set properly for an unknown
      # reason. TODO(googleuser): Why are the shapes not set?
      state_c.set_shape([tf.Dimension(None), num_units])
      state_h.set_shape([tf.Dimension(None), num_units])
      substates.append(tf.contrib.rnn.LSTMStateTuple(state_c, state_h))
    state = tuple(substates)

    input_tensor = dragnn.get_input_tensor(fixed_embeddings, linked_embeddings)
    cell = self._train_cell if during_training else self._inference_cell

    def _cell_closure(scope):
      """Applies the LSTM cell to the current inputs and state."""
      return cell(input_tensor, state, scope=scope)

    unused_h, state = self._apply_with_captured_variables(_cell_closure)

    # Return tensors to be put into the tensor arrays / used to compute
    # objective.
    output_tensors = []
    for new_substate in state:
      new_c, new_h = new_substate
      output_tensors.append(new_c)
      output_tensors.append(new_h)
    return self._append_base_layers(output_tensors)


class BulkBiLSTMNetwork(BaseLSTMNetwork):
  """Bulk wrapper around tf.contrib.rnn.stack_bidirectional_dynamic_rnn().

  Features:
    lengths: [stride, 1] sequence lengths per batch item.
    All other features are concatenated into input activations.

  Subclass-specific layers:
    outputs: [stride * num_steps, self._output_dim] bi-LSTM activations.
  """

  def __init__(self, component):
    """Initializes the bulk bi-LSTM.

    Parameters used:
      parallel_iterations (1): Parallelism of the underlying tf.while_loop().
        Defaults to 1 thread to encourage deterministic behavior, but can be
        increased to trade memory for speed.

    Args:
      component: parent ComponentBuilderBase object.
    """
    super(BulkBiLSTMNetwork, self).__init__(
        component, additional_attr_defaults={'parallel_iterations': 1})

    check.In('lengths', self._linked_feature_dims,
             'Missing required linked feature')
    check.Eq(self._linked_feature_dims['lengths'], 1,
             'Wrong dimension for "lengths" feature')
    self._input_dim = self._concatenated_input_dim - 1  # exclude 'lengths'
    self._output_dim = self.get_layer_size('outputs')
    tf.logging.info('[%s] Bulk bi-LSTM with input_dim=%d output_dim=%d',
                    component.name, self._input_dim, self._output_dim)

    # Create one training and inference cell per layer and direction.
    self._train_cells_forward = self._create_train_cells()
    self._train_cells_backward = self._create_train_cells()
    self._inference_cells_forward = self._create_inference_cells()
    self._inference_cells_backward = self._create_inference_cells()

    def _bilstm_closure(scope):
      """Applies the bi-LSTM to placeholder inputs and lengths."""
      # Use singleton |stride| and |steps| because their values don't affect the
      # weight variables.
      stride, steps = 1, 1
      placeholder_inputs = tf.placeholder(
          dtype=tf.float32, shape=[stride, steps, self._input_dim])
      placeholder_lengths = tf.placeholder(dtype=tf.int64, shape=[stride])

      # Omit the initial states and sequence lengths for simplicity; they don't
      # affect the weight variables.
      tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
          self._train_cells_forward,
          self._train_cells_backward,
          placeholder_inputs,
          dtype=tf.float32,
          sequence_length=placeholder_lengths,
          scope=scope)

    self._capture_variables_as_params(_bilstm_closure)

    # Allocate parameters for the initial states.  Note that an LSTM state is a
    # tuple of two substates (c, h), so there are 4 variables per layer.
    for index, num_units in enumerate(self._hidden_layer_sizes):
      for direction in ['forward', 'backward']:
        for substate in ['c', 'h']:
          self._params.append(
              tf.get_variable(
                  'initial_state_%s_%s_%d' % (direction, substate, index),
                  [1, num_units],  # leading 1 for later batch-wise tiling
                  dtype=tf.float32,
                  initializer=tf.constant_initializer(0.0)))

  def create_hidden_layers(self, component, hidden_layer_sizes):
    """See base class."""
    dim = 2 * hidden_layer_sizes[-1]
    return [dragnn.Layer(component, name='outputs', dim=dim)], []

  def create(self,
             fixed_embeddings,
             linked_embeddings,
             context_tensor_arrays,
             attention_tensor,
             during_training,
             stride=None):
    """Requires |stride|; otherwise see base class."""
    check.NotNone(stride,
                  'BulkBiLSTMNetwork requires "stride" and must be called '
                  'in the bulk feature extractor component.')

    # Flatten the lengths into a vector.
    lengths = dragnn.lookup_named_tensor('lengths', linked_embeddings)
    lengths_s = tf.squeeze(lengths.tensor, [1])

    # Collect all other inputs into a batched tensor.
    linked_embeddings = [
        named_tensor for named_tensor in linked_embeddings
        if named_tensor.name != 'lengths'
    ]
    inputs_sxnxd = dragnn.get_input_tensor_with_stride(
        fixed_embeddings, linked_embeddings, stride)

    # Since get_input_tensor_with_stride() concatenates the input embeddings, it
    # obscures the static activation dimension, which the RNN library requires.
    # Restore it using set_shape().  Note that set_shape() merges into the known
    # shape, so only specify the activation dimension.
    inputs_sxnxd.set_shape(
        [tf.Dimension(None), tf.Dimension(None), self._input_dim])

    initial_states_forward, initial_states_backward = (
        self._create_initial_states(stride))

    if during_training:
      cells_forward = self._train_cells_forward
      cells_backward = self._train_cells_backward
    else:
      cells_forward = self._inference_cells_forward
      cells_backward = self._inference_cells_backward

    def _bilstm_closure(scope):
      """Applies the bi-LSTM to the current inputs."""
      outputs_sxnxd, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
          cells_forward,
          cells_backward,
          inputs_sxnxd,
          initial_states_fw=initial_states_forward,
          initial_states_bw=initial_states_backward,
          sequence_length=lengths_s,
          parallel_iterations=self._attrs['parallel_iterations'],
          scope=scope)
      return outputs_sxnxd

    # Layer outputs are not batched; flatten out the batch dimension.
    outputs_sxnxd = self._apply_with_captured_variables(_bilstm_closure)
    outputs_snxd = tf.reshape(outputs_sxnxd, [-1, self._output_dim])
    return self._append_base_layers([outputs_snxd])

  def _create_initial_states(self, stride):
    """Returns stacked and batched initial states for the bi-LSTM."""
    initial_states_forward = []
    initial_states_backward = []
    for index in range(len(self._hidden_layer_sizes)):
      # Retrieve the initial states for this layer.
      states_sxd = []
      for direction in ['forward', 'backward']:
        for substate in ['c', 'h']:
          state_1xd = self._component.get_variable('initial_state_%s_%s_%d' %
                                                   (direction, substate, index))
          state_sxd = tf.tile(state_1xd, [stride, 1])  # tile across the batch
          states_sxd.append(state_sxd)

      # Assemble and append forward and backward LSTM states.
      initial_states_forward.append(
          tf.contrib.rnn.LSTMStateTuple(states_sxd[0], states_sxd[1]))
      initial_states_backward.append(
          tf.contrib.rnn.LSTMStateTuple(states_sxd[2], states_sxd[3]))
    return initial_states_forward, initial_states_backward
