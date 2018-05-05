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

"""Network units implementing the Transformer network (Vaswani et al. 2017).

Heavily adapted from the tensor2tensor implementation of the Transformer,
described in detail here: https://arxiv.org/abs/1706.03762.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from dragnn.python import network_units


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.

  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.

  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  expressed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.

  Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float

  Returns:
    a Tensor the same shape as x.
  """
  length = tf.shape(x)[1]
  channels = tf.shape(x)[2]
  pos = tf.to_float(tf.range(length))
  num_timescales = channels // 2
  log_timescale_increment = (
      np.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(pos, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return x + signal


def split_last_dimension(x, n):
  """Partitions x so that the last dimension becomes two dimensions.

  The first of these two dimensions is n.

  Args:
    x: a Tensor with shape [..., m]
    n: an integer.

  Returns:
    a Tensor with shape [..., n, m/n]
  """
  old_shape = x.get_shape().dims
  last = old_shape[-1]
  new_shape = old_shape[:-1] + [n] + [last // n if last else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
  ret.set_shape(new_shape)
  return ret


def combine_last_two_dimensions(x):
  """Reshape x so that the last two dimensions become one.

  Args:
    x: a Tensor with shape [..., a, b]

  Returns:
    a Tensor with shape [..., ab]
  """
  old_shape = x.get_shape().dims
  a, b = old_shape[-2:]
  new_shape = old_shape[:-2] + [a * b if a and b else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
  ret.set_shape(new_shape)
  return ret


def split_heads(x, num_heads):
  """Splits channels (dimension 3) into multiple heads (becomes dimension 1).

  Args:
    x: a Tensor with shape [batch, length, channels]
    num_heads: an integer

  Returns:
    a Tensor with shape [batch, num_heads, length, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def combine_heads(x):
  """Performs the inverse of split_heads.

  Args:
    x: a Tensor with shape [batch, num_heads, length, channels / num_heads]

  Returns:
    a Tensor with shape [batch, length, channels]
  """
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def compute_padding_mask(lengths):
  """Computes an additive mask for padding.

  Given the non-padded sequence lengths for the batch, computes a mask that will
  send padding attention to 0 when added to logits before applying a softmax.

  Args:
    lengths: a Tensor containing the sequence length of each batch element

  Returns:
    A Tensor of shape [batch_size, 1, 1, max_len] with zeros in non-padding
    entries and -1e9 in padding entries.
  """
  lengths = tf.reshape(lengths, [-1])
  mask = tf.sequence_mask(lengths)

  # This will be used as an additive mask, so we want the inverse of the mask
  # produced by tf.sequence_mask.
  inv_mask = tf.to_float(tf.logical_not(mask))

  mem_padding = inv_mask * -1e9
  return tf.expand_dims(tf.expand_dims(mem_padding, 1), 1)


def dot_product_attention(queries, keys, values, dropout_keep_rate, bias=None):
  """Computes dot-product attention.

  Args:
    queries: a Tensor with shape [batch, heads, seq_len, depth_keys]
    keys: a Tensor with shape [batch, heads, seq_len, depth_keys]
    values: a Tensor with shape [batch, heads, seq_len, depth_values]
    dropout_keep_rate: dropout proportion of units to keep
    bias: A bias to add before applying the softmax, or None. This can be used
          for masking padding in the batch.

  Returns:
    A Tensor with shape [batch, heads, seq_len, depth_values].
  """
  # [batch, num_heads, seq_len, seq_len]
  logits = tf.matmul(queries, keys, transpose_b=True)
  if bias is not None:
    logits += bias

  attn_weights = tf.nn.softmax(logits)

  # Dropping out the attention links for each of the heads
  attn_weights = network_units.maybe_apply_dropout(attn_weights,
                                                   dropout_keep_rate,
                                                   False)
  return tf.matmul(attn_weights, values)


def residual(old_input, new_input, dropout_keep_rate, layer_norm):
  """Residual layer combining old_input and new_input.

  Computes old_input + dropout(new_input) if layer_norm is None; otherwise:
  layer_norm(old_input + dropout(new_input)).

  Args:
    old_input: old float32 Tensor input to residual layer
    new_input: new float32 Tensor input to residual layer
    dropout_keep_rate: dropout proportion of units to keep
    layer_norm: network_units.LayerNorm to apply to residual output, or None

  Returns:
    float32 Tensor output of residual layer.
  """
  res_sum = old_input + network_units.maybe_apply_dropout(new_input,
                                                          dropout_keep_rate,
                                                          False)
  return layer_norm.normalize(res_sum) if layer_norm else res_sum


def mlp(component, input_tensor, dropout_keep_rate, depth):
  """Feed the input through an MLP.

  Each layer except the last is followed by a ReLU activation and dropout.

  Args:
    component: the DRAGNN Component containing parameters for the MLP
    input_tensor: the float32 Tensor input to the MLP.
    dropout_keep_rate: dropout proportion of units to keep
    depth: depth of the MLP.

  Returns:
    the float32 output Tensor
  """
  for i in range(depth):
    ff_weights = component.get_variable('ff_weights_%d' % i)
    input_tensor = tf.nn.conv2d(input_tensor,
                                ff_weights,
                                [1, 1, 1, 1],
                                padding='SAME')
    # Apply ReLU and dropout to all but the last layer
    if i < depth - 1:
      input_tensor = tf.nn.relu(input_tensor)
      input_tensor = network_units.maybe_apply_dropout(input_tensor,
                                                       dropout_keep_rate,
                                                       False)
  return input_tensor


class TransformerEncoderNetwork(network_units.NetworkUnitInterface):
  """Implementation of the Transformer network encoder."""

  def __init__(self, component):
    """Initializes parameters for this Transformer unit.

    Args:
      component: parent ComponentBuilderBase object.

    Parameters used to construct the network:
      num_layers: number of transformer layers (attention + MLP)
      hidden_size: size of hidden layers in MLPs
      filter_size: filter width for each attention head
      num_heads: number of attention heads
      residual_dropout: dropout keep rate for residual layers
      attention_dropout: dropout keep rate for attention weights
      mlp_dropout: dropout keep rate for mlp layers
      initialization: initialization scheme to use for model parameters
      bias_init: initial value for bias parameters
      scale_attention: whether to scale attention parameters by filter_size^-0.5
      layer_norm_residuals: whether to perform layer normalization on residual
        layers
      timing_signal: whether to add a position-wise timing signal to the input
      kernel: kernel width in middle MLP layers
      mlp_layers: number of MLP layers. Must be >= 2.

    Raises:
      ValueError: if mlp_layers < 2.

    The input depth of the first layer is inferred from the total concatenated
    size of the input features, minus 1 to account for the sequence lengths.

    Hyperparameters used:
      dropout_rate: The probability that an input is not dropped. This is the
          default when the |dropout_keep_prob| parameter is unset.
    """

    super(TransformerEncoderNetwork, self).__init__(component)
    default_dropout_rate = component.master.hyperparams.dropout_rate
    self._attrs = network_units.get_attrs_with_defaults(
        component.spec.network_unit.parameters, defaults={
            'num_layers': 4,
            'hidden_size': 256,
            'filter_size': 64,
            'num_heads': 8,
            'residual_drop': default_dropout_rate,
            'attention_drop': default_dropout_rate,
            'mlp_drop': default_dropout_rate,
            'initialization': 'xavier',
            'bias_init': 0.001,
            'scale_attention': True,
            'layer_norm_residuals': True,
            'timing_signal': True,
            'kernel': 1,
            'mlp_layers': 2})

    self._num_layers = self._attrs['num_layers']
    self._hidden_size = self._attrs['hidden_size']
    self._filter_size = self._attrs['filter_size']
    self._num_heads = self._attrs['num_heads']
    self._residual_dropout = self._attrs['residual_drop']
    self._attention_dropout = self._attrs['attention_drop']
    self._mlp_dropout = self._attrs['mlp_drop']
    self._initialization = self._attrs['initialization']
    self._bias_init = self._attrs['bias_init']
    self._scale_attn = self._attrs['scale_attention']
    self._layer_norm_res = self._attrs['layer_norm_residuals']
    self._timing_signal = self._attrs['timing_signal']
    self._kernel = self._attrs['kernel']
    self._mlp_depth = self._attrs['mlp_layers']

    if self._mlp_depth < 2:
      raise ValueError('TransformerEncoderNetwork needs mlp_layers >= 2')

    self._combined_filters = self._num_heads * self._filter_size

    self._weights = []
    self._biases = []
    self._layer_norms = {}

    # Hacky: one dimension comes from the lengths input; subtract it.
    self._concatenated_input_dim -= 1

    # Initial projection of inputs, this is mainly to project input down to the
    # right size for residual layers
    proj_shape = [1, 1, self._concatenated_input_dim, self._combined_filters]
    self._weights.append(
        network_units.add_var_initialized('init_proj', proj_shape,
                                          self._initialization))
    self._biases.append(tf.get_variable('init_bias',
                                        self._combined_filters,
                                        initializer=tf.constant_initializer(
                                            self._bias_init),
                                        dtype=tf.float32))

    for i in range(self._num_layers):
      with tf.variable_scope('transform_%d' % i):
        # Attention weights: 3 * self.combined_filters = (q, k, v)
        # We assume that q, k and v all have the same dimension
        attn_shape = [1, 1, self._combined_filters, 3 * self._combined_filters]
        self._weights.append(
            network_units.add_var_initialized('attn_weights',
                                              attn_shape,
                                              self._initialization))

        # Attention final projection weights
        proj_shape = [1, 1, self._combined_filters, self._combined_filters]
        self._weights.append(
            network_units.add_var_initialized('proj_weights',
                                              proj_shape,
                                              self._initialization))

        # MLP weights
        with tf.variable_scope('mlp'):
          ff_shape = [1, 1, self._combined_filters, self._hidden_size]
          self._weights.append(
              network_units.add_var_initialized('ff_weights_0',
                                                ff_shape,
                                                self._initialization))
          ff_shape = [1, self._kernel, self._hidden_size, self._hidden_size]
          for j in range(1, self._mlp_depth - 1):
            self._weights.append(
                network_units.add_var_initialized('ff_weights_%d' % j,
                                                  ff_shape,
                                                  self._initialization))
          ff_shape = [1, 1, self._hidden_size, self._combined_filters]
          self._weights.append(
              network_units.add_var_initialized('ff_weights_%d' %
                                                (self._mlp_depth - 1),
                                                ff_shape,
                                                self._initialization))

        # Layer normalization for residual layers
        if self._layer_norm_res:
          attn_layer_norm = network_units.LayerNorm(component,
                                                    'attn_layer_norm_%d' % i,
                                                    self._combined_filters,
                                                    tf.float32)
          self._layer_norms['attn_layer_norm_%d' % i] = attn_layer_norm

          ff_layer_norm = network_units.LayerNorm(component,
                                                  'ff_layer_norm_%d' % i,
                                                  self._combined_filters,
                                                  tf.float32)
          self._layer_norms['ff_layer_norm_%d' % i] = ff_layer_norm

          # Layer norm parameters are not added to self._weights,
          # which means that they are not l2 regularized
          self._params.extend(attn_layer_norm.params + ff_layer_norm.params)

    self._params.extend(self._weights)
    self._params.extend(self._biases)
    self._regularized_weights.extend(self._weights)
    self._layers.append(
        network_units.Layer(component, name='transformer_output',
                            dim=self._combined_filters))

  def create(self,
             fixed_embeddings,
             linked_embeddings,
             context_tensor_arrays,
             attention_tensor,
             during_training,
             stride=None):
    """Requires |stride|; otherwise see base class."""
    del context_tensor_arrays, attention_tensor
    if stride is None:
      raise RuntimeError("TransformerEncoderNetwork needs 'stride' and must be "
                         "called in the bulk feature extractor component.")

    lengths = network_units.lookup_named_tensor('lengths', linked_embeddings)
    lengths_s = tf.to_int32(tf.squeeze(lengths.tensor, [1]))
    num_steps = tf.reduce_max(lengths_s)

    in_tensor = network_units.lookup_named_tensor('features', linked_embeddings)
    input_tensor = tf.reshape(in_tensor.tensor, [stride, num_steps, -1])

    if self._timing_signal:
      input_tensor = add_timing_signal_1d(input_tensor)

    # Adds a dimension for conv2d
    input_tensor = tf.expand_dims(input_tensor, 1)

    # For masking padding in attention
    mask = compute_padding_mask(lengths_s)

    conv = tf.nn.conv2d(input_tensor,
                        self._component.get_variable('init_proj'),
                        [1, 1, 1, 1], padding='SAME')
    conv = tf.nn.bias_add(conv, self._component.get_variable('init_bias'))

    for i in range(self._num_layers):
      with tf.variable_scope('transform_%d' % i, reuse=True):
        attn_weights = self._component.get_variable('attn_weights')
        attn_combined = tf.nn.conv2d(conv,
                                     attn_weights,
                                     [1, 1, 1, 1],
                                     padding='SAME')
        attn_combined = tf.squeeze(attn_combined, 1)

        # Splits combined projection into queries, keys, and values
        queries, keys, values = tf.split(attn_combined,
                                         [self._combined_filters]*3,
                                         axis=2)

        # Splits each of queries, keys, values into attention heads
        queries = split_heads(queries, self._num_heads)
        keys = split_heads(keys, self._num_heads)
        values = split_heads(values, self._num_heads)
        if self._scale_attn:
          queries *= self._filter_size**-0.5

        # Performs dot product attention and concatenates the resulting heads
        attended = dot_product_attention(queries, keys, values,
                                         self._attention_dropout, mask)
        attended = combine_heads(attended)

        # Projects combined heads
        attended = tf.expand_dims(attended, 1)
        proj = tf.nn.conv2d(attended,
                            self._component.get_variable('proj_weights'),
                            [1, 1, 1, 1],
                            padding='SAME')

        # Residual connection between input and attended input
        attn_layer_norm_params = None
        if self._layer_norm_res:
          attn_layer_norm_params = self._layer_norms['attn_layer_norm_%d' % i]
        proj_res = residual(conv, proj, self._residual_dropout,
                            attn_layer_norm_params)

        # Feed forward
        with tf.variable_scope('mlp'):
          ff = mlp(self._component, proj_res, self._mlp_dropout,
                   self._mlp_depth)

        # Residual connection between attended input and feed forward layers
        ff_layer_norm_params = None
        if self._layer_norm_res:
          ff_layer_norm_params = self._layer_norms['ff_layer_norm_%d' % i]
        conv = residual(proj_res, ff, self._residual_dropout,
                        ff_layer_norm_params)

    return [tf.reshape(conv, [-1, self._combined_filters],
                       name='reshape_activations')]


class PairwiseBilinearLabelNetwork(network_units.NetworkUnitInterface):
  r"""Network unit that computes pairwise bilinear label scores.

  Given source and target representations for each token, this network unit
  computes bilinear scores for each label for each of the N^2 combinations of
  source and target tokens, rather than for only N already-computed
  source/target pairs (as is performed by the biaffine_units). The output is
  suitable as input to e.g. the heads_labels transition system.
  Specifically, a weights tensor W called `bilinear' is used to compute bilinear
  scores B for input tensors S and T:

    B_{bnml} = \sum_{i,j} S_{bni} W_{ilj} T{bmj}

  for batches b, steps n and m and labels l.

  Parameters:
    num_labels: The number of dependency labels, L.

  Features:
    sources: [B * N, S] matrix of batched activations for source tokens.
    targets: [B * N, T] matrix of batched activations for target tokens.

  Layers:
    bilinear_scores: [B * N, N * L] matrix where vector b*N*N*L+t contains
                     per-label scores for all N possible arcs from token t in
                     batch b.
  """

  def __init__(self, component):
    super(PairwiseBilinearLabelNetwork, self).__init__(component)
    parameters = component.spec.network_unit.parameters

    self._num_labels = int(parameters['num_labels'])

    self._source_dim = self._linked_feature_dims['sources']
    self._target_dim = self._linked_feature_dims['targets']

    self._weights = []
    self._weights.append(
        network_units.add_var_initialized('bilinear',
                                          [self._source_dim,
                                           self._num_labels,
                                           self._target_dim],
                                          'xavier'))

    self._params.extend(self._weights)
    self._regularized_weights.extend(self._weights)
    self._layers.append(network_units.Layer(component,
                                            name='bilinear_scores',
                                            dim=self._num_labels))

  def create(self,
             fixed_embeddings,
             linked_embeddings,
             context_tensor_arrays,
             attention_tensor,
             during_training,
             stride=None):
    """Requires |stride|; otherwise see base class."""
    del context_tensor_arrays, attention_tensor
    if stride is None:
      raise RuntimeError("PairwiseBilinearLabelNetwork needs 'stride' and must "
                         "be called in a bulk component.")

    sources = network_units.lookup_named_tensor('sources', linked_embeddings)
    sources_tensor = tf.reshape(sources.tensor, [stride, -1, self._source_dim])

    targets = network_units.lookup_named_tensor('targets', linked_embeddings)
    targets_tensor = tf.reshape(targets.tensor, [stride, -1, self._target_dim])

    # Dimensions: source_dim x num_labels x target_dim
    bilinear_params = self._component.get_variable('bilinear')

    # Ensures that num_steps is the same for both inputs
    num_steps = tf.shape(sources_tensor)[1]
    with tf.control_dependencies([tf.assert_equal(num_steps,
                                                  tf.shape(targets_tensor)[1],
                                                  name='num_steps_mismatch')]):
      # Dimensions:
      # (batch_size*num_steps x source_dim) *
      #   (source_dim x num_labels*target_dim)
      #     = (batch_size*num_steps x num_labels*target_dim)
      lin = tf.matmul(tf.reshape(sources_tensor, [-1, self._source_dim]),
                      tf.reshape(bilinear_params, [self._source_dim, -1]))

      # (batch_size x num_steps*num_labels x target_dim) *
      #   (batch_size x num_steps x target_dim)^T
      #     = (batch_size x num_steps*num_labels x num_steps)
      bilin = tf.matmul(
          tf.reshape(lin, [-1, num_steps*self._num_labels, self._target_dim]),
          targets_tensor, transpose_b=True)

    # (batch_size x num_steps*num_labels x num_steps) ->
    #   (batch_size x num_steps x num_steps*num_labels)
    scores = tf.transpose(bilin, [0, 2, 1])

    return [tf.reshape(scores, [-1, num_steps*self._num_labels],
                       name='reshape_activations')]
