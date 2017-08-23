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

"""Basic network units used in assembling DRAGNN graphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python.ops import tensor_array_ops as ta
from tensorflow.python.platform import tf_logging as logging

from dragnn.python import dragnn_ops
from syntaxnet import syntaxnet_ops
from syntaxnet.util import check
from syntaxnet.util import registry


def linked_embeddings_name(channel_id):
  """Returns the name of the linked embedding matrix for some channel ID."""
  return 'linked_embedding_matrix_%d' % channel_id


def fixed_embeddings_name(channel_id):
  """Returns the name of the fixed embedding matrix for some channel ID."""
  return 'fixed_embedding_matrix_%d' % channel_id


class StoredActivations(object):
  """Wrapper around stored activation vectors.

  Because activations are produced and consumed in different layouts by bulk
  vs. dynamic components, this class provides a simple common
  interface/conversion API. It can be constructed from either a TensorArray
  (dynamic) or a Tensor (bulk), and the resulting object to use for lookups is
  either bulk_tensor (for bulk components) or dynamic_tensor (for dynamic
  components).
  """

  def __init__(self, tensor=None, array=None, stride=None, dim=None):
    """Creates ops for converting the input to either format.

    If 'tensor' is used, then a conversion from [stride * steps, dim] to
    [steps + 1, stride, dim] is performed for dynamic_tensor reads.

    If 'array' is used, then a conversion from [steps + 1, stride, dim] to
    [stride * steps, dim] is performed for bulk_tensor reads.

    Args:
      tensor: Bulk tensor input.
      array: TensorArray dynamic input.
      stride: stride of bulk tensor. Not used for dynamic.
      dim: dim of bulk tensor. Not used for dynamic.
    """
    if tensor is not None:
      check.IsNone(array, 'Cannot initialize from tensor and array')
      check.NotNone(stride, 'Stride is required for bulk tensor')
      check.NotNone(dim, 'Dim is required for bulk tensor')

      self._bulk_tensor = tensor
      with tf.name_scope('convert_to_dyn'):
        tensor = tf.reshape(tensor, [stride, -1, dim])
        tensor = tf.transpose(tensor, perm=[1, 0, 2])
        pad = tf.zeros([1, stride, dim], dtype=tensor.dtype)
        self._array_tensor = tf.concat([pad, tensor], 0)

    if array is not None:
      check.IsNone(tensor, 'Cannot initialize from both tensor and array')
      with tf.name_scope('convert_to_bulk'):
        self._bulk_tensor = convert_network_state_tensorarray(array)
      with tf.name_scope('convert_to_dyn'):
        self._array_tensor = array.stack()

  @property
  def bulk_tensor(self):
    return self._bulk_tensor

  @property
  def dynamic_tensor(self):
    return self._array_tensor


class NamedTensor(object):
  """Container for a tensor with associated name and dimension attributes."""

  def __init__(self, tensor, name, dim=None):
    """Inits NamedTensor with tensor, name and optional dim."""
    self.tensor = tensor
    self.name = name
    self.dim = dim


def add_embeddings(channel_id, feature_spec, seed=None):
  """Adds a variable for the embedding of a given fixed feature.

  Supports pre-trained or randomly initialized embeddings In both cases, extra
  vector is reserved for out-of-vocabulary words, so the embedding matrix has
  the size of [feature_spec.vocabulary_size + 1, feature_spec.embedding_dim].

  Args:
    channel_id: Numeric id of the fixed feature channel
    feature_spec: Feature spec protobuf of type FixedFeatureChannel
    seed: used for random initializer

  Returns:
    tf.Variable object corresponding to the embedding for that feature.

  Raises:
    RuntimeError: if more the pretrained embeddings are specified in resources
        containing more than one part.
  """
  check.Gt(feature_spec.embedding_dim, 0,
           'Embeddings requested for non-embedded feature: %s' % feature_spec)
  name = fixed_embeddings_name(channel_id)
  shape = [feature_spec.vocabulary_size + 1, feature_spec.embedding_dim]
  if feature_spec.HasField('pretrained_embedding_matrix'):
    if len(feature_spec.pretrained_embedding_matrix.part) > 1:
      raise RuntimeError('pretrained_embedding_matrix resource contains '
                         'more than one part:\n%s',
                         str(feature_spec.pretrained_embedding_matrix))
    if len(feature_spec.vocab.part) > 1:
      raise RuntimeError('vocab resource contains more than one part:\n%s',
                         str(feature_spec.vocab))
    seed1, seed2 = tf.get_seed(seed)
    embeddings = syntaxnet_ops.word_embedding_initializer(
        vectors=feature_spec.pretrained_embedding_matrix.part[0].file_pattern,
        vocabulary=feature_spec.vocab.part[0].file_pattern,
        num_special_embeddings=1,
        embedding_init=1.0,
        seed=seed1,
        seed2=seed2)
    return tf.get_variable(name, initializer=tf.reshape(embeddings, shape))
  else:
    return tf.get_variable(
        name,
        shape,
        initializer=tf.random_normal_initializer(
            stddev=1.0 / feature_spec.embedding_dim**.5, seed=seed))


def embedding_lookup(embedding_matrix, indices, ids, weights, size):
  """Performs a weighted embedding lookup.

  Args:
    embedding_matrix: float Tensor from which to do the lookup.
    indices: int Tensor for the output rows of the looked up vectors.
    ids: int Tensor vectors to look up in the embedding_matrix.
    weights: float Tensor weights to apply to the looked up vectors.
    size: int number of output rows. Needed since some output rows may be
        empty.

  Returns:
    Weighted embedding vectors.
  """
  embeddings = tf.nn.embedding_lookup([embedding_matrix], ids)
  # TODO(googleuser): allow skipping weights.
  broadcast_weights_shape = tf.concat([tf.shape(weights), [1]], 0)
  embeddings *= tf.reshape(weights, broadcast_weights_shape)
  embeddings = tf.unsorted_segment_sum(embeddings, indices, size)
  return embeddings


def fixed_feature_lookup(component, state, channel_id, stride):
  """Looks up fixed features and passes them through embeddings.

  Embedding vectors may be scaled by weights if the features specify it.

  Args:
    component: Component object in which to look up the fixed features.
    state: MasterState object for the live nlp_saft::dragnn::MasterState.
    channel_id: int id of the fixed feature to look up.
    stride: int Tensor of current batch * beam size.

  Returns:
    NamedTensor object containing the embedding vectors.
  """
  feature_spec = component.spec.fixed_feature[channel_id]
  check.Gt(feature_spec.embedding_dim, 0,
           'Embeddings requested for non-embedded feature: %s' % feature_spec)
  embedding_matrix = component.get_variable(fixed_embeddings_name(channel_id))

  with tf.op_scope([embedding_matrix], 'fixed_embedding_' + feature_spec.name):
    indices, ids, weights = dragnn_ops.extract_fixed_features(
        state.handle, component=component.name, channel_id=channel_id)
    size = stride * feature_spec.size
    embeddings = embedding_lookup(embedding_matrix, indices, ids, weights, size)
    dim = feature_spec.size * feature_spec.embedding_dim
    return NamedTensor(
        tf.reshape(embeddings, [-1, dim]), feature_spec.name, dim=dim)


def get_input_tensor(fixed_embeddings, linked_embeddings):
  """Helper function for constructing an input tensor from all the features.

  Args:
    fixed_embeddings: list of NamedTensor objects for fixed feature channels
    linked_embeddings: list of NamedTensor objects for linked feature channels

  Returns:
    a tensor of shape [N, D], where D is the total input dimension of the
        concatenated feature channels

  Raises:
    RuntimeError: if no features, fixed or linked, are configured.
  """
  embeddings = fixed_embeddings + linked_embeddings
  if not embeddings:
    raise RuntimeError('There needs to be at least one feature set defined.')

  # Concat_v2 takes care of optimizing away the concatenation
  # operation in the case when there is exactly one embedding input.
  return tf.concat([e.tensor for e in embeddings], 1)


def get_input_tensor_with_stride(fixed_embeddings, linked_embeddings, stride):
  """Constructs an input tensor with a separate dimension for steps.

  Args:
    fixed_embeddings: list of NamedTensor objects for fixed feature channels
    linked_embeddings: list of NamedTensor objects for linked feature channels
    stride: int stride (i.e. beam * batch) to use to reshape the input

  Returns:
    a tensor of shape [stride, num_steps, D], where D is the total input
        dimension of the concatenated feature channels
  """
  input_tensor = get_input_tensor(fixed_embeddings, linked_embeddings)
  shape = tf.shape(input_tensor)
  return tf.reshape(input_tensor, [stride, -1, shape[1]])


def convert_network_state_tensorarray(tensorarray):
  """Converts a source TensorArray to a source Tensor.

  Performs a permutation between the steps * [stride, D] shape of a
  source TensorArray and the (flattened) [stride * steps, D] shape of
  a source Tensor.

  The TensorArrays used during recurrence have an additional zeroth step that
  needs to be removed.

  Args:
    tensorarray: TensorArray object to be converted.

  Returns:
    Tensor object after conversion.
  """
  tensor = tensorarray.stack()  # Results in a [steps, stride, D] tensor.
  tensor = tf.slice(tensor, [1, 0, 0], [-1, -1, -1])  # Lop off the 0th step.
  tensor = tf.transpose(tensor, [1, 0, 2])  # Switch steps and stride.
  return tf.reshape(tensor, [-1, tf.shape(tensor)[2]])


def pass_through_embedding_matrix(act_block, embedding_matrix, step_idx):
  """Passes the activations through the embedding_matrix.

  Takes care to handle out of bounds lookups.

  Args:
    act_block: matrix of activations.
    embedding_matrix: matrix of weights.
    step_idx: vector containing step indices, with -1 indicating out of bounds.

  Returns:
    the embedded activations.
  """
  # Indicator vector for out of bounds lookups.
  step_idx_mask = tf.expand_dims(tf.equal(step_idx, -1), -1)

  # Pad the last column of the activation vectors with the indicator.
  act_block = tf.concat([act_block, tf.to_float(step_idx_mask)], 1)
  return tf.matmul(act_block, embedding_matrix)


def lookup_named_tensor(name, named_tensors):
  """Retrieves a NamedTensor by name.

  Args:
    name: Name of the tensor to retrieve.
    named_tensors: List of NamedTensor objects to search.

  Returns:
    The NamedTensor in |named_tensors| with the |name|.

  Raises:
    KeyError: If the |name| is not found among the |named_tensors|.
  """
  for named_tensor in named_tensors:
    if named_tensor.name == name:
      return named_tensor
  raise KeyError('Name "%s" not found in named tensors: %s' %
                 (name, named_tensors))


def activation_lookup_recurrent(component, state, channel_id, source_array,
                                source_layer_size, stride):
  """Looks up activations from tensor arrays.

  If the linked feature's embedding_dim is set to -1, the feature vectors are
  not passed through (i.e. multiplied by) an embedding matrix.

  Args:
    component: Component object in which to look up the fixed features.
    state: MasterState object for the live nlp_saft::dragnn::MasterState.
    channel_id: int id of the fixed feature to look up.
    source_array: TensorArray from which to fetch feature vectors, expected to
        have size [steps + 1] elements of shape [stride, D] each.
    source_layer_size: int length of feature vectors before embedding.
    stride: int Tensor of current batch * beam size.

  Returns:
    NamedTensor object containing the embedding vectors.
  """
  feature_spec = component.spec.linked_feature[channel_id]

  with tf.name_scope('activation_lookup_recurrent_%s' % feature_spec.name):
    # Linked features are returned as a pair of tensors, one indexing into
    # steps, and one indexing within the activation tensor (beam x batch)
    # stored for a step.
    step_idx, idx = dragnn_ops.extract_link_features(
        state.handle, component=component.name, channel_id=channel_id)

    # We take the [steps, batch*beam, ...] tensor array, gather and concat
    # the steps we might need into a [some_steps*batch*beam, ...] tensor,
    # and flatten 'idx' to dereference this new tensor.
    #
    # The first element of each tensor array is reserved for an
    # initialization variable, so we offset all step indices by +1.
    #
    # TODO(googleuser): It would be great to not have to extract
    # the steps in their entirety, forcing a copy of much of the
    # TensorArray at each step. Better would be to support a
    # TensorArray.gather_nd to pick the specific elements directly.
    # TODO(googleuser): In the interim, a small optimization would
    # be to use tf.unique instead of tf.range.
    step_min = tf.reduce_min(step_idx)
    ta_range = tf.range(step_min + 1, tf.reduce_max(step_idx) + 2)
    act_block = source_array.gather(ta_range)
    act_block = tf.reshape(act_block,
                           tf.concat([[-1], tf.shape(act_block)[2:]], 0))
    flat_idx = (step_idx - step_min) * stride + idx
    act_block = tf.gather(act_block, flat_idx)
    act_block = tf.reshape(act_block, [-1, source_layer_size])

    if feature_spec.embedding_dim != -1:
      embedding_matrix = component.get_variable(
          linked_embeddings_name(channel_id))
      act_block = pass_through_embedding_matrix(act_block, embedding_matrix,
                                                step_idx)
      dim = feature_spec.size * feature_spec.embedding_dim
    else:
      # If embedding_dim is -1, just output concatenation of activations.
      dim = feature_spec.size * source_layer_size

    return NamedTensor(
        tf.reshape(act_block, [-1, dim]), feature_spec.name, dim=dim)


def activation_lookup_other(component, state, channel_id, source_tensor,
                            source_layer_size):
  """Looks up activations from tensors.

  If the linked feature's embedding_dim is set to -1, the feature vectors are
  not passed through (i.e. multiplied by) an embedding matrix.

  Args:
    component: Component object in which to look up the fixed features.
    state: MasterState object for the live nlp_saft::dragnn::MasterState.
    channel_id: int id of the fixed feature to look up.
    source_tensor: Tensor from which to fetch feature vectors. Expected to have
        have shape [steps + 1, stride, D].
    source_layer_size: int length of feature vectors before embedding (D). It
        would in principle be possible to get this dimension dynamically from
        the second dimension of source_tensor. However, having it statically is
        more convenient.

  Returns:
    NamedTensor object containing the embedding vectors.
  """
  feature_spec = component.spec.linked_feature[channel_id]

  with tf.name_scope('activation_lookup_other_%s' % feature_spec.name):
    # Linked features are returned as a pair of tensors, one indexing into
    # steps, and one indexing within the stride (beam x batch) of each step.
    step_idx, idx = dragnn_ops.extract_link_features(
        state.handle, component=component.name, channel_id=channel_id)

    # The first element of each tensor array is reserved for an
    # initialization variable, so we offset all step indices by +1.
    indices = tf.stack([step_idx + 1, idx], axis=1)
    act_block = tf.gather_nd(source_tensor, indices)
    act_block = tf.reshape(act_block, [-1, source_layer_size])

    if feature_spec.embedding_dim != -1:
      embedding_matrix = component.get_variable(
          linked_embeddings_name(channel_id))
      act_block = pass_through_embedding_matrix(act_block, embedding_matrix,
                                                step_idx)
      dim = feature_spec.size * feature_spec.embedding_dim
    else:
      # If embedding_dim is -1, just output concatenation of activations.
      dim = feature_spec.size * source_layer_size

    return NamedTensor(
        tf.reshape(act_block, [-1, dim]), feature_spec.name, dim=dim)


class LayerNorm(object):
  """Utility to add layer normalization to any tensor.

  Layer normalization implementation is based on:

    https://arxiv.org/abs/1607.06450. "Layer Normalization"
    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

  This object will construct additional variables that need to be optimized, and
  these variables can be accessed via params().

  Attributes:
    params: List of additional parameters to be trained.
  """

  def __init__(self, component, name, shape, dtype):
    """Construct variables to normalize an input of given shape.

    Arguments:
      component: ComponentBuilder handle.
      name: Human readable name to organize the variables.
      shape: Shape of the layer to be normalized.
      dtype: Type of the layer to be normalized.
    """
    self._name = name
    self._shape = shape
    self._component = component
    beta = tf.get_variable(
        'beta_%s' % name,
        shape=shape,
        dtype=dtype,
        initializer=tf.zeros_initializer())
    gamma = tf.get_variable(
        'gamma_%s' % name,
        shape=shape,
        dtype=dtype,
        initializer=tf.ones_initializer())
    self._params = [beta, gamma]

  @property
  def params(self):
    return self._params

  def normalize(self, inputs):
    """Apply normalization to input.

    The shape must match the declared shape in the constructor.
    [This is copied from tf.contrib.rnn.LayerNormBasicLSTMCell.]

    Args:
      inputs: Input tensor

    Returns:
      Normalized version of input tensor.

    Raises:
      ValueError: if inputs has undefined rank.
    """
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    axis = range(1, inputs_rank)

    beta = self._component.get_variable('beta_%s' % self._name)
    gamma = self._component.get_variable('gamma_%s' % self._name)

    with tf.variable_scope('layer_norm_%s' % self._name):
      # Calculate the moments on the last axis (layer activations).
      mean, variance = nn.moments(inputs, axis, keep_dims=True)

      # Compute layer normalization using the batch_normalization function.
      variance_epsilon = 1E-12
      outputs = nn.batch_normalization(
          inputs, mean, variance, beta, gamma, variance_epsilon)
      outputs.set_shape(inputs_shape)
      return outputs


class Layer(object):
  """A layer in a feed-forward network.

  Attributes:
    component: ComponentBuilderBase that produces this layer.
    name: Name of this layer.
    dim: Dimension of this layer, or negative if dynamic.
  """

  def __init__(self, component, name, dim):
    check.NotNone(dim, 'Dimension is required')
    self.component = component
    self.name = name
    self.dim = dim

  def __str__(self):
    return 'Layer: %s/%s[%d]' % (self.component.name, self.name, self.dim)

  def create_array(self, stride):
    """Creates a new tensor array to store this layer's activations.

    Arguments:
      stride: Possibly dynamic batch * beam size with which to initialize the
        tensor array

    Returns:
      TensorArray object
    """
    check.Gt(self.dim, 0, 'Cannot create array when dimension is dynamic')
    tensor_array = ta.TensorArray(dtype=tf.float32,
                                  size=0,
                                  dynamic_size=True,
                                  clear_after_read=False,
                                  infer_shape=False,
                                  name='%s_array' % self.name)

    # Start each array with all zeros. Special values will still be learned via
    # the extra embedding dimension stored for each linked feature channel.
    initial_value = tf.zeros([stride, self.dim])
    return tensor_array.write(0, initial_value)


def get_attrs_with_defaults(parameters, defaults):
  """Populates a dictionary with run-time attributes.

  Given defaults, populates any overrides from 'parameters' with their
  corresponding converted values. 'defaults' should be typed. This is useful
  for specifying NetworkUnit-specific configuration options.

  Args:
    parameters: a <string, string> map.
    defaults: a <string, value> typed set of default values.

  Returns:
    dictionary populated with any overrides.

  Raises:
    RuntimeError: if a key in parameters is not present in defaults.
  """
  attrs = defaults
  for key, value in parameters.iteritems():
    check.In(key, defaults, 'Unknown attribute: %s' % key)
    if isinstance(defaults[key], bool):
      attrs[key] = value.lower() == 'true'
    else:
      attrs[key] = type(defaults[key])(value)
  return attrs


def maybe_apply_dropout(inputs, keep_prob, per_sequence, stride=None):
  """Applies dropout, if so configured, to an input tensor.

  The input may be rank 2 or 3 depending on whether the stride (i.e., batch
  size) has been incorporated into the shape.

  Args:
    inputs: [stride * num_steps, dim] or [stride, num_steps, dim] input tensor.
    keep_prob: Scalar probability of keeping each input element.  If >= 1.0, no
        dropout is performed.
    per_sequence: If true, sample the dropout mask once per sequence, instead of
        once per step.  Requires |stride| when true.
    stride: Scalar batch size.  Optional if |per_sequence| is false.

  Returns:
    [stride * num_steps, dim] or [stride, num_steps, dim] tensor, matching the
    shape of |inputs|, containing the masked or original inputs, depending on
    whether dropout was actually performed.
  """
  check.Ge(inputs.get_shape().ndims, 2, 'inputs must be rank 2 or 3')
  check.Le(inputs.get_shape().ndims, 3, 'inputs must be rank 2 or 3')
  flat = (inputs.get_shape().ndims == 2)

  if keep_prob >= 1.0:
    return inputs

  if not per_sequence:
    return tf.nn.dropout(inputs, keep_prob)

  check.NotNone(stride, 'per-sequence dropout requires stride')
  dim = inputs.get_shape().as_list()[-1]
  check.NotNone(dim, 'inputs must have static activation dimension, but have '
                'static shape %s' % inputs.get_shape().as_list())

  # If needed, restore the batch dimension to separate the sequences.
  inputs_sxnxd = tf.reshape(inputs, [stride, -1, dim]) if flat else inputs

  # Replace |num_steps| with 1 in |noise_shape|, so the dropout mask broadcasts
  # to all steps for a particular sequence.
  noise_shape = [stride, 1, dim]
  masked_sxnxd = tf.nn.dropout(inputs_sxnxd, keep_prob, noise_shape)

  # If needed, flatten out the batch dimension in the return value.
  return tf.reshape(masked_sxnxd, [-1, dim]) if flat else masked_sxnxd


@registry.RegisteredClass
class NetworkUnitInterface(object):
  """Base class to implement NN specifications.

  This class contains the required functionality to build a network inside of a
  DRAGNN graph: (1) initializing TF variables during __init__(), and (2)
  creating particular instances from extracted features in create().

  Attributes:
    params (list): List of tf.Variable objects representing trainable
      parameters.
    layers (list): List of Layer objects to track network layers that should
      be written to Tensors during training and inference.
  """
  __metaclass__ = abc.ABCMeta  # required for @abstractmethod

  def __init__(self, component, init_layers=None, init_context_layers=None):
    """Initializes parameters for embedding matrices.

    The subclass may provide optional lists of initial layers and context layers
    to allow this base class constructor to use accessors like get_layer_size(),
    which is required for networks that may be used self-recurrently.

    Args:
      component: parent ComponentBuilderBase object.
      init_layers: optional initial layers.
      init_context_layers: optional initial context layers.
    """
    self._component = component
    self._params = []
    self._layers = init_layers if init_layers else []
    self._regularized_weights = []
    self._context_layers = init_context_layers if init_context_layers else []
    self._fixed_feature_dims = {}  # mapping from name to dimension
    self._linked_feature_dims = {}  # mapping from name to dimension

    # Allocate parameters for all embedding channels. Note that for both Fixed
    # and Linked embedding matrices, we store an additional +1 embedding that's
    # used when the index is out of scope.
    for channel_id, spec in enumerate(component.spec.fixed_feature):
      check.NotIn(spec.name, self._fixed_feature_dims,
                  'Duplicate fixed feature')
      check.Gt(spec.size, 0, 'Invalid fixed feature size')
      if spec.embedding_dim > 0:
        fixed_dim = spec.embedding_dim
        self._params.append(add_embeddings(channel_id, spec))
      else:
        fixed_dim = 1  # assume feature ID extraction; only one ID per step
      self._fixed_feature_dims[spec.name] = spec.size * fixed_dim

    for channel_id, spec in enumerate(component.spec.linked_feature):
      check.NotIn(spec.name, self._linked_feature_dims,
                  'Duplicate linked feature')
      check.Gt(spec.size, 0, 'Invalid linked feature size')
      if spec.source_component == component.name:
        source_array_dim = self.get_layer_size(spec.source_layer)
      else:
        source = component.master.lookup_component[spec.source_component]
        source_array_dim = source.network.get_layer_size(spec.source_layer)

      if spec.embedding_dim != -1:
        check.Gt(source_array_dim, 0,
                 'Cannot embed linked feature with dynamic dimension')
        self._params.append(
            tf.get_variable(
                linked_embeddings_name(channel_id),
                [source_array_dim + 1, spec.embedding_dim],
                initializer=tf.random_normal_initializer(
                    stddev=1 / spec.embedding_dim**.5)))

        self._linked_feature_dims[spec.name] = spec.size * spec.embedding_dim
      else:
        # If embedding_dim is -1, linked features are not embedded.
        self._linked_feature_dims[spec.name] = spec.size * source_array_dim

    # Compute the cumulative dimension of all inputs.  If any input has dynamic
    # dimension, then the result is -1.
    input_dims = (self._fixed_feature_dims.values() +
                  self._linked_feature_dims.values())
    if any(x < 0 for x in input_dims):
      self._concatenated_input_dim = -1
    else:
      self._concatenated_input_dim = sum(input_dims)
    tf.logging.info('component %s concat_input_dim %s', component.name,
                    self._concatenated_input_dim)

    # Allocate attention parameters.
    if self._component.spec.attention_component:
      attention_source_component = self._component.master.lookup_component[
          self._component.spec.attention_component]
      attention_hidden_layer_sizes = map(
          int, attention_source_component.spec.network_unit.parameters[
              'hidden_layer_sizes'].split(','))
      attention_hidden_layer_size = attention_hidden_layer_sizes[-1]

      hidden_layer_sizes = map(int, component.spec.network_unit.parameters[
          'hidden_layer_sizes'].split(','))
      # The attention function is built on the last layer of hidden embeddings.
      hidden_layer_size = hidden_layer_sizes[-1]
      self._params.append(
          tf.get_variable(
              'attention_weights_pm_0',
              [attention_hidden_layer_size, hidden_layer_size],
              initializer=tf.random_normal_initializer(stddev=1e-4)))

      self._params.append(
          tf.get_variable(
              'attention_weights_hm_0', [hidden_layer_size, hidden_layer_size],
              initializer=tf.random_normal_initializer(stddev=1e-4)))

      self._params.append(
          tf.get_variable(
              'attention_bias_0', [1, hidden_layer_size],
              initializer=tf.zeros_initializer()))

      self._params.append(
          tf.get_variable(
              'attention_bias_1', [1, hidden_layer_size],
              initializer=tf.zeros_initializer()))

      self._params.append(
          tf.get_variable(
              'attention_weights_pu',
              [attention_hidden_layer_size, component.num_actions],
              initializer=tf.random_normal_initializer(stddev=1e-4)))

  @abc.abstractmethod
  def create(self,
             fixed_embeddings,
             linked_embeddings,
             context_tensor_arrays,
             attention_tensor,
             during_training,
             stride=None):
    """Constructs a feed-forward unit based on the features and context tensors.

    Args:
      fixed_embeddings: list of NamedTensor objects
      linked_embeddings: list of NamedTensor objects
      context_tensor_arrays: optional list of TensorArray objects used for
          implicit recurrence.
      attention_tensor: optional Tensor used for attention.
      during_training: whether to create a network for training (vs inference).
      stride: int scalar tensor containing the stride required for
          bulk computation.

    Returns:
      A list of tensors corresponding to the list of layers.
    """
    pass

  @property
  def layers(self):
    return self._layers

  @property
  def params(self):
    return self._params

  @property
  def regularized_weights(self):
    return self._regularized_weights

  @property
  def context_layers(self):
    return self._context_layers

  def get_layer_index(self, layer_name):
    """Gets the index of the given named layer of the network."""
    return [x.name for x in self.layers].index(layer_name)

  def get_layer_size(self, layer_name):
    """Gets the size of the given named layer of the network.

    Args:
      layer_name: string name of layer to look update

    Returns:
      the size of the layer.

    Raises:
      KeyError: if the layer_name to look up doesn't exist.
    """
    for layer in self.layers:
      if layer.name == layer_name:
        return layer.dim
    raise KeyError('Layer {} not found in component {}'.format(
        layer_name, self._component.name))

  def get_logits(self, network_tensors):
    """Pulls out the logits from the tensors produced by this unit.

    Args:
      network_tensors: list of tensors as output by create().

    Raises:
      NotImplementedError: by default a 'logits' tensor need not be implemented.
    """
    raise NotImplementedError()

  def get_l2_regularized_weights(self):
    """Gets the weights that need to be regularized."""
    return self.regularized_weights

  def attention(self, last_layer, attention_tensor):
    """Compute the attention term for the network unit."""
    h_tensor = attention_tensor

    # Compute the attentions.
    # Using feed-forward net to map the two inputs into the same dimension
    focus_tensor = tf.nn.tanh(
        tf.matmul(
            h_tensor,
            self._component.get_variable('attention_weights_pm_0'),
            name='h_x_pm') + self._component.get_variable('attention_bias_0'))

    context_tensor = tf.nn.tanh(
        tf.matmul(
            last_layer,
            self._component.get_variable('attention_weights_hm_0'),
            name='l_x_hm') + self._component.get_variable('attention_bias_1'))
    # The tf.multiply in the following expression broadcasts along the 0 dim:
    z_vec = tf.reduce_sum(tf.multiply(focus_tensor, context_tensor), 1)
    p_vec = tf.nn.softmax(tf.reshape(z_vec, [1, -1]))
    # The tf.multiply in the following expression broadcasts along the 1 dim:
    r_vec = tf.expand_dims(
        tf.reduce_sum(
            tf.multiply(
                h_tensor, tf.reshape(p_vec, [-1, 1]), name='time_together2'),
            0),
        0)
    return tf.matmul(
        r_vec,
        self._component.get_variable('attention_weights_pu'),
        name='time_together3')


class IdentityNetwork(NetworkUnitInterface):
  """A network that returns concatenated input embeddings and activations."""

  def __init__(self, component):
    super(IdentityNetwork, self).__init__(component)
    self._layers = [
        Layer(
            component,
            name='input_embeddings',
            dim=self._concatenated_input_dim)
    ]

  def create(self,
             fixed_embeddings,
             linked_embeddings,
             context_tensor_arrays,
             attention_tensor,
             during_training,
             stride=None):
    return [get_input_tensor(fixed_embeddings, linked_embeddings)]

  def get_layer_size(self, layer_name):
    # Note that get_layer_size is called by super.__init__ before any layers are
    # constructed if and only if there are recurrent links.
    assert hasattr(self,
                   '_layers'), 'IdentityNetwork cannot have recurrent links'
    return super(IdentityNetwork, self).get_layer_size(layer_name)

  def get_logits(self, network_tensors):
    return network_tensors[-1]

  def get_context_layers(self):
    return []


class FeedForwardNetwork(NetworkUnitInterface):
  """Implementation of C&M style feedforward network.

  Supports dropout and optional layer normalization.

  Layers:
    layer_<i>: Activations for i'th hidden layer (0-origin).
    last_layer: Activations for the last hidden layer.  This is a convenience
        alias for "layer_<n-1>", where n is the number of hidden layers.
    logits: Logits associated with component actions.
  """

  def __init__(self, component):
    """Initializes parameters required to run this network.

    Args:
      component: parent ComponentBuilderBase object.

    Parameters used to construct the network:
      hidden_layer_sizes: comma-separated list of ints, indicating the
        number of hidden units in each hidden layer.
      layer_norm_input (False): Whether or not to apply layer normalization
        on the concatenated input to the network.
      layer_norm_hidden (False): Whether or not to apply layer normalization
        to the first set of hidden layer activations.
      nonlinearity ('relu'): Name of function from module "tf.nn" to apply to
        each hidden layer; e.g., "relu" or "elu".
      dropout_keep_prob (-1.0): The probability that an input is not dropped.
        If >= 1.0, disables dropout.  If < 0.0, uses the global |dropout_rate|
        hyperparameter.
      dropout_per_sequence (False): If true, sample the dropout mask once per
        sequence, instead of once per step.  See Gal and Ghahramani
        (https://arxiv.org/abs/1512.05287).
      dropout_all_layers (False): If true, apply dropout to the input of all
        hidden layers, instead of just applying it to the network input.

    Hyperparameters used:
      dropout_rate: The probability that an input is not dropped.  Only used
          when the |dropout_keep_prob| parameter is negative.
    """
    self._attrs = get_attrs_with_defaults(
        component.spec.network_unit.parameters, defaults={
            'hidden_layer_sizes': '',
            'layer_norm_input': False,
            'layer_norm_hidden': False,
            'nonlinearity': 'relu',
            'dropout_keep_prob': -1.0,
            'dropout_per_sequence': False,
            'dropout_all_layers': False})

    # Initialize the hidden layer sizes before running the base initializer, as
    # the base initializer may need to know the size of of the hidden layer for
    # recurrent connections.
    self._hidden_layer_sizes = (
        map(int, self._attrs['hidden_layer_sizes'].split(','))
        if self._attrs['hidden_layer_sizes'] else [])
    super(FeedForwardNetwork, self).__init__(component)

    # Infer dropout rate from network parameters and grid hyperparameters.
    self._dropout_rate = self._attrs['dropout_keep_prob']
    if self._dropout_rate < 0.0:
      self._dropout_rate = component.master.hyperparams.dropout_rate

    # Add layer norm if specified.
    self._layer_norm_input = None
    self._layer_norm_hidden = None
    if self._attrs['layer_norm_input']:
      self._layer_norm_input = LayerNorm(self._component, 'concat_input',
                                         self._concatenated_input_dim,
                                         tf.float32)
      self._params.extend(self._layer_norm_input.params)

    if self._attrs['layer_norm_hidden']:
      self._layer_norm_hidden = LayerNorm(self._component, 'layer_0',
                                          self._hidden_layer_sizes[0],
                                          tf.float32)
      self._params.extend(self._layer_norm_hidden.params)

    # Extract nonlinearity from |tf.nn|.
    self._nonlinearity = getattr(tf.nn, self._attrs['nonlinearity'])

    # TODO(googleuser): add initializer stddevs as part of the network unit's
    # configuration.
    self._weights = []
    last_layer_dim = self._concatenated_input_dim

    # Initialize variables for the parameters, and add Layer objects for
    # cross-component bookkeeping.
    for index, hidden_layer_size in enumerate(self._hidden_layer_sizes):
      weights = tf.get_variable(
          'weights_%d' % index, [last_layer_dim, hidden_layer_size],
          initializer=tf.random_normal_initializer(stddev=1e-4))
      self._params.append(weights)
      if index > 0 or self._layer_norm_hidden is None:
        self._params.append(
            tf.get_variable(
                'bias_%d' % index, [hidden_layer_size],
                initializer=tf.constant_initializer(
                    0.2, dtype=tf.float32)))

      self._weights.append(weights)
      self._layers.append(
          Layer(
              component, name='layer_%d' % index, dim=hidden_layer_size))
      last_layer_dim = hidden_layer_size

    # Add a convenience alias for the last hidden layer, if any.
    if self._hidden_layer_sizes:
      self._layers.append(Layer(component, 'last_layer', last_layer_dim))

    # By default, regularize only the weights.
    self._regularized_weights.extend(self._weights)

    if component.num_actions:
      self._params.append(
          tf.get_variable(
              'weights_softmax', [last_layer_dim, component.num_actions],
              initializer=tf.random_normal_initializer(stddev=1e-4)))
      self._params.append(
          tf.get_variable(
              'bias_softmax', [component.num_actions],
              initializer=tf.zeros_initializer()))
      self._layers.append(
          Layer(
              component, name='logits', dim=component.num_actions))

  def create(self,
             fixed_embeddings,
             linked_embeddings,
             context_tensor_arrays,
             attention_tensor,
             during_training,
             stride=None):
    """See base class."""
    input_tensor = get_input_tensor(fixed_embeddings, linked_embeddings)

    if during_training:
      input_tensor.set_shape([None, self._concatenated_input_dim])
      input_tensor = self._maybe_apply_dropout(input_tensor, stride)

    if self._layer_norm_input:
      input_tensor = self._layer_norm_input.normalize(input_tensor)

    tensors = []
    last_layer = input_tensor
    for index, hidden_layer_size in enumerate(self._hidden_layer_sizes):
      acts = tf.matmul(last_layer,
                       self._component.get_variable('weights_%d' % index))

      # Note that the first layer was already handled before this loop.
      # TODO(googleuser): Refactor this loop so dropout and layer normalization
      # are applied consistently.
      if during_training and self._attrs['dropout_all_layers'] and index > 0:
        acts.set_shape([None, hidden_layer_size])
        acts = self._maybe_apply_dropout(acts, stride)

      # Don't add a bias term if we're going to apply layer norm, since layer
      # norm includes a bias already.
      if index == 0 and self._layer_norm_hidden:
        acts = self._layer_norm_hidden.normalize(acts)
      else:
        acts = tf.nn.bias_add(acts,
                              self._component.get_variable('bias_%d' % index))

      last_layer = self._nonlinearity(acts)
      tensors.append(last_layer)

    # Add a convenience alias for the last hidden layer, if any.
    if self._hidden_layer_sizes:
      tensors.append(last_layer)

    if self._layers[-1].name == 'logits':
      logits = tf.matmul(
          last_layer, self._component.get_variable(
              'weights_softmax')) + self._component.get_variable('bias_softmax')

      if self._component.spec.attention_component:
        logits += self.attention(last_layer, attention_tensor)

      logits = tf.identity(logits, name=self._layers[-1].name)
      tensors.append(logits)
    return tensors

  def get_layer_size(self, layer_name):
    if layer_name == 'logits':
      return self._component.num_actions

    if layer_name == 'last_layer':
      return self._hidden_layer_sizes[-1]

    if not layer_name.startswith('layer_'):
      logging.fatal(
          'Invalid layer name: "%s" Can only retrieve from "logits", '
          '"last_layer", and "layer_*".',
          layer_name)

    # NOTE(danielandor): Since get_layer_size is called before the
    # model has been built, we compute the layer size directly from
    # the hyperparameters rather than from self._layers.
    layer_index = int(layer_name.split('_')[1])
    return self._hidden_layer_sizes[layer_index]

  def get_logits(self, network_tensors):
    return network_tensors[-1]

  def _maybe_apply_dropout(self, inputs, stride):
    return maybe_apply_dropout(inputs, self._dropout_rate,
                               self._attrs['dropout_per_sequence'], stride)


class LSTMNetwork(NetworkUnitInterface):
  """Implementation of action LSTM style network."""

  def __init__(self, component):
    assert component.num_actions > 0, 'Component num actions must be positive.'
    network_unit_spec = component.spec.network_unit
    self._hidden_layer_sizes = (
        int)(network_unit_spec.parameters['hidden_layer_sizes'])

    self._input_dropout_rate = component.master.hyperparams.dropout_rate
    self._recurrent_dropout_rate = (
        component.master.hyperparams.recurrent_dropout_rate)
    if self._recurrent_dropout_rate < 0.0:
      self._recurrent_dropout_rate = component.master.hyperparams.dropout_rate

    super(LSTMNetwork, self).__init__(component)
    layer_input_dim = self._concatenated_input_dim

    self._context_layers = []

    # TODO(googleuser): should we choose different initilizer,
    # e.g. truncated_normal_initializer?
    self._x2i = tf.get_variable(
        'x2i', [layer_input_dim, self._hidden_layer_sizes],
        initializer=tf.random_normal_initializer(stddev=1e-4))
    self._h2i = tf.get_variable(
        'h2i', [self._hidden_layer_sizes, self._hidden_layer_sizes],
        initializer=tf.random_normal_initializer(stddev=1e-4))
    self._c2i = tf.get_variable(
        'c2i', [self._hidden_layer_sizes, self._hidden_layer_sizes],
        initializer=tf.random_normal_initializer(stddev=1e-4))
    self._bi = tf.get_variable(
        'bi', [self._hidden_layer_sizes],
        initializer=tf.random_normal_initializer(stddev=1e-4))

    self._x2o = tf.get_variable(
        'x2o', [layer_input_dim, self._hidden_layer_sizes],
        initializer=tf.random_normal_initializer(stddev=1e-4))
    self._h2o = tf.get_variable(
        'h2o', [self._hidden_layer_sizes, self._hidden_layer_sizes],
        initializer=tf.random_normal_initializer(stddev=1e-4))
    self._c2o = tf.get_variable(
        'c2o', [self._hidden_layer_sizes, self._hidden_layer_sizes],
        initializer=tf.random_normal_initializer(stddev=1e-4))
    self._bo = tf.get_variable(
        'bo', [self._hidden_layer_sizes],
        initializer=tf.random_normal_initializer(stddev=1e-4))

    self._x2c = tf.get_variable(
        'x2c', [layer_input_dim, self._hidden_layer_sizes],
        initializer=tf.random_normal_initializer(stddev=1e-4))
    self._h2c = tf.get_variable(
        'h2c', [self._hidden_layer_sizes, self._hidden_layer_sizes],
        initializer=tf.random_normal_initializer(stddev=1e-4))
    self._bc = tf.get_variable(
        'bc', [self._hidden_layer_sizes],
        initializer=tf.random_normal_initializer(stddev=1e-4))

    self._params.extend([
        self._x2i, self._h2i, self._c2i, self._bi, self._x2o, self._h2o,
        self._c2o, self._bo, self._x2c, self._h2c, self._bc])

    lstm_h_layer = Layer(component, name='lstm_h', dim=self._hidden_layer_sizes)
    lstm_c_layer = Layer(component, name='lstm_c', dim=self._hidden_layer_sizes)

    self._context_layers.append(lstm_h_layer)
    self._context_layers.append(lstm_c_layer)

    self._layers.extend(self._context_layers)

    self._layers.append(
        Layer(
            component, name='layer_0', dim=self._hidden_layer_sizes))

    self.params.append(tf.get_variable(
        'weights_softmax', [self._hidden_layer_sizes, component.num_actions],
        initializer=tf.random_normal_initializer(stddev=1e-4)))
    self.params.append(
        tf.get_variable(
            'bias_softmax', [component.num_actions],
            initializer=tf.zeros_initializer()))

    self._layers.append(
        Layer(
            component, name='logits', dim=component.num_actions))

  def create(self,
             fixed_embeddings,
             linked_embeddings,
             context_tensor_arrays,
             attention_tensor,
             during_training,
             stride=None):
    """See base class."""
    input_tensor = get_input_tensor(fixed_embeddings, linked_embeddings)

    # context_tensor_arrays[0] is lstm_h
    # context_tensor_arrays[1] is lstm_c
    assert len(context_tensor_arrays) == 2
    length = context_tensor_arrays[0].size()

    # Get the (possibly averaged) parameters to execute the network.
    x2i = self._component.get_variable('x2i')
    h2i = self._component.get_variable('h2i')
    c2i = self._component.get_variable('c2i')
    bi = self._component.get_variable('bi')
    x2o = self._component.get_variable('x2o')
    h2o = self._component.get_variable('h2o')
    c2o = self._component.get_variable('c2o')
    bo = self._component.get_variable('bo')
    x2c = self._component.get_variable('x2c')
    h2c = self._component.get_variable('h2c')
    bc = self._component.get_variable('bc')

    # i_h_tm1, i_c_tm1 = h_{t-1}, c_{t-1}
    i_h_tm1 = context_tensor_arrays[0].read(length - 1)
    i_c_tm1 = context_tensor_arrays[1].read(length - 1)

    # apply dropout according to http://arxiv.org/pdf/1409.2329v5.pdf
    if during_training and self._input_dropout_rate < 1:
      input_tensor = tf.nn.dropout(input_tensor, self._input_dropout_rate)

    # input --  i_t = sigmoid(affine(x_t, h_{t-1}, c_{t-1}))
    i_ait = tf.matmul(input_tensor, x2i) + tf.matmul(i_h_tm1, h2i) + tf.matmul(
        i_c_tm1, c2i) + bi
    i_it = tf.sigmoid(i_ait)

    # forget -- f_t = 1 - i_t
    i_ft = tf.ones([1, 1]) - i_it

    # write memory cell -- tanh(affine(x_t, h_{t-1}))
    i_awt = tf.matmul(input_tensor, x2c) + tf.matmul(i_h_tm1, h2c) + bc
    i_wt = tf.tanh(i_awt)

    # c_t = f_t \odot c_{t-1} + i_t \odot tanh(affine(x_t, h_{t-1}))
    ct = tf.add(
        tf.multiply(i_it, i_wt), tf.multiply(i_ft, i_c_tm1), name='lstm_c')

    # output -- o_t = sigmoid(affine(x_t, h_{t-1}, c_t))
    i_aot = tf.matmul(input_tensor, x2o) + tf.matmul(ct, c2o) + tf.matmul(
        i_h_tm1, h2o) + bo

    i_ot = tf.sigmoid(i_aot)

    # ht = o_t \odot tanh(ct)
    ph_t = tf.tanh(ct)
    ht = tf.multiply(i_ot, ph_t, name='lstm_h')

    if during_training and self._recurrent_dropout_rate < 1:
      ht = tf.nn.dropout(
          ht, self._recurrent_dropout_rate, name='lstm_h_dropout')

    h = tf.identity(ht, name='layer_0')

    logits = tf.nn.xw_plus_b(ht, tf.get_variable('weights_softmax'),
                             tf.get_variable('bias_softmax'))

    if self._component.spec.attention_component:
      logits += self.attention(ht, attention_tensor)

    logits = tf.identity(logits, name='logits')
    # tensors will be consistent with the layers:
    # [lstm_h, lstm_c, layer_0, logits]
    tensors = [ht, ct, h, logits]
    return tensors

  def get_layer_size(self, layer_name):
    assert layer_name == 'layer_0', 'Can only retrieve from first hidden layer.'
    return self._hidden_layer_sizes

  def get_logits(self, network_tensors):
    return network_tensors[self.get_layer_index('logits')]


class ConvNetwork(NetworkUnitInterface):
  """Implementation of a convolutional feed forward network."""

  def __init__(self, component):
    """Initializes kernels and biases for this convolutional net.

    Args:
      component: parent ComponentBuilderBase object.

    Parameters used to construct the network:
      widths: comma separated list of ints, number of steps input to the
              convolutional kernel at every layer.
      depths: comma separated list of ints, number of channels input to the
              convolutional kernel at every layer.
      output_embedding_dim: int, number of output channels for the convolutional
              kernel of the last layer, which receives no ReLU activation and
              therefore can be used in a softmax output. If zero, this final
              layer is disabled entirely.
      nonlinearity ('relu'): Name of function from module "tf.nn" to apply to
        each hidden layer; e.g., "relu" or "elu".
      dropout_keep_prob (-1.0): The probability that an input is not dropped.
        If >= 1.0, disables dropout.  If < 0.0, uses the global |dropout_rate|
        hyperparameter.
      dropout_per_sequence (False): If true, sample the dropout mask once per
        sequence, instead of once per step.  See Gal and Ghahramani
        (https://arxiv.org/abs/1512.05287).

    Hyperparameters used:
      dropout_rate: The probability that an input is not dropped.  Only used
          when the |dropout_keep_prob| parameter is negative.
    """

    super(ConvNetwork, self).__init__(component)
    self._attrs = get_attrs_with_defaults(
        component.spec.network_unit.parameters, defaults={
            'widths': '',
            'depths': '',
            'output_embedding_dim': 0,
            'nonlinearity': 'relu',
            'dropout_keep_prob': -1.0,
            'dropout_per_sequence': False})

    self._weights = []
    self._biases = []
    self._widths = map(int, self._attrs['widths'].split(','))
    self._depths = map(int, self._attrs['depths'].split(','))
    self._output_dim = self._attrs['output_embedding_dim']
    if self._output_dim:
      self._depths.append(self._output_dim)
    self.kernel_shapes = []
    for i in range(len(self._depths) - 1):
      self.kernel_shapes.append(
          [1, self._widths[i], self._depths[i], self._depths[i + 1]])
    for i in range(len(self._depths) - 1):
      with tf.variable_scope('conv%d' % i):
        self._weights.append(
            tf.get_variable(
                'weights',
                self.kernel_shapes[i],
                initializer=tf.random_normal_initializer(stddev=1e-4),
                dtype=tf.float32))
        bias_init = 0.0 if (i == len(self._widths) - 1) else 0.2
        self._biases.append(
            tf.get_variable(
                'biases',
                self.kernel_shapes[i][-1],
                initializer=tf.constant_initializer(bias_init),
                dtype=tf.float32))

    # Extract nonlinearity from |tf.nn|.
    self._nonlinearity = getattr(tf.nn, self._attrs['nonlinearity'])

    # Infer dropout rate from network parameters and grid hyperparameters.
    self._dropout_rate = self._attrs['dropout_keep_prob']
    if self._dropout_rate < 0.0:
      self._dropout_rate = component.master.hyperparams.dropout_rate

    self._params.extend(self._weights + self._biases)
    self._layers.append(
        Layer(
            component, name='conv_output', dim=self._depths[-1]))
    self._regularized_weights.extend(self._weights[:-1] if self._output_dim else
                                     self._weights)

  def create(self,
             fixed_embeddings,
             linked_embeddings,
             context_tensor_arrays,
             attention_tensor,
             during_training,
             stride=None):
    """Requires |stride|; otherwise see base class."""
    if stride is None:
      raise RuntimeError("ConvNetwork needs 'stride' and must be called in the "
                         "bulk feature extractor component.")
    input_tensor = get_input_tensor_with_stride(fixed_embeddings,
                                                linked_embeddings, stride)

    # TODO(googleuser): Add context and attention.
    del context_tensor_arrays, attention_tensor

    # On CPU, add a dimension so that the 'image' has shape
    # [stride, 1, num_steps, D].
    conv = tf.expand_dims(input_tensor, 1)
    for i in range(len(self._depths) - 1):
      with tf.variable_scope('conv%d' % i, reuse=True) as scope:
        if during_training:
          conv.set_shape([None, 1, None, self._depths[i]])
          conv = self._maybe_apply_dropout(conv, stride)
        conv = tf.nn.conv2d(
            conv,
            self._component.get_variable('weights'), [1, 1, 1, 1],
            padding='SAME')
        conv = tf.nn.bias_add(conv, self._component.get_variable('biases'))
        if i < (len(self._weights) - 1) or not self._output_dim:
          conv = self._nonlinearity(conv, name=scope.name)
    return [
        tf.reshape(
            conv, [-1, self._depths[-1]], name='reshape_activations')
    ]

  def _maybe_apply_dropout(self, inputs, stride):
    # The |inputs| are rank 4 (one 1xN "image" per sequence).  Squeeze out and
    # restore the singleton image height, so dropout is applied to the normal
    # rank 3 batched input tensor.
    inputs = tf.squeeze(inputs, [1])
    inputs = maybe_apply_dropout(inputs, self._dropout_rate,
                                 self._attrs['dropout_per_sequence'], stride)
    inputs = tf.expand_dims(inputs, 1)
    return inputs


class PairwiseConvNetwork(NetworkUnitInterface):
  """Implementation of a pairwise 2D convolutional feed forward network.

  For a sequence of N tokens, all N^2 pairs of concatenated input features are
  constructed. If each input vector is of length D, then the sequence is
  represented by an image of dimensions [N, N] with 2*D channels per pixel.
  I.e. pixel [i, j] has a representation that is the concatenation of the
  representations of the tokens at i and at j.

  To use this network for graph edge scoring, for instance by using the "heads"
  transition system, the output layer needs to have dimensions [N, N] and only
  a single channel. The network takes care of outputting an [N, N] sized layer,
  but the user needs to ensure that the output depth equals 1.

  TODO(googleuser): Like Dozat and Manning, we will need an
  additional network to label the edges, and the ability to read head
  and modifier representations from different inputs.
  """

  def __init__(self, component):
    """Initializes kernels and biases for this convolutional net.

    Parameters used to construct the network:
      depths: comma separated list of ints, number of channels input to the
          convolutional kernel at every layer.
      widths: comma separated list of ints, number of steps input to the
          convolutional kernel at every layer.
      relu_layers: comma separate list of ints, the id of layers after which
          to apply a relu activation. *By default, all but the final layer will
          have a relu activation applied.*

    To generate a network with M layers, both 'depths' and 'widths' must be of
    length M. The input depth of the first layer is inferred from the total
    concatenated size of the input features.

    Args:
      component: parent ComponentBuilderBase object.

    Raises:
      RuntimeError: if the number of depths and weights are not equal.
      ValueError: if the final depth is not equal to 1.
    """
    parameters = component.spec.network_unit.parameters
    super(PairwiseConvNetwork, self).__init__(component)

    # Each input pixel will comprise the concatenation of two tokens, so the
    # input depth is double that for a single token.
    self._depths = [self._concatenated_input_dim * 2]
    self._depths.extend(map(int, parameters['depths'].split(',')))
    self._widths = map(int, parameters['widths'].split(','))
    self._num_layers = len(self._widths)
    if len(self._depths) != self._num_layers + 1:
      raise RuntimeError('Unmatched depths/weights %s/%s' %
                         (parameters['depths'], parameters['weights']))
    if self._depths[-1] != 1:
      raise ValueError('Final depth is not equal to 1 in %s' %
                       parameters['depths'])

    self._kernel_shapes = []
    for i, width in enumerate(self._widths):
      self._kernel_shapes.append(
          [width, width, self._depths[i], self._depths[i + 1]])
    if parameters['relu_layers']:
      self._relu_layers = set(map(int, parameters['relu_layers'].split(',')))
    else:
      self._relu_layers = set(range(self._num_layers - 1))

    self._weights = []
    self._biases = []
    for i, kernel_shape in enumerate(self._kernel_shapes):
      with tf.variable_scope('conv%d' % i):
        self._weights.append(
            tf.get_variable(
                'weights',
                kernel_shape,
                initializer=tf.random_normal_initializer(stddev=1e-4),
                dtype=tf.float32))
        bias_init = 0.0 if i in self._relu_layers else 0.2
        self._biases.append(
            tf.get_variable(
                'biases',
                kernel_shape[-1],
                initializer=tf.constant_initializer(bias_init),
                dtype=tf.float32))

    self._params.extend(self._weights + self._biases)
    self._layers.append(Layer(component, name='conv_output', dim=-1))
    self._regularized_weights.extend(self._weights[:-1])

  def create(self,
             fixed_embeddings,
             linked_embeddings,
             context_tensor_arrays,
             attention_tensor,
             during_training,
             stride=None):
    """Requires |stride|; otherwise see base class."""
    # TODO(googleuser): Normalize the arguments to create(). 'stride'
    # is unused by the recurrent network units, while 'context_tensor_arrays'
    # and 'attenion_tensor_array' is unused by bulk network units. b/33587044
    if stride is None:
      raise ValueError("PairwiseConvNetwork needs 'stride'")

    input_tensor = get_input_tensor_with_stride(fixed_embeddings,
                                                linked_embeddings, stride)

    # TODO(googleuser): Add dropout.
    del context_tensor_arrays, attention_tensor, during_training  # Unused.

    num_steps = tf.shape(input_tensor)[1]
    arg1 = tf.expand_dims(input_tensor, 1)
    arg1 = tf.tile(arg1, tf.stack([1, num_steps, 1, 1]))
    arg2 = tf.expand_dims(input_tensor, 2)
    arg2 = tf.tile(arg2, tf.stack([1, 1, num_steps, 1]))
    conv = tf.concat([arg1, arg2], 3)
    for i in xrange(self._num_layers):
      with tf.variable_scope('conv%d' % i, reuse=True) as scope:
        conv = tf.nn.conv2d(
            conv,
            self._component.get_variable('weights'), [1, 1, 1, 1],
            padding='SAME')
        conv = tf.nn.bias_add(conv, self._component.get_variable('biases'))
        if i in self._relu_layers:
          conv = tf.nn.relu(conv, name=scope.name)
    return [tf.reshape(conv, [-1, num_steps], name='reshape_activations')]


class ExportFixedFeaturesNetwork(NetworkUnitInterface):
  """A network that exports fixed features as layers.

  Each fixed feature embedding is output as a layer whose name and dimension are
  set to the name and dimension of the corresponding fixed feature.
  """

  def __init__(self, component):
    """Initializes exported layers."""
    super(ExportFixedFeaturesNetwork, self).__init__(component)
    for feature_spec in component.spec.fixed_feature:
      name = feature_spec.name
      dim = self._fixed_feature_dims[name]
      self._layers.append(Layer(component, name, dim))

  def create(self,
             fixed_embeddings,
             linked_embeddings,
             context_tensor_arrays,
             attention_tensor,
             during_training,
             stride=None):
    """See base class."""
    check.Eq(len(self.layers), len(fixed_embeddings))
    for index in range(len(fixed_embeddings)):
      check.Eq(self.layers[index].name, fixed_embeddings[index].name)
    return [fixed_embedding.tensor for fixed_embedding in fixed_embeddings]


class SplitNetwork(NetworkUnitInterface):
  """Network unit that splits its input into slices of equal dimension.

  Parameters:
    num_slices: The number of slices to split the input into, S.  The input must
                have static dimension D, where D % S == 0.

  Features:
    All inputs are concatenated before being split.

  Layers:
    slice_0: [B * N, D / S] The first slice of the input.
    slice_1: [B * N, D / S] The second slice of the input.
    ...
  """

  def __init__(self, component):
    """Initializes weights and layers.

    Args:
      component: Parent ComponentBuilderBase object.
    """
    super(SplitNetwork, self).__init__(component)

    parameters = component.spec.network_unit.parameters
    self._num_slices = int(parameters['num_slices'])
    check.Gt(self._num_slices, 0, 'Invalid number of slices.')
    check.Eq(self._concatenated_input_dim % self._num_slices, 0,
             'Input dimension %s does not evenly divide into %s slices' %
             (self._concatenated_input_dim, self._num_slices))
    self._slice_dim = int(self._concatenated_input_dim / self._num_slices)

    for slice_index in xrange(self._num_slices):
      self._layers.append(
          Layer(self, 'slice_%s' % slice_index, self._slice_dim))

  def create(self,
             fixed_embeddings,
             linked_embeddings,
             context_tensor_arrays,
             attention_tensor,
             during_training,
             stride=None):
    input_bnxd = get_input_tensor(fixed_embeddings, linked_embeddings)
    return tf.split(input_bnxd, self._num_slices, axis=1)
