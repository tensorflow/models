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
"""Utils for supporting the DRAGNN runtime from the TF side."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import re

import tensorflow as tf

from dragnn.python import network_units
from syntaxnet.util import check


def add_hooks(component, cell_subgraph_spec):
  """Adds "hook" nodes to the graph, for use by the runtime.

  The runtime hook nodes are not on the path to any required output, and will
  not be called when running TF-based DRAGNN.  As long as the TF graph is not
  pruned, however, the DRAGNN runtime can call them.

  Runtime hook nodes can perform any TF computation.  Possible uses include:
    * Applying stable names to existing tensors (e.g., via tf.identity()).
    * Converting variable data from a TF-friendly or training-friendly format
      into a runtime-friendly format.

  NB: There are several restrictions on the context in which this function is
  called.  In brief, call ComponentBuilderBase._add_runtime_hooks() at the top
  of each ComponentBuilderSubclass.build_*() method.  In detail, this:
    * Must be called in the variable scope of the |component|, so variable
      references in component.get_variable() work.
    * Must be called, possibly transitively, from one of the |component|'s
      build_*() methods, so MasterBuilder.read_from_avg is set properly for
      component.get_variable().
    * Must not be called from within a tf.while_loop(), or the hook nodes will
      not work.  In particular, NetworkUnitInterface.create() is called from a
      tf.while_loop() in DynamicComponentBuilder.

  Args:
    component: Component for which to add hooks.
    cell_subgraph_spec: CellSubgraphSpec for which to add hooks.
  """
  for channel_id, feature_spec in enumerate(component.spec.linked_feature):
    if feature_spec.embedding_dim != -1:
      _add_hooks_for_linked_embedding_matrix(component, channel_id)

  for channel_id, feature_spec in enumerate(component.spec.fixed_feature):
    if feature_spec.embedding_dim != -1:
      _add_hooks_for_fixed_embedding_matrix(component, channel_id)

  for params in component.network.params:
    _add_hooks_for_trainable_params(component, params)

  for parameter_getter in component.network.derived_params:
    _add_hooks_for_derived_parameter(parameter_getter)

  _add_hook_node(
      tf.constant(cell_subgraph_spec.SerializeToString(), tf.string),
      '{}/EXPORT/CellSubgraphSpec'.format(component.name))


def _blocked_and_dtype_transformations(tensor):
  """Yields variants of a tensor, for standard blocking/dtype variants.

  Args:
    tensor (tf.Tensor): Input tensor.

  Yields:
    (modified_tensor, suffix) pairs, where `modified_tensor` is a transformed
    version of the input, and `suffix` is a string like "/blocked32".
  """
  for blocking_level in (32, 48):
    blocked = make_padded_blocked_matrix(tensor, blocking_level)
    bfloat16_blocked = tf.to_bfloat16(bfloat16_permutation(blocked))
    yield blocked, '/blocked{}'.format(blocking_level)
    yield bfloat16_blocked, '/blocked{}/bfloat16'.format(blocking_level)


def _add_hooks_for_linked_embedding_matrix(component, channel_id):
  """Adds runtime hooks for a linked embedding matrix.

  The computation performed by network_units.pass_through_embedding_matrix() is
  equivalent to the following:

    for i in range(stride):
      if step_idx[i] == -1:
        outputs[i,:] = out_of_bounds_vector
      else:
        outputs[i,:] = tf.matmul(act_block[i,:], weight_matrix)

  The implementation uses clever arithmetic to do this in one matmul per batch.
  Specifically, the weight_matrix is extended with the out_of_bounds_vector and
  each activation vector is extended with a 0/1 out-of-bounds indicator.  Then,
  multiplying the two suffices, assuming that act_block[i,:] is set to zero for
  out-of-bounds links.

  While this works well for training and high-throughput batched computation, it
  isn't the best for the runtime:
    * Appending a 0/1 indicator to the input activation vector requires a copy.
      Ideally, we could use the input activation vector by reference alone.
    * In order to access to the |out_of_bounds_vector| as a contiguous array,
      the runtime must load the linked embedding matrix in row-major format,
      which may not be the fastest format for arithmetic.
    * The dimensions of the extended-by-1 matrix and vector are likely to be
      pessimal.  Most dimensions are specified as 2^n, and adding one element
      produces maximal padding on the trailing elements, which in turn wastes
      memory, reduces cache utilization, etc.

  Therefore, in the runtime we split the linked embedding matrix into a separate
  weight matrix and out-of-bounds vector.

  Args:
    component: Component for which to add hooks.
    channel_id: Linked embedding channel for which to add hooks.
  """
  var_name = network_units.linked_embeddings_name(channel_id)
  extended_matrix = component.get_variable(var_name)
  extended_num_rows = tf.shape(extended_matrix)[0]
  matrix, vector = tf.split(extended_matrix, [extended_num_rows - 1, 1], 0)
  transposed = tf.transpose(matrix)

  hook_name = functools.partial(_get_hook_name, component, var_name)

  _add_hook_node(matrix, hook_name('/weights'))
  _add_hook_node(transposed, hook_name('/weights/transposed'))

  # Add blocked versions of the matrix and its transpose.
  for blocked, blocked_suffix in _blocked_and_dtype_transformations(matrix):
    blocked_name = hook_name('/weights/matrix' + blocked_suffix)
    _add_hook_node(blocked, blocked_name)
  for blocked, blocked_suffix in _blocked_and_dtype_transformations(transposed):
    blocked_name = hook_name('/weights/transposed' + blocked_suffix)
    _add_hook_node(blocked, blocked_name)

  # Add shape and out-of-bounds information.
  _add_hook_node(tf.shape(transposed), hook_name('/weights/transposed/shape'))
  _add_hook_node(vector, _get_hook_name(component, var_name, '/out_of_bounds'))


def _add_hooks_for_fixed_embedding_matrix(component, channel_id):
  """Adds runtime hooks for a fixed embedding matrix.

  The hooks remove the last row from the embedding matrix.  The extra row was
  probably intended for out-of-vocabulary items, but those are handled in the
  feature system and the extra row is never used.

  Args:
    component: Component for which to add hooks.
    channel_id: Fixed embedding channel for which to add hooks.
  """
  var_name = network_units.fixed_embeddings_name(channel_id)
  extended_matrix = component.get_variable(var_name)
  extended_num_rows = tf.shape(extended_matrix)[0]
  matrix = tf.slice(extended_matrix, [0, 0], [extended_num_rows - 1, -1])

  # TODO(googleuser): If the extra row is removed from the variable itself, remove
  # the tf.slice() and point the hook directly at the variable.
  _add_hook_node(matrix, _get_hook_name(component, var_name, '/trimmed'))


def _add_hooks_for_derived_parameter(getter):
  """Adds hooks for derived parameters.

  Derived parameters are typically slight format modifications of regular
  parameters, exposed because doing the computation in Python is more convenient
  than as VariableStore wrappers.

  Args:
    getter: Function which, when called, will return the derived tensor.
  """
  parameter = getter()
  full_name = parameter.op.name

  def _hook_name(base_name):
    """Returns a hook node name constructed from a base name."""
    return full_name + base_name

  if parameter.shape.ndims != 2:
    tf.logging.info('Not adding matrix hooks for derived parameter %s',
                    full_name)
    return

  _add_hook_node(tf.transpose(parameter), _hook_name('/transposed'))
  for blocked, blocked_suffix in _blocked_and_dtype_transformations(parameter):
    _add_hook_node(blocked, _hook_name('/matrix' + blocked_suffix))


def _add_hooks_for_trainable_params(component, params):
  """Adds runtime hooks for a variable of trainable parameters.

  Ignores parameters that are not statically-deducible as matrices.

  Args:
    component: Component for which to add hooks.
    params: Variable for which to add hooks.
  """
  full_name = params.op.name
  matrix = component.get_variable(var_params=params)

  # Only add hooks for tensors that are statically-deducible as matrices.
  if params.shape.ndims != 2:
    tf.logging.info('Not adding hooks for trainable params %s', full_name)
    return

  # Infer the suffix to append to variable names, if any, based on whether the
  # possibly-averaged |matrix| is named differently than the |params|.
  suffix = re.sub('^' + re.escape(full_name), '', matrix.op.name)
  check.Ne(suffix, matrix.op.name,
           'Failed to find suffix for params %s' % full_name)

  def _hook_name(base_name):
    """Returns a hook node name constructed from a base name."""
    return full_name + base_name + suffix

  # Add the matrix and its transpose.
  transposed = tf.transpose(matrix)
  _add_hook_node(matrix, _hook_name('/matrix'))
  _add_hook_node(transposed, _hook_name('/transposed'))

  # Add blocked versions of the matrix and its transpose.
  for blocked, blocked_suffix in _blocked_and_dtype_transformations(matrix):
    _add_hook_node(blocked, _hook_name('/matrix' + blocked_suffix))
  for blocked, blocked_suffix in _blocked_and_dtype_transformations(transposed):
    _add_hook_node(blocked, _hook_name('/transposed' + blocked_suffix))

  # Also add hooks for the original shapes, which are obscured by padding.
  _add_hook_node(tf.shape(matrix), _hook_name('/matrix/shape'))
  _add_hook_node(tf.shape(transposed), _hook_name('/transposed/shape'))


def make_padded_blocked_matrix(matrix, block_size):
  """Converts a matrix to padded column-blocked format.

  For example, given a [64,127] matrix and block_size=16, this function returns
  an [8,64,16] tensor where the 8 inner sub-matrices, when concatenated left to
  right, re-constitute the original matrix.  Note that the 8th sub-matrix has a
  final column of padding.

  Args:
    matrix: The matrix to convert.
    block_size: The number of columns per block.

  Returns:
    Padded column-blocked matrix.
  """
  shape = tf.shape(matrix)
  num_rows = shape[0]
  num_columns = shape[1]

  # Compute the amount of padding and resulting number of blocks.
  last_block_size = num_columns % block_size
  padding_size = (block_size - last_block_size) % block_size
  num_blocks = (num_columns + padding_size) // block_size

  # Somehow the obvious approach based on tf.split() and tf.stack() doesn't work
  # (seems that the number of splits needs to be statically-known), but this
  # alternative based on tf.transpose() and tf.reshape() does.  Continuing the
  # example from the docstring...
  padded = tf.pad(matrix, [[0, 0], [0, padding_size]])  # [64,127] => [64,128]
  transposed = tf.transpose(padded)  # => [128,64]
  blocked = tf.reshape(transposed, [num_blocks, block_size,
                                    num_rows])  # => [8,16,64]
  return tf.transpose(blocked, [0, 2, 1])  # => [8,64,16]


def bfloat16_permutation(tensor):
  """Permutes values in the last dimension of a tensor.

  This permutation is used so that we can directly use unpacklo/unpackhi AVX2
  instructions on the matrix coefficients. These unpacking instructions
  effectively permute the data. See FastUnpackPermutation() and
  AvxFloatVecArray::Load(const TruncatedFloat16 *) in avx_vector_array.h for
  more details.

  Args:
    tensor: Blocked matrix, the result of make_padded_blocked_matrix(). Must
      have its last dimension a multiple of 16.

  Returns:
    Permuted matrix, suitable for calling tf.to_bfloat16() on. For testing
    convenience we don't do so in this method.

  Raises:
    ValueError: If the matrix's block dimension is not a multiple of 16.
  """
  orig_shape = tensor.shape
  if tensor.shape[-1] % 16 != 0:
    raise ValueError('Bad block dimension, must be divisible by 16')
  permutation = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]
  indices = tf.constant(
      [16 * (i // 16) + permutation[i % 16] for i in xrange(orig_shape[-1])])
  return tf.gather(tensor, indices, axis=len(orig_shape) - 1)


def _get_hook_name(component, variable_name, suffix):
  """Builds the name of a hook node.

  Specifically, the name of the hook node is:

    <component.name>/<variable_name><suffix><remainder>

  where <remainder> is whatever follows <variable_name> in the name of the op
  that produces the named variable.  Recall that component.get_variable() may
  return either the original variable or its moving average.  These might have
  names like:

    foo_component/bar_variable
    foo_component/bar_variable/ExponentialMovingAverage

  In the examples above, the <remainder> is "" for the original variable and
  "/ExponentialMovingAverage" for its moving average.  Calling this function
  with suffix="/baz_suffix" in either case would add hook nodes named:

    foo_component/bar_variable/baz_suffix
    foo_component/bar_variable/baz_suffix/ExponentialMovingAverage

  Note that the suffix is inserted after the variable name, not necessarily at
  the end of the entire op name.

  Args:
    component: Component that the hook node belongs to.
    variable_name: Variable that the hook node name is based on.
    suffix: Suffix to append to the variable name.

  Returns:
    Name of the hook node.
  """
  variable = component.get_variable(variable_name)
  full_name = variable.op.name
  prefix = component.name + '/' + variable_name
  hook_name = re.sub('^' + re.escape(prefix), prefix + suffix, full_name)

  # If re.sub() did not match anything, it returns the unmodified input (i.e.,
  # |full_name|).  Enforce that some change was made.
  check.Ne(
      full_name, hook_name,
      'Failed to match expected variable prefix "{}" in variable "{}"'.format(
          prefix, full_name))

  return hook_name


def _add_hook_node(tensor, fully_qualified_name):
  """Adds a hook node that outputs a tensor with a fully-qualified name."""
  # Since the name is fully-qualified, insert the hook node into the top-level
  # name scope.
  with tf.name_scope(None):
    tf.identity(tensor, name=fully_qualified_name)
