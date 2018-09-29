# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
"""A class for managing OpRegularizers.

OpRegularizerManager creates the required regulrizers and manages the
association between ops and their regularizers. OpRegularizerManager handles the
logic associated with the graph topology:
- Concatenating tensors is reflected in concatenating their regularizers.
- Skip-connections (aka residual connections), RNNs and other structures where
  the shapes of two (or more) tensors are tied together are reflected in
  grouping their regularizers together.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging
import tensorflow as tf

from morph_net.framework import concat_and_slice_regularizers
from morph_net.framework import generic_regularizers
from morph_net.framework import grouping_regularizers


# When an op has two (or more) inputs, that haveregularizers, the latter need to
# be grouped. _GROUPING_OPS is a whitelist of ops that are allowed to group, as
# a form of verification of the correctness of the code. The list is not
# exhaustive, feel free to add other grouping ops as needed.
_GROUPING_OPS = ('Add', 'Sub', 'Mul', 'Div', 'Maximum', 'Minimum',
                 'SquaredDifference', 'RealDiv')  # TODO: Is Div needed?
# Ops that are not pass-through, that necessarily modify the regularizer.
# These are the Ops that should not have an regularizer that is identifical to
# one of its input. When we recursively look for regularizers along the graph
# the recursion will always stop at these Ops even if no regularizer factory
# is provided, and never assume that they pass the regularizer of their input
# through.
NON_PASS_THROUGH_OPS = ('Conv2D', 'Conv2DBackpropInput', 'MatMul')


def _remove_nones_and_dups(items):
  result = []
  for i in items:
    if i is not None and i not in result:
      result.append(i)
  return result


def _raise_type_error_if_not_operation(op):
  if not isinstance(op, tf.Operation):
    raise TypeError('\'op\' must be of type tf.Operation, not %s' %
                    str(type(op)))


class OpRegularizerManager(object):
  """A class for managing OpRegularizers."""

  # Public methods -------------------------------------------------------------
  def __init__(self, ops, op_regularizer_factory_dict,
               create_grouping_regularizer=None):
    """Creates an instance.

    Args:
      ops: A list of tf.Operation-s. An OpRegularizer will be created for all
        the ops in `ops`, and recursively for all ops they depend on via data
        dependency. Typically `ops` would contain a single tf.Operation, which
        is the output of the network.
      op_regularizer_factory_dict: A dictionary, where the keys are strings
        representing TensorFlow Op types, and the values are callables that
        create the respective OpRegularizers. For every op encountered during
        the recursion, if op.type is in op_regularizer_factory_dict, the
        respective callable will be used to create an OpRegularizer. The
        signature of the callables is the following args;
          op; a tf.Operation for which to create a regularizer.
          opreg_manager; A reference to an OpRegularizerManager object. Can be
            None if the callable does not need access to OpRegularizerManager.
      create_grouping_regularizer: A callable that has the signature of
        grouping_regularizers.MaxGroupingRegularizer's constructor. Will be
        called whenever a grouping op (see _GROUPING_OPS) is encountered.
        Defaults to MaxGroupingRegularizer if None.

    Raises:
      ValueError: If ops is not a list.
    """
    self._constructed = False
    if not isinstance(ops, list):
      raise ValueError(
          'Input %s ops is not a list. Should probably use []' % str(ops))
    self._op_to_regularizer = {}
    self._regularizer_to_ops = collections.defaultdict(list)
    self._op_regularizer_factory_dict = op_regularizer_factory_dict
    for op_type in NON_PASS_THROUGH_OPS:
      if op_type not in self._op_regularizer_factory_dict:
        self._op_regularizer_factory_dict[op_type] = lambda x, y: None
    self._create_grouping_regularizer = (
        create_grouping_regularizer or
        grouping_regularizers.MaxGroupingRegularizer)
    self._visited = set()
    for op in ops:
      self._get_regularizer(op)
    self._constructed = True

  def get_regularizer(self, op):
    """Looks up or creates an OpRegularizer for a tf.Operation.

    Args:
      op: A tf.Operation.

    - If `self` has an OpRegularizer for `op`, it will be returned.
      Otherwise:
    - If called before construction of `self` was completed (that is, from the
      constructor), an attempt to create an OpRegularizer for `op` will be made.
      Otherwise:
    - If called after contstruction of `self` was completed, an exception will
      be raised.

    Returns:
      An OpRegularizer for `op`. Can be None if `op` is not regularized (e.g.
      `op` is a constant).

    Raises:
      RuntimeError: If `self` object has no OpRegularizer for `op` in its
        lookup table, and the construction of `self` has already been completed
        (because them `self` is immutable and an OpRegularizer cannot be
        created).
    """
    try:
      return self._op_to_regularizer[op]
    except KeyError:
      if self._constructed:
        raise ValueError('Op %s does not have a regularizer.' % op.name)
      else:
        return self._get_regularizer(op)

  @property
  def ops(self):
    return self._op_to_regularizer.keys()

  # ---- Public MUTABLE methods ------------------------------------------------
  #
  # These methods are intended to be called by OpRegularizer factory functions,
  # in the constructor of OpRegularizerManager. OpRegularizerManager is
  # immutable after construction, so calling these methods after construction
  # has been completed will raise an exception.

  def group_and_replace_regularizers(self, regularizers):
    """Groups a list of OpRegularizers and replaces them by the grouped one.

    Args:
      regularizers: A list of OpRegularizer objects to be grouped.

    Returns:
      An OpRegularizer object formed by the grouping.

    Raises:
      RuntimeError: group_and_replace_regularizers was called affter
         construction of the OpRegularizerManager object was completed.
    """
    if self._constructed:
      raise RuntimeError('group_and_replace_regularizers can only be called '
                         'before construction of the OpRegularizerManager was '
                         'completed.')
    grouped = self._create_grouping_regularizer(regularizers)
    # Replace all the references to the regularizers by the new grouped
    # regularizer.
    for r in regularizers:
      self._replace_regularizer(r, grouped)
    return grouped

  # Private methods ------------------------------------------------------------
  def _get_regularizer(self, op):
    """Fetches the regularizer of `op` if exists, creates it otherwise.

    This function calls itself recursively, directly or via _create_regularizer
    (which in turn calls _get_regularizer). It performs DFS along the data
    dependencies of the graph, and uses a self._visited set to detect loops. The
    use of self._visited makes it not thread safe, but _get_regularizer is a
    private method that is supposed to only be called form the constructor, so
    execution in multiple threads (for the same object) is not expected.

    Args:
      op: A Tf.Operation.

    Returns:
      An OpRegularizer that corresponds to `op`, or None if op does not have
      a regularizer (e. g. it's a constant op).
    """
    _raise_type_error_if_not_operation(op)
    if op not in self._op_to_regularizer:
      if op in self._visited:
        # In while loops, the data dependencies form a loop.
        # TODO: RNNs have "legit" loops - will this still work?
        return None
      self._visited.add(op)
      regularizer = self._create_regularizer(op)
      self._op_to_regularizer[op] = regularizer
      self._regularizer_to_ops[regularizer].append(op)
      # Make sure that there is a regularizer (or None) for every op on which
      # `op` depends via data dependency.
      for i in op.inputs:
        self._get_regularizer(i.op)
      self._visited.remove(op)
    return self._op_to_regularizer[op]

  def _create_regularizer(self, op):
    """Creates an OpRegularizer for `op`.

    Args:
      op: A Tf.Operation.

    Returns:
      An OpRegularizer that corresponds to `op`, or None if op does not have
      a regularizer.

    Raises:
      RuntimeError: Grouping is attempted at op which is not whitelisted for
        grouping (in _GROUPING_OPS).
    """
    # First we see if there is a factory function for creating the regularizer
    # in the op_regularizer_factory_dict (supplied in the constructor).
    if op.type in self._op_regularizer_factory_dict:
      regularizer = self._op_regularizer_factory_dict[op.type](op, self)
      if regularizer is None:
        logging.warning('Failed to create regularizer for %s.', op.name)
      else:
        logging.info('Created regularizer for %s.', op.name)
      return regularizer
    # Unless overridden in op_regularizer_factory_dict, we assume that ops
    # without inputs have no regularizers. These are 'leaf' ops, typically
    # constants and variables.
    if not op.inputs:
      return None
    if op.type == 'ConcatV2':
      return self._create_concat_regularizer(op)

    inputs_regularizers = _remove_nones_and_dups(
        [self._get_regularizer(i.op) for i in op.inputs])

    # Ops whose inputs have no regularizers, and that are not in
    # op_regularizer_factory_dict, have no regularizer either (think of ops that
    # only involve constants as an example).
    if not inputs_regularizers:
      return None

    # Ops that have one input with a regularizer, and are not in
    # op_regularizer_factory_dict, are assumed to be pass-through, that is, to
    # carry over the regularizer of their inputs. Examples:
    # - Unary ops, such as as RELU.
    # - BiasAdd, or similar ops, that involve a constant/variable and a
    #   regularized op (e.g. the convolution that comes before the bias).
    elif len(inputs_regularizers) == 1:
      return inputs_regularizers[0]

    # Group if we have more than one regularizer in the inputs of `op` and if it
    # is white-listed for grouping.
    elif op.type in _GROUPING_OPS:
      return self.group_and_replace_regularizers(inputs_regularizers)

    raise RuntimeError('Grouping is attempted at op which is not whitelisted '
                       'for grouping: %s' % str(op.type))

  def _create_concat_regularizer(self, concat_op):
    """Creates an OpRegularizer for a concat op.

    Args:
      concat_op: A tf.Operation of type ConcatV2.

    Returns:
      An OpRegularizer for `concat_op`.
    """
    # We omit the last input, because it's the concat dimension. Others are
    # the tensors to be concatenated.
    input_ops = [i.op for i in concat_op.inputs[:-1]]
    regularizers_to_concat = [self._get_regularizer(op) for op in input_ops]
    # If all inputs have no regularizer, so does the concat op.
    if regularizers_to_concat == [None] * len(regularizers_to_concat):
      return None
    offset = 0

    # Replace the regularizers_to_concat by SlicingReferenceRegularizer-s that
    # slice the concatenated regularizer.
    ops_to_concat = []
    for r, op in zip(regularizers_to_concat, input_ops):
      if r is None:
        length = op.outputs[0].shape.as_list()[-1]
        offset += length
        ops_to_concat.append(self._ConstantOpReg(length))
      else:
        length = tf.shape(r.alive_vector)[0]
        slice_ref = concat_and_slice_regularizers.SlicingReferenceRegularizer(
            lambda: self._get_regularizer(concat_op), offset, length)
        offset += length
        self._replace_regularizer(r, slice_ref)
        ops_to_concat.append(r)

    # Create the concatenated regularizer itself.
    return concat_and_slice_regularizers.ConcatRegularizer(ops_to_concat)

  def _replace_regularizer(self, source, target):
    """Replaces `source` by 'target' in self's lookup tables."""
    for op in self._regularizer_to_ops[source]:
      assert self._op_to_regularizer[op] is source
      self._op_to_regularizer[op] = target
      self._regularizer_to_ops[target].append(op)
    del self._regularizer_to_ops[source]

  class _ConstantOpReg(generic_regularizers.OpRegularizer):
    """A class with the constant alive property, and zero regularization."""

    def __init__(self, size):
      self._regularization_vector = tf.zeros(size)
      self._alive_vector = tf.cast(tf.ones(size), tf.bool)

    @property
    def regularization_vector(self):
      return self._regularization_vector

    @property
    def alive_vector(self):
      return self._alive_vector
