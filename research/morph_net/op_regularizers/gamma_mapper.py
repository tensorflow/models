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
"""Classes for mapping convolutions to their batch-norm gammas."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import tensorflow as tf

from morph_net.framework import op_regularizer_manager


class GenericConvGammaMapper(object):
  """An interface for mapping convolutions to their batch-norm gammas."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def get_gamma(self, conv_op):
    """Returns the BatchNorm gamma tensor associated with `conv_op`, or None.

    Args:
      conv_op: A tf.Operation of type Conv2D.

    Returns:
      A tf.Tensor containing the BatchNorm gamma associated with `conv_op`, or
      None if `conv_op` has no BatchNorm gamma.

    Raises:
      ValueError: `conv_op` is not a tf.Operation of type `Conv2D`.
      KeyError: `conv_op` is not in the graph that was used to construct `self`
    """

  @abc.abstractproperty
  def all_conv_ops(self):
    """Return all Conv2D ops that were in the graph when `self` was created."""
    pass


def _get_existing_variable(name):
  """Fetches a variable by name (like tf.get_variable with reuse=True).

  The reason why we can't simply use tf.get_variable with reuse=True is that
  when variable partitioner is used, tf.get_variable requires knowing the shape
  of the variable (even though it knows it and thus shouldn't require it). This
  helper is a convenience function to solve this problem.

  Args:
    name: A string, the name of the variable.

  Returns:
    A tf.Tensor which is the result of convert_to_tensor of the variable, or
    None if the variable does not exist.
  """
  try:
    op = tf.get_default_graph().get_operation_by_name(name)
  except KeyError:
    return None

  # Among all cases (partitioned variable, resource variable, or regular one),
  # we assume that there's either a shape attribute to the op or to its output.
  try:
    shape = tf.TensorShape(op.get_attr('shape'))
  except ValueError:
    shape = op.outputs[0].shape

  with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    try:
      # tf.Variable and tf.PartitionedVariable are not polymorphic, but
      # both support convert_to_tensor. The result is thus always a
      # tf.Tensor.
      return tf.convert_to_tensor(tf.get_variable(name, shape=shape))
    except ValueError as e:
      if 'Variable %s does not exist' % name in str(e):
        return None
      else:
        raise e  # pass through any other exceptions.


class ConvGammaMapperByName(GenericConvGammaMapper):
  """Maps a convolution to its BatchNorm gamma.

  Assumes that the convolutions and their respective gammas conform to the
  naming convention of tf.contrib.layers: A convolution's name ends with
  `<BASE_NAME>/Conv2D`, and the respective batch-norm gamma ends with
  `<BASE_NAME>/BatchNorm/gamma`
  """

  def __init__(self):
    """Constructs an instance. Builds mapping from Conv2D ops to their Gamma."""
    self._conv_to_gamma = {}
    # We use get_variable under a reuse=True scope because this is a way to
    # capture both a regular tf.Variable and a PartitionedVariable.
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      for op in tf.get_default_graph().get_operations():
        if op.type != 'Conv2D' and op.type != 'DepthwiseConv2dNative':
          continue
        base_name = op.name.rsplit('/', 1)[0]
        self._conv_to_gamma[op] = _get_existing_variable(base_name +
                                                         '/BatchNorm/gamma')

  def get_gamma(self, conv_op):
    _raise_if_not_conv(conv_op)
    return self._conv_to_gamma[conv_op]

  @property
  def all_conv_ops(self):
    return self._conv_to_gamma.keys()


class ConvGammaMapperByConnectivity(GenericConvGammaMapper):
  """Maps a convolution to its BatchNorm gammas based on graph connectivity.

  Given a batch-norm gamma, propagates along the graph to find the convolutions
  that are batch-nomalized by this gamma. It can me more than one convolution
  that are normalized by the same batch-norm gamma in ResNet-s, where
  un-normalized convolutions are first summed and then their sum is normalized.
  The converse is also true - a single convolution can be connected (through
  residual connections) to multiple batch-norms.

  Only fused batch-norm is supported: there seems to be significant variability
  in the way non-fused batch-norm manifests in the tensorflow graph.
  """

  def __init__(self):
    """Constructs an instance. Builds mapping from Conv2D ops to their Gamma."""
    self._conv_to_gamma = collections.defaultdict(set)
    for op in tf.get_default_graph().get_operations():
      if op.type != 'FusedBatchNorm':
        continue

      convs = _dfs(op)
      for conv in convs:
        if conv.type == 'Conv2D':
          self._conv_to_gamma[conv].add(op.inputs[1])  # Input #1 is gamma.

    for op in tf.get_default_graph().get_operations():
      if op.type == 'Conv2D' and op not in self._conv_to_gamma:
        self._conv_to_gamma[op] = None

  def get_gamma(self, conv_op):
    _raise_if_not_conv(conv_op)
    if conv_op not in self._conv_to_gamma:
      raise KeyError
    gammas = self._conv_to_gamma[conv_op]
    if gammas and len(gammas) == 1:
      # For a single element, return the element itself, to conform with
      # ConvGammaMapperByName.
      return list(gammas)[0]
    return gammas

  @property
  def all_conv_ops(self):
    return self._conv_to_gamma.keys()


def _dfs(op, visited=None):
  """Perform DFS on a graph.

  Args:
    op: A tf.Operation, the root node for the DFS.
    visited: A set, used in the recursion.

  Returns:
    A list of the tf.Operations of type Conv2D that were encountered.
  """
  visited = visited or set()
  ret = []
  for child in op.inputs:
    if child.op in visited:
      return ret
    visited.add(child.op)
    if child.op.type not in op_regularizer_manager.NON_PASS_THROUGH_OPS:
      ret.extend(_dfs(child.op, visited))
    if child.op.type in ('Conv2D',):  # TODO: support depthwise conv.
      ret.append(child.op)
  return ret


def _raise_if_not_conv(op):
  if not isinstance(op, tf.Operation):
    raise ValueError('conv_op must be a tf.Operation, not %s' % type(op))
  if op.type != 'Conv2D' and op.type != 'DepthwiseConv2dNative':
    raise ValueError('conv_op must be a Conv2D or DepthwiseConv2dNative,'
                     'not %s' % op.type)
