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
"""Helpers for Network Regularizers that are bilinear in their inputs/outputs.

Examples: The number of FLOPs and the number weights of a convolution are both
a bilinear expression in the number of its inputs and outputs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from morph_net.framework import generic_regularizers


_CONV2D_OPS = ('Conv2D', 'Conv2DBackpropInput', 'DepthwiseConv2dNative')
_SUPPORTED_OPS = _CONV2D_OPS + ('MatMul',)


def _raise_if_not_supported(op):
  if not isinstance(op, tf.Operation):
    raise ValueError('conv_op must be a tf.Operation, not %s' % type(op))
  if op.type not in _SUPPORTED_OPS:
    raise ValueError('conv_op must be a Conv2D or a MatMul, not %s' % op.type)


def _get_conv_filter_size(conv_op):
  assert conv_op.type in _CONV2D_OPS
  conv_weights = conv_op.inputs[1]
  filter_shape = conv_weights.shape.as_list()[:2]
  return filter_shape[0] * filter_shape[1]


def flop_coeff(op):
  """Computes the coefficient of number of flops associated with a convolution.

  The FLOPs cost of a convolution is given by C * output_depth * input_depth,
  where C = 2 * output_width * output_height * filter_size. The 2 is because we
  have one multiplication and one addition for each convolution weight and
  pixel. This function returns C.

  Args:
    op: A tf.Operation of type 'Conv2D' or 'MatMul'.

  Returns:
    A float, the coefficient that when multiplied by the input depth and by the
    output depth gives the number of flops needed to compute the convolution.

  Raises:
    ValueError: conv_op is not a tf.Operation of type Conv2D.
  """
  _raise_if_not_supported(op)
  if op.type in _CONV2D_OPS:
    # Looking at the output shape makes it easy to automatically take into
    # account strides and the type of padding.
    if op.type == 'Conv2D' or op.type == 'DepthwiseConv2dNative':
      shape = op.outputs[0].shape.as_list()
    else:  # Conv2DBackpropInput
      # For a transposed convolution, the input and the output are swapped (as
      # far as shapes are concerned). In other words, for a given filter shape
      # and stride, if Conv2D maps from shapeX to shapeY, Conv2DBackpropInput
      # maps from shapeY to shapeX. Therefore wherever we use the output shape
      # for Conv2D, we use the input shape for Conv2DBackpropInput.
      shape = _get_input(op).shape.as_list()
    size = shape[1] * shape[2]
    return 2.0 * size * _get_conv_filter_size(op)
  else:  # MatMul
    # A MatMul is like a 1x1 conv with an output size of 1x1, so from the factor
    # above only the 2.0 remains.
    return 2.0


def num_weights_coeff(op):
  """The number of weights of a conv is C * output_depth * input_depth. Finds C.

  Args:
    op: A tf.Operation of type 'Conv2D' or 'MatMul'

  Returns:
    A float, the coefficient that when multiplied by the input depth and by the
    output depth gives the number of flops needed to compute the convolution.

  Raises:
    ValueError: conv_op is not a tf.Operation of type Conv2D.
  """
  _raise_if_not_supported(op)
  return _get_conv_filter_size(op) if op.type in _CONV2D_OPS else 1.0


class BilinearNetworkRegularizer(generic_regularizers.NetworkRegularizer):
  """A NetworkRegularizer with bilinear cost and loss.

  Can be used for FLOPs regularization or for model size regularization.
  """

  def __init__(self, opreg_manager, coeff_func):
    """Creates an instance.

    Args:
      opreg_manager: An OpRegularizerManager object that will be used to query
        OpRegularizers of the various ops in the graph.
      coeff_func: A callable that receives a tf.Operation of type Conv2D and
        returns a bilinear coefficient of its cost. Examples:
        - Use conv_flop_coeff for a FLOP regularizer.
        - Use conv_num_weights_coeff for a number-of-weights regularizer.
    """
    self._opreg_manager = opreg_manager
    self._coeff_func = coeff_func

  def _get_cost_or_regularization_term(self, is_regularization, ops=None):
    total = 0.0
    if not ops:
      ops = self._opreg_manager.ops
    for op in ops:
      if op.type not in _SUPPORTED_OPS:
        continue
      # We use the following expression for thr regularizer:
      #
      # coeff * (number_of_inputs_alive * sum_of_output_regularizers +
      #          number_of_outputs_alive * sum_of_input_regularizers)
      #
      # where 'coeff' is a coefficient (for a particular convolution) such that
      # the number of flops of that convolution is given by:
      # number_of_flops = coeff * number_of_inputs * number_of_outputs.
      input_op = _get_input(op).op
      input_op_reg = self._opreg_manager.get_regularizer(input_op)
      output_op_reg = self._opreg_manager.get_regularizer(op)
      coeff = self._coeff_func(op)
      num_alive_inputs = _count_alive(input_op, input_op_reg)
      num_alive_outputs = _count_alive(op, output_op_reg)
      if op.type == 'DepthwiseConv2dNative':
        if is_regularization:
          reg_inputs = _sum_of_reg_vector(input_op_reg)
          reg_outputs = _sum_of_reg_vector(output_op_reg)
          # reg_inputs and reg_outputs are often identical since they should
          # come from the same reguarlizer. Duplicate them for symmetry.
          # When the input doesn't have a regularizer (e.g. input), only the
          # second term is used.
          # TODO: revisit this expression after experiments.
          total += coeff * (reg_inputs + reg_outputs)
        else:
          # num_alive_inputs may not always equals num_alive_outputs because the
          # input (e.g. the image) may not have a gamma regularizer. In this
          # case the computation is porportional only to num_alive_outputs.
          total += coeff * num_alive_outputs
      else:
        if is_regularization:
          reg_inputs = _sum_of_reg_vector(input_op_reg)
          reg_outputs = _sum_of_reg_vector(output_op_reg)
          total += coeff * (
              num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
        else:
          total += coeff * num_alive_inputs * num_alive_outputs
    return total

  def get_cost(self, ops=None):
    return self._get_cost_or_regularization_term(False, ops)

  def get_regularization_term(self, ops=None):
    return self._get_cost_or_regularization_term(True, ops)


def _get_input(op):
  """Returns the input to that op that represents the activations.

  (as opposed to e.g. weights.)

  Args:
    op: A tf.Operation object with type in _SUPPORTED_OPS.

  Returns:
    A tf.Tensor representing the input activations.

  Raises:
    ValueError: MatMul is used with transposition (unsupported).
  """
  assert op.type in _SUPPORTED_OPS, 'Op type %s is not supported.' % op.type
  if op.type == 'Conv2D' or op.type == 'DepthwiseConv2dNative':
    return op.inputs[0]
  if op.type == 'Conv2DBackpropInput':
    return op.inputs[2]
  if op.type == 'MatMul':
    if op.get_attr('transpose_a') or op.get_attr('transpose_b'):
      raise ValueError('MatMul with transposition is not yet supported.')
    return op.inputs[0]


def _count_alive(op, opreg):
  if opreg:
    return tf.reduce_sum(tf.cast(opreg.alive_vector, tf.float32))
  else:
    return float(op.outputs[0].shape.as_list()[-1])


def _sum_of_reg_vector(opreg):
  if opreg:
    return tf.reduce_sum(opreg.regularization_vector)
  else:
    return 0.0
