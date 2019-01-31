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
"""An OpRegularizer that applies L1 regularization on batch-norm gammas."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from morph_net.framework import generic_regularizers
from morph_net.op_regularizers import gamma_mapper


class GammaL1Regularizer(generic_regularizers.OpRegularizer):
  """An OpRegularizer that L1-regularizes batch-norm gamma."""

  def __init__(self, gamma, gamma_threshold):
    """Creates an instance.

    Args:
      gamma: a tf.Tensor of rank 1 with the gammas.
      gamma_threshold: A float scalar, the threshold above which a gamma is
        considered 'alive'.
    """
    self._gamma = gamma
    self._gamma_threshold = gamma_threshold
    abs_gamma = tf.abs(gamma)
    self._alive_vector = abs_gamma > gamma_threshold
    self._regularization_vector = abs_gamma

  @property
  def regularization_vector(self):
    return self._regularization_vector

  @property
  def alive_vector(self):
    return self._alive_vector


class GammaL1RegularizerFactory(object):
  """A class for creating a GammaL1Regularizer for convolutions."""

  def __init__(self, gamma_threshold):
    """Creates an instance.

    Args:
      gamma_threshold: A float scalar, will be used as a 'gamma_threshold' for
        all the GammaL1Regularizer-s created by this class.
    """
    self._gamma_conv_mapper = gamma_mapper.ConvGammaMapperByName()
    self._gamma_threshold = gamma_threshold

  def create_regularizer(self, op, opreg_manager):
    """Creates a GammaL1Regularizer for `op`.

    Args:
      op: A tf.Operation of type 'Conv2D' or 'DepthwiseConv2dNative'.
      opreg_manager: An OpRegularizerManager object that will host the created
        OpRegularizer object.

    Returns:
      a GammaL1Regularizer that corresponds to `op`.

    Raises:
      ValueError: If `op` does not have a Gamma that corresponds to it.
    """
    gamma = self._gamma_conv_mapper.get_gamma(op)
    if gamma is None:
      regularizer = None
    else:
      regularizer = GammaL1Regularizer(gamma, self._gamma_threshold)

    if op.type == 'DepthwiseConv2dNative':
      regularizer = _group_depthwise_conv_regularizer(op, regularizer,
                                                      opreg_manager)
    return regularizer


def _group_depthwise_conv_regularizer(op, regularizer, opreg_manager):
  """Groups the regularizer of a depthwise convolution if needed."""
  # If its first input doesn't have regularizers, return None. While pruning
  # the depthwise_conv still effectively shut down input channels, fluid_net
  # currently has not implemented a way to interpret it. In particular, the
  # union of active channels of dw's input and output will always return all
  # channels.
  # TODO: update the interpretation to discover channels that are
  # effectively pruned by dw.
  input_reg = opreg_manager.get_regularizer(op.inputs[0].op)
  if input_reg is None:
    return None
  # Check for cases where the depthwise convolution has a multiplier that
  # is not one. Do not regularize this case.
  # TODO: add support for depthwise with multiplier != 1.
  if (op.inputs[0].shape.as_list()[-1] !=
      op.outputs[0].shape.as_list()[-1]):
    return None
  # If the op is not regularized, return the regularizer of its input.
  # This applies to cases in separable convolutions where the pointwise
  # is batchnormed but the depthwise is not.
  if regularizer is None:
    return input_reg
  # If both the input and depthwise have regularizers, we group them.
  # This applies to Mobilenets where both 1x1 and depthwise have
  # batchnorms.
  else:
    return opreg_manager.group_and_replace_regularizers(
        [regularizer, input_reg])
