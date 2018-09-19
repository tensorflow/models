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
"""Tests for compute_cost_estimator.

Note that BilinearNetworkRegularizer is not tested here - its specific
instantiation is tested in flop_regularizer_test.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops

from morph_net.network_regularizers import bilinear_cost_utils

layers = tf.contrib.layers


def _flops(op):
  """Get the number of flops of a convolution, from the ops stats registry.

  Args:
    op: A tf.Operation object.

  Returns:
    The number os flops needed to evaluate conv_op.
  """
  return (ops.get_stats_for_node_def(tf.get_default_graph(), op.node_def,
                                     'flops').value)


def _output_depth(conv_op):
  return conv_op.outputs[0].shape.as_list()[-1]


def _input_depth(conv_op):
  conv_weights = conv_op.inputs[1]
  return conv_weights.shape.as_list()[2]


class BilinearCostUtilTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()
    image = tf.constant(0.0, shape=[1, 11, 13, 17])
    net = layers.conv2d(
        image, 19, [7, 5], stride=2, padding='SAME', scope='conv1')
    layers.conv2d_transpose(
        image, 29, [7, 5], stride=2, padding='SAME', scope='convt2')
    net = tf.reduce_mean(net, axis=(1, 2))
    layers.fully_connected(net, 23, scope='FC')
    net = layers.conv2d(
        image, 10, [7, 5], stride=2, padding='SAME', scope='conv2')
    layers.separable_conv2d(
        net, None, [3, 2], depth_multiplier=1, padding='SAME', scope='dw1')
    self.conv_op = tf.get_default_graph().get_operation_by_name('conv1/Conv2D')
    self.convt_op = tf.get_default_graph().get_operation_by_name(
        'convt2/conv2d_transpose')
    self.matmul_op = tf.get_default_graph().get_operation_by_name(
        'FC/MatMul')
    self.dw_op = tf.get_default_graph().get_operation_by_name(
        'dw1/depthwise')

  def assertNearRelatively(self, expected, actual):
    self.assertNear(expected, actual, expected * 1e-6)

  def testConvFlopsCoeff(self):
    # Divide by the input depth and the output depth to get the coefficient.
    expected_coeff = _flops(self.conv_op) / (17.0 * 19.0)
    actual_coeff = bilinear_cost_utils.flop_coeff(self.conv_op)
    self.assertNearRelatively(expected_coeff, actual_coeff)

  def testConvTransposeFlopsCoeff(self):
    # Divide by the input depth and the output depth to get the coefficient.
    expected_coeff = _flops(self.convt_op) / (17.0 * 29.0)
    actual_coeff = bilinear_cost_utils.flop_coeff(self.convt_op)
    self.assertNearRelatively(expected_coeff, actual_coeff)

  def testFcFlopsCoeff(self):
    expected_coeff = _flops(self.matmul_op) / (19.0 * 23.0)
    actual_coeff = bilinear_cost_utils.flop_coeff(self.matmul_op)
    self.assertNearRelatively(expected_coeff, actual_coeff)

  def testConvNumWeightsCoeff(self):
    actual_coeff = bilinear_cost_utils.num_weights_coeff(self.conv_op)
    # The coefficient is just the filter size - 7 * 5 = 35:
    self.assertNearRelatively(35, actual_coeff)

  def testFcNumWeightsCoeff(self):
    actual_coeff = bilinear_cost_utils.num_weights_coeff(self.matmul_op)
    # The coefficient is 1.0, the number of weights is just inputs x outputs.
    self.assertNearRelatively(1.0, actual_coeff)

  def testDepthwiseConvFlopsCoeff(self):
    # Divide by the input depth (which is also the output depth) to get the
    # coefficient.
    expected_coeff = _flops(self.dw_op) / (10.0)
    actual_coeff = bilinear_cost_utils.flop_coeff(self.dw_op)
    self.assertNearRelatively(expected_coeff, actual_coeff)


if __name__ == '__main__':
  tf.test.main()
