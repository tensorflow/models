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
"""Tests for flop_regularizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np

import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v1
from morph_net.network_regularizers import bilinear_cost_utils
from morph_net.network_regularizers import flop_regularizer

arg_scope = tf.contrib.framework.arg_scope
layers = tf.contrib.layers
_coeff = bilinear_cost_utils.flop_coeff
NUM_CHANNELS = 3


class GammaFlopLossTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()
    self.BuildWithBatchNorm()
    with self.test_session():
      self.Init()

  def BuildWithBatchNorm(self):
    params = {
        'trainable': True,
        'normalizer_fn': layers.batch_norm,
        'normalizer_params': {
            'scale': True
        }
    }

    with arg_scope([layers.conv2d], **params):
      self.BuildModel()

  def BuildModel(self):
    # Our test model is:
    #
    #         -> conv1 --+     -> conv3 -->
    #        /           |    /
    #  image          [concat]
    #        \           |    \
    #         -> conv2 --+     -> conv4 -->
    #
    # (the model has two "outputs", conv3 and conv4).
    #
    image = tf.constant(0.0, shape=[1, 17, 19, NUM_CHANNELS])
    conv1 = layers.conv2d(image, 13, [7, 5], padding='SAME', scope='conv1')
    conv2 = layers.conv2d(image, 23, [1, 1], padding='SAME', scope='conv2')
    concat = tf.concat([conv1, conv2], 3)
    self.conv3 = layers.conv2d(
        concat, 29, [3, 3], stride=2, padding='SAME', scope='conv3')
    self.conv4 = layers.conv2d(
        concat, 31, [1, 1], stride=1, padding='SAME', scope='conv4')
    self.name_to_var = {v.op.name: v for v in tf.global_variables()}

    self.gamma_flop_reg = flop_regularizer.GammaFlopsRegularizer(
        [self.conv3.op, self.conv4.op], gamma_threshold=0.45)

  def GetConv(self, name):
    return tf.get_default_graph().get_operation_by_name(name + '/Conv2D')

  def Init(self):
    tf.global_variables_initializer().run()
    gamma1 = self.name_to_var['conv1/BatchNorm/gamma']
    gamma1.assign([0.8] * 7 + [0.2] * 6).eval()
    gamma2 = self.name_to_var['conv2/BatchNorm/gamma']
    gamma2.assign([-0.7] * 11 + [0.1] * 12).eval()
    gamma3 = self.name_to_var['conv3/BatchNorm/gamma']
    gamma3.assign([0.6] * 10 + [-0.3] * 19).eval()
    gamma4 = self.name_to_var['conv4/BatchNorm/gamma']
    gamma4.assign([-0.5] * 17 + [-0.4] * 14).eval()

  def cost(self, conv):
    with self.test_session():
      return self.gamma_flop_reg.get_cost(conv).eval()

  def loss(self, conv):
    with self.test_session():
      return self.gamma_flop_reg.get_regularization_term(conv).eval()

  def testCost(self):
    # Conv1 has 7 gammas above 0.45, and NUM_CHANNELS inputs (from the image).
    conv = self.GetConv('conv1')
    self.assertEqual(_coeff(conv) * 7 * NUM_CHANNELS, self.cost([conv]))

    # Conv2 has 11 gammas above 0.45, and NUM_CHANNELS inputs (from the image).
    conv = self.GetConv('conv2')
    self.assertEqual(_coeff(conv) * 11 * NUM_CHANNELS, self.cost([conv]))

    # Conv3 has 10 gammas above 0.45, and 7 + 11 inputs from conv1 and conv2.
    conv = self.GetConv('conv3')
    self.assertEqual(_coeff(conv) * 10 * 18, self.cost([conv]))

    # Conv4 has 17 gammas above 0.45, and 7 + 11 inputs from conv1 and conv2.
    conv = self.GetConv('conv4')
    self.assertEqual(_coeff(conv) * 17 * 18, self.cost([conv]))

    # Test that passing a list of convs sums their contributions:
    convs = [self.GetConv('conv3'), self.GetConv('conv4')]
    self.assertEqual(
        self.cost(convs[:1]) + self.cost(convs[1:]), self.cost(convs))


class GammaFlopLossWithDepthwiseConvTestBase(object):
  """Test flop_regularizer for a network with depthwise convolutions."""
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def GetSession(self):
    return

  def BuildWithBatchNorm(self):
    params = {
        'trainable': True,
        'normalizer_fn': layers.batch_norm,
        'normalizer_params': {
            'scale': True
        }
    }
    ops_with_batchnorm = [layers.conv2d]
    if self._depthwise_use_batchnorm:
      ops_with_batchnorm.append(layers.separable_conv2d)

    with arg_scope(ops_with_batchnorm, **params):
      self.BuildModel()

  def BuildModel(self):
    # Our test model is:
    #
    #         -> dw1 --> conv1 --+
    #        /                   |
    #  image                     [concat] --> conv3
    #        \                   |
    #         -> conv2 --> dw2 --+
    #
    # (the model has one "output", conv3).
    #
    image = tf.constant(0.0, shape=[1, 17, 19, NUM_CHANNELS])
    dw1 = layers.separable_conv2d(
        image, None, [3, 3], depth_multiplier=1, stride=1, scope='dw1')
    conv1 = layers.conv2d(dw1, 13, [7, 5], padding='SAME', scope='conv1')
    conv2 = layers.conv2d(image, 23, [1, 1], padding='SAME', scope='conv2')
    dw2 = layers.separable_conv2d(
        conv2, None, [5, 5], depth_multiplier=1, stride=1, scope='dw2')
    concat = tf.concat([conv1, dw2], 3)
    self.conv3 = layers.conv2d(
        concat, 29, [3, 3], stride=2, padding='SAME', scope='conv3')
    self.name_to_var = {v.op.name: v for v in tf.global_variables()}

    self.gamma_flop_reg = flop_regularizer.GammaFlopsRegularizer(
        [self.conv3.op], gamma_threshold=0.45)

  def GetConv(self, name):
    return tf.get_default_graph().get_operation_by_name(
        name + ('/Conv2D' if 'conv' in name else '/depthwise'))

  def GetGammaAbsValue(self, name):
    gamma_op = tf.get_default_graph().get_operation_by_name(name +
                                                            '/BatchNorm/gamma')
    with self.GetSession():  # pylint: disable=not-context-manager
      gamma = gamma_op.outputs[0].eval()
    return np.abs(gamma)

  def Init(self):
    tf.global_variables_initializer().run()
    gamma1 = self.name_to_var['conv1/BatchNorm/gamma']
    gamma1.assign([0.8] * 7 + [0.2] * 6).eval()
    gamma2 = self.name_to_var['conv2/BatchNorm/gamma']
    gamma2.assign([-0.7] * 11 + [0.1] * 12).eval()
    gamma3 = self.name_to_var['conv3/BatchNorm/gamma']
    gamma3.assign([0.6] * 10 + [-0.3] * 19).eval()
    # Initialize gamma for depthwise convs only if there are Batchnorm for them.
    if self._depthwise_use_batchnorm:
      gammad1 = self.name_to_var['dw1/BatchNorm/gamma']
      gammad1.assign([-0.3] * 1 + [-0.9] * 2).eval()
      gammad2 = self.name_to_var['dw2/BatchNorm/gamma']
      gammad2.assign([0.3] * 5 + [0.9] * 10 + [-0.1] * 8).eval()

  def cost(self, conv):  # pylint: disable=invalid-name
    with self.GetSession():  # pylint: disable=not-context-manager
      cost = self.gamma_flop_reg.get_cost(conv)
      return cost.eval() if isinstance(cost, tf.Tensor) else cost

  def loss(self, conv):  # pylint: disable=invalid-name
    with self.GetSession():  # pylint: disable=not-context-manager
      reg = self.gamma_flop_reg.get_regularization_term(conv)
      return reg.eval() if isinstance(reg, tf.Tensor) else reg


class GammaFlopLossWithDepthwiseConvTest(
    tf.test.TestCase, GammaFlopLossWithDepthwiseConvTestBase):
  """Test flop_regularizer for a network with depthwise convolutions."""

  def setUp(self):
    self._depthwise_use_batchnorm = True
    tf.reset_default_graph()
    self.BuildWithBatchNorm()
    with self.test_session():
      self.Init()

  def GetSession(self):
    return self.test_session()

  def testCost(self):
    # Dw1 has 2 gammas above 0.45 out of NUM_CHANNELS inputs (from the image),
    # but because the input doesn't have a regularizer, it has no way of
    # removing the channels, so the channel count is still NUM_CHANNELS.
    conv = self.GetConv('dw1')
    self.assertEqual(_coeff(conv) * NUM_CHANNELS, self.cost([conv]))

    # Conv1 has 7 gammas above 0.45, and NUM_CHANNELS inputs (from dw1).
    conv = self.GetConv('conv1')
    self.assertEqual(_coeff(conv) * 7 * NUM_CHANNELS, self.cost([conv]))

    # Conv2 has 11 active + 12 inactive, while Dw2 has 5 inactive, 10 active and
    # 8 active. Their max (or) has 15 active and 8 inactive.
    # Conv2 has NUM_CHANNELS inputs (from the image).
    conv = self.GetConv('conv2')
    self.assertEqual(_coeff(conv) * 15 * NUM_CHANNELS, self.cost([conv]))

    # Dw2 has 15 out of 23 inputs (from the Conv2).
    conv = self.GetConv('dw2')
    self.assertEqual(_coeff(conv) * 15, self.cost([conv]))

    # Conv3 has 10 gammas above 0.45, and 7 + 15 inputs from conv1 and dw2.
    conv = self.GetConv('conv3')
    self.assertEqual(_coeff(conv) * 10 * 22, self.cost([conv]))

  def testRegularizer(self):
    # Dw1 depthwise convolution is connected to the input (no regularizer).
    conv = self.GetConv('dw1')
    # Although the effective regularizer for dw is computed as below:
    # gamma = self.GetGammaAbsValue('dw1')
    # expected_loss = _coeff(conv) * gamma.sum()
    # Since the input is not regularized, dw does not return a regularizer.
    expected_loss = 0.0
    self.assertNear(expected_loss, self.loss([conv]), expected_loss * 1e-5)

    # Conv1 takes Dw1 as input, its input regularizer is from dw1.
    conv = self.GetConv('conv1')
    gamma = self.GetGammaAbsValue('conv1')
    # The effective size for dw can be computed from its gamma, and
    # the loss may be computed as follows:
    # gamma_dw = self.GetGammaAbsValue('dw1')
    # expected_loss = _coeff(conv) * (
    #     gamma.sum() * (gamma_dw > 0.45).sum() + gamma_dw.sum() *
    #     (gamma > 0.45).sum())
    # However, since dw cannot change shape because its input doesn't have a
    # regularizer, the real loss we expect should be:
    expected_loss = _coeff(conv) * (gamma.sum() * NUM_CHANNELS)
    self.assertNear(expected_loss, self.loss([conv]), expected_loss * 1e-5)

    # Dw2 depthwise convolution is connected to conv2 (grouped regularizer).
    conv = self.GetConv('conv2')
    gamma_conv = self.GetGammaAbsValue('conv2')
    dw = self.GetConv('dw2')
    gamma_dw = self.GetGammaAbsValue('dw2')
    gamma = np.maximum(gamma_dw, gamma_conv).sum()
    expected_loss = _coeff(conv) * (gamma * 3 + (gamma > 0.45).sum() * 0)
    self.assertNear(expected_loss, self.loss([conv]), expected_loss * 1e-5)
    expected_loss = _coeff(dw) * gamma * 2
    self.assertNear(expected_loss, self.loss([dw]), expected_loss * 1e-5)


class GammaFlopLossWithDepthwiseConvNoBatchNormTest(
    tf.test.TestCase, GammaFlopLossWithDepthwiseConvTestBase):
  """Test flop_regularizer for un-batchnormed depthwise convolutions.

  This test is used to confirm that when depthwise convolution is not BNed, it
  will not be considered towards the regularizer, but it will be counted towards
  the cost.
  This design choice is for backward compatibility for users who did not
  regularize depthwise convolutions. However, the cost will be reported
  regardless in order to be faithful to the real computation complexity.
  """

  def setUp(self):
    self._depthwise_use_batchnorm = False
    tf.reset_default_graph()
    self.BuildWithBatchNorm()
    with self.test_session():
      self.Init()

  def GetSession(self):
    return self.test_session()

  def testCost(self):
    # Dw1 has NUM_CHANNELS inputs (from the image).
    conv = self.GetConv('dw1')
    self.assertEqual(_coeff(conv) * 3, self.cost([conv]))

    # Conv1 has 7 gammas above 0.45, and 3 inputs (from dw1).
    conv = self.GetConv('conv1')
    self.assertEqual(_coeff(conv) * 7 * 3, self.cost([conv]))

    # Conv2 has 11 active outputs and NUM_CHANNELS inputs (from the image).
    conv = self.GetConv('conv2')
    self.assertEqual(_coeff(conv) * 11 * NUM_CHANNELS, self.cost([conv]))

    # Dw2 has 11 inputs (pass-through from the Conv2).
    conv = self.GetConv('dw2')
    self.assertEqual(_coeff(conv) * 11, self.cost([conv]))

    # Conv3 has 10 gammas above 0.45, and 7 + 11 inputs from conv1 and dw2.
    conv = self.GetConv('conv3')
    self.assertEqual(_coeff(conv) * 10 * 18, self.cost([conv]))

  def testRegularizer(self):
    # Dw1 depthwise convolution is connected to the input (no regularizer).
    conv = self.GetConv('dw1')
    expected_loss = 0.0
    self.assertNear(expected_loss, self.loss([conv]), expected_loss * 1e-5)

    # Conv1 takes Dw1 as input, but it's not affected by dw1 because depthwise
    # is not BNed.
    conv = self.GetConv('conv1')
    gamma = self.GetGammaAbsValue('conv1')
    expected_loss = _coeff(conv) * (gamma.sum() * NUM_CHANNELS)
    self.assertNear(expected_loss, self.loss([conv]), expected_loss * 1e-5)

    # Dw2 depthwise convolution is connected to conv2 (pass through).
    dw = self.GetConv('dw2')
    gamma = self.GetGammaAbsValue('conv2')
    expected_loss = _coeff(dw) * gamma.sum() * 2
    self.assertNear(expected_loss, self.loss([dw]), expected_loss * 1e-5)


class GammaFlopResidualConnectionsLossTest(tf.test.TestCase):
  """Tests flop_regularizer for a network with residual connections."""

  def setUp(self):
    tf.reset_default_graph()
    tf.set_random_seed(7)
    self._threshold = 0.6

  def buildModel(self, resnet_fn, block_fn):
    # We use this model as a test case because the slim.nets.resnet module is
    # used in some production.
    #
    # The model looks as follows:
    #
    # Image --> unit_1/shortcut
    # Image --> unit_1/conv1 --> unit_1/conv2 --> unit_1/conv3
    #
    # unit_1/shortcut + unit_1/conv3 --> unit_1 (residual connection)
    #
    # unit_1 --> unit_2/conv1  -> unit_2/conv2 --> unit_2/conv3
    #
    # unit_1 + unit_2/conv3 --> unit_2 (residual connection)
    #
    # In between, there are strided convolutions and pooling ops, but these
    # should not affect the regularizer.
    blocks = [
        block_fn('block1', base_depth=7, num_units=2, stride=2),
    ]
    image = tf.constant(0.0, shape=[1, 2, 2, NUM_CHANNELS])
    net = resnet_fn(
        image, blocks, include_root_block=False, is_training=False)[0]
    net = tf.reduce_mean(net, axis=(1, 2))
    return layers.fully_connected(net, 23, scope='FC')

  def buildGraphWithBatchNorm(self, resnet_fn, block_fn):
    params = {
        'trainable': True,
        'normalizer_fn': layers.batch_norm,
        'normalizer_params': {
            'scale': True
        }
    }

    with arg_scope([layers.conv2d, layers.separable_conv2d], **params):
      self.net = self.buildModel(resnet_fn, block_fn)

  def initGamma(self):
    assignments = []
    gammas = {}
    for v in tf.global_variables():
      if v.op.name.endswith('/gamma'):
        assignments.append(v.assign(tf.random_uniform(v.shape)))
        gammas[v.op.name] = v
    with self.test_session() as s:
      s.run(assignments)
      self._gammas = s.run(gammas)

  def getGamma(self, short_name):
    tokens = short_name.split('/')
    name = ('resnet_v1/block1/' + tokens[0] + '/bottleneck_v1/' + tokens[1] +
            '/BatchNorm/gamma')
    return self._gammas[name]

  def getOp(self, short_name):
    if short_name == 'FC':
      return tf.get_default_graph().get_operation_by_name('FC/MatMul')
    tokens = short_name.split('/')
    name = ('resnet_v1/block1/' + tokens[0] + '/bottleneck_v1/' + tokens[1] +
            '/Conv2D')
    return tf.get_default_graph().get_operation_by_name(name)

  def numAlive(self, short_name):
    return np.sum(self.getGamma(short_name) > self._threshold)

  def getCoeff(self, short_name):
    return _coeff(self.getOp(short_name))

  def testCost(self):
    self.buildGraphWithBatchNorm(resnet_v1.resnet_v1, resnet_v1.resnet_v1_block)
    self.initGamma()
    res_alive = np.logical_or(
        np.logical_or(
            self.getGamma('unit_1/shortcut') > self._threshold,
            self.getGamma('unit_1/conv3') > self._threshold),
        self.getGamma('unit_2/conv3') > self._threshold)

    self.gamma_flop_reg = flop_regularizer.GammaFlopsRegularizer(
        [self.net.op], self._threshold)

    expected = {}
    expected['unit_1/shortcut'] = (
        self.getCoeff('unit_1/shortcut') * np.sum(res_alive) * NUM_CHANNELS)
    expected['unit_1/conv1'] = (
        self.getCoeff('unit_1/conv1') * self.numAlive('unit_1/conv1') *
        NUM_CHANNELS)
    expected['unit_1/conv2'] = (
        self.getCoeff('unit_1/conv2') * self.numAlive('unit_1/conv2') *
        self.numAlive('unit_1/conv1'))
    expected['unit_1/conv3'] = (
        self.getCoeff('unit_1/conv3') * np.sum(res_alive) *
        self.numAlive('unit_1/conv2'))
    expected['unit_2/conv1'] = (
        self.getCoeff('unit_2/conv1') * self.numAlive('unit_2/conv1') *
        np.sum(res_alive))
    expected['unit_2/conv2'] = (
        self.getCoeff('unit_2/conv2') * self.numAlive('unit_2/conv2') *
        self.numAlive('unit_2/conv1'))
    expected['unit_2/conv3'] = (
        self.getCoeff('unit_2/conv3') * np.sum(res_alive) *
        self.numAlive('unit_2/conv2'))
    expected['FC'] = 2.0 * np.sum(res_alive) * 23.0

    # TODO: Is there a way to use Parametrized Tests to make this more
    # elegant?
    with self.test_session():
      for short_name in expected:
        cost = self.gamma_flop_reg.get_cost([self.getOp(short_name)]).eval()
        self.assertEqual(expected[short_name], cost)

      self.assertEqual(
          sum(expected.values()),
          self.gamma_flop_reg.get_cost().eval())


class GroupLassoFlopRegTest(tf.test.TestCase):

  def assertNearRelatively(self, expected, actual):
    self.assertNear(expected, actual, expected * 1e-6)

  def testFlopRegularizer(self):
    tf.reset_default_graph()
    tf.set_random_seed(7907)
    with arg_scope(
        [layers.conv2d, layers.conv2d_transpose],
        weights_initializer=tf.random_normal_initializer):
      # Our test model is:
      #
      #         -> conv1 --+
      #        /           |--[concat]
      #  image --> conv2 --+
      #        \
      #         -> convt
      #
      # (the model has two "outputs", convt and concat).
      #
      image = tf.constant(0.0, shape=[1, 17, 19, NUM_CHANNELS])
      conv1 = layers.conv2d(
          image, 13, [7, 5], padding='SAME', scope='conv1')
      conv2 = layers.conv2d(
          image, 23, [1, 1], padding='SAME', scope='conv2')
      self.concat = tf.concat([conv1, conv2], 3)
      self.convt = layers.conv2d_transpose(
          image, 29, [7, 5], stride=3, padding='SAME', scope='convt')
      self.name_to_var = {v.op.name: v for v in tf.global_variables()}
    with self.test_session():
      tf.global_variables_initializer().run()

    threshold = 1.0
    flop_reg = flop_regularizer.GroupLassoFlopsRegularizer(
        [self.concat.op, self.convt.op], threshold=threshold)

    with self.test_session() as s:
      evaluated_vars = s.run(self.name_to_var)

    def group_norm(weights, axis=(0, 1, 2)):  # pylint: disable=invalid-name
      return np.sqrt(np.mean(weights**2, axis=axis))

    reg_vectors = {
        'conv1': group_norm(evaluated_vars['conv1/weights'], (0, 1, 2)),
        'conv2': group_norm(evaluated_vars['conv2/weights'], (0, 1, 2)),
        'convt': group_norm(evaluated_vars['convt/weights'], (0, 1, 3))
    }

    num_alive = {k: np.sum(r > threshold) for k, r in reg_vectors.iteritems()}
    total_outputs = (
        reg_vectors['conv1'].shape[0] + reg_vectors['conv2'].shape[0])
    total_alive_outputs = sum(num_alive.values())
    assert total_alive_outputs > 0, (
        'All outputs are dead - test is trivial. Decrease the threshold.')
    assert total_alive_outputs < total_outputs, (
        'All outputs are alive - test is trivial. Increase the threshold.')

    coeff1 = _coeff(_get_op('conv1/Conv2D'))
    coeff2 = _coeff(_get_op('conv2/Conv2D'))
    coefft = _coeff(_get_op('convt/conv2d_transpose'))

    expected_flop_cost = NUM_CHANNELS * (
        coeff1 * num_alive['conv1'] + coeff2 * num_alive['conv2'] +
        coefft * num_alive['convt'])
    expected_reg_term = NUM_CHANNELS * (
        coeff1 * np.sum(reg_vectors['conv1']) + coeff2 * np.sum(
            reg_vectors['conv2']) + coefft * np.sum(reg_vectors['convt']))
    with self.test_session():
      self.assertEqual(
          round(expected_flop_cost), round(flop_reg.get_cost().eval()))
      self.assertNearRelatively(expected_reg_term,
                                flop_reg.get_regularization_term().eval())


def _get_op(name):  # pylint: disable=invalid-name
  return tf.get_default_graph().get_operation_by_name(name)


if __name__ == '__main__':
  tf.test.main()
