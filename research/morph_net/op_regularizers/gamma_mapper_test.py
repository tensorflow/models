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
"""Tests for gamma_mapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.python.platform import flags
from morph_net.op_regularizers import gamma_mapper


FLAGS = flags.FLAGS


layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope


NUM_CHANNELS = 3


def get_op(name):
  return tf.get_default_graph().get_operation_by_name(name)


CONV1_GAMMA = [0.1 * x for x in range(13)]
SEP_CONV_GAMMA = [0.07 * x for x in range(23)]
CKPT_FILE_NAME = 'ckpt'


def build_model():
  image = tf.constant(0.0, shape=[1, 17, 19, 3])
  conv1 = layers.conv2d(image, 13, (3, 3), padding='SAME', scope='conv1')
  layers.separable_conv2d(conv1, 23, (3, 3), 1, scope='sep_conv')


def setUpModule():
  """Save a model for later loading it.

  This is the only way we're aware of for assigning values to variables
  irrespectively of their type (regular or partitioned), since partitioned
  variables do not support assignment.
  """
  with tf.Graph().as_default():
    params = {
        'normalizer_fn': layers.batch_norm,
        'normalizer_params': {
            'scale': True,
        }
    }
    with tf.contrib.framework.arg_scope(
        [layers.conv2d, layers.separable_conv2d], **params):
      build_model()

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      conv_gamma = tf.get_variable('conv1/BatchNorm/gamma')
      sep_gamma = tf.get_variable('sep_conv/BatchNorm/gamma')
    s = tf.Session()
    s.run(tf.global_variables_initializer())
    s.run([conv_gamma.assign(CONV1_GAMMA), sep_gamma.assign(SEP_CONV_GAMMA)])
    saver = tf.train.Saver()
    saver.save(s, os.path.join(FLAGS.test_tmpdir, CKPT_FILE_NAME))


class ConvGammaMapperTest(parameterized.TestCase, tf.test.TestCase):

  def createMapper(self, connectivity):
    if connectivity:
      return gamma_mapper.ConvGammaMapperByConnectivity()
    return gamma_mapper.ConvGammaMapperByName()

  def setUp(self):
    tf.reset_default_graph()

  def TestSuccess(self, connectivity, partitioning, fused, use_resource):
    params = {
        'trainable': True,
        'normalizer_fn': layers.batch_norm,
        'normalizer_params': {
            'scale': True,
            'fused': fused
        }
    }

    partitioner = tf.fixed_size_partitioner(2) if partitioning else None
    with tf.variable_scope(
        tf.get_variable_scope(),
        partitioner=partitioner,
        use_resource=use_resource):
      with tf.contrib.framework.arg_scope(
          [layers.conv2d, layers.separable_conv2d], **params):
        build_model()

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(FLAGS.test_tmpdir, CKPT_FILE_NAME))
    mapper = self.createMapper(connectivity)
    conv = get_op('conv1/Conv2D')
    sep_conv = get_op('sep_conv/separable_conv2d')
    with sess.as_default():
      self.assertAllClose(CONV1_GAMMA, mapper.get_gamma(conv).eval())
      self.assertAllClose(SEP_CONV_GAMMA, mapper.get_gamma(sep_conv).eval())

  def testSuccess(self):
    for connectivity in (False, True):
      for partitioning in (False, True):
        for fused in (False, True):
          if connectivity and not fused:  # This combination is not supported
            continue
          for use_resource in (False, True):
            tf.reset_default_graph()
            self.TestSuccess(connectivity, partitioning, fused, use_resource)

  @parameterized.named_parameters(
      ('_name_nopart', False, False), ('_name_part', False, True),
      ('_conn_nopart', True, False), ('_conn_part', True, True))
  def testNoBatchNorm(self, connectivity, partitioning):
    partitioner = tf.fixed_size_partitioner(2) if partitioning else None
    with tf.variable_scope(
        tf.get_variable_scope(), partitioner=partitioner):
      build_model()
    mapper = self.createMapper(connectivity)
    conv = get_op('conv1/Conv2D')
    self.assertEqual(None, mapper.get_gamma(conv))

  @parameterized.named_parameters(('_name_nopart', False),
                                  ('_conn_nopart', True))
  def testNotAConv(self, connectivity):
    build_model()
    mapper = self.createMapper(connectivity)
    bias_add = get_op('conv1/BiasAdd')
    with self.assertRaises(ValueError):
      mapper.get_gamma(bias_add)

  @parameterized.named_parameters(('_name_nopart', False),
                                  ('_conn_nopart', True))
  def testNotAnOpButATensor(self, connectivity):
    build_model()
    mapper = self.createMapper(connectivity)
    conv = get_op('conv1/Conv2D')
    with self.assertRaises(ValueError):
      mapper.get_gamma(conv.outputs[0])

  @parameterized.named_parameters(('_name_nopart', False),
                                  ('_conn_nopart', True))
  def testNotInGraph(self, connectivity):
    mapper = self.createMapper(connectivity)
    # Graph is built after the mapper
    build_model()
    conv = get_op('conv1/Conv2D')
    with self.assertRaises(KeyError):
      mapper.get_gamma(conv)


def build_resnet(block_fn, resnet_fn):
  params = {
      'trainable': True,
      'normalizer_fn': layers.batch_norm,
      'normalizer_params': {
          'is_training': True,
          'scale': True,
          'fused': True
      }
  }

  with arg_scope([layers.conv2d], **params):
    with arg_scope([layers.batch_norm], **(params['normalizer_params'])):
      # Each block looks like:
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
          block_fn('block2', base_depth=13, num_units=2, stride=2),
      ]
      image = tf.constant(0.0, shape=[1, 2, 2, NUM_CHANNELS])
      return resnet_fn(
          image, blocks, include_root_block=False, is_training=False)[0]


class ConvGammaMapperByConnectivityResnetTest(parameterized.TestCase,
                                              tf.test.TestCase):

  def assertGammaMatchesConv(self, mapper, prefix):
    conv = get_op(prefix + '/Conv2D')
    gamma = mapper.get_gamma(conv)
    self.assertTrue(gamma.op.name.startswith(prefix + '/BatchNorm/gamma'))

  def assertConvsConnectedToGammas(self, conv_names, gamma_prefixes, mapper):
    """Asserts that each convolution is connected to each gamma.

    Args:
      conv_names: A list of strings representing names of Conv2D operations.
      gamma_prefixes: A list of strings representing name prefixes of gamma
        variables (we only verify prefixes because suffixes may depend on
        whether we have partitioning or no).
      mapper: a ConvGammaMapperByConnectivity object
    """
    def make_set(item):
      return item if isinstance(item, set) else set([item,])

    convs = [get_op(conv_name) for conv_name in conv_names]
    gamma_sets = [make_set(mapper.get_gamma(conv)) for conv in convs]
    if len(gamma_sets) > 1:
      for i in range(1, len(gamma_sets)):
        self.assertEqual(gamma_sets[i], gamma_sets[0])

    actual_gamma_names = sorted([g.op.name for g in gamma_sets[0]])
    gamma_prefixes = sorted(gamma_prefixes)
    for expected, actual in zip(gamma_prefixes, actual_gamma_names):
      self.assertTrue(actual.startswith(expected))

  def testSuccessResnetV2(self):
    build_resnet(resnet_v2.resnet_v2_block, resnet_v2.resnet_v2)
    mapper = gamma_mapper.ConvGammaMapperByConnectivity()
    # Check all "regular" convs, that are connected to their own batch norm,
    # without residual connecitons involved.
    for block in (1, 2):
      for unit in (1, 2):
        for conv in (1, 2):
          self.assertGammaMatchesConv(
              mapper, 'resnet_v2/block%d/unit_%d/bottleneck_v2/conv%d' %
              (block, unit, conv))

    # This diagram depicts all the convs and the batch-norm that don't have a
    # one to one mapping:
    #
    #                  CONVS                        BATCH-NORMS
    #
    #           block1/unit_1/shortcut --+
    #                                    |
    #           block1/unit_1/conv3  ----+-->  block1/unit_2/preact
    #                                    |
    #           block1/unit_2/conv3  ----+-->  block2/unit_1/preact
    #
    #
    #           block2/unit_1/shortcut --+
    #                                    |
    #           block2/unit_1/conv3  ----+-->  block2/unit_1/preact
    #                                    |
    #           block2/unit_2/conv3  ----+-->  postnorm
    #
    # This connectivity is tested below.

    self.assertConvsConnectedToGammas([
        'resnet_v2/block1/unit_1/bottleneck_v2/shortcut/Conv2D',
        'resnet_v2/block1/unit_1/bottleneck_v2/conv3/Conv2D'
    ], [
        'resnet_v2/block1/unit_2/bottleneck_v2/preact/gamma',
        'resnet_v2/block2/unit_1/bottleneck_v2/preact/gamma'
    ], mapper)

    self.assertConvsConnectedToGammas([
        'resnet_v2/block1/unit_2/bottleneck_v2/conv3/Conv2D',
    ], [
        'resnet_v2/block2/unit_1/bottleneck_v2/preact/gamma',
    ], mapper)

    self.assertConvsConnectedToGammas([
        'resnet_v2/block2/unit_1/bottleneck_v2/shortcut/Conv2D',
        'resnet_v2/block2/unit_1/bottleneck_v2/conv3/Conv2D'
    ], [
        'resnet_v2/block2/unit_2/bottleneck_v2/preact/gamma',
        'resnet_v2/postnorm/gamma'
    ], mapper)

    self.assertConvsConnectedToGammas([
        'resnet_v2/block2/unit_2/bottleneck_v2/conv3/Conv2D',
    ], [
        'resnet_v2/postnorm/gamma',
    ], mapper)

  def testSuccessResnetV1(self):
    build_resnet(resnet_v1.resnet_v1_block, resnet_v1.resnet_v1)
    mapper = gamma_mapper.ConvGammaMapperByConnectivity()
    # Here the mapping between convolutions and batch-norms is simple one to
    # one.
    for block in (1, 2):
      self.assertGammaMatchesConv(
          mapper, 'resnet_v1/block%d/unit_1/bottleneck_v1/shortcut' % block)

      for unit in (1, 2):
        for conv in (1, 2, 3):
          self.assertGammaMatchesConv(
              mapper, 'resnet_v1/block%d/unit_%d/bottleneck_v1/conv%d' %
              (block, unit, conv))


if __name__ == '__main__':
  tf.test.main()
