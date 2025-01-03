# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for slim.nets.resnet_v1."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim as slim

from nets import resnet_utils
from nets import resnet_v1

tf.disable_resource_variables()


def create_test_input(batch_size, height, width, channels):
  """Create test input tensor.

  Args:
    batch_size: The number of images per batch or `None` if unknown.
    height: The height of each image or `None` if unknown.
    width: The width of each image or `None` if unknown.
    channels: The number of channels per image or `None` if unknown.

  Returns:
    Either a placeholder `Tensor` of dimension
      [batch_size, height, width, channels] if any of the inputs are `None` or a
    constant `Tensor` with the mesh grid values along the spatial dimensions.
  """
  if None in [batch_size, height, width, channels]:
    return tf.placeholder(tf.float32, (batch_size, height, width, channels))
  else:
    return tf.cast(
        np.tile(
            np.reshape(
                np.reshape(np.arange(height), [height, 1]) +
                np.reshape(np.arange(width), [1, width]),
                [1, height, width, 1]), [batch_size, 1, 1, channels]),
        dtype=tf.float32)


class ResnetUtilsTest(tf.test.TestCase):

  def testSubsampleThreeByThree(self):
    x = tf.reshape(tf.cast(tf.range(9), dtype=tf.float32), [1, 3, 3, 1])
    x = resnet_utils.subsample(x, 2)
    expected = tf.reshape(tf.constant([0, 2, 6, 8]), [1, 2, 2, 1])
    with self.test_session():
      self.assertAllClose(x.eval(), expected.eval())

  def testSubsampleFourByFour(self):
    x = tf.reshape(tf.cast(tf.range(16), dtype=tf.float32), [1, 4, 4, 1])
    x = resnet_utils.subsample(x, 2)
    expected = tf.reshape(tf.constant([0, 2, 8, 10]), [1, 2, 2, 1])
    with self.test_session():
      self.assertAllClose(x.eval(), expected.eval())

  def testConv2DSameEven(self):
    n, n2 = 4, 2

    # Input image.
    x = create_test_input(1, n, n, 1)

    # Convolution kernel.
    w = create_test_input(1, 3, 3, 1)
    w = tf.reshape(w, [3, 3, 1, 1])

    tf.get_variable('Conv/weights', initializer=w)
    tf.get_variable('Conv/biases', initializer=tf.zeros([1]))
    tf.get_variable_scope().reuse_variables()

    y1 = slim.conv2d(x, 1, [3, 3], stride=1, scope='Conv')
    y1_expected = tf.cast([[14, 28, 43, 26], [28, 48, 66, 37], [43, 66, 84, 46],
                           [26, 37, 46, 22]],
                          dtype=tf.float32)
    y1_expected = tf.reshape(y1_expected, [1, n, n, 1])

    y2 = resnet_utils.subsample(y1, 2)
    y2_expected = tf.cast([[14, 43], [43, 84]], dtype=tf.float32)
    y2_expected = tf.reshape(y2_expected, [1, n2, n2, 1])

    y3 = resnet_utils.conv2d_same(x, 1, 3, stride=2, scope='Conv')
    y3_expected = y2_expected

    y4 = slim.conv2d(x, 1, [3, 3], stride=2, scope='Conv')
    y4_expected = tf.cast([[48, 37], [37, 22]], dtype=tf.float32)
    y4_expected = tf.reshape(y4_expected, [1, n2, n2, 1])

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(y1.eval(), y1_expected.eval())
      self.assertAllClose(y2.eval(), y2_expected.eval())
      self.assertAllClose(y3.eval(), y3_expected.eval())
      self.assertAllClose(y4.eval(), y4_expected.eval())

  def testConv2DSameOdd(self):
    n, n2 = 5, 3

    # Input image.
    x = create_test_input(1, n, n, 1)

    # Convolution kernel.
    w = create_test_input(1, 3, 3, 1)
    w = tf.reshape(w, [3, 3, 1, 1])

    tf.get_variable('Conv/weights', initializer=w)
    tf.get_variable('Conv/biases', initializer=tf.zeros([1]))
    tf.get_variable_scope().reuse_variables()

    y1 = slim.conv2d(x, 1, [3, 3], stride=1, scope='Conv')
    y1_expected = tf.cast(
        [[14, 28, 43, 58, 34], [28, 48, 66, 84, 46], [43, 66, 84, 102, 55],
         [58, 84, 102, 120, 64], [34, 46, 55, 64, 30]],
        dtype=tf.float32)
    y1_expected = tf.reshape(y1_expected, [1, n, n, 1])

    y2 = resnet_utils.subsample(y1, 2)
    y2_expected = tf.cast([[14, 43, 34], [43, 84, 55], [34, 55, 30]],
                          dtype=tf.float32)
    y2_expected = tf.reshape(y2_expected, [1, n2, n2, 1])

    y3 = resnet_utils.conv2d_same(x, 1, 3, stride=2, scope='Conv')
    y3_expected = y2_expected

    y4 = slim.conv2d(x, 1, [3, 3], stride=2, scope='Conv')
    y4_expected = y2_expected

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(y1.eval(), y1_expected.eval())
      self.assertAllClose(y2.eval(), y2_expected.eval())
      self.assertAllClose(y3.eval(), y3_expected.eval())
      self.assertAllClose(y4.eval(), y4_expected.eval())

  def _resnet_plain(self, inputs, blocks, output_stride=None, scope=None):
    """A plain ResNet without extra layers before or after the ResNet blocks."""
    with tf.variable_scope(scope, values=[inputs]):
      with slim.arg_scope([slim.conv2d], outputs_collections='end_points'):
        net = resnet_utils.stack_blocks_dense(inputs, blocks, output_stride)
        end_points = slim.utils.convert_collection_to_dict('end_points')
        return net, end_points

  def testEndPointsV1(self):
    """Test the end points of a tiny v1 bottleneck network."""
    blocks = [
        resnet_v1.resnet_v1_block(
            'block1', base_depth=1, num_units=2, stride=2),
        resnet_v1.resnet_v1_block(
            'block2', base_depth=2, num_units=2, stride=1),
    ]
    inputs = create_test_input(2, 32, 16, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, end_points = self._resnet_plain(inputs, blocks, scope='tiny')
    expected = [
        'tiny/block1/unit_1/bottleneck_v1/shortcut',
        'tiny/block1/unit_1/bottleneck_v1/conv1',
        'tiny/block1/unit_1/bottleneck_v1/conv2',
        'tiny/block1/unit_1/bottleneck_v1/conv3',
        'tiny/block1/unit_2/bottleneck_v1/conv1',
        'tiny/block1/unit_2/bottleneck_v1/conv2',
        'tiny/block1/unit_2/bottleneck_v1/conv3',
        'tiny/block2/unit_1/bottleneck_v1/shortcut',
        'tiny/block2/unit_1/bottleneck_v1/conv1',
        'tiny/block2/unit_1/bottleneck_v1/conv2',
        'tiny/block2/unit_1/bottleneck_v1/conv3',
        'tiny/block2/unit_2/bottleneck_v1/conv1',
        'tiny/block2/unit_2/bottleneck_v1/conv2',
        'tiny/block2/unit_2/bottleneck_v1/conv3']
    self.assertItemsEqual(expected, end_points.keys())

  def _stack_blocks_nondense(self, net, blocks):
    """A simplified ResNet Block stacker without output stride control."""
    for block in blocks:
      with tf.variable_scope(block.scope, 'block', [net]):
        for i, unit in enumerate(block.args):
          with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
            net = block.unit_fn(net, rate=1, **unit)
    return net

  def testAtrousValuesBottleneck(self):
    """Verify the values of dense feature extraction by atrous convolution.

    Make sure that dense feature extraction by stack_blocks_dense() followed by
    subsampling gives identical results to feature extraction at the nominal
    network output stride using the simple self._stack_blocks_nondense() above.
    """
    block = resnet_v1.resnet_v1_block
    blocks = [
        block('block1', base_depth=1, num_units=2, stride=2),
        block('block2', base_depth=2, num_units=2, stride=2),
        block('block3', base_depth=4, num_units=2, stride=2),
        block('block4', base_depth=8, num_units=2, stride=1),
    ]
    nominal_stride = 8

    # Test both odd and even input dimensions.
    height = 30
    width = 31
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      with slim.arg_scope([slim.batch_norm], is_training=False):
        for output_stride in [1, 2, 4, 8, None]:
          with tf.Graph().as_default():
            with self.test_session() as sess:
              tf.set_random_seed(0)
              inputs = create_test_input(1, height, width, 3)
              # Dense feature extraction followed by subsampling.
              output = resnet_utils.stack_blocks_dense(inputs,
                                                       blocks,
                                                       output_stride)
              if output_stride is None:
                factor = 1
              else:
                factor = nominal_stride // output_stride

              output = resnet_utils.subsample(output, factor)
              # Make the two networks use the same weights.
              tf.get_variable_scope().reuse_variables()
              # Feature extraction at the nominal network rate.
              expected = self._stack_blocks_nondense(inputs, blocks)
              sess.run(tf.global_variables_initializer())
              output, expected = sess.run([output, expected])
              self.assertAllClose(output, expected, atol=1e-4, rtol=1e-4)

  def testStridingLastUnitVsSubsampleBlockEnd(self):
    """Compares subsampling at the block's last unit or block's end.

    Makes sure that the final output is the same when we use a stride at the
    last unit of a block vs. we subsample activations at the end of a block.
    """
    block = resnet_v1.resnet_v1_block

    blocks = [
        block('block1', base_depth=1, num_units=2, stride=2),
        block('block2', base_depth=2, num_units=2, stride=2),
        block('block3', base_depth=4, num_units=2, stride=2),
        block('block4', base_depth=8, num_units=2, stride=1),
    ]

    # Test both odd and even input dimensions.
    height = 30
    width = 31
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      with slim.arg_scope([slim.batch_norm], is_training=False):
        for output_stride in [1, 2, 4, 8, None]:
          with tf.Graph().as_default():
            with self.test_session() as sess:
              tf.set_random_seed(0)
              inputs = create_test_input(1, height, width, 3)

              # Subsampling at the last unit of the block.
              output = resnet_utils.stack_blocks_dense(
                  inputs, blocks, output_stride,
                  store_non_strided_activations=False,
                  outputs_collections='output')
              output_end_points = slim.utils.convert_collection_to_dict(
                  'output')

              # Make the two networks use the same weights.
              tf.get_variable_scope().reuse_variables()

              # Subsample activations at the end of the blocks.
              expected = resnet_utils.stack_blocks_dense(
                  inputs, blocks, output_stride,
                  store_non_strided_activations=True,
                  outputs_collections='expected')
              expected_end_points = slim.utils.convert_collection_to_dict(
                  'expected')

              sess.run(tf.global_variables_initializer())

              # Make sure that the final output is the same.
              output, expected = sess.run([output, expected])
              self.assertAllClose(output, expected, atol=1e-4, rtol=1e-4)

              # Make sure that intermediate block activations in
              # output_end_points are subsampled versions of the corresponding
              # ones in expected_end_points.
              for i, block in enumerate(blocks[:-1:]):
                output = output_end_points[block.scope]
                expected = expected_end_points[block.scope]
                atrous_activated = (output_stride is not None and
                                    2 ** i >= output_stride)
                if not atrous_activated:
                  expected = resnet_utils.subsample(expected, 2)
                output, expected = sess.run([output, expected])
                self.assertAllClose(output, expected, atol=1e-4, rtol=1e-4)


class ResnetCompleteNetworkTest(tf.test.TestCase):
  """Tests with complete small ResNet v1 networks."""

  def _resnet_small(self,
                    inputs,
                    num_classes=None,
                    is_training=True,
                    global_pool=True,
                    output_stride=None,
                    include_root_block=True,
                    spatial_squeeze=True,
                    reuse=None,
                    scope='resnet_v1_small'):
    """A shallow and thin ResNet v1 for faster tests."""
    block = resnet_v1.resnet_v1_block
    blocks = [
        block('block1', base_depth=1, num_units=3, stride=2),
        block('block2', base_depth=2, num_units=3, stride=2),
        block('block3', base_depth=4, num_units=3, stride=2),
        block('block4', base_depth=8, num_units=2, stride=1),
    ]
    return resnet_v1.resnet_v1(inputs, blocks, num_classes,
                               is_training=is_training,
                               global_pool=global_pool,
                               output_stride=output_stride,
                               include_root_block=include_root_block,
                               spatial_squeeze=spatial_squeeze,
                               reuse=reuse,
                               scope=scope)

  def testClassificationEndPoints(self):
    global_pool = True
    num_classes = 10
    inputs = create_test_input(2, 224, 224, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      logits, end_points = self._resnet_small(inputs, num_classes,
                                              global_pool=global_pool,
                                              spatial_squeeze=False,
                                              scope='resnet')
    self.assertTrue(logits.op.name.startswith('resnet/logits'))
    self.assertListEqual(logits.get_shape().as_list(), [2, 1, 1, num_classes])
    self.assertTrue('predictions' in end_points)
    self.assertListEqual(end_points['predictions'].get_shape().as_list(),
                         [2, 1, 1, num_classes])
    self.assertTrue('global_pool' in end_points)
    self.assertListEqual(end_points['global_pool'].get_shape().as_list(),
                         [2, 1, 1, 32])

  def testClassificationEndPointsWithNoBatchNormArgscope(self):
    global_pool = True
    num_classes = 10
    inputs = create_test_input(2, 224, 224, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      logits, end_points = self._resnet_small(inputs, num_classes,
                                              global_pool=global_pool,
                                              spatial_squeeze=False,
                                              is_training=None,
                                              scope='resnet')
    self.assertTrue(logits.op.name.startswith('resnet/logits'))
    self.assertListEqual(logits.get_shape().as_list(), [2, 1, 1, num_classes])
    self.assertTrue('predictions' in end_points)
    self.assertListEqual(end_points['predictions'].get_shape().as_list(),
                         [2, 1, 1, num_classes])
    self.assertTrue('global_pool' in end_points)
    self.assertListEqual(end_points['global_pool'].get_shape().as_list(),
                         [2, 1, 1, 32])

  def testEndpointNames(self):
    # Like ResnetUtilsTest.testEndPointsV1(), but for the public API.
    global_pool = True
    num_classes = 10
    inputs = create_test_input(2, 224, 224, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, end_points = self._resnet_small(inputs, num_classes,
                                         global_pool=global_pool,
                                         scope='resnet')
    expected = ['resnet/conv1']
    for block in range(1, 5):
      for unit in range(1, 4 if block < 4 else 3):
        for conv in range(1, 4):
          expected.append('resnet/block%d/unit_%d/bottleneck_v1/conv%d' %
                          (block, unit, conv))
        expected.append('resnet/block%d/unit_%d/bottleneck_v1' % (block, unit))
      expected.append('resnet/block%d/unit_1/bottleneck_v1/shortcut' % block)
      expected.append('resnet/block%d' % block)
    expected.extend(['global_pool', 'resnet/logits', 'resnet/spatial_squeeze',
                     'predictions'])
    self.assertItemsEqual(end_points.keys(), expected)

  def testClassificationShapes(self):
    global_pool = True
    num_classes = 10
    inputs = create_test_input(2, 224, 224, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, end_points = self._resnet_small(inputs, num_classes,
                                         global_pool=global_pool,
                                         scope='resnet')
      endpoint_to_shape = {
          'resnet/block1': [2, 28, 28, 4],
          'resnet/block2': [2, 14, 14, 8],
          'resnet/block3': [2, 7, 7, 16],
          'resnet/block4': [2, 7, 7, 32]}
      for endpoint in endpoint_to_shape:
        shape = endpoint_to_shape[endpoint]
        self.assertListEqual(end_points[endpoint].get_shape().as_list(), shape)

  def testFullyConvolutionalEndpointShapes(self):
    global_pool = False
    num_classes = 10
    inputs = create_test_input(2, 321, 321, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, end_points = self._resnet_small(inputs, num_classes,
                                         global_pool=global_pool,
                                         spatial_squeeze=False,
                                         scope='resnet')
      endpoint_to_shape = {
          'resnet/block1': [2, 41, 41, 4],
          'resnet/block2': [2, 21, 21, 8],
          'resnet/block3': [2, 11, 11, 16],
          'resnet/block4': [2, 11, 11, 32]}
      for endpoint in endpoint_to_shape:
        shape = endpoint_to_shape[endpoint]
        self.assertListEqual(end_points[endpoint].get_shape().as_list(), shape)

  def testRootlessFullyConvolutionalEndpointShapes(self):
    global_pool = False
    num_classes = 10
    inputs = create_test_input(2, 128, 128, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, end_points = self._resnet_small(inputs, num_classes,
                                         global_pool=global_pool,
                                         include_root_block=False,
                                         spatial_squeeze=False,
                                         scope='resnet')
      endpoint_to_shape = {
          'resnet/block1': [2, 64, 64, 4],
          'resnet/block2': [2, 32, 32, 8],
          'resnet/block3': [2, 16, 16, 16],
          'resnet/block4': [2, 16, 16, 32]}
      for endpoint in endpoint_to_shape:
        shape = endpoint_to_shape[endpoint]
        self.assertListEqual(end_points[endpoint].get_shape().as_list(), shape)

  def testAtrousFullyConvolutionalEndpointShapes(self):
    global_pool = False
    num_classes = 10
    output_stride = 8
    inputs = create_test_input(2, 321, 321, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, end_points = self._resnet_small(inputs,
                                         num_classes,
                                         global_pool=global_pool,
                                         output_stride=output_stride,
                                         spatial_squeeze=False,
                                         scope='resnet')
      endpoint_to_shape = {
          'resnet/block1': [2, 41, 41, 4],
          'resnet/block2': [2, 41, 41, 8],
          'resnet/block3': [2, 41, 41, 16],
          'resnet/block4': [2, 41, 41, 32]}
      for endpoint in endpoint_to_shape:
        shape = endpoint_to_shape[endpoint]
        self.assertListEqual(end_points[endpoint].get_shape().as_list(), shape)

  def testAtrousFullyConvolutionalValues(self):
    """Verify dense feature extraction with atrous convolution."""
    nominal_stride = 32
    for output_stride in [4, 8, 16, 32, None]:
      with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        with tf.Graph().as_default():
          with self.test_session() as sess:
            tf.set_random_seed(0)
            inputs = create_test_input(2, 81, 81, 3)
            # Dense feature extraction followed by subsampling.
            output, _ = self._resnet_small(inputs, None, is_training=False,
                                           global_pool=False,
                                           output_stride=output_stride)
            if output_stride is None:
              factor = 1
            else:
              factor = nominal_stride // output_stride
            output = resnet_utils.subsample(output, factor)
            # Make the two networks use the same weights.
            tf.get_variable_scope().reuse_variables()
            # Feature extraction at the nominal network rate.
            expected, _ = self._resnet_small(inputs, None, is_training=False,
                                             global_pool=False)
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(output.eval(), expected.eval(),
                                atol=1e-4, rtol=1e-4)

  def testUnknownBatchSize(self):
    batch = 2
    height, width = 65, 65
    global_pool = True
    num_classes = 10
    inputs = create_test_input(None, height, width, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      logits, _ = self._resnet_small(inputs, num_classes,
                                     global_pool=global_pool,
                                     spatial_squeeze=False,
                                     scope='resnet')
    self.assertTrue(logits.op.name.startswith('resnet/logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [None, 1, 1, num_classes])
    images = create_test_input(batch, height, width, 3)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(logits, {inputs: images.eval()})
      self.assertEqual(output.shape, (batch, 1, 1, num_classes))

  def testFullyConvolutionalUnknownHeightWidth(self):
    batch = 2
    height, width = 65, 65
    global_pool = False
    inputs = create_test_input(batch, None, None, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      output, _ = self._resnet_small(inputs, None, global_pool=global_pool)
    self.assertListEqual(output.get_shape().as_list(),
                         [batch, None, None, 32])
    images = create_test_input(batch, height, width, 3)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output, {inputs: images.eval()})
      self.assertEqual(output.shape, (batch, 3, 3, 32))

  def testAtrousFullyConvolutionalUnknownHeightWidth(self):
    batch = 2
    height, width = 65, 65
    global_pool = False
    output_stride = 8
    inputs = create_test_input(batch, None, None, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      output, _ = self._resnet_small(inputs,
                                     None,
                                     global_pool=global_pool,
                                     output_stride=output_stride)
    self.assertListEqual(output.get_shape().as_list(),
                         [batch, None, None, 32])
    images = create_test_input(batch, height, width, 3)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output, {inputs: images.eval()})
      self.assertEqual(output.shape, (batch, 9, 9, 32))

  def testDepthMultiplier(self):
    resnets = [
        resnet_v1.resnet_v1_50, resnet_v1.resnet_v1_101,
        resnet_v1.resnet_v1_152, resnet_v1.resnet_v1_200
    ]
    resnet_names = [
        'resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152', 'resnet_v1_200'
    ]
    for resnet, resnet_name in zip(resnets, resnet_names):
      depth_multiplier = 0.25
      global_pool = True
      num_classes = 10
      inputs = create_test_input(2, 224, 224, 3)
      with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        scope_base = resnet_name + '_base'
        _, end_points_base = resnet(
            inputs,
            num_classes,
            global_pool=global_pool,
            min_base_depth=1,
            scope=scope_base)
        scope_test = resnet_name + '_test'
        _, end_points_test = resnet(
            inputs,
            num_classes,
            global_pool=global_pool,
            min_base_depth=1,
            depth_multiplier=depth_multiplier,
            scope=scope_test)
        for block in ['block1', 'block2', 'block3', 'block4']:
          block_name_base = scope_base + '/' + block
          block_name_test = scope_test + '/' + block
          self.assertTrue(block_name_base in end_points_base)
          self.assertTrue(block_name_test in end_points_test)
          self.assertEqual(
              len(end_points_base[block_name_base].get_shape().as_list()), 4)
          self.assertEqual(
              len(end_points_test[block_name_test].get_shape().as_list()), 4)
          self.assertListEqual(
              end_points_base[block_name_base].get_shape().as_list()[:3],
              end_points_test[block_name_test].get_shape().as_list()[:3])
          self.assertEqual(
              int(depth_multiplier *
                  end_points_base[block_name_base].get_shape().as_list()[3]),
              end_points_test[block_name_test].get_shape().as_list()[3])

  def testMinBaseDepth(self):
    resnets = [
        resnet_v1.resnet_v1_50, resnet_v1.resnet_v1_101,
        resnet_v1.resnet_v1_152, resnet_v1.resnet_v1_200
    ]
    resnet_names = [
        'resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152', 'resnet_v1_200'
    ]
    for resnet, resnet_name in zip(resnets, resnet_names):
      min_base_depth = 5
      global_pool = True
      num_classes = 10
      inputs = create_test_input(2, 224, 224, 3)
      with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        _, end_points = resnet(
            inputs,
            num_classes,
            global_pool=global_pool,
            min_base_depth=min_base_depth,
            depth_multiplier=0,
            scope=resnet_name)
        for block in ['block1', 'block2', 'block3', 'block4']:
          block_name = resnet_name + '/' + block
          self.assertTrue(block_name in end_points)
          self.assertEqual(
              len(end_points[block_name].get_shape().as_list()), 4)
          # The output depth is 4 times base_depth.
          depth_expected = min_base_depth * 4
          self.assertEqual(
              end_points[block_name].get_shape().as_list()[3], depth_expected)

if __name__ == '__main__':
  tf.test.main()
