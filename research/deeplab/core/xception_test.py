# Lint as: python2, python3
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

"""Tests for xception.py."""
import numpy as np
import six
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

from deeplab.core import xception
from tensorflow.contrib.slim.nets import resnet_utils

slim = contrib_slim


def create_test_input(batch, height, width, channels):
  """Create test input tensor."""
  if None in [batch, height, width, channels]:
    return tf.placeholder(tf.float32, (batch, height, width, channels))
  else:
    return tf.cast(
        np.tile(
            np.reshape(
                np.reshape(np.arange(height), [height, 1]) +
                np.reshape(np.arange(width), [1, width]),
                [1, height, width, 1]),
            [batch, 1, 1, channels]),
        tf.float32)


class UtilityFunctionTest(tf.test.TestCase):

  def testSeparableConv2DSameWithInputEvenSize(self):
    n, n2 = 4, 2

    # Input image.
    x = create_test_input(1, n, n, 1)

    # Convolution kernel.
    dw = create_test_input(1, 3, 3, 1)
    dw = tf.reshape(dw, [3, 3, 1, 1])

    tf.get_variable('Conv/depthwise_weights', initializer=dw)
    tf.get_variable('Conv/pointwise_weights',
                    initializer=tf.ones([1, 1, 1, 1]))
    tf.get_variable('Conv/biases', initializer=tf.zeros([1]))
    tf.get_variable_scope().reuse_variables()

    y1 = slim.separable_conv2d(x, 1, [3, 3], depth_multiplier=1,
                               stride=1, scope='Conv')
    y1_expected = tf.cast([[14, 28, 43, 26],
                           [28, 48, 66, 37],
                           [43, 66, 84, 46],
                           [26, 37, 46, 22]], tf.float32)
    y1_expected = tf.reshape(y1_expected, [1, n, n, 1])

    y2 = resnet_utils.subsample(y1, 2)
    y2_expected = tf.cast([[14, 43],
                           [43, 84]], tf.float32)
    y2_expected = tf.reshape(y2_expected, [1, n2, n2, 1])

    y3 = xception.separable_conv2d_same(x, 1, 3, depth_multiplier=1,
                                        regularize_depthwise=True,
                                        stride=2, scope='Conv')
    y3_expected = y2_expected

    y4 = slim.separable_conv2d(x, 1, [3, 3], depth_multiplier=1,
                               stride=2, scope='Conv')
    y4_expected = tf.cast([[48, 37],
                           [37, 22]], tf.float32)
    y4_expected = tf.reshape(y4_expected, [1, n2, n2, 1])

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(y1.eval(), y1_expected.eval())
      self.assertAllClose(y2.eval(), y2_expected.eval())
      self.assertAllClose(y3.eval(), y3_expected.eval())
      self.assertAllClose(y4.eval(), y4_expected.eval())

  def testSeparableConv2DSameWithInputOddSize(self):
    n, n2 = 5, 3

    # Input image.
    x = create_test_input(1, n, n, 1)

    # Convolution kernel.
    dw = create_test_input(1, 3, 3, 1)
    dw = tf.reshape(dw, [3, 3, 1, 1])

    tf.get_variable('Conv/depthwise_weights', initializer=dw)
    tf.get_variable('Conv/pointwise_weights',
                    initializer=tf.ones([1, 1, 1, 1]))
    tf.get_variable('Conv/biases', initializer=tf.zeros([1]))
    tf.get_variable_scope().reuse_variables()

    y1 = slim.separable_conv2d(x, 1, [3, 3], depth_multiplier=1,
                               stride=1, scope='Conv')
    y1_expected = tf.cast([[14, 28, 43, 58, 34],
                           [28, 48, 66, 84, 46],
                           [43, 66, 84, 102, 55],
                           [58, 84, 102, 120, 64],
                           [34, 46, 55, 64, 30]], tf.float32)
    y1_expected = tf.reshape(y1_expected, [1, n, n, 1])

    y2 = resnet_utils.subsample(y1, 2)
    y2_expected = tf.cast([[14, 43, 34],
                           [43, 84, 55],
                           [34, 55, 30]], tf.float32)
    y2_expected = tf.reshape(y2_expected, [1, n2, n2, 1])

    y3 = xception.separable_conv2d_same(x, 1, 3, depth_multiplier=1,
                                        regularize_depthwise=True,
                                        stride=2, scope='Conv')
    y3_expected = y2_expected

    y4 = slim.separable_conv2d(x, 1, [3, 3], depth_multiplier=1,
                               stride=2, scope='Conv')
    y4_expected = y2_expected

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(y1.eval(), y1_expected.eval())
      self.assertAllClose(y2.eval(), y2_expected.eval())
      self.assertAllClose(y3.eval(), y3_expected.eval())
      self.assertAllClose(y4.eval(), y4_expected.eval())


class XceptionNetworkTest(tf.test.TestCase):
  """Tests with small Xception network."""

  def _xception_small(self,
                      inputs,
                      num_classes=None,
                      is_training=True,
                      global_pool=True,
                      output_stride=None,
                      regularize_depthwise=True,
                      reuse=None,
                      scope='xception_small'):
    """A shallow and thin Xception for faster tests."""
    block = xception.xception_block
    blocks = [
        block('entry_flow/block1',
              depth_list=[1, 1, 1],
              skip_connection_type='conv',
              activation_fn_in_separable_conv=False,
              regularize_depthwise=regularize_depthwise,
              num_units=1,
              stride=2),
        block('entry_flow/block2',
              depth_list=[2, 2, 2],
              skip_connection_type='conv',
              activation_fn_in_separable_conv=False,
              regularize_depthwise=regularize_depthwise,
              num_units=1,
              stride=2),
        block('entry_flow/block3',
              depth_list=[4, 4, 4],
              skip_connection_type='conv',
              activation_fn_in_separable_conv=False,
              regularize_depthwise=regularize_depthwise,
              num_units=1,
              stride=1),
        block('entry_flow/block4',
              depth_list=[4, 4, 4],
              skip_connection_type='conv',
              activation_fn_in_separable_conv=False,
              regularize_depthwise=regularize_depthwise,
              num_units=1,
              stride=2),
        block('middle_flow/block1',
              depth_list=[4, 4, 4],
              skip_connection_type='sum',
              activation_fn_in_separable_conv=False,
              regularize_depthwise=regularize_depthwise,
              num_units=2,
              stride=1),
        block('exit_flow/block1',
              depth_list=[8, 8, 8],
              skip_connection_type='conv',
              activation_fn_in_separable_conv=False,
              regularize_depthwise=regularize_depthwise,
              num_units=1,
              stride=2),
        block('exit_flow/block2',
              depth_list=[16, 16, 16],
              skip_connection_type='none',
              activation_fn_in_separable_conv=True,
              regularize_depthwise=regularize_depthwise,
              num_units=1,
              stride=1),
    ]
    return xception.xception(inputs,
                             blocks=blocks,
                             num_classes=num_classes,
                             is_training=is_training,
                             global_pool=global_pool,
                             output_stride=output_stride,
                             reuse=reuse,
                             scope=scope)

  def testClassificationEndPoints(self):
    global_pool = True
    num_classes = 3
    inputs = create_test_input(2, 32, 32, 3)
    with slim.arg_scope(xception.xception_arg_scope()):
      logits, end_points = self._xception_small(
          inputs,
          num_classes=num_classes,
          global_pool=global_pool,
          scope='xception')
    self.assertTrue(
        logits.op.name.startswith('xception/logits'))
    self.assertListEqual(logits.get_shape().as_list(), [2, 1, 1, num_classes])
    self.assertTrue('predictions' in end_points)
    self.assertListEqual(end_points['predictions'].get_shape().as_list(),
                         [2, 1, 1, num_classes])
    self.assertTrue('global_pool' in end_points)
    self.assertListEqual(end_points['global_pool'].get_shape().as_list(),
                         [2, 1, 1, 16])

  def testEndpointNames(self):
    global_pool = True
    num_classes = 3
    inputs = create_test_input(2, 32, 32, 3)
    with slim.arg_scope(xception.xception_arg_scope()):
      _, end_points = self._xception_small(
          inputs,
          num_classes=num_classes,
          global_pool=global_pool,
          scope='xception')
    expected = [
        'xception/entry_flow/conv1_1',
        'xception/entry_flow/conv1_2',
        'xception/entry_flow/block1/unit_1/xception_module/separable_conv1',
        'xception/entry_flow/block1/unit_1/xception_module/separable_conv2',
        'xception/entry_flow/block1/unit_1/xception_module/separable_conv3',
        'xception/entry_flow/block1/unit_1/xception_module/shortcut',
        'xception/entry_flow/block1/unit_1/xception_module',
        'xception/entry_flow/block1',
        'xception/entry_flow/block2/unit_1/xception_module/separable_conv1',
        'xception/entry_flow/block2/unit_1/xception_module/separable_conv2',
        'xception/entry_flow/block2/unit_1/xception_module/separable_conv3',
        'xception/entry_flow/block2/unit_1/xception_module/shortcut',
        'xception/entry_flow/block2/unit_1/xception_module',
        'xception/entry_flow/block2',
        'xception/entry_flow/block3/unit_1/xception_module/separable_conv1',
        'xception/entry_flow/block3/unit_1/xception_module/separable_conv2',
        'xception/entry_flow/block3/unit_1/xception_module/separable_conv3',
        'xception/entry_flow/block3/unit_1/xception_module/shortcut',
        'xception/entry_flow/block3/unit_1/xception_module',
        'xception/entry_flow/block3',
        'xception/entry_flow/block4/unit_1/xception_module/separable_conv1',
        'xception/entry_flow/block4/unit_1/xception_module/separable_conv2',
        'xception/entry_flow/block4/unit_1/xception_module/separable_conv3',
        'xception/entry_flow/block4/unit_1/xception_module/shortcut',
        'xception/entry_flow/block4/unit_1/xception_module',
        'xception/entry_flow/block4',
        'xception/middle_flow/block1/unit_1/xception_module/separable_conv1',
        'xception/middle_flow/block1/unit_1/xception_module/separable_conv2',
        'xception/middle_flow/block1/unit_1/xception_module/separable_conv3',
        'xception/middle_flow/block1/unit_1/xception_module',
        'xception/middle_flow/block1/unit_2/xception_module/separable_conv1',
        'xception/middle_flow/block1/unit_2/xception_module/separable_conv2',
        'xception/middle_flow/block1/unit_2/xception_module/separable_conv3',
        'xception/middle_flow/block1/unit_2/xception_module',
        'xception/middle_flow/block1',
        'xception/exit_flow/block1/unit_1/xception_module/separable_conv1',
        'xception/exit_flow/block1/unit_1/xception_module/separable_conv2',
        'xception/exit_flow/block1/unit_1/xception_module/separable_conv3',
        'xception/exit_flow/block1/unit_1/xception_module/shortcut',
        'xception/exit_flow/block1/unit_1/xception_module',
        'xception/exit_flow/block1',
        'xception/exit_flow/block2/unit_1/xception_module/separable_conv1',
        'xception/exit_flow/block2/unit_1/xception_module/separable_conv2',
        'xception/exit_flow/block2/unit_1/xception_module/separable_conv3',
        'xception/exit_flow/block2/unit_1/xception_module',
        'xception/exit_flow/block2',
        'global_pool',
        'xception/logits',
        'predictions',
    ]
    self.assertItemsEqual(list(end_points.keys()), expected)

  def testClassificationShapes(self):
    global_pool = True
    num_classes = 3
    inputs = create_test_input(2, 64, 64, 3)
    with slim.arg_scope(xception.xception_arg_scope()):
      _, end_points = self._xception_small(
          inputs,
          num_classes,
          global_pool=global_pool,
          scope='xception')
      endpoint_to_shape = {
          'xception/entry_flow/conv1_1': [2, 32, 32, 32],
          'xception/entry_flow/block1': [2, 16, 16, 1],
          'xception/entry_flow/block2': [2, 8, 8, 2],
          'xception/entry_flow/block4': [2, 4, 4, 4],
          'xception/middle_flow/block1': [2, 4, 4, 4],
          'xception/exit_flow/block1': [2, 2, 2, 8],
          'xception/exit_flow/block2': [2, 2, 2, 16]}
      for endpoint, shape in six.iteritems(endpoint_to_shape):
        self.assertListEqual(end_points[endpoint].get_shape().as_list(), shape)

  def testFullyConvolutionalEndpointShapes(self):
    global_pool = False
    num_classes = 3
    inputs = create_test_input(2, 65, 65, 3)
    with slim.arg_scope(xception.xception_arg_scope()):
      _, end_points = self._xception_small(
          inputs,
          num_classes,
          global_pool=global_pool,
          scope='xception')
      endpoint_to_shape = {
          'xception/entry_flow/conv1_1': [2, 33, 33, 32],
          'xception/entry_flow/block1': [2, 17, 17, 1],
          'xception/entry_flow/block2': [2, 9, 9, 2],
          'xception/entry_flow/block4': [2, 5, 5, 4],
          'xception/middle_flow/block1': [2, 5, 5, 4],
          'xception/exit_flow/block1': [2, 3, 3, 8],
          'xception/exit_flow/block2': [2, 3, 3, 16]}
      for endpoint, shape in six.iteritems(endpoint_to_shape):
        self.assertListEqual(end_points[endpoint].get_shape().as_list(), shape)

  def testAtrousFullyConvolutionalEndpointShapes(self):
    global_pool = False
    num_classes = 3
    output_stride = 8
    inputs = create_test_input(2, 65, 65, 3)
    with slim.arg_scope(xception.xception_arg_scope()):
      _, end_points = self._xception_small(
          inputs,
          num_classes,
          global_pool=global_pool,
          output_stride=output_stride,
          scope='xception')
      endpoint_to_shape = {
          'xception/entry_flow/block1': [2, 17, 17, 1],
          'xception/entry_flow/block2': [2, 9, 9, 2],
          'xception/entry_flow/block4': [2, 9, 9, 4],
          'xception/middle_flow/block1': [2, 9, 9, 4],
          'xception/exit_flow/block1': [2, 9, 9, 8],
          'xception/exit_flow/block2': [2, 9, 9, 16]}
      for endpoint, shape in six.iteritems(endpoint_to_shape):
        self.assertListEqual(end_points[endpoint].get_shape().as_list(), shape)

  def testAtrousFullyConvolutionalValues(self):
    """Verify dense feature extraction with atrous convolution."""
    nominal_stride = 32
    for output_stride in [4, 8, 16, 32, None]:
      with slim.arg_scope(xception.xception_arg_scope()):
        with tf.Graph().as_default():
          with self.test_session() as sess:
            tf.set_random_seed(0)
            inputs = create_test_input(2, 96, 97, 3)
            # Dense feature extraction followed by subsampling.
            output, _ = self._xception_small(
                inputs,
                None,
                is_training=False,
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
            expected, _ = self._xception_small(
                inputs,
                None,
                is_training=False,
                global_pool=False)
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(output.eval(), expected.eval(),
                                atol=1e-5, rtol=1e-5)

  def testUnknownBatchSize(self):
    batch = 2
    height, width = 65, 65
    global_pool = True
    num_classes = 10
    inputs = create_test_input(None, height, width, 3)
    with slim.arg_scope(xception.xception_arg_scope()):
      logits, _ = self._xception_small(
          inputs,
          num_classes,
          global_pool=global_pool,
          scope='xception')
    self.assertTrue(logits.op.name.startswith('xception/logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [None, 1, 1, num_classes])
    images = create_test_input(batch, height, width, 3)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(logits, {inputs: images.eval()})
      self.assertEquals(output.shape, (batch, 1, 1, num_classes))

  def testFullyConvolutionalUnknownHeightWidth(self):
    batch = 2
    height, width = 65, 65
    global_pool = False
    inputs = create_test_input(batch, None, None, 3)
    with slim.arg_scope(xception.xception_arg_scope()):
      output, _ = self._xception_small(
          inputs,
          None,
          global_pool=global_pool)
    self.assertListEqual(output.get_shape().as_list(),
                         [batch, None, None, 16])
    images = create_test_input(batch, height, width, 3)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output, {inputs: images.eval()})
      self.assertEquals(output.shape, (batch, 3, 3, 16))

  def testAtrousFullyConvolutionalUnknownHeightWidth(self):
    batch = 2
    height, width = 65, 65
    global_pool = False
    output_stride = 8
    inputs = create_test_input(batch, None, None, 3)
    with slim.arg_scope(xception.xception_arg_scope()):
      output, _ = self._xception_small(
          inputs,
          None,
          global_pool=global_pool,
          output_stride=output_stride)
    self.assertListEqual(output.get_shape().as_list(),
                         [batch, None, None, 16])
    images = create_test_input(batch, height, width, 3)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output, {inputs: images.eval()})
      self.assertEquals(output.shape, (batch, 9, 9, 16))

  def testEndpointsReuse(self):
    inputs = create_test_input(2, 32, 32, 3)
    with slim.arg_scope(xception.xception_arg_scope()):
      _, end_points0 = xception.xception_65(
          inputs,
          num_classes=10,
          reuse=False)
    with slim.arg_scope(xception.xception_arg_scope()):
      _, end_points1 = xception.xception_65(
          inputs,
          num_classes=10,
          reuse=True)
    self.assertItemsEqual(list(end_points0.keys()), list(end_points1.keys()))

  def testUseBoundedAcitvation(self):
    global_pool = False
    num_classes = 3
    output_stride = 16
    for use_bounded_activation in (True, False):
      tf.reset_default_graph()
      inputs = create_test_input(2, 65, 65, 3)
      with slim.arg_scope(xception.xception_arg_scope(
          use_bounded_activation=use_bounded_activation)):
        _, _ = self._xception_small(
            inputs,
            num_classes,
            global_pool=global_pool,
            output_stride=output_stride,
            scope='xception')
        for node in tf.get_default_graph().as_graph_def().node:
          if node.op.startswith('Relu'):
            self.assertEqual(node.op == 'Relu6', use_bounded_activation)

if __name__ == '__main__':
  tf.test.main()
