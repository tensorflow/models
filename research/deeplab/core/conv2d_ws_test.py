# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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

"""Tests for conv2d_ws."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers
from deeplab.core import conv2d_ws


class ConvolutionTest(tf.test.TestCase):

  def testInvalidShape(self):
    with self.cached_session():
      images_3d = tf.random_uniform((5, 6, 7, 9, 3), seed=1)
      with self.assertRaisesRegexp(
          ValueError, 'Convolution expects input with rank 4, got 5'):
        conv2d_ws.conv2d(images_3d, 32, 3)

  def testInvalidDataFormat(self):
    height, width = 7, 9
    with self.cached_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      with self.assertRaisesRegexp(ValueError, 'data_format'):
        conv2d_ws.conv2d(images, 32, 3, data_format='CHWN')

  def testCreateConv(self):
    height, width = 7, 9
    with self.cached_session():
      images = np.random.uniform(size=(5, height, width, 4)).astype(np.float32)
      output = conv2d_ws.conv2d(images, 32, [3, 3])
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 32])
      weights = contrib_framework.get_variables_by_name('weights')[0]
      self.assertListEqual(weights.get_shape().as_list(), [3, 3, 4, 32])
      biases = contrib_framework.get_variables_by_name('biases')[0]
      self.assertListEqual(biases.get_shape().as_list(), [32])

  def testCreateConvWithWS(self):
    height, width = 7, 9
    with self.cached_session():
      images = np.random.uniform(size=(5, height, width, 4)).astype(np.float32)
      output = conv2d_ws.conv2d(
          images, 32, [3, 3], use_weight_standardization=True)
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 32])
      weights = contrib_framework.get_variables_by_name('weights')[0]
      self.assertListEqual(weights.get_shape().as_list(), [3, 3, 4, 32])
      biases = contrib_framework.get_variables_by_name('biases')[0]
      self.assertListEqual(biases.get_shape().as_list(), [32])

  def testCreateConvNCHW(self):
    height, width = 7, 9
    with self.cached_session():
      images = np.random.uniform(size=(5, 4, height, width)).astype(np.float32)
      output = conv2d_ws.conv2d(images, 32, [3, 3], data_format='NCHW')
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, 32, height, width])
      weights = contrib_framework.get_variables_by_name('weights')[0]
      self.assertListEqual(weights.get_shape().as_list(), [3, 3, 4, 32])
      biases = contrib_framework.get_variables_by_name('biases')[0]
      self.assertListEqual(biases.get_shape().as_list(), [32])

  def testCreateSquareConv(self):
    height, width = 7, 9
    with self.cached_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = conv2d_ws.conv2d(images, 32, 3)
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 32])

  def testCreateConvWithTensorShape(self):
    height, width = 7, 9
    with self.cached_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = conv2d_ws.conv2d(images, 32, images.get_shape()[1:3])
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 32])

  def testCreateFullyConv(self):
    height, width = 7, 9
    with self.cached_session():
      images = tf.random_uniform((5, height, width, 32), seed=1)
      output = conv2d_ws.conv2d(
          images, 64, images.get_shape()[1:3], padding='VALID')
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 64])
      biases = contrib_framework.get_variables_by_name('biases')[0]
      self.assertListEqual(biases.get_shape().as_list(), [64])

  def testFullyConvWithCustomGetter(self):
    height, width = 7, 9
    with self.cached_session():
      called = [0]

      def custom_getter(getter, *args, **kwargs):
        called[0] += 1
        return getter(*args, **kwargs)

      with tf.variable_scope('test', custom_getter=custom_getter):
        images = tf.random_uniform((5, height, width, 32), seed=1)
        conv2d_ws.conv2d(images, 64, images.get_shape()[1:3])
      self.assertEqual(called[0], 2)  # Custom getter called twice.

  def testCreateVerticalConv(self):
    height, width = 7, 9
    with self.cached_session():
      images = tf.random_uniform((5, height, width, 4), seed=1)
      output = conv2d_ws.conv2d(images, 32, [3, 1])
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 32])
      weights = contrib_framework.get_variables_by_name('weights')[0]
      self.assertListEqual(weights.get_shape().as_list(), [3, 1, 4, 32])
      biases = contrib_framework.get_variables_by_name('biases')[0]
      self.assertListEqual(biases.get_shape().as_list(), [32])

  def testCreateHorizontalConv(self):
    height, width = 7, 9
    with self.cached_session():
      images = tf.random_uniform((5, height, width, 4), seed=1)
      output = conv2d_ws.conv2d(images, 32, [1, 3])
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 32])
      weights = contrib_framework.get_variables_by_name('weights')[0]
      self.assertListEqual(weights.get_shape().as_list(), [1, 3, 4, 32])

  def testCreateConvWithStride(self):
    height, width = 6, 8
    with self.cached_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = conv2d_ws.conv2d(images, 32, [3, 3], stride=2)
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(),
                           [5, height / 2, width / 2, 32])

  def testCreateConvCreatesWeightsAndBiasesVars(self):
    height, width = 7, 9
    images = tf.random_uniform((5, height, width, 3), seed=1)
    with self.cached_session():
      self.assertFalse(contrib_framework.get_variables('conv1/weights'))
      self.assertFalse(contrib_framework.get_variables('conv1/biases'))
      conv2d_ws.conv2d(images, 32, [3, 3], scope='conv1')
      self.assertTrue(contrib_framework.get_variables('conv1/weights'))
      self.assertTrue(contrib_framework.get_variables('conv1/biases'))

  def testCreateConvWithScope(self):
    height, width = 7, 9
    with self.cached_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = conv2d_ws.conv2d(images, 32, [3, 3], scope='conv1')
      self.assertEqual(output.op.name, 'conv1/Relu')

  def testCreateConvWithCollection(self):
    height, width = 7, 9
    images = tf.random_uniform((5, height, width, 3), seed=1)
    with tf.name_scope('fe'):
      conv = conv2d_ws.conv2d(
          images, 32, [3, 3], outputs_collections='outputs', scope='Conv')
    output_collected = tf.get_collection('outputs')[0]
    self.assertEqual(output_collected.aliases, ['Conv'])
    self.assertEqual(output_collected, conv)

  def testCreateConvWithoutActivation(self):
    height, width = 7, 9
    with self.cached_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = conv2d_ws.conv2d(images, 32, [3, 3], activation_fn=None)
      self.assertEqual(output.op.name, 'Conv/BiasAdd')

  def testCreateConvValid(self):
    height, width = 7, 9
    with self.cached_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = conv2d_ws.conv2d(images, 32, [3, 3], padding='VALID')
      self.assertListEqual(output.get_shape().as_list(), [5, 5, 7, 32])

  def testCreateConvWithWD(self):
    height, width = 7, 9
    weight_decay = 0.01
    with self.cached_session() as sess:
      images = tf.random_uniform((5, height, width, 3), seed=1)
      regularizer = contrib_layers.l2_regularizer(weight_decay)
      conv2d_ws.conv2d(images, 32, [3, 3], weights_regularizer=regularizer)
      l2_loss = tf.nn.l2_loss(
          contrib_framework.get_variables_by_name('weights')[0])
      wd = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)[0]
      self.assertEqual(wd.op.name, 'Conv/kernel/Regularizer/l2_regularizer')
      sess.run(tf.global_variables_initializer())
      self.assertAlmostEqual(sess.run(wd), weight_decay * l2_loss.eval())

  def testCreateConvNoRegularizers(self):
    height, width = 7, 9
    with self.cached_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      conv2d_ws.conv2d(images, 32, [3, 3])
      self.assertEqual(
          tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), [])

  def testReuseVars(self):
    height, width = 7, 9
    with self.cached_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      conv2d_ws.conv2d(images, 32, [3, 3], scope='conv1')
      self.assertEqual(len(contrib_framework.get_variables()), 2)
      conv2d_ws.conv2d(images, 32, [3, 3], scope='conv1', reuse=True)
      self.assertEqual(len(contrib_framework.get_variables()), 2)

  def testNonReuseVars(self):
    height, width = 7, 9
    with self.cached_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      conv2d_ws.conv2d(images, 32, [3, 3])
      self.assertEqual(len(contrib_framework.get_variables()), 2)
      conv2d_ws.conv2d(images, 32, [3, 3])
      self.assertEqual(len(contrib_framework.get_variables()), 4)

  def testReuseConvWithWD(self):
    height, width = 7, 9
    with self.cached_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      weight_decay = contrib_layers.l2_regularizer(0.01)
      with contrib_framework.arg_scope([conv2d_ws.conv2d],
                                       weights_regularizer=weight_decay):
        conv2d_ws.conv2d(images, 32, [3, 3], scope='conv1')
        self.assertEqual(len(contrib_framework.get_variables()), 2)
        self.assertEqual(
            len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), 1)
        conv2d_ws.conv2d(images, 32, [3, 3], scope='conv1', reuse=True)
        self.assertEqual(len(contrib_framework.get_variables()), 2)
        self.assertEqual(
            len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), 1)

  def testConvWithBatchNorm(self):
    height, width = 7, 9
    with self.cached_session():
      images = tf.random_uniform((5, height, width, 32), seed=1)
      with contrib_framework.arg_scope([conv2d_ws.conv2d],
                                       normalizer_fn=contrib_layers.batch_norm,
                                       normalizer_params={'decay': 0.9}):
        net = conv2d_ws.conv2d(images, 32, [3, 3])
        net = conv2d_ws.conv2d(net, 32, [3, 3])
      self.assertEqual(len(contrib_framework.get_variables()), 8)
      self.assertEqual(
          len(contrib_framework.get_variables('Conv/BatchNorm')), 3)
      self.assertEqual(
          len(contrib_framework.get_variables('Conv_1/BatchNorm')), 3)

  def testReuseConvWithBatchNorm(self):
    height, width = 7, 9
    with self.cached_session():
      images = tf.random_uniform((5, height, width, 32), seed=1)
      with contrib_framework.arg_scope([conv2d_ws.conv2d],
                                       normalizer_fn=contrib_layers.batch_norm,
                                       normalizer_params={'decay': 0.9}):
        net = conv2d_ws.conv2d(images, 32, [3, 3], scope='Conv')
        net = conv2d_ws.conv2d(net, 32, [3, 3], scope='Conv', reuse=True)
      self.assertEqual(len(contrib_framework.get_variables()), 4)
      self.assertEqual(
          len(contrib_framework.get_variables('Conv/BatchNorm')), 3)
      self.assertEqual(
          len(contrib_framework.get_variables('Conv_1/BatchNorm')), 0)

  def testCreateConvCreatesWeightsAndBiasesVarsWithRateTwo(self):
    height, width = 7, 9
    images = tf.random_uniform((5, height, width, 3), seed=1)
    with self.cached_session():
      self.assertFalse(contrib_framework.get_variables('conv1/weights'))
      self.assertFalse(contrib_framework.get_variables('conv1/biases'))
      conv2d_ws.conv2d(images, 32, [3, 3], rate=2, scope='conv1')
      self.assertTrue(contrib_framework.get_variables('conv1/weights'))
      self.assertTrue(contrib_framework.get_variables('conv1/biases'))

  def testOutputSizeWithRateTwoSamePadding(self):
    num_filters = 32
    input_size = [5, 10, 12, 3]
    expected_size = [5, 10, 12, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = conv2d_ws.conv2d(
        images, num_filters, [3, 3], rate=2, padding='SAME')
    self.assertListEqual(list(output.get_shape().as_list()), expected_size)
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWithRateTwoValidPadding(self):
    num_filters = 32
    input_size = [5, 10, 12, 3]
    expected_size = [5, 6, 8, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = conv2d_ws.conv2d(
        images, num_filters, [3, 3], rate=2, padding='VALID')
    self.assertListEqual(list(output.get_shape().as_list()), expected_size)
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testOutputSizeWithRateTwoThreeValidPadding(self):
    num_filters = 32
    input_size = [5, 10, 12, 3]
    expected_size = [5, 6, 6, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = conv2d_ws.conv2d(
        images, num_filters, [3, 3], rate=[2, 3], padding='VALID')
    self.assertListEqual(list(output.get_shape().as_list()), expected_size)
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testDynamicOutputSizeWithRateOneValidPadding(self):
    num_filters = 32
    input_size = [5, 9, 11, 3]
    expected_size = [None, None, None, num_filters]
    expected_size_dynamic = [5, 7, 9, num_filters]

    with self.cached_session():
      images = tf.placeholder(np.float32, [None, None, None, input_size[3]])
      output = conv2d_ws.conv2d(
          images, num_filters, [3, 3], rate=1, padding='VALID')
      tf.global_variables_initializer().run()
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), expected_size)
      eval_output = output.eval({images: np.zeros(input_size, np.float32)})
      self.assertListEqual(list(eval_output.shape), expected_size_dynamic)

  def testDynamicOutputSizeWithRateOneValidPaddingNCHW(self):
    if tf.test.is_gpu_available(cuda_only=True):
      num_filters = 32
      input_size = [5, 3, 9, 11]
      expected_size = [None, num_filters, None, None]
      expected_size_dynamic = [5, num_filters, 7, 9]

      with self.session(use_gpu=True):
        images = tf.placeholder(np.float32, [None, input_size[1], None, None])
        output = conv2d_ws.conv2d(
            images,
            num_filters, [3, 3],
            rate=1,
            padding='VALID',
            data_format='NCHW')
        tf.global_variables_initializer().run()
        self.assertEqual(output.op.name, 'Conv/Relu')
        self.assertListEqual(output.get_shape().as_list(), expected_size)
        eval_output = output.eval({images: np.zeros(input_size, np.float32)})
        self.assertListEqual(list(eval_output.shape), expected_size_dynamic)

  def testDynamicOutputSizeWithRateTwoValidPadding(self):
    num_filters = 32
    input_size = [5, 9, 11, 3]
    expected_size = [None, None, None, num_filters]
    expected_size_dynamic = [5, 5, 7, num_filters]

    with self.cached_session():
      images = tf.placeholder(np.float32, [None, None, None, input_size[3]])
      output = conv2d_ws.conv2d(
          images, num_filters, [3, 3], rate=2, padding='VALID')
      tf.global_variables_initializer().run()
      self.assertEqual(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), expected_size)
      eval_output = output.eval({images: np.zeros(input_size, np.float32)})
      self.assertListEqual(list(eval_output.shape), expected_size_dynamic)

  def testWithScope(self):
    num_filters = 32
    input_size = [5, 9, 11, 3]
    expected_size = [5, 5, 7, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = conv2d_ws.conv2d(
        images, num_filters, [3, 3], rate=2, padding='VALID', scope='conv7')
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(output.op.name, 'conv7/Relu')
      self.assertListEqual(list(output.eval().shape), expected_size)

  def testWithScopeWithoutActivation(self):
    num_filters = 32
    input_size = [5, 9, 11, 3]
    expected_size = [5, 5, 7, num_filters]

    images = tf.random_uniform(input_size, seed=1)
    output = conv2d_ws.conv2d(
        images,
        num_filters, [3, 3],
        rate=2,
        padding='VALID',
        activation_fn=None,
        scope='conv7')
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(output.op.name, 'conv7/BiasAdd')
      self.assertListEqual(list(output.eval().shape), expected_size)


if __name__ == '__main__':
  tf.test.main()
