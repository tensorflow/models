# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Tests for slim.ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops

from inception.slim import ops
from inception.slim import scopes
from inception.slim import variables


class ConvTest(tf.test.TestCase):

  def testCreateConv(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.conv2d(images, 32, [3, 3])
      self.assertEquals(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 32])

  def testCreateSquareConv(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.conv2d(images, 32, 3)
      self.assertEquals(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 32])

  def testCreateConvWithTensorShape(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.conv2d(images, 32, images.get_shape()[1:3])
      self.assertEquals(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 32])

  def testCreateFullyConv(self):
    height, width = 6, 6
    with self.test_session():
      images = tf.random_uniform((5, height, width, 32), seed=1)
      output = ops.conv2d(images, 64, images.get_shape()[1:3], padding='VALID')
      self.assertEquals(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 64])

  def testCreateVerticalConv(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.conv2d(images, 32, [3, 1])
      self.assertEquals(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(),
                           [5, height, width, 32])

  def testCreateHorizontalConv(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.conv2d(images, 32, [1, 3])
      self.assertEquals(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(),
                           [5, height, width, 32])

  def testCreateConvWithStride(self):
    height, width = 6, 6
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.conv2d(images, 32, [3, 3], stride=2)
      self.assertEquals(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(),
                           [5, height/2, width/2, 32])

  def testCreateConvCreatesWeightsAndBiasesVars(self):
    height, width = 3, 3
    images = tf.random_uniform((5, height, width, 3), seed=1)
    with self.test_session():
      self.assertFalse(variables.get_variables('conv1/weights'))
      self.assertFalse(variables.get_variables('conv1/biases'))
      ops.conv2d(images, 32, [3, 3], scope='conv1')
      self.assertTrue(variables.get_variables('conv1/weights'))
      self.assertTrue(variables.get_variables('conv1/biases'))

  def testCreateConvWithScope(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.conv2d(images, 32, [3, 3], scope='conv1')
      self.assertEquals(output.op.name, 'conv1/Relu')

  def testCreateConvWithoutActivation(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.conv2d(images, 32, [3, 3], activation=None)
      self.assertEquals(output.op.name, 'Conv/BiasAdd')

  def testCreateConvValid(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.conv2d(images, 32, [3, 3], padding='VALID')
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 32])

  def testCreateConvWithWD(self):
    height, width = 3, 3
    with self.test_session() as sess:
      images = tf.random_uniform((5, height, width, 3), seed=1)
      ops.conv2d(images, 32, [3, 3], weight_decay=0.01)
      wd = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)[0]
      self.assertEquals(wd.op.name,
                        'Conv/weights/Regularizer/L2Regularizer/value')
      sess.run(tf.initialize_all_variables())
      self.assertTrue(sess.run(wd) <= 0.01)

  def testCreateConvWithoutWD(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      ops.conv2d(images, 32, [3, 3], weight_decay=0)
      self.assertEquals(
          tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), [])

  def testReuseVars(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      ops.conv2d(images, 32, [3, 3], scope='conv1')
      self.assertEquals(len(variables.get_variables()), 2)
      ops.conv2d(images, 32, [3, 3], scope='conv1', reuse=True)
      self.assertEquals(len(variables.get_variables()), 2)

  def testNonReuseVars(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      ops.conv2d(images, 32, [3, 3])
      self.assertEquals(len(variables.get_variables()), 2)
      ops.conv2d(images, 32, [3, 3])
      self.assertEquals(len(variables.get_variables()), 4)

  def testReuseConvWithWD(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      ops.conv2d(images, 32, [3, 3], weight_decay=0.01, scope='conv1')
      self.assertEquals(len(variables.get_variables()), 2)
      self.assertEquals(
          len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), 1)
      ops.conv2d(images, 32, [3, 3], weight_decay=0.01, scope='conv1',
                 reuse=True)
      self.assertEquals(len(variables.get_variables()), 2)
      self.assertEquals(
          len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), 1)

  def testConvWithBatchNorm(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 32), seed=1)
      with scopes.arg_scope([ops.conv2d], batch_norm_params={'decay': 0.9}):
        net = ops.conv2d(images, 32, [3, 3])
        net = ops.conv2d(net, 32, [3, 3])
      self.assertEquals(len(variables.get_variables()), 8)
      self.assertEquals(len(variables.get_variables('Conv/BatchNorm')), 3)
      self.assertEquals(len(variables.get_variables('Conv_1/BatchNorm')), 3)

  def testReuseConvWithBatchNorm(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 32), seed=1)
      with scopes.arg_scope([ops.conv2d], batch_norm_params={'decay': 0.9}):
        net = ops.conv2d(images, 32, [3, 3], scope='Conv')
        net = ops.conv2d(net, 32, [3, 3], scope='Conv', reuse=True)
      self.assertEquals(len(variables.get_variables()), 4)
      self.assertEquals(len(variables.get_variables('Conv/BatchNorm')), 3)
      self.assertEquals(len(variables.get_variables('Conv_1/BatchNorm')), 0)


class FCTest(tf.test.TestCase):

  def testCreateFC(self):
    height, width = 3, 3
    with self.test_session():
      inputs = tf.random_uniform((5, height * width * 3), seed=1)
      output = ops.fc(inputs, 32)
      self.assertEquals(output.op.name, 'FC/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, 32])

  def testCreateFCWithScope(self):
    height, width = 3, 3
    with self.test_session():
      inputs = tf.random_uniform((5, height * width * 3), seed=1)
      output = ops.fc(inputs, 32, scope='fc1')
      self.assertEquals(output.op.name, 'fc1/Relu')

  def testCreateFcCreatesWeightsAndBiasesVars(self):
    height, width = 3, 3
    inputs = tf.random_uniform((5, height * width * 3), seed=1)
    with self.test_session():
      self.assertFalse(variables.get_variables('fc1/weights'))
      self.assertFalse(variables.get_variables('fc1/biases'))
      ops.fc(inputs, 32, scope='fc1')
      self.assertTrue(variables.get_variables('fc1/weights'))
      self.assertTrue(variables.get_variables('fc1/biases'))

  def testReuseVars(self):
    height, width = 3, 3
    inputs = tf.random_uniform((5, height * width * 3), seed=1)
    with self.test_session():
      ops.fc(inputs, 32, scope='fc1')
      self.assertEquals(len(variables.get_variables('fc1')), 2)
      ops.fc(inputs, 32, scope='fc1', reuse=True)
      self.assertEquals(len(variables.get_variables('fc1')), 2)

  def testNonReuseVars(self):
    height, width = 3, 3
    inputs = tf.random_uniform((5, height * width * 3), seed=1)
    with self.test_session():
      ops.fc(inputs, 32)
      self.assertEquals(len(variables.get_variables('FC')), 2)
      ops.fc(inputs, 32)
      self.assertEquals(len(variables.get_variables('FC')), 4)

  def testCreateFCWithoutActivation(self):
    height, width = 3, 3
    with self.test_session():
      inputs = tf.random_uniform((5, height * width * 3), seed=1)
      output = ops.fc(inputs, 32, activation=None)
      self.assertEquals(output.op.name, 'FC/xw_plus_b')

  def testCreateFCWithWD(self):
    height, width = 3, 3
    with self.test_session() as sess:
      inputs = tf.random_uniform((5, height * width * 3), seed=1)
      ops.fc(inputs, 32, weight_decay=0.01)
      wd = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)[0]
      self.assertEquals(wd.op.name,
                        'FC/weights/Regularizer/L2Regularizer/value')
      sess.run(tf.initialize_all_variables())
      self.assertTrue(sess.run(wd) <= 0.01)

  def testCreateFCWithoutWD(self):
    height, width = 3, 3
    with self.test_session():
      inputs = tf.random_uniform((5, height * width * 3), seed=1)
      ops.fc(inputs, 32, weight_decay=0)
      self.assertEquals(
          tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), [])

  def testReuseFCWithWD(self):
    height, width = 3, 3
    with self.test_session():
      inputs = tf.random_uniform((5, height * width * 3), seed=1)
      ops.fc(inputs, 32, weight_decay=0.01, scope='fc')
      self.assertEquals(len(variables.get_variables()), 2)
      self.assertEquals(
          len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), 1)
      ops.fc(inputs, 32, weight_decay=0.01, scope='fc', reuse=True)
      self.assertEquals(len(variables.get_variables()), 2)
      self.assertEquals(
          len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), 1)

  def testFCWithBatchNorm(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height * width * 3), seed=1)
      with scopes.arg_scope([ops.fc], batch_norm_params={}):
        net = ops.fc(images, 27)
        net = ops.fc(net, 27)
      self.assertEquals(len(variables.get_variables()), 8)
      self.assertEquals(len(variables.get_variables('FC/BatchNorm')), 3)
      self.assertEquals(len(variables.get_variables('FC_1/BatchNorm')), 3)

  def testReuseFCWithBatchNorm(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height * width * 3), seed=1)
      with scopes.arg_scope([ops.fc], batch_norm_params={'decay': 0.9}):
        net = ops.fc(images, 27, scope='fc1')
        net = ops.fc(net, 27, scope='fc1', reuse=True)
      self.assertEquals(len(variables.get_variables()), 4)
      self.assertEquals(len(variables.get_variables('fc1/BatchNorm')), 3)


class MaxPoolTest(tf.test.TestCase):

  def testCreateMaxPool(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.max_pool(images, [3, 3])
      self.assertEquals(output.op.name, 'MaxPool/MaxPool')
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 3])

  def testCreateSquareMaxPool(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.max_pool(images, 3)
      self.assertEquals(output.op.name, 'MaxPool/MaxPool')
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 3])

  def testCreateMaxPoolWithScope(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.max_pool(images, [3, 3], scope='pool1')
      self.assertEquals(output.op.name, 'pool1/MaxPool')

  def testCreateMaxPoolSAME(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.max_pool(images, [3, 3], padding='SAME')
      self.assertListEqual(output.get_shape().as_list(), [5, 2, 2, 3])

  def testCreateMaxPoolStrideSAME(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.max_pool(images, [3, 3], stride=1, padding='SAME')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 3])

  def testGlobalMaxPool(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.max_pool(images, images.get_shape()[1:3], stride=1)
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 3])


class AvgPoolTest(tf.test.TestCase):

  def testCreateAvgPool(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.avg_pool(images, [3, 3])
      self.assertEquals(output.op.name, 'AvgPool/AvgPool')
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 3])

  def testCreateSquareAvgPool(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.avg_pool(images, 3)
      self.assertEquals(output.op.name, 'AvgPool/AvgPool')
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 3])

  def testCreateAvgPoolWithScope(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.avg_pool(images, [3, 3], scope='pool1')
      self.assertEquals(output.op.name, 'pool1/AvgPool')

  def testCreateAvgPoolSAME(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.avg_pool(images, [3, 3], padding='SAME')
      self.assertListEqual(output.get_shape().as_list(), [5, 2, 2, 3])

  def testCreateAvgPoolStrideSAME(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.avg_pool(images, [3, 3], stride=1, padding='SAME')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 3])

  def testGlobalAvgPool(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.avg_pool(images, images.get_shape()[1:3], stride=1)
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 3])


class OneHotEncodingTest(tf.test.TestCase):

  def testOneHotEncodingCreate(self):
    with self.test_session():
      labels = tf.constant([0, 1, 2])
      output = ops.one_hot_encoding(labels, num_classes=3)
      self.assertEquals(output.op.name, 'OneHotEncoding/SparseToDense')
      self.assertListEqual(output.get_shape().as_list(), [3, 3])

  def testOneHotEncoding(self):
    with self.test_session():
      labels = tf.constant([0, 1, 2])
      one_hot_labels = tf.constant([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])
      output = ops.one_hot_encoding(labels, num_classes=3)
      self.assertAllClose(output.eval(), one_hot_labels.eval())


class DropoutTest(tf.test.TestCase):

  def testCreateDropout(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.dropout(images)
      self.assertEquals(output.op.name, 'Dropout/dropout/mul_1')
      output.get_shape().assert_is_compatible_with(images.get_shape())

  def testCreateDropoutNoTraining(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      output = ops.dropout(images, is_training=False)
      self.assertEquals(output, images)


class FlattenTest(tf.test.TestCase):

  def testFlatten4D(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      output = ops.flatten(images)
      self.assertEquals(output.get_shape().num_elements(),
                        images.get_shape().num_elements())
      self.assertEqual(output.get_shape()[0], images.get_shape()[0])

  def testFlatten3D(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width), seed=1, name='images')
      output = ops.flatten(images)
      self.assertEquals(output.get_shape().num_elements(),
                        images.get_shape().num_elements())
      self.assertEqual(output.get_shape()[0], images.get_shape()[0])

  def testFlattenBatchSize(self):
    height, width = 3, 3
    with self.test_session() as sess:
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      inputs = tf.placeholder(tf.int32, (None, height, width, 3))
      output = ops.flatten(inputs)
      self.assertEquals(output.get_shape().as_list(),
                        [None, height * width * 3])
      output = sess.run(output, {inputs: images.eval()})
      self.assertEquals(output.size,
                        images.get_shape().num_elements())
      self.assertEqual(output.shape[0], images.get_shape()[0])


class BatchNormTest(tf.test.TestCase):

  def testCreateOp(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = ops.batch_norm(images)
      self.assertTrue(output.op.name.startswith('BatchNorm/batchnorm'))
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 3])

  def testCreateVariables(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      ops.batch_norm(images)
      beta = variables.get_variables_by_name('beta')[0]
      self.assertEquals(beta.op.name, 'BatchNorm/beta')
      gamma = variables.get_variables_by_name('gamma')
      self.assertEquals(gamma, [])
      moving_mean = tf.moving_average_variables()[0]
      moving_variance = tf.moving_average_variables()[1]
      self.assertEquals(moving_mean.op.name, 'BatchNorm/moving_mean')
      self.assertEquals(moving_variance.op.name, 'BatchNorm/moving_variance')

  def testCreateVariablesWithScale(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      ops.batch_norm(images, scale=True)
      beta = variables.get_variables_by_name('beta')[0]
      gamma = variables.get_variables_by_name('gamma')[0]
      self.assertEquals(beta.op.name, 'BatchNorm/beta')
      self.assertEquals(gamma.op.name, 'BatchNorm/gamma')
      moving_mean = tf.moving_average_variables()[0]
      moving_variance = tf.moving_average_variables()[1]
      self.assertEquals(moving_mean.op.name, 'BatchNorm/moving_mean')
      self.assertEquals(moving_variance.op.name, 'BatchNorm/moving_variance')

  def testCreateVariablesWithoutCenterWithScale(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      ops.batch_norm(images, center=False, scale=True)
      beta = variables.get_variables_by_name('beta')
      self.assertEquals(beta, [])
      gamma = variables.get_variables_by_name('gamma')[0]
      self.assertEquals(gamma.op.name, 'BatchNorm/gamma')
      moving_mean = tf.moving_average_variables()[0]
      moving_variance = tf.moving_average_variables()[1]
      self.assertEquals(moving_mean.op.name, 'BatchNorm/moving_mean')
      self.assertEquals(moving_variance.op.name, 'BatchNorm/moving_variance')

  def testCreateVariablesWithoutCenterWithoutScale(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      ops.batch_norm(images, center=False, scale=False)
      beta = variables.get_variables_by_name('beta')
      self.assertEquals(beta, [])
      gamma = variables.get_variables_by_name('gamma')
      self.assertEquals(gamma, [])
      moving_mean = tf.moving_average_variables()[0]
      moving_variance = tf.moving_average_variables()[1]
      self.assertEquals(moving_mean.op.name, 'BatchNorm/moving_mean')
      self.assertEquals(moving_variance.op.name, 'BatchNorm/moving_variance')

  def testMovingAverageVariables(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      ops.batch_norm(images, scale=True)
      moving_mean = tf.moving_average_variables()[0]
      moving_variance = tf.moving_average_variables()[1]
      self.assertEquals(moving_mean.op.name, 'BatchNorm/moving_mean')
      self.assertEquals(moving_variance.op.name, 'BatchNorm/moving_variance')

  def testUpdateOps(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      ops.batch_norm(images)
      update_ops = tf.get_collection(ops.UPDATE_OPS_COLLECTION)
      update_moving_mean = update_ops[0]
      update_moving_variance = update_ops[1]
      self.assertEquals(update_moving_mean.op.name,
                        'BatchNorm/AssignMovingAvg')
      self.assertEquals(update_moving_variance.op.name,
                        'BatchNorm/AssignMovingAvg_1')

  def testReuseVariables(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      ops.batch_norm(images, scale=True, scope='bn')
      ops.batch_norm(images, scale=True, scope='bn', reuse=True)
      beta = variables.get_variables_by_name('beta')
      gamma = variables.get_variables_by_name('gamma')
      self.assertEquals(len(beta), 1)
      self.assertEquals(len(gamma), 1)
      moving_vars = tf.get_collection('moving_vars')
      self.assertEquals(len(moving_vars), 2)

  def testReuseUpdateOps(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      ops.batch_norm(images, scope='bn')
      self.assertEquals(len(tf.get_collection(ops.UPDATE_OPS_COLLECTION)), 2)
      ops.batch_norm(images, scope='bn', reuse=True)
      self.assertEquals(len(tf.get_collection(ops.UPDATE_OPS_COLLECTION)), 4)

  def testCreateMovingVars(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      _ = ops.batch_norm(images, moving_vars='moving_vars')
      moving_mean = tf.get_collection('moving_vars',
                                      'BatchNorm/moving_mean')
      self.assertEquals(len(moving_mean), 1)
      self.assertEquals(moving_mean[0].op.name, 'BatchNorm/moving_mean')
      moving_variance = tf.get_collection('moving_vars',
                                          'BatchNorm/moving_variance')
      self.assertEquals(len(moving_variance), 1)
      self.assertEquals(moving_variance[0].op.name, 'BatchNorm/moving_variance')

  def testComputeMovingVars(self):
    height, width = 3, 3
    with self.test_session() as sess:
      image_shape = (10, height, width, 3)
      image_values = np.random.rand(*image_shape)
      expected_mean = np.mean(image_values, axis=(0, 1, 2))
      expected_var = np.var(image_values, axis=(0, 1, 2))
      images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
      output = ops.batch_norm(images, decay=0.1)
      update_ops = tf.get_collection(ops.UPDATE_OPS_COLLECTION)
      with tf.control_dependencies(update_ops):
        barrier = tf.no_op(name='gradient_barrier')
        output = control_flow_ops.with_dependencies([barrier], output)
      # Initialize all variables
      sess.run(tf.initialize_all_variables())
      moving_mean = variables.get_variables('BatchNorm/moving_mean')[0]
      moving_variance = variables.get_variables('BatchNorm/moving_variance')[0]
      mean, variance = sess.run([moving_mean, moving_variance])
      # After initialization moving_mean == 0 and moving_variance == 1.
      self.assertAllClose(mean, [0] * 3)
      self.assertAllClose(variance, [1] * 3)
      for _ in range(10):
        sess.run([output])
      mean = moving_mean.eval()
      variance = moving_variance.eval()
      # After 10 updates with decay 0.1 moving_mean == expected_mean and
      # moving_variance == expected_var.
      self.assertAllClose(mean, expected_mean)
      self.assertAllClose(variance, expected_var)

  def testEvalMovingVars(self):
    height, width = 3, 3
    with self.test_session() as sess:
      image_shape = (10, height, width, 3)
      image_values = np.random.rand(*image_shape)
      expected_mean = np.mean(image_values, axis=(0, 1, 2))
      expected_var = np.var(image_values, axis=(0, 1, 2))
      images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
      output = ops.batch_norm(images, decay=0.1, is_training=False)
      update_ops = tf.get_collection(ops.UPDATE_OPS_COLLECTION)
      with tf.control_dependencies(update_ops):
        barrier = tf.no_op(name='gradient_barrier')
        output = control_flow_ops.with_dependencies([barrier], output)
      # Initialize all variables
      sess.run(tf.initialize_all_variables())
      moving_mean = variables.get_variables('BatchNorm/moving_mean')[0]
      moving_variance = variables.get_variables('BatchNorm/moving_variance')[0]
      mean, variance = sess.run([moving_mean, moving_variance])
      # After initialization moving_mean == 0 and moving_variance == 1.
      self.assertAllClose(mean, [0] * 3)
      self.assertAllClose(variance, [1] * 3)
      # Simulate assigment from saver restore.
      init_assigns = [tf.assign(moving_mean, expected_mean),
                      tf.assign(moving_variance, expected_var)]
      sess.run(init_assigns)
      for _ in range(10):
        sess.run([output], {images: np.random.rand(*image_shape)})
      mean = moving_mean.eval()
      variance = moving_variance.eval()
      # Although we feed different images, the moving_mean and moving_variance
      # shouldn't change.
      self.assertAllClose(mean, expected_mean)
      self.assertAllClose(variance, expected_var)

  def testReuseVars(self):
    height, width = 3, 3
    with self.test_session() as sess:
      image_shape = (10, height, width, 3)
      image_values = np.random.rand(*image_shape)
      expected_mean = np.mean(image_values, axis=(0, 1, 2))
      expected_var = np.var(image_values, axis=(0, 1, 2))
      images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
      output = ops.batch_norm(images, decay=0.1, is_training=False)
      update_ops = tf.get_collection(ops.UPDATE_OPS_COLLECTION)
      with tf.control_dependencies(update_ops):
        barrier = tf.no_op(name='gradient_barrier')
        output = control_flow_ops.with_dependencies([barrier], output)
      # Initialize all variables
      sess.run(tf.initialize_all_variables())
      moving_mean = variables.get_variables('BatchNorm/moving_mean')[0]
      moving_variance = variables.get_variables('BatchNorm/moving_variance')[0]
      mean, variance = sess.run([moving_mean, moving_variance])
      # After initialization moving_mean == 0 and moving_variance == 1.
      self.assertAllClose(mean, [0] * 3)
      self.assertAllClose(variance, [1] * 3)
      # Simulate assigment from saver restore.
      init_assigns = [tf.assign(moving_mean, expected_mean),
                      tf.assign(moving_variance, expected_var)]
      sess.run(init_assigns)
      for _ in range(10):
        sess.run([output], {images: np.random.rand(*image_shape)})
      mean = moving_mean.eval()
      variance = moving_variance.eval()
      # Although we feed different images, the moving_mean and moving_variance
      # shouldn't change.
      self.assertAllClose(mean, expected_mean)
      self.assertAllClose(variance, expected_var)

if __name__ == '__main__':
  tf.test.main()
