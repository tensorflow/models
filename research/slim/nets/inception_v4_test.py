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
"""Tests for slim.inception_v4."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tf_slim as slim

from nets import inception


class InceptionTest(tf.test.TestCase):

  def testBuildLogits(self):
    batch_size = 5
    height, width = 299, 299
    num_classes = 1000
    inputs = tf.random.uniform((batch_size, height, width, 3))
    logits, end_points = inception.inception_v4(inputs, num_classes)
    auxlogits = end_points['AuxLogits']
    predictions = end_points['Predictions']
    self.assertTrue(auxlogits.op.name.startswith('InceptionV4/AuxLogits'))
    self.assertListEqual(auxlogits.get_shape().as_list(),
                         [batch_size, num_classes])
    self.assertTrue(logits.op.name.startswith('InceptionV4/Logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])
    self.assertTrue(predictions.op.name.startswith(
        'InceptionV4/Logits/Predictions'))
    self.assertListEqual(predictions.get_shape().as_list(),
                         [batch_size, num_classes])

  def testBuildPreLogitsNetwork(self):
    batch_size = 5
    height, width = 299, 299
    num_classes = None
    inputs = tf.random.uniform((batch_size, height, width, 3))
    net, end_points = inception.inception_v4(inputs, num_classes)
    self.assertTrue(net.op.name.startswith('InceptionV4/Logits/AvgPool'))
    self.assertListEqual(net.get_shape().as_list(), [batch_size, 1, 1, 1536])
    self.assertFalse('Logits' in end_points)
    self.assertFalse('Predictions' in end_points)

  def testBuildWithoutAuxLogits(self):
    batch_size = 5
    height, width = 299, 299
    num_classes = 1000
    inputs = tf.random.uniform((batch_size, height, width, 3))
    logits, endpoints = inception.inception_v4(inputs, num_classes,
                                               create_aux_logits=False)
    self.assertFalse('AuxLogits' in endpoints)
    self.assertTrue(logits.op.name.startswith('InceptionV4/Logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])

  def testAllEndPointsShapes(self):
    batch_size = 5
    height, width = 299, 299
    num_classes = 1000
    inputs = tf.random.uniform((batch_size, height, width, 3))
    _, end_points = inception.inception_v4(inputs, num_classes)
    endpoints_shapes = {'Conv2d_1a_3x3': [batch_size, 149, 149, 32],
                        'Conv2d_2a_3x3': [batch_size, 147, 147, 32],
                        'Conv2d_2b_3x3': [batch_size, 147, 147, 64],
                        'Mixed_3a': [batch_size, 73, 73, 160],
                        'Mixed_4a': [batch_size, 71, 71, 192],
                        'Mixed_5a': [batch_size, 35, 35, 384],
                        # 4 x Inception-A blocks
                        'Mixed_5b': [batch_size, 35, 35, 384],
                        'Mixed_5c': [batch_size, 35, 35, 384],
                        'Mixed_5d': [batch_size, 35, 35, 384],
                        'Mixed_5e': [batch_size, 35, 35, 384],
                        # Reduction-A block
                        'Mixed_6a': [batch_size, 17, 17, 1024],
                        # 7 x Inception-B blocks
                        'Mixed_6b': [batch_size, 17, 17, 1024],
                        'Mixed_6c': [batch_size, 17, 17, 1024],
                        'Mixed_6d': [batch_size, 17, 17, 1024],
                        'Mixed_6e': [batch_size, 17, 17, 1024],
                        'Mixed_6f': [batch_size, 17, 17, 1024],
                        'Mixed_6g': [batch_size, 17, 17, 1024],
                        'Mixed_6h': [batch_size, 17, 17, 1024],
                        # Reduction-A block
                        'Mixed_7a': [batch_size, 8, 8, 1536],
                        # 3 x Inception-C blocks
                        'Mixed_7b': [batch_size, 8, 8, 1536],
                        'Mixed_7c': [batch_size, 8, 8, 1536],
                        'Mixed_7d': [batch_size, 8, 8, 1536],
                        # Logits and predictions
                        'AuxLogits': [batch_size, num_classes],
                        'global_pool': [batch_size, 1, 1, 1536],
                        'PreLogitsFlatten': [batch_size, 1536],
                        'Logits': [batch_size, num_classes],
                        'Predictions': [batch_size, num_classes]}
    self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
    for endpoint_name in endpoints_shapes:
      expected_shape = endpoints_shapes[endpoint_name]
      self.assertTrue(endpoint_name in end_points)
      self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                           expected_shape)

  def testBuildBaseNetwork(self):
    batch_size = 5
    height, width = 299, 299
    inputs = tf.random.uniform((batch_size, height, width, 3))
    net, end_points = inception.inception_v4_base(inputs)
    self.assertTrue(net.op.name.startswith(
        'InceptionV4/Mixed_7d'))
    self.assertListEqual(net.get_shape().as_list(), [batch_size, 8, 8, 1536])
    expected_endpoints = [
        'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'Mixed_3a',
        'Mixed_4a', 'Mixed_5a', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d',
        'Mixed_5e', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d',
        'Mixed_6e', 'Mixed_6f', 'Mixed_6g', 'Mixed_6h', 'Mixed_7a',
        'Mixed_7b', 'Mixed_7c', 'Mixed_7d']
    self.assertItemsEqual(end_points.keys(), expected_endpoints)
    for name, op in end_points.items():
      self.assertTrue(op.name.startswith('InceptionV4/' + name))

  def testBuildOnlyUpToFinalEndpoint(self):
    batch_size = 5
    height, width = 299, 299
    all_endpoints = [
        'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'Mixed_3a',
        'Mixed_4a', 'Mixed_5a', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d',
        'Mixed_5e', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d',
        'Mixed_6e', 'Mixed_6f', 'Mixed_6g', 'Mixed_6h', 'Mixed_7a',
        'Mixed_7b', 'Mixed_7c', 'Mixed_7d']
    for index, endpoint in enumerate(all_endpoints):
      with tf.Graph().as_default():
        inputs = tf.random.uniform((batch_size, height, width, 3))
        out_tensor, end_points = inception.inception_v4_base(
            inputs, final_endpoint=endpoint)
        self.assertTrue(out_tensor.op.name.startswith(
            'InceptionV4/' + endpoint))
        self.assertItemsEqual(all_endpoints[:index + 1], end_points.keys())

  def testVariablesSetDevice(self):
    batch_size = 5
    height, width = 299, 299
    num_classes = 1000
    inputs = tf.random.uniform((batch_size, height, width, 3))
    # Force all Variables to reside on the device.
    with tf.variable_scope('on_cpu'), tf.device('/cpu:0'):
      inception.inception_v4(inputs, num_classes)
    with tf.variable_scope('on_gpu'), tf.device('/gpu:0'):
      inception.inception_v4(inputs, num_classes)
    for v in tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='on_cpu'):
      self.assertDeviceEqual(v.device, '/cpu:0')
    for v in tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='on_gpu'):
      self.assertDeviceEqual(v.device, '/gpu:0')

  def testHalfSizeImages(self):
    batch_size = 5
    height, width = 150, 150
    num_classes = 1000
    inputs = tf.random.uniform((batch_size, height, width, 3))
    logits, end_points = inception.inception_v4(inputs, num_classes)
    self.assertTrue(logits.op.name.startswith('InceptionV4/Logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])
    pre_pool = end_points['Mixed_7d']
    self.assertListEqual(pre_pool.get_shape().as_list(),
                         [batch_size, 3, 3, 1536])

  def testGlobalPool(self):
    batch_size = 1
    height, width = 350, 400
    num_classes = 1000
    inputs = tf.random.uniform((batch_size, height, width, 3))
    logits, end_points = inception.inception_v4(inputs, num_classes)
    self.assertTrue(logits.op.name.startswith('InceptionV4/Logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])
    pre_pool = end_points['Mixed_7d']
    self.assertListEqual(pre_pool.get_shape().as_list(),
                         [batch_size, 9, 11, 1536])

  def testGlobalPoolUnknownImageShape(self):
    batch_size = 1
    height, width = 350, 400
    num_classes = 1000
    with self.test_session() as sess:
      inputs = tf.placeholder(tf.float32, (batch_size, None, None, 3))
      logits, end_points = inception.inception_v4(
          inputs, num_classes, create_aux_logits=False)
      self.assertTrue(logits.op.name.startswith('InceptionV4/Logits'))
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])
      pre_pool = end_points['Mixed_7d']
      images = tf.random.uniform((batch_size, height, width, 3))
      sess.run(tf.global_variables_initializer())
      logits_out, pre_pool_out = sess.run([logits, pre_pool],
                                          {inputs: images.eval()})
      self.assertTupleEqual(logits_out.shape, (batch_size, num_classes))
      self.assertTupleEqual(pre_pool_out.shape, (batch_size, 9, 11, 1536))

  def testUnknownBatchSize(self):
    batch_size = 1
    height, width = 299, 299
    num_classes = 1000
    with self.test_session() as sess:
      inputs = tf.placeholder(tf.float32, (None, height, width, 3))
      logits, _ = inception.inception_v4(inputs, num_classes)
      self.assertTrue(logits.op.name.startswith('InceptionV4/Logits'))
      self.assertListEqual(logits.get_shape().as_list(),
                           [None, num_classes])
      images = tf.random.uniform((batch_size, height, width, 3))
      sess.run(tf.global_variables_initializer())
      output = sess.run(logits, {inputs: images.eval()})
      self.assertEquals(output.shape, (batch_size, num_classes))

  def testEvaluation(self):
    batch_size = 2
    height, width = 299, 299
    num_classes = 1000
    with self.test_session() as sess:
      eval_inputs = tf.random.uniform((batch_size, height, width, 3))
      logits, _ = inception.inception_v4(eval_inputs,
                                         num_classes,
                                         is_training=False)
      predictions = tf.argmax(input=logits, axis=1)
      sess.run(tf.global_variables_initializer())
      output = sess.run(predictions)
      self.assertEquals(output.shape, (batch_size,))

  def testTrainEvalWithReuse(self):
    train_batch_size = 5
    eval_batch_size = 2
    height, width = 150, 150
    num_classes = 1000
    with self.test_session() as sess:
      train_inputs = tf.random.uniform((train_batch_size, height, width, 3))
      inception.inception_v4(train_inputs, num_classes)
      eval_inputs = tf.random.uniform((eval_batch_size, height, width, 3))
      logits, _ = inception.inception_v4(eval_inputs,
                                         num_classes,
                                         is_training=False,
                                         reuse=True)
      predictions = tf.argmax(input=logits, axis=1)
      sess.run(tf.global_variables_initializer())
      output = sess.run(predictions)
      self.assertEquals(output.shape, (eval_batch_size,))

  def testNoBatchNormScaleByDefault(self):
    height, width = 299, 299
    num_classes = 1000
    inputs = tf.placeholder(tf.float32, (1, height, width, 3))
    with slim.arg_scope(inception.inception_v4_arg_scope()):
      inception.inception_v4(inputs, num_classes, is_training=False)

    self.assertEqual(tf.global_variables('.*/BatchNorm/gamma:0$'), [])

  def testBatchNormScale(self):
    height, width = 299, 299
    num_classes = 1000
    inputs = tf.placeholder(tf.float32, (1, height, width, 3))
    with slim.arg_scope(
        inception.inception_v4_arg_scope(batch_norm_scale=True)):
      inception.inception_v4(inputs, num_classes, is_training=False)

    gamma_names = set(
        v.op.name
        for v in tf.global_variables('.*/BatchNorm/gamma:0$'))
    self.assertGreater(len(gamma_names), 0)
    for v in tf.global_variables('.*/BatchNorm/moving_mean:0$'):
      self.assertIn(v.op.name[:-len('moving_mean')] + 'gamma', gamma_names)


if __name__ == '__main__':
  tf.test.main()
