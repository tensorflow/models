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
import google3

import tensorflow as tf

from nets import inception


class InceptionTest(tf.test.TestCase):

  def testBuildLogits(self):
    batch_size = 5
    height, width = 299, 299
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = inception.inception_v4(inputs, num_classes)
      self.assertTrue(logits.op.name.startswith('InceptionV4/Logits'))
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])

  def testAllEndPointsShapes(self):
    batch_size = 5
    height, width = 299, 299
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
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
                          'PreLogitsFlatten': [batch_size, 1536],
                          'Logits': [batch_size, num_classes],
                          'Predictions': [batch_size, num_classes]}
      self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
      for endpoint_name in endpoints_shapes:
        expected_shape = endpoints_shapes[endpoint_name]
        self.assertTrue(endpoint_name in end_points)
        self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                             expected_shape)

  def testAllEndPointsOpNames(self):
    batch_size = 5
    height, width = 299, 299
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      _, end_points = inception.inception_v4(inputs, num_classes)
      for name, op in end_points.iteritems():
        if name == 'PreLogitsFlatten':
          self.assertTrue(op.op.name.startswith(
              'InceptionV4/Logits/PreLogitsFlatten'))
        elif name == 'Predictions':
          self.assertEquals(op.op.name, 'InceptionV4/Logits/Predictions')
        else:
          self.assertTrue(op.op.name.startswith('InceptionV4/' + name))

  def testVariablesSetDevice(self):
    batch_size = 5
    height, width = 299, 299
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      # Force all Variables to reside on the device.
      with tf.variable_scope('on_cpu'), tf.device('/cpu:0'):
        inception.inception_v4(inputs, num_classes)
      with tf.variable_scope('on_gpu'), tf.device('/gpu:0'):
        inception.inception_v4(inputs, num_classes)
      for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope='on_cpu'):
        self.assertDeviceEqual(v.device, '/cpu:0')
      for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope='on_gpu'):
        self.assertDeviceEqual(v.device, '/gpu:0')

  def testHalfSizeImages(self):
    batch_size = 5
    height, width = 150, 150
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, end_points = inception.inception_v4(inputs, num_classes)
      self.assertTrue(logits.op.name.startswith('InceptionV4/Logits'))
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])
      pre_pool = end_points['Mixed_7d']
      self.assertListEqual(pre_pool.get_shape().as_list(),
                           [batch_size, 3, 3, 1536])

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
      images = tf.random_uniform((batch_size, height, width, 3))
      sess.run(tf.initialize_all_variables())
      output = sess.run(logits, {inputs: images.eval()})
      self.assertEquals(output.shape, (batch_size, num_classes))

  def testEvaluation(self):
    batch_size = 2
    height, width = 299, 299
    num_classes = 1000
    with self.test_session() as sess:
      eval_inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = inception.inception_v4(eval_inputs,
                                         num_classes,
                                         is_training=False)
      predictions = tf.argmax(logits, 1)
      sess.run(tf.initialize_all_variables())
      output = sess.run(predictions)
      self.assertEquals(output.shape, (batch_size,))

  def testTrainEvalWithReuse(self):
    train_batch_size = 5
    eval_batch_size = 2
    height, width = 150, 150
    num_classes = 1000
    with self.test_session() as sess:
      train_inputs = tf.random_uniform((train_batch_size, height, width, 3))
      inception.inception_v4(train_inputs, num_classes)
      eval_inputs = tf.random_uniform((eval_batch_size, height, width, 3))
      logits, _ = inception.inception_v4(eval_inputs,
                                         num_classes,
                                         is_training=False,
                                         reuse=True)
      predictions = tf.argmax(logits, 1)
      sess.run(tf.initialize_all_variables())
      output = sess.run(predictions)
      self.assertEquals(output.shape, (eval_batch_size,))


if __name__ == '__main__':
  tf.test.main()
