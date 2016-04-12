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
"""Tests for slim.inception."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception.slim import inception_model as inception


class InceptionTest(tf.test.TestCase):

  def testBuildLogits(self):
    batch_size = 5
    height, width = 299, 299
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = inception.inception_v3(inputs, num_classes)
      self.assertTrue(logits.op.name.startswith('logits'))
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])

  def testBuildEndPoints(self):
    batch_size = 5
    height, width = 299, 299
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      _, end_points = inception.inception_v3(inputs, num_classes)
      self.assertTrue('logits' in end_points)
      logits = end_points['logits']
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])
      self.assertTrue('aux_logits' in end_points)
      aux_logits = end_points['aux_logits']
      self.assertListEqual(aux_logits.get_shape().as_list(),
                           [batch_size, num_classes])
      pre_pool = end_points['mixed_8x8x2048b']
      self.assertListEqual(pre_pool.get_shape().as_list(),
                           [batch_size, 8, 8, 2048])

  def testVariablesSetDevice(self):
    batch_size = 5
    height, width = 299, 299
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      # Force all Variables to reside on the device.
      with tf.variable_scope('on_cpu'), tf.device('/cpu:0'):
        inception.inception_v3(inputs, num_classes)
      with tf.variable_scope('on_gpu'), tf.device('/gpu:0'):
        inception.inception_v3(inputs, num_classes)
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
      logits, end_points = inception.inception_v3(inputs, num_classes)
      self.assertTrue(logits.op.name.startswith('logits'))
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])
      pre_pool = end_points['mixed_8x8x2048b']
      self.assertListEqual(pre_pool.get_shape().as_list(),
                           [batch_size, 3, 3, 2048])

  def testUnknowBatchSize(self):
    batch_size = 1
    height, width = 299, 299
    num_classes = 1000
    with self.test_session() as sess:
      inputs = tf.placeholder(tf.float32, (None, height, width, 3))
      logits, _ = inception.inception_v3(inputs, num_classes)
      self.assertTrue(logits.op.name.startswith('logits'))
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
      logits, _ = inception.inception_v3(eval_inputs, num_classes,
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
      inception.inception_v3(train_inputs, num_classes)
      tf.get_variable_scope().reuse_variables()
      eval_inputs = tf.random_uniform((eval_batch_size, height, width, 3))
      logits, _ = inception.inception_v3(eval_inputs, num_classes,
                                         is_training=False)
      predictions = tf.argmax(logits, 1)
      sess.run(tf.initialize_all_variables())
      output = sess.run(predictions)
      self.assertEquals(output.shape, (eval_batch_size,))


if __name__ == '__main__':
  tf.test.main()
