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
"""Tests for slim.inception_resnet_v2."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import inception


class InceptionTest(tf.test.TestCase):

  def testBuildLogits(self):
    batch_size = 5
    height, width = 299, 299
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, endpoints = inception.inception_resnet_v2(inputs, num_classes)
      self.assertTrue('AuxLogits' in endpoints)
      auxlogits = endpoints['AuxLogits']
      self.assertTrue(
          auxlogits.op.name.startswith('InceptionResnetV2/AuxLogits'))
      self.assertListEqual(auxlogits.get_shape().as_list(),
                           [batch_size, num_classes])
      self.assertTrue(logits.op.name.startswith('InceptionResnetV2/Logits'))
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])

  def testBuildWithoutAuxLogits(self):
    batch_size = 5
    height, width = 299, 299
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, endpoints = inception.inception_resnet_v2(inputs, num_classes,
                                                        create_aux_logits=False)
      self.assertTrue('AuxLogits' not in endpoints)
      self.assertTrue(logits.op.name.startswith('InceptionResnetV2/Logits'))
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])

  def testBuildEndPoints(self):
    batch_size = 5
    height, width = 299, 299
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      _, end_points = inception.inception_resnet_v2(inputs, num_classes)
      self.assertTrue('Logits' in end_points)
      logits = end_points['Logits']
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])
      self.assertTrue('AuxLogits' in end_points)
      aux_logits = end_points['AuxLogits']
      self.assertListEqual(aux_logits.get_shape().as_list(),
                           [batch_size, num_classes])
      pre_pool = end_points['Conv2d_7b_1x1']
      self.assertListEqual(pre_pool.get_shape().as_list(),
                           [batch_size, 8, 8, 1536])

  def testBuildBaseNetwork(self):
    batch_size = 5
    height, width = 299, 299

    inputs = tf.random_uniform((batch_size, height, width, 3))
    net, end_points = inception.inception_resnet_v2_base(inputs)
    self.assertTrue(net.op.name.startswith('InceptionResnetV2/Conv2d_7b_1x1'))
    self.assertListEqual(net.get_shape().as_list(),
                         [batch_size, 8, 8, 1536])
    expected_endpoints = ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
                          'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3',
                          'MaxPool_5a_3x3', 'Mixed_5b', 'Mixed_6a',
                          'PreAuxLogits', 'Mixed_7a', 'Conv2d_7b_1x1']
    self.assertItemsEqual(end_points.keys(), expected_endpoints)

  def testBuildOnlyUptoFinalEndpoint(self):
    batch_size = 5
    height, width = 299, 299
    endpoints = ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
                 'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3',
                 'MaxPool_5a_3x3', 'Mixed_5b', 'Mixed_6a',
                 'PreAuxLogits', 'Mixed_7a', 'Conv2d_7b_1x1']
    for index, endpoint in enumerate(endpoints):
      with tf.Graph().as_default():
        inputs = tf.random_uniform((batch_size, height, width, 3))
        out_tensor, end_points = inception.inception_resnet_v2_base(
            inputs, final_endpoint=endpoint)
        if endpoint != 'PreAuxLogits':
          self.assertTrue(out_tensor.op.name.startswith(
              'InceptionResnetV2/' + endpoint))
        self.assertItemsEqual(endpoints[:index+1], end_points)

  def testBuildAndCheckAllEndPointsUptoPreAuxLogits(self):
    batch_size = 5
    height, width = 299, 299

    inputs = tf.random_uniform((batch_size, height, width, 3))
    _, end_points = inception.inception_resnet_v2_base(
        inputs, final_endpoint='PreAuxLogits')
    endpoints_shapes = {'Conv2d_1a_3x3': [5, 149, 149, 32],
                        'Conv2d_2a_3x3': [5, 147, 147, 32],
                        'Conv2d_2b_3x3': [5, 147, 147, 64],
                        'MaxPool_3a_3x3': [5, 73, 73, 64],
                        'Conv2d_3b_1x1': [5, 73, 73, 80],
                        'Conv2d_4a_3x3': [5, 71, 71, 192],
                        'MaxPool_5a_3x3': [5, 35, 35, 192],
                        'Mixed_5b': [5, 35, 35, 320],
                        'Mixed_6a': [5, 17, 17, 1088],
                        'PreAuxLogits': [5, 17, 17, 1088]
                       }

    self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
    for endpoint_name in endpoints_shapes:
      expected_shape = endpoints_shapes[endpoint_name]
      self.assertTrue(endpoint_name in end_points)
      self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                           expected_shape)

  def testBuildAndCheckAllEndPointsUptoPreAuxLogitsWithAlignedFeatureMaps(self):
    batch_size = 5
    height, width = 299, 299

    inputs = tf.random_uniform((batch_size, height, width, 3))
    _, end_points = inception.inception_resnet_v2_base(
        inputs, final_endpoint='PreAuxLogits', align_feature_maps=True)
    endpoints_shapes = {'Conv2d_1a_3x3': [5, 150, 150, 32],
                        'Conv2d_2a_3x3': [5, 150, 150, 32],
                        'Conv2d_2b_3x3': [5, 150, 150, 64],
                        'MaxPool_3a_3x3': [5, 75, 75, 64],
                        'Conv2d_3b_1x1': [5, 75, 75, 80],
                        'Conv2d_4a_3x3': [5, 75, 75, 192],
                        'MaxPool_5a_3x3': [5, 38, 38, 192],
                        'Mixed_5b': [5, 38, 38, 320],
                        'Mixed_6a': [5, 19, 19, 1088],
                        'PreAuxLogits': [5, 19, 19, 1088]
                       }

    self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
    for endpoint_name in endpoints_shapes:
      expected_shape = endpoints_shapes[endpoint_name]
      self.assertTrue(endpoint_name in end_points)
      self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                           expected_shape)

  def testBuildAndCheckAllEndPointsUptoPreAuxLogitsWithOutputStrideEight(self):
    batch_size = 5
    height, width = 299, 299

    inputs = tf.random_uniform((batch_size, height, width, 3))
    _, end_points = inception.inception_resnet_v2_base(
        inputs, final_endpoint='PreAuxLogits', output_stride=8)
    endpoints_shapes = {'Conv2d_1a_3x3': [5, 149, 149, 32],
                        'Conv2d_2a_3x3': [5, 147, 147, 32],
                        'Conv2d_2b_3x3': [5, 147, 147, 64],
                        'MaxPool_3a_3x3': [5, 73, 73, 64],
                        'Conv2d_3b_1x1': [5, 73, 73, 80],
                        'Conv2d_4a_3x3': [5, 71, 71, 192],
                        'MaxPool_5a_3x3': [5, 35, 35, 192],
                        'Mixed_5b': [5, 35, 35, 320],
                        'Mixed_6a': [5, 33, 33, 1088],
                        'PreAuxLogits': [5, 33, 33, 1088]
                       }

    self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
    for endpoint_name in endpoints_shapes:
      expected_shape = endpoints_shapes[endpoint_name]
      self.assertTrue(endpoint_name in end_points)
      self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                           expected_shape)

  def testVariablesSetDevice(self):
    batch_size = 5
    height, width = 299, 299
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      # Force all Variables to reside on the device.
      with tf.variable_scope('on_cpu'), tf.device('/cpu:0'):
        inception.inception_resnet_v2(inputs, num_classes)
      with tf.variable_scope('on_gpu'), tf.device('/gpu:0'):
        inception.inception_resnet_v2(inputs, num_classes)
      for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='on_cpu'):
        self.assertDeviceEqual(v.device, '/cpu:0')
      for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='on_gpu'):
        self.assertDeviceEqual(v.device, '/gpu:0')

  def testHalfSizeImages(self):
    batch_size = 5
    height, width = 150, 150
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, end_points = inception.inception_resnet_v2(inputs, num_classes)
      self.assertTrue(logits.op.name.startswith('InceptionResnetV2/Logits'))
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])
      pre_pool = end_points['Conv2d_7b_1x1']
      self.assertListEqual(pre_pool.get_shape().as_list(),
                           [batch_size, 3, 3, 1536])

  def testUnknownBatchSize(self):
    batch_size = 1
    height, width = 299, 299
    num_classes = 1000
    with self.test_session() as sess:
      inputs = tf.placeholder(tf.float32, (None, height, width, 3))
      logits, _ = inception.inception_resnet_v2(inputs, num_classes)
      self.assertTrue(logits.op.name.startswith('InceptionResnetV2/Logits'))
      self.assertListEqual(logits.get_shape().as_list(),
                           [None, num_classes])
      images = tf.random_uniform((batch_size, height, width, 3))
      sess.run(tf.global_variables_initializer())
      output = sess.run(logits, {inputs: images.eval()})
      self.assertEquals(output.shape, (batch_size, num_classes))

  def testEvaluation(self):
    batch_size = 2
    height, width = 299, 299
    num_classes = 1000
    with self.test_session() as sess:
      eval_inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = inception.inception_resnet_v2(eval_inputs,
                                                num_classes,
                                                is_training=False)
      predictions = tf.argmax(logits, 1)
      sess.run(tf.global_variables_initializer())
      output = sess.run(predictions)
      self.assertEquals(output.shape, (batch_size,))

  def testTrainEvalWithReuse(self):
    train_batch_size = 5
    eval_batch_size = 2
    height, width = 150, 150
    num_classes = 1000
    with self.test_session() as sess:
      train_inputs = tf.random_uniform((train_batch_size, height, width, 3))
      inception.inception_resnet_v2(train_inputs, num_classes)
      eval_inputs = tf.random_uniform((eval_batch_size, height, width, 3))
      logits, _ = inception.inception_resnet_v2(eval_inputs,
                                                num_classes,
                                                is_training=False,
                                                reuse=True)
      predictions = tf.argmax(logits, 1)
      sess.run(tf.global_variables_initializer())
      output = sess.run(predictions)
      self.assertEquals(output.shape, (eval_batch_size,))


if __name__ == '__main__':
  tf.test.main()
