# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for networks.s3dg."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import s3dg


class S3DGTest(tf.test.TestCase):

  def testBuildClassificationNetwork(self):
    batch_size = 5
    num_frames = 64
    height, width = 224, 224
    num_classes = 1000

    inputs = tf.random.uniform((batch_size, num_frames, height, width, 3))
    logits, end_points = s3dg.s3dg(inputs, num_classes)
    self.assertTrue(logits.op.name.startswith('InceptionV1/Logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])
    self.assertTrue('Predictions' in end_points)
    self.assertListEqual(end_points['Predictions'].get_shape().as_list(),
                         [batch_size, num_classes])

  def testBuildBaseNetwork(self):
    batch_size = 5
    num_frames = 64
    height, width = 224, 224

    inputs = tf.random.uniform((batch_size, num_frames, height, width, 3))
    mixed_6c, end_points = s3dg.s3dg_base(inputs)
    self.assertTrue(mixed_6c.op.name.startswith('InceptionV1/Mixed_5c'))
    self.assertListEqual(mixed_6c.get_shape().as_list(),
                         [batch_size, 8, 7, 7, 1024])
    expected_endpoints = ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
                          'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b',
                          'Mixed_3c', 'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c',
                          'Mixed_4d', 'Mixed_4e', 'Mixed_4f', 'MaxPool_5a_2x2',
                          'Mixed_5b', 'Mixed_5c']
    self.assertItemsEqual(end_points.keys(), expected_endpoints)

  def testBuildOnlyUptoFinalEndpointNoGating(self):
    batch_size = 5
    num_frames = 64
    height, width = 224, 224
    endpoints = ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
                 'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
                 'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d',
                 'Mixed_4e', 'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b',
                 'Mixed_5c']
    for index, endpoint in enumerate(endpoints):
      with tf.Graph().as_default():
        inputs = tf.random.uniform((batch_size, num_frames, height, width, 3))
        out_tensor, end_points = s3dg.s3dg_base(
            inputs, final_endpoint=endpoint, gating_startat=None)
        print(endpoint, out_tensor.op.name)
        self.assertTrue(out_tensor.op.name.startswith(
            'InceptionV1/' + endpoint))
        self.assertItemsEqual(endpoints[:index+1], end_points)

  def testBuildAndCheckAllEndPointsUptoMixed5c(self):
    batch_size = 5
    num_frames = 64
    height, width = 224, 224

    inputs = tf.random.uniform((batch_size, num_frames, height, width, 3))
    _, end_points = s3dg.s3dg_base(inputs,
                                   final_endpoint='Mixed_5c')
    endpoints_shapes = {'Conv2d_1a_7x7': [5, 32, 112, 112, 64],
                        'MaxPool_2a_3x3': [5, 32, 56, 56, 64],
                        'Conv2d_2b_1x1': [5, 32, 56, 56, 64],
                        'Conv2d_2c_3x3': [5, 32, 56, 56, 192],
                        'MaxPool_3a_3x3': [5, 32, 28, 28, 192],
                        'Mixed_3b': [5, 32, 28, 28, 256],
                        'Mixed_3c': [5, 32, 28, 28, 480],
                        'MaxPool_4a_3x3': [5, 16, 14, 14, 480],
                        'Mixed_4b': [5, 16, 14, 14, 512],
                        'Mixed_4c': [5, 16, 14, 14, 512],
                        'Mixed_4d': [5, 16, 14, 14, 512],
                        'Mixed_4e': [5, 16, 14, 14, 528],
                        'Mixed_4f': [5, 16, 14, 14, 832],
                        'MaxPool_5a_2x2': [5, 8, 7, 7, 832],
                        'Mixed_5b': [5, 8, 7, 7, 832],
                        'Mixed_5c': [5, 8, 7, 7, 1024]}

    self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
    for endpoint_name, expected_shape in endpoints_shapes.iteritems():
      self.assertTrue(endpoint_name in end_points)
      self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                           expected_shape)

  def testHalfSizeImages(self):
    batch_size = 5
    num_frames = 64
    height, width = 112, 112

    inputs = tf.random.uniform((batch_size, num_frames, height, width, 3))
    mixed_5c, _ = s3dg.s3dg_base(inputs)
    self.assertTrue(mixed_5c.op.name.startswith('InceptionV1/Mixed_5c'))
    self.assertListEqual(mixed_5c.get_shape().as_list(),
                         [batch_size, 8, 4, 4, 1024])

  def testTenFrames(self):
    batch_size = 5
    num_frames = 10
    height, width = 224, 224

    inputs = tf.random.uniform((batch_size, num_frames, height, width, 3))
    mixed_5c, _ = s3dg.s3dg_base(inputs)
    self.assertTrue(mixed_5c.op.name.startswith('InceptionV1/Mixed_5c'))
    self.assertListEqual(mixed_5c.get_shape().as_list(),
                         [batch_size, 2, 7, 7, 1024])

  def testEvaluation(self):
    batch_size = 2
    num_frames = 64
    height, width = 224, 224
    num_classes = 1000

    eval_inputs = tf.random.uniform((batch_size, num_frames, height, width, 3))
    logits, _ = s3dg.s3dg(eval_inputs, num_classes,
                          is_training=False)
    predictions = tf.argmax(input=logits, axis=1)

    with self.test_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      output = sess.run(predictions)
      self.assertEquals(output.shape, (batch_size,))


if __name__ == '__main__':
  tf.test.main()
