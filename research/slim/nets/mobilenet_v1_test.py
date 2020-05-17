# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Tests for MobileNet v1."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim as slim

from nets import mobilenet_v1


class MobilenetV1Test(tf.test.TestCase):

  def testBuildClassificationNetwork(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000

    inputs = tf.random.uniform((batch_size, height, width, 3))
    logits, end_points = mobilenet_v1.mobilenet_v1(inputs, num_classes)
    self.assertTrue(logits.op.name.startswith(
        'MobilenetV1/Logits/SpatialSqueeze'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])
    self.assertTrue('Predictions' in end_points)
    self.assertListEqual(end_points['Predictions'].get_shape().as_list(),
                         [batch_size, num_classes])

  def testBuildPreLogitsNetwork(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = None

    inputs = tf.random.uniform((batch_size, height, width, 3))
    net, end_points = mobilenet_v1.mobilenet_v1(inputs, num_classes)
    self.assertTrue(net.op.name.startswith('MobilenetV1/Logits/AvgPool'))
    self.assertListEqual(net.get_shape().as_list(), [batch_size, 1, 1, 1024])
    self.assertFalse('Logits' in end_points)
    self.assertFalse('Predictions' in end_points)

  def testBuildBaseNetwork(self):
    batch_size = 5
    height, width = 224, 224

    inputs = tf.random.uniform((batch_size, height, width, 3))
    net, end_points = mobilenet_v1.mobilenet_v1_base(inputs)
    self.assertTrue(net.op.name.startswith('MobilenetV1/Conv2d_13'))
    self.assertListEqual(net.get_shape().as_list(),
                         [batch_size, 7, 7, 1024])
    expected_endpoints = ['Conv2d_0',
                          'Conv2d_1_depthwise', 'Conv2d_1_pointwise',
                          'Conv2d_2_depthwise', 'Conv2d_2_pointwise',
                          'Conv2d_3_depthwise', 'Conv2d_3_pointwise',
                          'Conv2d_4_depthwise', 'Conv2d_4_pointwise',
                          'Conv2d_5_depthwise', 'Conv2d_5_pointwise',
                          'Conv2d_6_depthwise', 'Conv2d_6_pointwise',
                          'Conv2d_7_depthwise', 'Conv2d_7_pointwise',
                          'Conv2d_8_depthwise', 'Conv2d_8_pointwise',
                          'Conv2d_9_depthwise', 'Conv2d_9_pointwise',
                          'Conv2d_10_depthwise', 'Conv2d_10_pointwise',
                          'Conv2d_11_depthwise', 'Conv2d_11_pointwise',
                          'Conv2d_12_depthwise', 'Conv2d_12_pointwise',
                          'Conv2d_13_depthwise', 'Conv2d_13_pointwise']
    self.assertItemsEqual(end_points.keys(), expected_endpoints)

  def testBuildOnlyUptoFinalEndpoint(self):
    batch_size = 5
    height, width = 224, 224
    endpoints = ['Conv2d_0',
                 'Conv2d_1_depthwise', 'Conv2d_1_pointwise',
                 'Conv2d_2_depthwise', 'Conv2d_2_pointwise',
                 'Conv2d_3_depthwise', 'Conv2d_3_pointwise',
                 'Conv2d_4_depthwise', 'Conv2d_4_pointwise',
                 'Conv2d_5_depthwise', 'Conv2d_5_pointwise',
                 'Conv2d_6_depthwise', 'Conv2d_6_pointwise',
                 'Conv2d_7_depthwise', 'Conv2d_7_pointwise',
                 'Conv2d_8_depthwise', 'Conv2d_8_pointwise',
                 'Conv2d_9_depthwise', 'Conv2d_9_pointwise',
                 'Conv2d_10_depthwise', 'Conv2d_10_pointwise',
                 'Conv2d_11_depthwise', 'Conv2d_11_pointwise',
                 'Conv2d_12_depthwise', 'Conv2d_12_pointwise',
                 'Conv2d_13_depthwise', 'Conv2d_13_pointwise']
    for index, endpoint in enumerate(endpoints):
      with tf.Graph().as_default():
        inputs = tf.random.uniform((batch_size, height, width, 3))
        out_tensor, end_points = mobilenet_v1.mobilenet_v1_base(
            inputs, final_endpoint=endpoint)
        self.assertTrue(out_tensor.op.name.startswith(
            'MobilenetV1/' + endpoint))
        self.assertItemsEqual(endpoints[:index + 1], end_points.keys())

  def testBuildCustomNetworkUsingConvDefs(self):
    batch_size = 5
    height, width = 224, 224
    conv_defs = [
        mobilenet_v1.Conv(kernel=[3, 3], stride=2, depth=32),
        mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=64),
        mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=2, depth=128),
        mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=512)
    ]

    inputs = tf.random.uniform((batch_size, height, width, 3))
    net, end_points = mobilenet_v1.mobilenet_v1_base(
        inputs, final_endpoint='Conv2d_3_pointwise', conv_defs=conv_defs)
    self.assertTrue(net.op.name.startswith('MobilenetV1/Conv2d_3'))
    self.assertListEqual(net.get_shape().as_list(),
                         [batch_size, 56, 56, 512])
    expected_endpoints = ['Conv2d_0',
                          'Conv2d_1_depthwise', 'Conv2d_1_pointwise',
                          'Conv2d_2_depthwise', 'Conv2d_2_pointwise',
                          'Conv2d_3_depthwise', 'Conv2d_3_pointwise']
    self.assertItemsEqual(end_points.keys(), expected_endpoints)

  def testBuildAndCheckAllEndPointsUptoConv2d_13(self):
    batch_size = 5
    height, width = 224, 224

    inputs = tf.random.uniform((batch_size, height, width, 3))
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        normalizer_fn=slim.batch_norm):
      _, end_points = mobilenet_v1.mobilenet_v1_base(
          inputs, final_endpoint='Conv2d_13_pointwise')
      _, explicit_padding_end_points = mobilenet_v1.mobilenet_v1_base(
          inputs, final_endpoint='Conv2d_13_pointwise',
          use_explicit_padding=True)
    endpoints_shapes = {'Conv2d_0': [batch_size, 112, 112, 32],
                        'Conv2d_1_depthwise': [batch_size, 112, 112, 32],
                        'Conv2d_1_pointwise': [batch_size, 112, 112, 64],
                        'Conv2d_2_depthwise': [batch_size, 56, 56, 64],
                        'Conv2d_2_pointwise': [batch_size, 56, 56, 128],
                        'Conv2d_3_depthwise': [batch_size, 56, 56, 128],
                        'Conv2d_3_pointwise': [batch_size, 56, 56, 128],
                        'Conv2d_4_depthwise': [batch_size, 28, 28, 128],
                        'Conv2d_4_pointwise': [batch_size, 28, 28, 256],
                        'Conv2d_5_depthwise': [batch_size, 28, 28, 256],
                        'Conv2d_5_pointwise': [batch_size, 28, 28, 256],
                        'Conv2d_6_depthwise': [batch_size, 14, 14, 256],
                        'Conv2d_6_pointwise': [batch_size, 14, 14, 512],
                        'Conv2d_7_depthwise': [batch_size, 14, 14, 512],
                        'Conv2d_7_pointwise': [batch_size, 14, 14, 512],
                        'Conv2d_8_depthwise': [batch_size, 14, 14, 512],
                        'Conv2d_8_pointwise': [batch_size, 14, 14, 512],
                        'Conv2d_9_depthwise': [batch_size, 14, 14, 512],
                        'Conv2d_9_pointwise': [batch_size, 14, 14, 512],
                        'Conv2d_10_depthwise': [batch_size, 14, 14, 512],
                        'Conv2d_10_pointwise': [batch_size, 14, 14, 512],
                        'Conv2d_11_depthwise': [batch_size, 14, 14, 512],
                        'Conv2d_11_pointwise': [batch_size, 14, 14, 512],
                        'Conv2d_12_depthwise': [batch_size, 7, 7, 512],
                        'Conv2d_12_pointwise': [batch_size, 7, 7, 1024],
                        'Conv2d_13_depthwise': [batch_size, 7, 7, 1024],
                        'Conv2d_13_pointwise': [batch_size, 7, 7, 1024]}
    self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
    for endpoint_name, expected_shape in endpoints_shapes.items():
      self.assertTrue(endpoint_name in end_points)
      self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                           expected_shape)
    self.assertItemsEqual(endpoints_shapes.keys(),
                          explicit_padding_end_points.keys())
    for endpoint_name, expected_shape in endpoints_shapes.items():
      self.assertTrue(endpoint_name in explicit_padding_end_points)
      self.assertListEqual(
          explicit_padding_end_points[endpoint_name].get_shape().as_list(),
          expected_shape)

  def testOutputStride16BuildAndCheckAllEndPointsUptoConv2d_13(self):
    batch_size = 5
    height, width = 224, 224
    output_stride = 16

    inputs = tf.random.uniform((batch_size, height, width, 3))
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        normalizer_fn=slim.batch_norm):
      _, end_points = mobilenet_v1.mobilenet_v1_base(
          inputs, output_stride=output_stride,
          final_endpoint='Conv2d_13_pointwise')
      _, explicit_padding_end_points = mobilenet_v1.mobilenet_v1_base(
          inputs, output_stride=output_stride,
          final_endpoint='Conv2d_13_pointwise', use_explicit_padding=True)
    endpoints_shapes = {'Conv2d_0': [batch_size, 112, 112, 32],
                        'Conv2d_1_depthwise': [batch_size, 112, 112, 32],
                        'Conv2d_1_pointwise': [batch_size, 112, 112, 64],
                        'Conv2d_2_depthwise': [batch_size, 56, 56, 64],
                        'Conv2d_2_pointwise': [batch_size, 56, 56, 128],
                        'Conv2d_3_depthwise': [batch_size, 56, 56, 128],
                        'Conv2d_3_pointwise': [batch_size, 56, 56, 128],
                        'Conv2d_4_depthwise': [batch_size, 28, 28, 128],
                        'Conv2d_4_pointwise': [batch_size, 28, 28, 256],
                        'Conv2d_5_depthwise': [batch_size, 28, 28, 256],
                        'Conv2d_5_pointwise': [batch_size, 28, 28, 256],
                        'Conv2d_6_depthwise': [batch_size, 14, 14, 256],
                        'Conv2d_6_pointwise': [batch_size, 14, 14, 512],
                        'Conv2d_7_depthwise': [batch_size, 14, 14, 512],
                        'Conv2d_7_pointwise': [batch_size, 14, 14, 512],
                        'Conv2d_8_depthwise': [batch_size, 14, 14, 512],
                        'Conv2d_8_pointwise': [batch_size, 14, 14, 512],
                        'Conv2d_9_depthwise': [batch_size, 14, 14, 512],
                        'Conv2d_9_pointwise': [batch_size, 14, 14, 512],
                        'Conv2d_10_depthwise': [batch_size, 14, 14, 512],
                        'Conv2d_10_pointwise': [batch_size, 14, 14, 512],
                        'Conv2d_11_depthwise': [batch_size, 14, 14, 512],
                        'Conv2d_11_pointwise': [batch_size, 14, 14, 512],
                        'Conv2d_12_depthwise': [batch_size, 14, 14, 512],
                        'Conv2d_12_pointwise': [batch_size, 14, 14, 1024],
                        'Conv2d_13_depthwise': [batch_size, 14, 14, 1024],
                        'Conv2d_13_pointwise': [batch_size, 14, 14, 1024]}
    self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
    for endpoint_name, expected_shape in endpoints_shapes.items():
      self.assertTrue(endpoint_name in end_points)
      self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                           expected_shape)
    self.assertItemsEqual(endpoints_shapes.keys(),
                          explicit_padding_end_points.keys())
    for endpoint_name, expected_shape in endpoints_shapes.items():
      self.assertTrue(endpoint_name in explicit_padding_end_points)
      self.assertListEqual(
          explicit_padding_end_points[endpoint_name].get_shape().as_list(),
          expected_shape)

  def testOutputStride8BuildAndCheckAllEndPointsUptoConv2d_13(self):
    batch_size = 5
    height, width = 224, 224
    output_stride = 8

    inputs = tf.random.uniform((batch_size, height, width, 3))
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        normalizer_fn=slim.batch_norm):
      _, end_points = mobilenet_v1.mobilenet_v1_base(
          inputs, output_stride=output_stride,
          final_endpoint='Conv2d_13_pointwise')
      _, explicit_padding_end_points = mobilenet_v1.mobilenet_v1_base(
          inputs, output_stride=output_stride,
          final_endpoint='Conv2d_13_pointwise', use_explicit_padding=True)
    endpoints_shapes = {'Conv2d_0': [batch_size, 112, 112, 32],
                        'Conv2d_1_depthwise': [batch_size, 112, 112, 32],
                        'Conv2d_1_pointwise': [batch_size, 112, 112, 64],
                        'Conv2d_2_depthwise': [batch_size, 56, 56, 64],
                        'Conv2d_2_pointwise': [batch_size, 56, 56, 128],
                        'Conv2d_3_depthwise': [batch_size, 56, 56, 128],
                        'Conv2d_3_pointwise': [batch_size, 56, 56, 128],
                        'Conv2d_4_depthwise': [batch_size, 28, 28, 128],
                        'Conv2d_4_pointwise': [batch_size, 28, 28, 256],
                        'Conv2d_5_depthwise': [batch_size, 28, 28, 256],
                        'Conv2d_5_pointwise': [batch_size, 28, 28, 256],
                        'Conv2d_6_depthwise': [batch_size, 28, 28, 256],
                        'Conv2d_6_pointwise': [batch_size, 28, 28, 512],
                        'Conv2d_7_depthwise': [batch_size, 28, 28, 512],
                        'Conv2d_7_pointwise': [batch_size, 28, 28, 512],
                        'Conv2d_8_depthwise': [batch_size, 28, 28, 512],
                        'Conv2d_8_pointwise': [batch_size, 28, 28, 512],
                        'Conv2d_9_depthwise': [batch_size, 28, 28, 512],
                        'Conv2d_9_pointwise': [batch_size, 28, 28, 512],
                        'Conv2d_10_depthwise': [batch_size, 28, 28, 512],
                        'Conv2d_10_pointwise': [batch_size, 28, 28, 512],
                        'Conv2d_11_depthwise': [batch_size, 28, 28, 512],
                        'Conv2d_11_pointwise': [batch_size, 28, 28, 512],
                        'Conv2d_12_depthwise': [batch_size, 28, 28, 512],
                        'Conv2d_12_pointwise': [batch_size, 28, 28, 1024],
                        'Conv2d_13_depthwise': [batch_size, 28, 28, 1024],
                        'Conv2d_13_pointwise': [batch_size, 28, 28, 1024]}
    self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
    for endpoint_name, expected_shape in endpoints_shapes.items():
      self.assertTrue(endpoint_name in end_points)
      self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                           expected_shape)
    self.assertItemsEqual(endpoints_shapes.keys(),
                          explicit_padding_end_points.keys())
    for endpoint_name, expected_shape in endpoints_shapes.items():
      self.assertTrue(endpoint_name in explicit_padding_end_points)
      self.assertListEqual(
          explicit_padding_end_points[endpoint_name].get_shape().as_list(),
          expected_shape)

  def testBuildAndCheckAllEndPointsApproximateFaceNet(self):
    batch_size = 5
    height, width = 128, 128

    inputs = tf.random.uniform((batch_size, height, width, 3))
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        normalizer_fn=slim.batch_norm):
      _, end_points = mobilenet_v1.mobilenet_v1_base(
          inputs, final_endpoint='Conv2d_13_pointwise', depth_multiplier=0.75)
      _, explicit_padding_end_points = mobilenet_v1.mobilenet_v1_base(
          inputs, final_endpoint='Conv2d_13_pointwise', depth_multiplier=0.75,
          use_explicit_padding=True)
    # For the Conv2d_0 layer FaceNet has depth=16
    endpoints_shapes = {'Conv2d_0': [batch_size, 64, 64, 24],
                        'Conv2d_1_depthwise': [batch_size, 64, 64, 24],
                        'Conv2d_1_pointwise': [batch_size, 64, 64, 48],
                        'Conv2d_2_depthwise': [batch_size, 32, 32, 48],
                        'Conv2d_2_pointwise': [batch_size, 32, 32, 96],
                        'Conv2d_3_depthwise': [batch_size, 32, 32, 96],
                        'Conv2d_3_pointwise': [batch_size, 32, 32, 96],
                        'Conv2d_4_depthwise': [batch_size, 16, 16, 96],
                        'Conv2d_4_pointwise': [batch_size, 16, 16, 192],
                        'Conv2d_5_depthwise': [batch_size, 16, 16, 192],
                        'Conv2d_5_pointwise': [batch_size, 16, 16, 192],
                        'Conv2d_6_depthwise': [batch_size, 8, 8, 192],
                        'Conv2d_6_pointwise': [batch_size, 8, 8, 384],
                        'Conv2d_7_depthwise': [batch_size, 8, 8, 384],
                        'Conv2d_7_pointwise': [batch_size, 8, 8, 384],
                        'Conv2d_8_depthwise': [batch_size, 8, 8, 384],
                        'Conv2d_8_pointwise': [batch_size, 8, 8, 384],
                        'Conv2d_9_depthwise': [batch_size, 8, 8, 384],
                        'Conv2d_9_pointwise': [batch_size, 8, 8, 384],
                        'Conv2d_10_depthwise': [batch_size, 8, 8, 384],
                        'Conv2d_10_pointwise': [batch_size, 8, 8, 384],
                        'Conv2d_11_depthwise': [batch_size, 8, 8, 384],
                        'Conv2d_11_pointwise': [batch_size, 8, 8, 384],
                        'Conv2d_12_depthwise': [batch_size, 4, 4, 384],
                        'Conv2d_12_pointwise': [batch_size, 4, 4, 768],
                        'Conv2d_13_depthwise': [batch_size, 4, 4, 768],
                        'Conv2d_13_pointwise': [batch_size, 4, 4, 768]}
    self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
    for endpoint_name, expected_shape in endpoints_shapes.items():
      self.assertTrue(endpoint_name in end_points)
      self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                           expected_shape)
    self.assertItemsEqual(endpoints_shapes.keys(),
                          explicit_padding_end_points.keys())
    for endpoint_name, expected_shape in endpoints_shapes.items():
      self.assertTrue(endpoint_name in explicit_padding_end_points)
      self.assertListEqual(
          explicit_padding_end_points[endpoint_name].get_shape().as_list(),
          expected_shape)

  def testModelHasExpectedNumberOfParameters(self):
    batch_size = 5
    height, width = 224, 224
    inputs = tf.random.uniform((batch_size, height, width, 3))
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        normalizer_fn=slim.batch_norm):
      mobilenet_v1.mobilenet_v1_base(inputs)
      total_params, _ = slim.model_analyzer.analyze_vars(
          slim.get_model_variables())
      self.assertAlmostEqual(3217920, total_params)

  def testBuildEndPointsWithDepthMultiplierLessThanOne(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000

    inputs = tf.random.uniform((batch_size, height, width, 3))
    _, end_points = mobilenet_v1.mobilenet_v1(inputs, num_classes)

    endpoint_keys = [key for key in end_points.keys() if key.startswith('Conv')]

    _, end_points_with_multiplier = mobilenet_v1.mobilenet_v1(
        inputs, num_classes, scope='depth_multiplied_net',
        depth_multiplier=0.5)

    for key in endpoint_keys:
      original_depth = end_points[key].get_shape().as_list()[3]
      new_depth = end_points_with_multiplier[key].get_shape().as_list()[3]
      self.assertEqual(0.5 * original_depth, new_depth)

  def testBuildEndPointsWithDepthMultiplierGreaterThanOne(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000

    inputs = tf.random.uniform((batch_size, height, width, 3))
    _, end_points = mobilenet_v1.mobilenet_v1(inputs, num_classes)

    endpoint_keys = [key for key in end_points.keys()
                     if key.startswith('Mixed') or key.startswith('Conv')]

    _, end_points_with_multiplier = mobilenet_v1.mobilenet_v1(
        inputs, num_classes, scope='depth_multiplied_net',
        depth_multiplier=2.0)

    for key in endpoint_keys:
      original_depth = end_points[key].get_shape().as_list()[3]
      new_depth = end_points_with_multiplier[key].get_shape().as_list()[3]
      self.assertEqual(2.0 * original_depth, new_depth)

  def testRaiseValueErrorWithInvalidDepthMultiplier(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000

    inputs = tf.random.uniform((batch_size, height, width, 3))
    with self.assertRaises(ValueError):
      _ = mobilenet_v1.mobilenet_v1(
          inputs, num_classes, depth_multiplier=-0.1)
    with self.assertRaises(ValueError):
      _ = mobilenet_v1.mobilenet_v1(
          inputs, num_classes, depth_multiplier=0.0)

  def testHalfSizeImages(self):
    batch_size = 5
    height, width = 112, 112
    num_classes = 1000

    inputs = tf.random.uniform((batch_size, height, width, 3))
    logits, end_points = mobilenet_v1.mobilenet_v1(inputs, num_classes)
    self.assertTrue(logits.op.name.startswith('MobilenetV1/Logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])
    pre_pool = end_points['Conv2d_13_pointwise']
    self.assertListEqual(pre_pool.get_shape().as_list(),
                         [batch_size, 4, 4, 1024])

  def testUnknownImageShape(self):
    tf.reset_default_graph()
    batch_size = 2
    height, width = 224, 224
    num_classes = 1000
    input_np = np.random.uniform(0, 1, (batch_size, height, width, 3))
    with self.test_session() as sess:
      inputs = tf.placeholder(
          tf.float32, shape=(batch_size, None, None, 3))
      logits, end_points = mobilenet_v1.mobilenet_v1(inputs, num_classes)
      self.assertTrue(logits.op.name.startswith('MobilenetV1/Logits'))
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])
      pre_pool = end_points['Conv2d_13_pointwise']
      feed_dict = {inputs: input_np}
      tf.global_variables_initializer().run()
      pre_pool_out = sess.run(pre_pool, feed_dict=feed_dict)
      self.assertListEqual(list(pre_pool_out.shape), [batch_size, 7, 7, 1024])

  def testGlobalPoolUnknownImageShape(self):
    tf.reset_default_graph()
    batch_size = 1
    height, width = 250, 300
    num_classes = 1000
    input_np = np.random.uniform(0, 1, (batch_size, height, width, 3))
    with self.test_session() as sess:
      inputs = tf.placeholder(
          tf.float32, shape=(batch_size, None, None, 3))
      logits, end_points = mobilenet_v1.mobilenet_v1(inputs, num_classes,
                                                     global_pool=True)
      self.assertTrue(logits.op.name.startswith('MobilenetV1/Logits'))
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])
      pre_pool = end_points['Conv2d_13_pointwise']
      feed_dict = {inputs: input_np}
      tf.global_variables_initializer().run()
      pre_pool_out = sess.run(pre_pool, feed_dict=feed_dict)
      self.assertListEqual(list(pre_pool_out.shape), [batch_size, 8, 10, 1024])

  def testUnknowBatchSize(self):
    batch_size = 1
    height, width = 224, 224
    num_classes = 1000

    inputs = tf.placeholder(tf.float32, (None, height, width, 3))
    logits, _ = mobilenet_v1.mobilenet_v1(inputs, num_classes)
    self.assertTrue(logits.op.name.startswith('MobilenetV1/Logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [None, num_classes])
    images = tf.random.uniform((batch_size, height, width, 3))

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(logits, {inputs: images.eval()})
      self.assertEquals(output.shape, (batch_size, num_classes))

  def testEvaluation(self):
    batch_size = 2
    height, width = 224, 224
    num_classes = 1000

    eval_inputs = tf.random.uniform((batch_size, height, width, 3))
    logits, _ = mobilenet_v1.mobilenet_v1(eval_inputs, num_classes,
                                          is_training=False)
    predictions = tf.argmax(input=logits, axis=1)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(predictions)
      self.assertEquals(output.shape, (batch_size,))

  def testTrainEvalWithReuse(self):
    train_batch_size = 5
    eval_batch_size = 2
    height, width = 150, 150
    num_classes = 1000

    train_inputs = tf.random.uniform((train_batch_size, height, width, 3))
    mobilenet_v1.mobilenet_v1(train_inputs, num_classes)
    eval_inputs = tf.random.uniform((eval_batch_size, height, width, 3))
    logits, _ = mobilenet_v1.mobilenet_v1(eval_inputs, num_classes,
                                          reuse=True)
    predictions = tf.argmax(input=logits, axis=1)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(predictions)
      self.assertEquals(output.shape, (eval_batch_size,))

  def testLogitsNotSqueezed(self):
    num_classes = 25
    images = tf.random.uniform([1, 224, 224, 3])
    logits, _ = mobilenet_v1.mobilenet_v1(images,
                                          num_classes=num_classes,
                                          spatial_squeeze=False)

    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      logits_out = sess.run(logits)
      self.assertListEqual(list(logits_out.shape), [1, 1, 1, num_classes])

  def testBatchNormScopeDoesNotHaveIsTrainingWhenItsSetToNone(self):
    sc = mobilenet_v1.mobilenet_v1_arg_scope(is_training=None)
    self.assertNotIn('is_training', sc[slim.arg_scope_func_key(
        slim.batch_norm)])

  def testBatchNormScopeDoesHasIsTrainingWhenItsNotNone(self):
    sc = mobilenet_v1.mobilenet_v1_arg_scope(is_training=True)
    self.assertIn('is_training', sc[slim.arg_scope_func_key(slim.batch_norm)])
    sc = mobilenet_v1.mobilenet_v1_arg_scope(is_training=False)
    self.assertIn('is_training', sc[slim.arg_scope_func_key(slim.batch_norm)])
    sc = mobilenet_v1.mobilenet_v1_arg_scope()
    self.assertIn('is_training', sc[slim.arg_scope_func_key(slim.batch_norm)])

if __name__ == '__main__':
  tf.test.main()
