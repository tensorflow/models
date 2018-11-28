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
# ==============================================================================
"""Tests for slim.nasnet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets.nasnet import nasnet

slim = tf.contrib.slim


class NASNetTest(tf.test.TestCase):

  def testBuildLogitsCifarModel(self):
    batch_size = 5
    height, width = 32, 32
    num_classes = 10
    inputs = tf.random_uniform((batch_size, height, width, 3))
    tf.train.create_global_step()
    with slim.arg_scope(nasnet.nasnet_cifar_arg_scope()):
      logits, end_points = nasnet.build_nasnet_cifar(inputs, num_classes)
    auxlogits = end_points['AuxLogits']
    predictions = end_points['Predictions']
    self.assertListEqual(auxlogits.get_shape().as_list(),
                         [batch_size, num_classes])
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])
    self.assertListEqual(predictions.get_shape().as_list(),
                         [batch_size, num_classes])

  def testBuildLogitsMobileModel(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000
    inputs = tf.random_uniform((batch_size, height, width, 3))
    tf.train.create_global_step()
    with slim.arg_scope(nasnet.nasnet_mobile_arg_scope()):
      logits, end_points = nasnet.build_nasnet_mobile(inputs, num_classes)
    auxlogits = end_points['AuxLogits']
    predictions = end_points['Predictions']
    self.assertListEqual(auxlogits.get_shape().as_list(),
                         [batch_size, num_classes])
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])
    self.assertListEqual(predictions.get_shape().as_list(),
                         [batch_size, num_classes])

  def testBuildLogitsLargeModel(self):
    batch_size = 5
    height, width = 331, 331
    num_classes = 1000
    inputs = tf.random_uniform((batch_size, height, width, 3))
    tf.train.create_global_step()
    with slim.arg_scope(nasnet.nasnet_large_arg_scope()):
      logits, end_points = nasnet.build_nasnet_large(inputs, num_classes)
    auxlogits = end_points['AuxLogits']
    predictions = end_points['Predictions']
    self.assertListEqual(auxlogits.get_shape().as_list(),
                         [batch_size, num_classes])
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])
    self.assertListEqual(predictions.get_shape().as_list(),
                         [batch_size, num_classes])

  def testBuildPreLogitsCifarModel(self):
    batch_size = 5
    height, width = 32, 32
    num_classes = None
    inputs = tf.random_uniform((batch_size, height, width, 3))
    tf.train.create_global_step()
    with slim.arg_scope(nasnet.nasnet_cifar_arg_scope()):
      net, end_points = nasnet.build_nasnet_cifar(inputs, num_classes)
    self.assertFalse('AuxLogits' in end_points)
    self.assertFalse('Predictions' in end_points)
    self.assertTrue(net.op.name.startswith('final_layer/Mean'))
    self.assertListEqual(net.get_shape().as_list(), [batch_size, 768])

  def testBuildPreLogitsMobileModel(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = None
    inputs = tf.random_uniform((batch_size, height, width, 3))
    tf.train.create_global_step()
    with slim.arg_scope(nasnet.nasnet_mobile_arg_scope()):
      net, end_points = nasnet.build_nasnet_mobile(inputs, num_classes)
    self.assertFalse('AuxLogits' in end_points)
    self.assertFalse('Predictions' in end_points)
    self.assertTrue(net.op.name.startswith('final_layer/Mean'))
    self.assertListEqual(net.get_shape().as_list(), [batch_size, 1056])

  def testBuildPreLogitsLargeModel(self):
    batch_size = 5
    height, width = 331, 331
    num_classes = None
    inputs = tf.random_uniform((batch_size, height, width, 3))
    tf.train.create_global_step()
    with slim.arg_scope(nasnet.nasnet_large_arg_scope()):
      net, end_points = nasnet.build_nasnet_large(inputs, num_classes)
    self.assertFalse('AuxLogits' in end_points)
    self.assertFalse('Predictions' in end_points)
    self.assertTrue(net.op.name.startswith('final_layer/Mean'))
    self.assertListEqual(net.get_shape().as_list(), [batch_size, 4032])

  def testAllEndPointsShapesCifarModel(self):
    batch_size = 5
    height, width = 32, 32
    num_classes = 10
    inputs = tf.random_uniform((batch_size, height, width, 3))
    tf.train.create_global_step()
    with slim.arg_scope(nasnet.nasnet_cifar_arg_scope()):
      _, end_points = nasnet.build_nasnet_cifar(inputs, num_classes)
    endpoints_shapes = {'Stem': [batch_size, 32, 32, 96],
                        'Cell_0': [batch_size, 32, 32, 192],
                        'Cell_1': [batch_size, 32, 32, 192],
                        'Cell_2': [batch_size, 32, 32, 192],
                        'Cell_3': [batch_size, 32, 32, 192],
                        'Cell_4': [batch_size, 32, 32, 192],
                        'Cell_5': [batch_size, 32, 32, 192],
                        'Cell_6': [batch_size, 16, 16, 384],
                        'Cell_7': [batch_size, 16, 16, 384],
                        'Cell_8': [batch_size, 16, 16, 384],
                        'Cell_9': [batch_size, 16, 16, 384],
                        'Cell_10': [batch_size, 16, 16, 384],
                        'Cell_11': [batch_size, 16, 16, 384],
                        'Cell_12': [batch_size, 8, 8, 768],
                        'Cell_13': [batch_size, 8, 8, 768],
                        'Cell_14': [batch_size, 8, 8, 768],
                        'Cell_15': [batch_size, 8, 8, 768],
                        'Cell_16': [batch_size, 8, 8, 768],
                        'Cell_17': [batch_size, 8, 8, 768],
                        'Reduction_Cell_0': [batch_size, 16, 16, 256],
                        'Reduction_Cell_1': [batch_size, 8, 8, 512],
                        'global_pool': [batch_size, 768],
                        # Logits and predictions
                        'AuxLogits': [batch_size, num_classes],
                        'Logits': [batch_size, num_classes],
                        'Predictions': [batch_size, num_classes]}
    self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
    for endpoint_name in endpoints_shapes:
      tf.logging.info('Endpoint name: {}'.format(endpoint_name))
      expected_shape = endpoints_shapes[endpoint_name]
      self.assertTrue(endpoint_name in end_points)
      self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                           expected_shape)

  def testNoAuxHeadCifarModel(self):
    batch_size = 5
    height, width = 32, 32
    num_classes = 10
    for use_aux_head in (True, False):
      tf.reset_default_graph()
      inputs = tf.random_uniform((batch_size, height, width, 3))
      tf.train.create_global_step()
      config = nasnet.cifar_config()
      config.set_hparam('use_aux_head', int(use_aux_head))
      with slim.arg_scope(nasnet.nasnet_cifar_arg_scope()):
        _, end_points = nasnet.build_nasnet_cifar(inputs, num_classes,
                                                  config=config)
      self.assertEqual('AuxLogits' in end_points, use_aux_head)

  def testAllEndPointsShapesMobileModel(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000
    inputs = tf.random_uniform((batch_size, height, width, 3))
    tf.train.create_global_step()
    with slim.arg_scope(nasnet.nasnet_mobile_arg_scope()):
      _, end_points = nasnet.build_nasnet_mobile(inputs, num_classes)
    endpoints_shapes = {'Stem': [batch_size, 28, 28, 88],
                        'Cell_0': [batch_size, 28, 28, 264],
                        'Cell_1': [batch_size, 28, 28, 264],
                        'Cell_2': [batch_size, 28, 28, 264],
                        'Cell_3': [batch_size, 28, 28, 264],
                        'Cell_4': [batch_size, 14, 14, 528],
                        'Cell_5': [batch_size, 14, 14, 528],
                        'Cell_6': [batch_size, 14, 14, 528],
                        'Cell_7': [batch_size, 14, 14, 528],
                        'Cell_8': [batch_size, 7, 7, 1056],
                        'Cell_9': [batch_size, 7, 7, 1056],
                        'Cell_10': [batch_size, 7, 7, 1056],
                        'Cell_11': [batch_size, 7, 7, 1056],
                        'Reduction_Cell_0': [batch_size, 14, 14, 352],
                        'Reduction_Cell_1': [batch_size, 7, 7, 704],
                        'global_pool': [batch_size, 1056],
                        # Logits and predictions
                        'AuxLogits': [batch_size, num_classes],
                        'Logits': [batch_size, num_classes],
                        'Predictions': [batch_size, num_classes]}
    self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
    for endpoint_name in endpoints_shapes:
      tf.logging.info('Endpoint name: {}'.format(endpoint_name))
      expected_shape = endpoints_shapes[endpoint_name]
      self.assertTrue(endpoint_name in end_points)
      self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                           expected_shape)

  def testNoAuxHeadMobileModel(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000
    for use_aux_head in (True, False):
      tf.reset_default_graph()
      inputs = tf.random_uniform((batch_size, height, width, 3))
      tf.train.create_global_step()
      config = nasnet.mobile_imagenet_config()
      config.set_hparam('use_aux_head', int(use_aux_head))
      with slim.arg_scope(nasnet.nasnet_mobile_arg_scope()):
        _, end_points = nasnet.build_nasnet_mobile(inputs, num_classes,
                                                   config=config)
      self.assertEqual('AuxLogits' in end_points, use_aux_head)

  def testAllEndPointsShapesLargeModel(self):
    batch_size = 5
    height, width = 331, 331
    num_classes = 1000
    inputs = tf.random_uniform((batch_size, height, width, 3))
    tf.train.create_global_step()
    with slim.arg_scope(nasnet.nasnet_large_arg_scope()):
      _, end_points = nasnet.build_nasnet_large(inputs, num_classes)
    endpoints_shapes = {'Stem': [batch_size, 42, 42, 336],
                        'Cell_0': [batch_size, 42, 42, 1008],
                        'Cell_1': [batch_size, 42, 42, 1008],
                        'Cell_2': [batch_size, 42, 42, 1008],
                        'Cell_3': [batch_size, 42, 42, 1008],
                        'Cell_4': [batch_size, 42, 42, 1008],
                        'Cell_5': [batch_size, 42, 42, 1008],
                        'Cell_6': [batch_size, 21, 21, 2016],
                        'Cell_7': [batch_size, 21, 21, 2016],
                        'Cell_8': [batch_size, 21, 21, 2016],
                        'Cell_9': [batch_size, 21, 21, 2016],
                        'Cell_10': [batch_size, 21, 21, 2016],
                        'Cell_11': [batch_size, 21, 21, 2016],
                        'Cell_12': [batch_size, 11, 11, 4032],
                        'Cell_13': [batch_size, 11, 11, 4032],
                        'Cell_14': [batch_size, 11, 11, 4032],
                        'Cell_15': [batch_size, 11, 11, 4032],
                        'Cell_16': [batch_size, 11, 11, 4032],
                        'Cell_17': [batch_size, 11, 11, 4032],
                        'Reduction_Cell_0': [batch_size, 21, 21, 1344],
                        'Reduction_Cell_1': [batch_size, 11, 11, 2688],
                        'global_pool': [batch_size, 4032],
                        # Logits and predictions
                        'AuxLogits': [batch_size, num_classes],
                        'Logits': [batch_size, num_classes],
                        'Predictions': [batch_size, num_classes]}
    self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
    for endpoint_name in endpoints_shapes:
      tf.logging.info('Endpoint name: {}'.format(endpoint_name))
      expected_shape = endpoints_shapes[endpoint_name]
      self.assertTrue(endpoint_name in end_points)
      self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                           expected_shape)

  def testNoAuxHeadLargeModel(self):
    batch_size = 5
    height, width = 331, 331
    num_classes = 1000
    for use_aux_head in (True, False):
      tf.reset_default_graph()
      inputs = tf.random_uniform((batch_size, height, width, 3))
      tf.train.create_global_step()
      config = nasnet.large_imagenet_config()
      config.set_hparam('use_aux_head', int(use_aux_head))
      with slim.arg_scope(nasnet.nasnet_large_arg_scope()):
        _, end_points = nasnet.build_nasnet_large(inputs, num_classes,
                                                  config=config)
      self.assertEqual('AuxLogits' in end_points, use_aux_head)

  def testVariablesSetDeviceMobileModel(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000
    inputs = tf.random_uniform((batch_size, height, width, 3))
    tf.train.create_global_step()
    # Force all Variables to reside on the device.
    with tf.variable_scope('on_cpu'), tf.device('/cpu:0'):
      with slim.arg_scope(nasnet.nasnet_mobile_arg_scope()):
        nasnet.build_nasnet_mobile(inputs, num_classes)
    with tf.variable_scope('on_gpu'), tf.device('/gpu:0'):
      with slim.arg_scope(nasnet.nasnet_mobile_arg_scope()):
        nasnet.build_nasnet_mobile(inputs, num_classes)
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='on_cpu'):
      self.assertDeviceEqual(v.device, '/cpu:0')
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='on_gpu'):
      self.assertDeviceEqual(v.device, '/gpu:0')

  def testUnknownBatchSizeMobileModel(self):
    batch_size = 1
    height, width = 224, 224
    num_classes = 1000
    with self.test_session() as sess:
      inputs = tf.placeholder(tf.float32, (None, height, width, 3))
      with slim.arg_scope(nasnet.nasnet_mobile_arg_scope()):
        logits, _ = nasnet.build_nasnet_mobile(inputs, num_classes)
      self.assertListEqual(logits.get_shape().as_list(),
                           [None, num_classes])
      images = tf.random_uniform((batch_size, height, width, 3))
      sess.run(tf.global_variables_initializer())
      output = sess.run(logits, {inputs: images.eval()})
      self.assertEquals(output.shape, (batch_size, num_classes))

  def testEvaluationMobileModel(self):
    batch_size = 2
    height, width = 224, 224
    num_classes = 1000
    with self.test_session() as sess:
      eval_inputs = tf.random_uniform((batch_size, height, width, 3))
      with slim.arg_scope(nasnet.nasnet_mobile_arg_scope()):
        logits, _ = nasnet.build_nasnet_mobile(eval_inputs,
                                               num_classes,
                                               is_training=False)
      predictions = tf.argmax(logits, 1)
      sess.run(tf.global_variables_initializer())
      output = sess.run(predictions)
      self.assertEquals(output.shape, (batch_size,))

  def testOverrideHParamsCifarModel(self):
    batch_size = 5
    height, width = 32, 32
    num_classes = 10
    inputs = tf.random_uniform((batch_size, height, width, 3))
    tf.train.create_global_step()
    config = nasnet.cifar_config()
    config.set_hparam('data_format', 'NCHW')
    with slim.arg_scope(nasnet.nasnet_cifar_arg_scope()):
      _, end_points = nasnet.build_nasnet_cifar(
          inputs, num_classes, config=config)
    self.assertListEqual(
        end_points['Stem'].shape.as_list(), [batch_size, 96, 32, 32])

  def testOverrideHParamsMobileModel(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000
    inputs = tf.random_uniform((batch_size, height, width, 3))
    tf.train.create_global_step()
    config = nasnet.mobile_imagenet_config()
    config.set_hparam('data_format', 'NCHW')
    with slim.arg_scope(nasnet.nasnet_mobile_arg_scope()):
      _, end_points = nasnet.build_nasnet_mobile(
          inputs, num_classes, config=config)
    self.assertListEqual(
        end_points['Stem'].shape.as_list(), [batch_size, 88, 28, 28])

  def testOverrideHParamsLargeModel(self):
    batch_size = 5
    height, width = 331, 331
    num_classes = 1000
    inputs = tf.random_uniform((batch_size, height, width, 3))
    tf.train.create_global_step()
    config = nasnet.large_imagenet_config()
    config.set_hparam('data_format', 'NCHW')
    with slim.arg_scope(nasnet.nasnet_large_arg_scope()):
      _, end_points = nasnet.build_nasnet_large(
          inputs, num_classes, config=config)
    self.assertListEqual(
        end_points['Stem'].shape.as_list(), [batch_size, 336, 42, 42])

  def testCurrentStepCifarModel(self):
    batch_size = 5
    height, width = 32, 32
    num_classes = 10
    inputs = tf.random_uniform((batch_size, height, width, 3))
    global_step = tf.train.create_global_step()
    with slim.arg_scope(nasnet.nasnet_cifar_arg_scope()):
      logits, end_points = nasnet.build_nasnet_cifar(inputs,
                                                     num_classes,
                                                     current_step=global_step)
    auxlogits = end_points['AuxLogits']
    predictions = end_points['Predictions']
    self.assertListEqual(auxlogits.get_shape().as_list(),
                         [batch_size, num_classes])
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])
    self.assertListEqual(predictions.get_shape().as_list(),
                         [batch_size, num_classes])

  def testUseBoundedAcitvationCifarModel(self):
    batch_size = 1
    height, width = 32, 32
    num_classes = 10
    for use_bounded_activation in (True, False):
      tf.reset_default_graph()
      inputs = tf.random_uniform((batch_size, height, width, 3))
      config = nasnet.cifar_config()
      config.set_hparam('use_bounded_activation', use_bounded_activation)
      with slim.arg_scope(nasnet.nasnet_cifar_arg_scope()):
        _, _ = nasnet.build_nasnet_cifar(
            inputs, num_classes, config=config)
      for node in tf.get_default_graph().as_graph_def().node:
        if node.op.startswith('Relu'):
          self.assertEqual(node.op == 'Relu6', use_bounded_activation)

if __name__ == '__main__':
  tf.test.main()
