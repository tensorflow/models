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

"""Tests for resnet_v1_beta module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
import tensorflow as tf

from deeplab.core import resnet_v1_beta
from tensorflow.contrib.slim.nets import resnet_utils

slim = tf.contrib.slim


def create_test_input(batch, height, width, channels):
  """Create test input tensor."""
  if None in [batch, height, width, channels]:
    return tf.placeholder(tf.float32, (batch, height, width, channels))
  else:
    return tf.to_float(
        np.tile(
            np.reshape(
                np.reshape(np.arange(height), [height, 1]) +
                np.reshape(np.arange(width), [1, width]),
                [1, height, width, 1]),
            [batch, 1, 1, channels]))


class ResnetCompleteNetworkTest(tf.test.TestCase):
  """Tests with complete small ResNet v1 networks."""

  def _resnet_small(self,
                    inputs,
                    num_classes=None,
                    is_training=True,
                    global_pool=True,
                    output_stride=None,
                    multi_grid=None,
                    reuse=None,
                    scope='resnet_v1_small'):
    """A shallow and thin ResNet v1 for faster tests."""
    if multi_grid is None:
      multi_grid = [1, 1, 1]
    else:
      if len(multi_grid) != 3:
        raise ValueError('Expect multi_grid to have length 3.')

    block = resnet_v1_beta.resnet_v1_beta_block
    blocks = [
        block('block1', base_depth=1, num_units=3, stride=2),
        block('block2', base_depth=2, num_units=3, stride=2),
        block('block3', base_depth=4, num_units=3, stride=2),
        resnet_utils.Block('block4', resnet_v1_beta.bottleneck, [
            {'depth': 32,
             'depth_bottleneck': 8,
             'stride': 1,
             'unit_rate': rate} for rate in multi_grid])]

    return resnet_v1_beta.resnet_v1_beta(
        inputs,
        blocks,
        num_classes=num_classes,
        is_training=is_training,
        global_pool=global_pool,
        output_stride=output_stride,
        root_block_fn=functools.partial(
            resnet_v1_beta.root_block_fn_for_beta_variant),
        reuse=reuse,
        scope=scope)

  def testClassificationEndPoints(self):
    global_pool = True
    num_classes = 10
    inputs = create_test_input(2, 224, 224, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      logits, end_points = self._resnet_small(inputs,
                                              num_classes,
                                              global_pool=global_pool,
                                              scope='resnet')

    self.assertTrue(logits.op.name.startswith('resnet/logits'))
    self.assertListEqual(logits.get_shape().as_list(), [2, 1, 1, num_classes])
    self.assertTrue('predictions' in end_points)
    self.assertListEqual(end_points['predictions'].get_shape().as_list(),
                         [2, 1, 1, num_classes])

  def testClassificationEndPointsWithMultigrid(self):
    global_pool = True
    num_classes = 10
    inputs = create_test_input(2, 224, 224, 3)
    multi_grid = [1, 2, 4]
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      logits, end_points = self._resnet_small(inputs,
                                              num_classes,
                                              global_pool=global_pool,
                                              multi_grid=multi_grid,
                                              scope='resnet')

    self.assertTrue(logits.op.name.startswith('resnet/logits'))
    self.assertListEqual(logits.get_shape().as_list(), [2, 1, 1, num_classes])
    self.assertTrue('predictions' in end_points)
    self.assertListEqual(end_points['predictions'].get_shape().as_list(),
                         [2, 1, 1, num_classes])

  def testClassificationShapes(self):
    global_pool = True
    num_classes = 10
    inputs = create_test_input(2, 224, 224, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, end_points = self._resnet_small(inputs,
                                         num_classes,
                                         global_pool=global_pool,
                                         scope='resnet')
      endpoint_to_shape = {
          'resnet/conv1_1': [2, 112, 112, 64],
          'resnet/conv1_2': [2, 112, 112, 64],
          'resnet/conv1_3': [2, 112, 112, 128],
          'resnet/block1': [2, 28, 28, 4],
          'resnet/block2': [2, 14, 14, 8],
          'resnet/block3': [2, 7, 7, 16],
          'resnet/block4': [2, 7, 7, 32]}
      for endpoint, shape in endpoint_to_shape.iteritems():
        self.assertListEqual(end_points[endpoint].get_shape().as_list(), shape)

  def testFullyConvolutionalEndpointShapes(self):
    global_pool = False
    num_classes = 10
    inputs = create_test_input(2, 321, 321, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, end_points = self._resnet_small(inputs,
                                         num_classes,
                                         global_pool=global_pool,
                                         scope='resnet')
      endpoint_to_shape = {
          'resnet/conv1_1': [2, 161, 161, 64],
          'resnet/conv1_2': [2, 161, 161, 64],
          'resnet/conv1_3': [2, 161, 161, 128],
          'resnet/block1': [2, 41, 41, 4],
          'resnet/block2': [2, 21, 21, 8],
          'resnet/block3': [2, 11, 11, 16],
          'resnet/block4': [2, 11, 11, 32]}
      for endpoint, shape in endpoint_to_shape.iteritems():
        self.assertListEqual(end_points[endpoint].get_shape().as_list(), shape)

  def testAtrousFullyConvolutionalEndpointShapes(self):
    global_pool = False
    num_classes = 10
    output_stride = 8
    inputs = create_test_input(2, 321, 321, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, end_points = self._resnet_small(inputs,
                                         num_classes,
                                         global_pool=global_pool,
                                         output_stride=output_stride,
                                         scope='resnet')
      endpoint_to_shape = {
          'resnet/conv1_1': [2, 161, 161, 64],
          'resnet/conv1_2': [2, 161, 161, 64],
          'resnet/conv1_3': [2, 161, 161, 128],
          'resnet/block1': [2, 41, 41, 4],
          'resnet/block2': [2, 41, 41, 8],
          'resnet/block3': [2, 41, 41, 16],
          'resnet/block4': [2, 41, 41, 32]}
      for endpoint, shape in endpoint_to_shape.iteritems():
        self.assertListEqual(end_points[endpoint].get_shape().as_list(), shape)

  def testAtrousFullyConvolutionalValues(self):
    """Verify dense feature extraction with atrous convolution."""
    nominal_stride = 32
    for output_stride in [4, 8, 16, 32, None]:
      with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        with tf.Graph().as_default():
          with self.test_session() as sess:
            tf.set_random_seed(0)
            inputs = create_test_input(2, 81, 81, 3)
            # Dense feature extraction followed by subsampling.
            output, _ = self._resnet_small(inputs,
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
            expected, _ = self._resnet_small(inputs,
                                             None,
                                             is_training=False,
                                             global_pool=False)
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(output.eval(), expected.eval(),
                                atol=1e-4, rtol=1e-4)

  def testUnknownBatchSize(self):
    batch = 2
    height, width = 65, 65
    global_pool = True
    num_classes = 10
    inputs = create_test_input(None, height, width, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      logits, _ = self._resnet_small(inputs,
                                     num_classes,
                                     global_pool=global_pool,
                                     scope='resnet')
    self.assertTrue(logits.op.name.startswith('resnet/logits'))
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
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      output, _ = self._resnet_small(inputs,
                                     None,
                                     global_pool=global_pool)
    self.assertListEqual(output.get_shape().as_list(),
                         [batch, None, None, 32])
    images = create_test_input(batch, height, width, 3)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output, {inputs: images.eval()})
      self.assertEquals(output.shape, (batch, 3, 3, 32))

  def testAtrousFullyConvolutionalUnknownHeightWidth(self):
    batch = 2
    height, width = 65, 65
    global_pool = False
    output_stride = 8
    inputs = create_test_input(batch, None, None, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      output, _ = self._resnet_small(inputs,
                                     None,
                                     global_pool=global_pool,
                                     output_stride=output_stride)
    self.assertListEqual(output.get_shape().as_list(),
                         [batch, None, None, 32])
    images = create_test_input(batch, height, width, 3)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output, {inputs: images.eval()})
      self.assertEquals(output.shape, (batch, 9, 9, 32))


if __name__ == '__main__':
  tf.test.main()
