# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for google3.third_party.tensorflow_models.slim.nets.mobilenet.mobilenet_v3."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import tensorflow as tf

from nets.mobilenet import mobilenet_v3


class MobilenetV3Test(absltest.TestCase):

  def setUp(self):
    super(MobilenetV3Test, self).setUp()
    tf.compat.v1.reset_default_graph()

  def testMobilenetV3Large(self):
    logits, endpoints = mobilenet_v3.mobilenet(
        tf.compat.v1.placeholder(tf.float32, (1, 224, 224, 3)))
    self.assertEqual(endpoints['layer_19'].shape, [1, 1, 1, 1280])
    self.assertEqual(logits.shape, [1, 1001])

  def testMobilenetV3Small(self):
    _, endpoints = mobilenet_v3.mobilenet(
        tf.compat.v1.placeholder(tf.float32, (1, 224, 224, 3)),
        conv_defs=mobilenet_v3.V3_SMALL)
    self.assertEqual(endpoints['layer_15'].shape, [1, 1, 1, 1024])

  def testMobilenetEdgeTpu(self):
    _, endpoints = mobilenet_v3.edge_tpu(
        tf.compat.v1.placeholder(tf.float32, (1, 224, 224, 3)))
    self.assertIn('Inference mode is created by default',
                  mobilenet_v3.edge_tpu.__doc__)
    self.assertEqual(endpoints['layer_24'].shape, [1, 7, 7, 1280])
    self.assertStartsWith(
        endpoints['layer_24'].name, 'MobilenetEdgeTPU')

  def testMobilenetEdgeTpuChangeScope(self):
    _, endpoints = mobilenet_v3.edge_tpu(
        tf.compat.v1.placeholder(tf.float32, (1, 224, 224, 3)), scope='Scope')
    self.assertStartsWith(
        endpoints['layer_24'].name, 'Scope')

  def testMobilenetV3BaseOnly(self):
    result, endpoints = mobilenet_v3.mobilenet(
        tf.compat.v1.placeholder(tf.float32, (1, 224, 224, 3)),
        conv_defs=mobilenet_v3.V3_LARGE,
        base_only=True,
        final_endpoint='layer_17')
    # Get the latest layer before average pool.
    self.assertEqual(endpoints['layer_17'].shape, [1, 7, 7, 960])
    self.assertEqual(result, endpoints['layer_17'])

  def testMobilenetV3BaseOnly_VariableInput(self):
    result, endpoints = mobilenet_v3.mobilenet(
        tf.placeholder(tf.float32, (None, None, None, 3)),
        conv_defs=mobilenet_v3.V3_LARGE,
        base_only=True,
        final_endpoint='layer_17')
    # Get the latest layer before average pool.
    self.assertEqual(endpoints['layer_17'].shape.as_list(),
                     [None, None, None, 960])
    self.assertEqual(result, endpoints['layer_17'])

if __name__ == '__main__':
  absltest.main()
