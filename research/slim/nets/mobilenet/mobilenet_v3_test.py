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

import tensorflow.compat.v1 as tf

from nets.mobilenet import mobilenet_v3
from google3.testing.pybase import parameterized


class MobilenetV3Test(tf.test.TestCase, parameterized.TestCase):

  # pylint: disable = g-unreachable-test-method
  def assertVariablesHaveNormalizerFn(self, use_groupnorm):
    global_variables = [v.name for v in tf.global_variables()]
    has_batch_norm = False
    has_group_norm = False
    for global_variable in global_variables:
      if 'BatchNorm' in global_variable:
        has_batch_norm = True
      if 'GroupNorm' in global_variable:
        has_group_norm = True
    if use_groupnorm:
      self.assertFalse(has_batch_norm)
      self.assertTrue(has_group_norm)
    else:
      self.assertTrue(has_batch_norm)
      self.assertFalse(has_group_norm)

  @parameterized.named_parameters(('without_groupnorm', False),
                                  ('with_groupnorm', True))
  def testMobilenetV3Large(self, use_groupnorm):
    logits, endpoints = mobilenet_v3.mobilenet(
        tf.placeholder(tf.float32, (1, 224, 224, 3)),
        use_groupnorm=use_groupnorm)
    self.assertEqual(endpoints['layer_19'].shape, [1, 1, 1, 1280])
    self.assertEqual(logits.shape, [1, 1001])
    self.assertVariablesHaveNormalizerFn(use_groupnorm)

  @parameterized.named_parameters(('without_groupnorm', False),
                                  ('with_groupnorm', True))
  def testMobilenetV3Small(self, use_groupnorm):
    _, endpoints = mobilenet_v3.mobilenet(
        tf.placeholder(tf.float32, (1, 224, 224, 3)),
        conv_defs=mobilenet_v3.V3_SMALL,
        use_groupnorm=use_groupnorm)
    self.assertEqual(endpoints['layer_15'].shape, [1, 1, 1, 1024])
    self.assertVariablesHaveNormalizerFn(use_groupnorm)

  @parameterized.named_parameters(('without_groupnorm', False),
                                  ('with_groupnorm', True))
  def testMobilenetEdgeTpu(self, use_groupnorm):
    _, endpoints = mobilenet_v3.edge_tpu(
        tf.placeholder(tf.float32, (1, 224, 224, 3)),
        use_groupnorm=use_groupnorm)
    self.assertIn('Inference mode is created by default',
                  mobilenet_v3.edge_tpu.__doc__)
    self.assertEqual(endpoints['layer_24'].shape, [1, 7, 7, 1280])
    self.assertStartsWith(
        endpoints['layer_24'].name, 'MobilenetEdgeTPU')
    self.assertVariablesHaveNormalizerFn(use_groupnorm)

  def testMobilenetEdgeTpuChangeScope(self):
    _, endpoints = mobilenet_v3.edge_tpu(
        tf.placeholder(tf.float32, (1, 224, 224, 3)), scope='Scope')
    self.assertStartsWith(
        endpoints['layer_24'].name, 'Scope')

  @parameterized.named_parameters(('without_groupnorm', False),
                                  ('with_groupnorm', True))
  def testMobilenetV3BaseOnly(self, use_groupnorm):
    result, endpoints = mobilenet_v3.mobilenet(
        tf.placeholder(tf.float32, (1, 224, 224, 3)),
        conv_defs=mobilenet_v3.V3_LARGE,
        use_groupnorm=use_groupnorm,
        base_only=True,
        final_endpoint='layer_17')
    # Get the latest layer before average pool.
    self.assertEqual(endpoints['layer_17'].shape, [1, 7, 7, 960])
    self.assertEqual(result, endpoints['layer_17'])
    self.assertVariablesHaveNormalizerFn(use_groupnorm)

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

  # Use reduce mean for pooling and check for operation 'ReduceMean' in graph
  @parameterized.named_parameters(('without_groupnorm', False),
                                  ('with_groupnorm', True))
  def testMobilenetV3WithReduceMean(self, use_groupnorm):
    _, _ = mobilenet_v3.mobilenet(
        tf.placeholder(tf.float32, (1, 224, 224, 3)),
        conv_defs=mobilenet_v3.V3_SMALL,
        use_groupnorm=use_groupnorm,
        use_reduce_mean_for_pooling=True)
    g = tf.get_default_graph()
    reduce_mean = [v for v in g.get_operations() if 'ReduceMean' in v.name]
    self.assertNotEmpty(reduce_mean)
    self.assertVariablesHaveNormalizerFn(use_groupnorm)

  @parameterized.named_parameters(('without_groupnorm', False),
                                  ('with_groupnorm', True))
  def testMobilenetV3WithOutReduceMean(self, use_groupnorm):
    _, _ = mobilenet_v3.mobilenet(
        tf.placeholder(tf.float32, (1, 224, 224, 3)),
        conv_defs=mobilenet_v3.V3_SMALL,
        use_groupnorm=use_groupnorm,
        use_reduce_mean_for_pooling=False)
    g = tf.get_default_graph()
    reduce_mean = [v for v in g.get_operations() if 'ReduceMean' in v.name]
    self.assertEmpty(reduce_mean)
    self.assertVariablesHaveNormalizerFn(use_groupnorm)


if __name__ == '__main__':
  # absltest.main()
  tf.test.main()
