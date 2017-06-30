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

from nets import nets_factory

slim = tf.contrib.slim


class NetworksTest(tf.test.TestCase):

  def testGetNetworkFn(self):
    batch_size = 5
    num_classes = 1000
    for net in nets_factory.networks_map:
      with self.test_session():
        net_fn = nets_factory.get_network_fn(net, num_classes)
        # Most networks use 224 as their default_image_size
        image_size = getattr(net_fn, 'default_image_size', 224)
        inputs = tf.random_uniform((batch_size, image_size, image_size, 3))
        logits, end_points = net_fn(inputs)
        self.assertTrue(isinstance(logits, tf.Tensor))
        self.assertTrue(isinstance(end_points, dict))
        self.assertEqual(logits.get_shape().as_list()[0], batch_size)
        self.assertEqual(logits.get_shape().as_list()[-1], num_classes)

  def testGetNetworkFnArgScope(self):
    batch_size = 5
    num_classes = 10
    net = 'cifarnet'
    with self.test_session(use_gpu=True):
      net_fn = nets_factory.get_network_fn(net, num_classes)
      image_size = getattr(net_fn, 'default_image_size', 224)
      with slim.arg_scope([slim.model_variable, slim.variable],
                          device='/CPU:0'):
        inputs = tf.random_uniform((batch_size, image_size, image_size, 3))
        net_fn(inputs)
      weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'CifarNet/conv1')[0]
      self.assertDeviceEqual('/CPU:0', weights.device)

if __name__ == '__main__':
  tf.test.main()
