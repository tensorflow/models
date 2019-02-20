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


class NetworksTest(tf.test.TestCase):

  def testGetNetworkFnFirstHalf(self):
    batch_size = 5
    num_classes = 1000
    for net in list(nets_factory.networks_map.keys())[:10]:
      with tf.Graph().as_default() as g, self.test_session(g):
        net_fn = nets_factory.get_network_fn(net, num_classes=num_classes)
        # Most networks use 224 as their default_image_size
        image_size = getattr(net_fn, 'default_image_size', 224)
        if net not in ['i3d', 's3dg']:
          inputs = tf.random_uniform(
              (batch_size, image_size, image_size, 3))
          logits, end_points = net_fn(inputs)
          self.assertTrue(isinstance(logits, tf.Tensor))
          self.assertTrue(isinstance(end_points, dict))
          self.assertEqual(logits.get_shape().as_list()[0], batch_size)
          self.assertEqual(logits.get_shape().as_list()[-1], num_classes)

  def testGetNetworkFnSecondHalf(self):
    batch_size = 5
    num_classes = 1000
    for net in list(nets_factory.networks_map.keys())[10:]:
      with tf.Graph().as_default() as g, self.test_session(g):
        net_fn = nets_factory.get_network_fn(net, num_classes=num_classes)
        # Most networks use 224 as their default_image_size
        image_size = getattr(net_fn, 'default_image_size', 224)
        if net not in ['i3d', 's3dg']:
          inputs = tf.random_uniform(
              (batch_size, image_size, image_size, 3))
          logits, end_points = net_fn(inputs)
          self.assertTrue(isinstance(logits, tf.Tensor))
          self.assertTrue(isinstance(end_points, dict))
          self.assertEqual(logits.get_shape().as_list()[0], batch_size)
          self.assertEqual(logits.get_shape().as_list()[-1], num_classes)

  def testGetNetworkFnVideoModels(self):
    batch_size = 5
    num_classes = 400
    for net in ['i3d', 's3dg']:
      with tf.Graph().as_default() as g, self.test_session(g):
        net_fn = nets_factory.get_network_fn(net, num_classes=num_classes)
        # Most networks use 224 as their default_image_size
        image_size = getattr(net_fn, 'default_image_size', 224) // 2
        inputs = tf.random_uniform(
            (batch_size, 10, image_size, image_size, 3))
        logits, end_points = net_fn(inputs)
        self.assertTrue(isinstance(logits, tf.Tensor))
        self.assertTrue(isinstance(end_points, dict))
        self.assertEqual(logits.get_shape().as_list()[0], batch_size)
        self.assertEqual(logits.get_shape().as_list()[-1], num_classes)

if __name__ == '__main__':
  tf.test.main()
