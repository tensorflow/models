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
"""Tests for resnet.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from absl.testing import parameterized
from official.vision.detection.modeling.architecture import resnet


class ResnetTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (50, None, None, True),
      (101, None, None, True),
      (152, None, None, True),
      (200, None, None, True),
      (50, 0.9, 7, False),
      (50, 0.9, 7, True),
  )
  def testResNetOutputShape(self, resnet_depth, dropblock_keep_prob,
                            dropblock_size, is_training):

    inputs = tf.zeros([1, 256, 256, 3])
    resnet_fn = resnet.Resnet(
        resnet_depth,
        dropblock_keep_prob=dropblock_keep_prob,
        dropblock_size=dropblock_size)
    features = resnet_fn(inputs, is_training=is_training)
    self.assertEqual(features[2].get_shape().as_list(), [1, 64, 64, 256])
    self.assertEqual(features[3].get_shape().as_list(), [1, 32, 32, 512])
    self.assertEqual(features[4].get_shape().as_list(), [1, 16, 16, 1024])
    self.assertEqual(features[5].get_shape().as_list(), [1, 8, 8, 2048])

if __name__ == '__main__':
  assert tf.version.VERSION.startswith('2.')
  tf.test.main()
