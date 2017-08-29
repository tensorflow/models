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
"""Tests for slim.nets.densenet"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from nets import densenet

slim = tf.contrib.slim


class DensenetTest(tf.test.TestCase):

  def testBuildClassificationNetwork(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000

    inputs = tf.random_uniform((batch_size, height, width, 3))
    logits, end_points = densenet.densenet(inputs, num_classes)
    self.assertTrue(logits.op.name.startswith('densenet/logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])
    self.assertTrue('predictions' in end_points)
    self.assertListEqual(end_points['predictions'].get_shape().as_list(),
                         [batch_size, num_classes])

if __name__ == '__main__':
  tf.test.main()
