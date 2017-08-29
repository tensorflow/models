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


from nets import fc_densenet
from nets import densenet_utils

densenet_arg_scope = densenet_utils.densenet_arg_scope

slim = tf.contrib.slim


class FCDensenetTest(tf.test.TestCase):

  def testTransitionUp(self):
    batch_size = 5
    height, width = 100, 100
    channels = 16
    skip_connection_channels = 48
    block_to_upsample = tf.random_uniform((batch_size, height, width,
                                           channels))
    skip_connection = tf.random_uniform((batch_size, 2*height, 2*width,
                                         skip_connection_channels))
    n_filters_keep = 8

    output = densenet_utils.TransitionUp(block_to_upsample, skip_connection,
                                n_filters_keep)
    output_shape = [batch_size, 2*height, 2*width,
                    n_filters_keep + skip_connection_channels]
    self.assertListEqual(output.get_shape().as_list(), output_shape)

  def testBuildNetwork(self):
    batch_size = 5
    height, width = 200, 200
    num_classes = 1000
    n_pool = 2

    inputs = tf.random_uniform((batch_size, height, width, 3))
    with slim.arg_scope(densenet_arg_scope()) as sc:
      logits, end_points = fc_densenet.fc_densenet(inputs, num_classes,
                                                   n_pool=n_pool)
    self.assertTrue(logits.op.name.startswith('fc_densenet/logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, height, width, num_classes])
    self.assertTrue('predictions' in end_points)
    self.assertListEqual(end_points['predictions'].get_shape().as_list(),
                         [batch_size, height, width, num_classes])

if __name__ == '__main__':
  tf.test.main()
