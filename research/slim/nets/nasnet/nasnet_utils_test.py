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
"""Tests for slim.nets.nasnet.nasnet_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets.nasnet import nasnet_utils


class NasnetUtilsTest(tf.test.TestCase):

  def testCalcReductionLayers(self):
    num_cells = 18
    num_reduction_layers = 2
    reduction_layers = nasnet_utils.calc_reduction_layers(
        num_cells, num_reduction_layers)
    self.assertEqual(len(reduction_layers), 2)
    self.assertEqual(reduction_layers[0], 6)
    self.assertEqual(reduction_layers[1], 12)

  def testGetChannelIndex(self):
    data_formats = ['NHWC', 'NCHW']
    for data_format in data_formats:
      index = nasnet_utils.get_channel_index(data_format)
      correct_index = 3 if data_format == 'NHWC' else 1
      self.assertEqual(index, correct_index)

  def testGetChannelDim(self):
    data_formats = ['NHWC', 'NCHW']
    shape = [10, 20, 30, 40]
    for data_format in data_formats:
      dim = nasnet_utils.get_channel_dim(shape, data_format)
      correct_dim = shape[3] if data_format == 'NHWC' else shape[1]
      self.assertEqual(dim, correct_dim)

  def testGlobalAvgPool(self):
    data_formats = ['NHWC', 'NCHW']
    inputs = tf.compat.v1.placeholder(tf.float32, (5, 10, 20, 10))
    for data_format in data_formats:
      output = nasnet_utils.global_avg_pool(
          inputs, data_format)
      self.assertEqual(output.shape, [5, 10])


if __name__ == '__main__':
  tf.test.main()
