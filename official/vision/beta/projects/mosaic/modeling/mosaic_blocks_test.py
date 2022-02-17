# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

# Lint as: python3
"""Tests for mosaic_blocks."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.projects.mosaic.modeling import mosaic_blocks


class MosaicBlocksTest(parameterized.TestCase, tf.test.TestCase):

  def test_multi_kernel_group_conv_block(self):
    block = mosaic_blocks.MultiKernelGroupConvBlock([64, 64], [3, 5])
    inputs = tf.ones([1, 4, 4, 448])
    outputs = block(inputs)
    self.assertAllEqual(outputs.shape, [1, 4, 4, 128])

  def test_mosaic_encoder_block(self):
    block = mosaic_blocks.MosaicEncoderBlock([64, 64], [3, 5], [1, 4, 8, 16])
    inputs = tf.ones([1, 32, 32, 448])
    outputs = block(inputs)
    self.assertAllEqual(outputs.shape, [1, 32, 32, 128])

  def test_mosaic_encoder_block_odd_input_overlap_pool(self):
    block = mosaic_blocks.MosaicEncoderBlock([64, 64], [3, 5], [1, 4, 8, 16])
    inputs = tf.ones([1, 31, 31, 448])
    outputs = block(inputs)
    self.assertAllEqual(outputs.shape, [1, 31, 31, 128])

  def test_mosaic_encoder_non_separable_block(self):
    block = mosaic_blocks.MosaicEncoderBlock([64, 64], [3, 5], [1, 4, 8, 16],
                                             use_depthwise_convolution=False)
    inputs = tf.ones([1, 32, 32, 448])
    outputs = block(inputs)
    self.assertAllEqual(outputs.shape, [1, 32, 32, 128])

if __name__ == '__main__':
  tf.test.main()
