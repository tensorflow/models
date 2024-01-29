# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for mosaic_blocks."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from official.projects.mosaic.modeling import mosaic_blocks


class MosaicBlocksTest(parameterized.TestCase, tf.test.TestCase):

  def test_multi_kernel_group_conv_block(self):
    block = mosaic_blocks.MultiKernelGroupConvBlock([64, 64], [3, 5])
    inputs = tf.ones([1, 4, 4, 448])
    outputs = block(inputs)
    self.assertAllEqual(outputs.shape, [1, 4, 4, 128])

  def test_mosaic_encoder_block(self):
    block = mosaic_blocks.MosaicEncoderBlock(
        encoder_input_level=4,
        branch_filter_depths=[64, 64],
        conv_kernel_sizes=[3, 5],
        pyramid_pool_bin_nums=[1, 4, 8, 16])
    inputs = tf.ones([1, 32, 32, 448])
    outputs = block(inputs)
    self.assertAllEqual(outputs.shape, [1, 32, 32, 128])

  def test_mosaic_encoder_block_odd_input_overlap_pool(self):
    block = mosaic_blocks.MosaicEncoderBlock(
        encoder_input_level=4,
        branch_filter_depths=[64, 64],
        conv_kernel_sizes=[3, 5],
        pyramid_pool_bin_nums=[1, 4, 8, 16])
    inputs = tf.ones([1, 31, 31, 448])
    outputs = block(inputs)
    self.assertAllEqual(outputs.shape, [1, 31, 31, 128])

  def test_mosaic_encoder_non_separable_block(self):
    block = mosaic_blocks.MosaicEncoderBlock(
        encoder_input_level=4,
        branch_filter_depths=[64, 64],
        conv_kernel_sizes=[3, 5],
        pyramid_pool_bin_nums=[1, 4, 8, 16],
        use_depthwise_convolution=False)
    inputs = tf.ones([1, 32, 32, 448])
    outputs = block(inputs)
    self.assertAllEqual(outputs.shape, [1, 32, 32, 128])

  def test_mosaic_decoder_concat_merge_block(self):
    concat_merge_block = mosaic_blocks.DecoderConcatMergeBlock(64, 32, [64, 64])
    inputs = [tf.ones([1, 32, 32, 128]), tf.ones([1, 64, 64, 192])]
    outputs = concat_merge_block(inputs)
    self.assertAllEqual(outputs.shape, [1, 64, 64, 32])

  def test_mosaic_decoder_concat_merge_block_default_output_size(self):
    concat_merge_block = mosaic_blocks.DecoderConcatMergeBlock(64, 32)
    inputs = [tf.ones([1, 32, 32, 128]), tf.ones([1, 64, 64, 192])]
    outputs = concat_merge_block(inputs)
    self.assertAllEqual(outputs.shape, [1, 64, 64, 32])

  def test_mosaic_decoder_concat_merge_block_default_output_size_4x(self):
    concat_merge_block = mosaic_blocks.DecoderConcatMergeBlock(64, 32)
    inputs = [tf.ones([1, 32, 32, 128]), tf.ones([1, 128, 128, 192])]
    outputs = concat_merge_block(inputs)
    self.assertAllEqual(outputs.shape, [1, 128, 128, 32])

  def test_mosaic_decoder_concat_merge_block_default_output_size_4x_rec(self):
    concat_merge_block = mosaic_blocks.DecoderConcatMergeBlock(64, 32)
    inputs = [tf.ones([1, 32, 64, 128]), tf.ones([1, 128, 256, 64])]
    outputs = concat_merge_block(inputs)
    self.assertAllEqual(outputs.shape, [1, 128, 256, 32])

  def test_mosaic_decoder_sum_merge_block(self):
    concat_merge_block = mosaic_blocks.DecoderSumMergeBlock(32, [128, 128])
    inputs = [tf.ones([1, 64, 64, 32]), tf.ones([1, 128, 128, 64])]
    outputs = concat_merge_block(inputs)
    self.assertAllEqual(outputs.shape, [1, 128, 128, 32])

  def test_mosaic_decoder_sum_merge_block_default_output_size(self):
    concat_merge_block = mosaic_blocks.DecoderSumMergeBlock(32)
    inputs = [tf.ones([1, 64, 64, 32]), tf.ones([1, 128, 128, 64])]
    outputs = concat_merge_block(inputs)
    self.assertAllEqual(outputs.shape, [1, 128, 128, 32])

if __name__ == '__main__':
  tf.test.main()
