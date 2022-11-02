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

"""Tests for nn_blocks."""

from typing import Any, Iterable, Tuple
# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.projects.mosaic.qat.modeling.layers import nn_blocks


def distribution_strategy_combinations() -> Iterable[Tuple[Any, ...]]:
  """Returns the combinations of end-to-end tests to run."""
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.cloud_tpu_strategy,
          strategy_combinations.one_device_strategy_gpu,
      ],
  )


class NNBlocksTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (nn_blocks.MultiKernelGroupConvBlockQuantized, [32, 64]),
      (nn_blocks.MultiKernelGroupConvBlockQuantized, [64, 128]),
  )
  def test_multi_kernel_grouped_convolution_block_creation(
      self, block_fn, output_filter_depths):
    input_size = 32
    inputs = tf.keras.Input(shape=(input_size, input_size, 16), batch_size=1)
    block = block_fn(
        output_filter_depths=output_filter_depths, kernel_sizes=[3, 3])

    features = block(inputs)

    self.assertAllEqual([1, input_size, input_size,
                         sum(output_filter_depths)], features.shape.as_list())

  @parameterized.parameters(
      (nn_blocks.MosaicEncoderBlockQuantized, [32, 64], [3, 3], [2, 2]),
      (nn_blocks.MosaicEncoderBlockQuantized, [64, 128], [3, 1], [2, 4]),
      (nn_blocks.MosaicEncoderBlockQuantized, [128, 256], [1, 1], [1, 1]),
      (nn_blocks.MosaicEncoderBlockQuantized, [128, 256], [3, 3], [4, 4]),
  )
  def test_mosaic_encoder_block_creation(self, block_fn, branch_filter_depths,
                                         conv_kernel_sizes,
                                         pyramid_pool_bin_nums):
    input_size = 128
    in_filters = 24
    inputs = tf.keras.Input(
        shape=(input_size, input_size, in_filters), batch_size=1)
    block = block_fn(
        branch_filter_depths=branch_filter_depths,
        conv_kernel_sizes=conv_kernel_sizes,
        pyramid_pool_bin_nums=pyramid_pool_bin_nums)

    features = block(inputs)

    self.assertAllEqual([1, input_size, input_size,
                         sum(branch_filter_depths)], features.shape.as_list())

  @parameterized.parameters(
      (nn_blocks.DecoderSumMergeBlockQuantized, 32, [128, 64]),
      (nn_blocks.DecoderSumMergeBlockQuantized, 16, [32, 32]),
  )
  def test_decoder_sum_merge_block_creation(self, block_fn,
                                            decoder_projected_depth,
                                            output_size):
    inputs = (tf.keras.Input(shape=(64, 64, 128), batch_size=1),
              tf.keras.Input(shape=(16, 16, 256), batch_size=1))
    block = block_fn(
        decoder_projected_depth=decoder_projected_depth,
        output_size=output_size)

    features = block(inputs)

    self.assertAllEqual(
        [1, output_size[0], output_size[1], decoder_projected_depth],
        features.shape.as_list())

  @parameterized.parameters(
      (nn_blocks.DecoderConcatMergeBlockQuantized, 64, 32, [128, 64]),
      (nn_blocks.DecoderConcatMergeBlockQuantized, 256, 16, [32, 32]),
  )
  def test_decoder_concat_merge_block_creation(self, block_fn,
                                               decoder_internal_depth,
                                               decoder_projected_depth,
                                               output_size):
    inputs = (tf.keras.Input(shape=(64, 64, 128), batch_size=1),
              tf.keras.Input(shape=(16, 16, 256), batch_size=1))
    block = block_fn(
        decoder_internal_depth=decoder_internal_depth,
        decoder_projected_depth=decoder_projected_depth,
        output_size=output_size)

    features = block(inputs)

    self.assertAllEqual(
        [1, output_size[0], output_size[1], decoder_projected_depth],
        features.shape.as_list())

if __name__ == '__main__':
  tf.test.main()
