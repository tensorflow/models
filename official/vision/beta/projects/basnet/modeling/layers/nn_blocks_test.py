# Lint as: python3
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
"""Tests for nn_blocks."""

from typing import Any, Iterable, Tuple
# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.vision.beta.modeling.layers import nn_blocks


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
      (nn_blocks.ResidualBlock, 1, False, 0.0, None),
      (nn_blocks.ResidualBlock, 2, True, 0.2, 0.25),
  )
  def test_residual_block_creation(self, block_fn, strides, use_projection,
                                   stochastic_depth_drop_rate, se_ratio):
    input_size = 128
    filter_size = 256
    inputs = tf.keras.Input(
        shape=(input_size, input_size, filter_size), batch_size=1)
    block = block_fn(
        filter_size,
        strides,
        use_projection=use_projection,
        se_ratio=se_ratio,
        stochastic_depth_drop_rate=stochastic_depth_drop_rate,
    )

    features = block(inputs)

    self.assertAllEqual(
        [1, input_size // strides, input_size // strides, filter_size],
        features.shape.as_list())

if __name__ == '__main__':
  tf.test.main()
