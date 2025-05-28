# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for resnet."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.vision.modeling.layers import nn_blocks_3d


class NNBlocksTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (nn_blocks_3d.BottleneckBlock3D, 1, 1, 2, True, 0.2, 0.1),
      (nn_blocks_3d.BottleneckBlock3D, 3, 2, 1, False, 0.0, 0.0),
  )
  def test_bottleneck_block_creation(self, block_fn, temporal_kernel_size,
                                     temporal_strides, spatial_strides,
                                     use_self_gating, se_ratio,
                                     stochastic_depth):
    temporal_size = 16
    spatial_size = 128
    filters = 256
    inputs = tf_keras.Input(
        shape=(temporal_size, spatial_size, spatial_size, filters * 4),
        batch_size=1)
    block = block_fn(
        filters=filters,
        temporal_kernel_size=temporal_kernel_size,
        temporal_strides=temporal_strides,
        spatial_strides=spatial_strides,
        use_self_gating=use_self_gating,
        se_ratio=se_ratio,
        stochastic_depth_drop_rate=stochastic_depth)

    features = block(inputs)

    self.assertAllEqual([
        1, temporal_size // temporal_strides, spatial_size // spatial_strides,
        spatial_size // spatial_strides, filters * 4
    ], features.shape.as_list())


if __name__ == '__main__':
  tf.test.main()
