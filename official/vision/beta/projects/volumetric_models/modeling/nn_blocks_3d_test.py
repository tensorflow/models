# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for 3D volumeric convoluion blocks."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.projects.volumetric_models.modeling import nn_blocks_3d


class NNBlocks3DTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((128, 128, 32, 1), (256, 256, 16, 2))
  def test_bottleneck_block_3d_volume_creation(self, spatial_size, volume_size,
                                               filters, strides):
    inputs = tf.keras.Input(
        shape=(spatial_size, spatial_size, volume_size, filters * 4),
        batch_size=1)
    block = nn_blocks_3d.BottleneckBlock3DVolume(
        filters=filters, strides=strides, use_projection=True)

    features = block(inputs)

    self.assertAllEqual([
        1, spatial_size // strides, spatial_size // strides,
        volume_size // strides, filters * 4
    ], features.shape.as_list())

  @parameterized.parameters((128, 128, 32, 1), (256, 256, 64, 2))
  def test_residual_block_3d_volume_creation(self, spatial_size, volume_size,
                                             filters, strides):
    inputs = tf.keras.Input(
        shape=(spatial_size, spatial_size, volume_size, filters), batch_size=1)
    block = nn_blocks_3d.ResidualBlock3DVolume(
        filters=filters, strides=strides, use_projection=True)

    features = block(inputs)

    self.assertAllEqual([
        1, spatial_size // strides, spatial_size // strides,
        volume_size // strides, filters
    ], features.shape.as_list())

  @parameterized.parameters((128, 128, 64, 1, 3), (256, 256, 128, 2, 1))
  def test_basic_block_3d_volume_creation(self, spatial_size, volume_size,
                                          filters, strides, kernel_size):
    inputs = tf.keras.Input(
        shape=(spatial_size, spatial_size, volume_size, filters), batch_size=1)
    block = nn_blocks_3d.BasicBlock3DVolume(
        filters=filters, strides=strides, kernel_size=kernel_size)

    features = block(inputs)

    self.assertAllEqual([
        1, spatial_size // strides, spatial_size // strides,
        volume_size // strides, filters
    ], features.shape.as_list())


if __name__ == '__main__':
  tf.test.main()
