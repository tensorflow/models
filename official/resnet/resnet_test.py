# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf


from official.resnet import resnet_model  # pylint: disable=g-bad-import-order


class BlockTest(tf.test.TestCase):

  def dense_run(self, tf_seed):
    """Simple generation of one random float and a single node dense network.

      The subsequent more involved tests depend on the ability to correctly seed
    TensorFlow. In the event that that process does not function as expected,
    the simple dense tests will fail indicating that the issue is with the
    tests rather than the ResNet functions.

    Args:
      tf_seed: Random seed for TensorFlow

    Returns:
      The generated random number and result of the dense network.
    """
    with self.test_session(graph=tf.Graph()) as sess:
      tf.set_random_seed(tf_seed)

      x = tf.random_uniform((1, 1))
      y = tf.layers.dense(inputs=x, units=1)

      init = tf.global_variables_initializer()
      sess.run(init)
      return x.eval()[0, 0], y.eval()[0, 0]

  def make_projection(self, filters_out, strides, data_format):
    """1D convolution with stride projector.

    Args:
      filters_out: Number of filters in the projection.
      strides: Stride length for convolution.
      data_format: channels_first or channels_last

    Returns:
      A 1 wide CNN projector function.
    """
    def projection_shortcut(inputs):
      return resnet_model.conv2d_fixed_padding(
          inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
          data_format=data_format)
    return projection_shortcut

  def resnet_block_run(self, tf_seed, batch_size, bottleneck, projection,
                       version, width, channels):
    """Test whether resnet block construction has changed.

      This function runs ResNet block construction under a variety of different
    conditions.

    Args:
      tf_seed: Random seed for TensorFlow
      batch_size: Number of points in the fake image. This is needed due to
        batch normalization.
      bottleneck: Whether or not to use bottleneck layers.
      projection: Whether or not to project the input.
      version: Which version of ResNet to test.
      width: The width of the fake image.
      channels: The number of channels in the fake image.

    Returns:
      The size of the block output, as well as several check values.
    """
    data_format = "channels_last"

    if version == 1:
      block_fn = resnet_model._building_block_v1
      if bottleneck:
        block_fn = resnet_model._bottleneck_block_v1
    else:
      block_fn = resnet_model._building_block_v2
      if bottleneck:
        block_fn = resnet_model._bottleneck_block_v2

    with self.test_session(graph=tf.Graph()) as sess:
      tf.set_random_seed(tf_seed)

      strides = 1
      channels_out = channels
      projection_shortcut = None
      if projection:
        strides = 2
        channels_out *= strides
        projection_shortcut = self.make_projection(
            filters_out=channels_out, strides=strides, data_format=data_format)

      filters = channels_out
      if bottleneck:
        filters = channels_out // 4

      x = tf.random_uniform((batch_size, width, width, channels))

      y = block_fn(inputs=x, filters=filters, training=True,
                   projection_shortcut=projection_shortcut, strides=strides,
                   data_format=data_format)

      init = tf.global_variables_initializer()
      sess.run(init)

      y_array = y.eval()
      y_flat = y_array.flatten()
      return y_array.shape, (y_flat[0], y_flat[-1], np.sum(y_flat))

  def test_dense_0(self):
    """Sanity check 0 on dense layer."""
    computed = self.dense_run(1813835975)
    tf.assert_equal(computed, (0.8760674, 0.2547844))

  def test_dense_1(self):
    """Sanity check 1 on dense layer."""
    computed = self.dense_run(3574260356)
    tf.assert_equal(computed, (0.75590825, 0.5339718))

  def test_bottleneck_v1_width_32_channels_64_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        599400476, batch_size=32, bottleneck=True, projection=True,
        version=1, width=32, channels=64)
    tf.assert_equal(computed_size, (32, 16, 16, 128))
    tf.assert_equal(computed_values, (0.0, 0.92648625, 587702.4))

  def test_bottleneck_v2_width_32_channels_64_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        309580726, batch_size=32, bottleneck=True, projection=True,
        version=2, width=32, channels=64)
    tf.assert_equal(computed_size, (32, 16, 16, 128))
    tf.assert_equal(computed_values, (-1.8759897, -0.5546854, -12860.312))

  def test_bottleneck_v1_width_32_channels_64_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        1969060699, batch_size=32, bottleneck=True, projection=False,
        version=1, width=32, channels=64)
    tf.assert_equal(computed_size, (32, 32, 32, 64))
    tf.assert_equal(computed_values, (0.10141289, 0.0, 1483393.0))

  def test_bottleneck_v2_width_32_channels_64_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        1716369119, batch_size=32, bottleneck=True, projection=False,
        version=2, width=32, channels=64)
    tf.assert_equal(computed_size, (32, 32, 32, 64))
    tf.assert_equal(computed_values, (1.4106897, 0.7455499, 834762.75))

  def test_building_v1_width_32_channels_64_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        1455996458, batch_size=32, bottleneck=False, projection=True,
        version=1, width=32, channels=64)
    tf.assert_equal(computed_size, (32, 16, 16, 128))
    tf.assert_equal(computed_values, (0.0, 0.0, 591701.3))

  def test_building_v2_width_32_channels_64_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        2770738568, batch_size=32, bottleneck=False, projection=True,
        version=2, width=32, channels=64)
    tf.assert_equal(computed_size, (32, 16, 16, 128))
    tf.assert_equal(computed_values, (-0.1908517, 0.2792631, -45776.055))

  def test_building_v1_width_32_channels_64_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        1262621774, batch_size=32, bottleneck=False, projection=False,
        version=1, width=32, channels=64)
    tf.assert_equal(computed_size, (32, 32, 32, 64))
    tf.assert_equal(computed_values, (0.0, 0.0, 1493558.9))

  def test_building_v2_width_32_channels_64_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        3856195393, batch_size=32, bottleneck=False, projection=False,
        version=2, width=32, channels=64)
    tf.assert_equal(computed_size, (32, 32, 32, 64))
    tf.assert_equal(computed_values, (-0.12920928, 0.38566422, 1157867.9))


if __name__ == "__main__":
  tf.test.main()
