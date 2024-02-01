# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for ASPP."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.vision.modeling.layers import deeplab


class DeeplabTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (None,),
      ([32, 32],),
      )
  def test_aspp(self, pool_kernel_size):
    del pool_kernel_size
    inputs = tf_keras.Input(shape=(64, 64, 128), dtype=tf.float32)
    layer = deeplab.SpatialPyramidPooling(output_channels=256,
                                          dilation_rates=[6, 12, 18],
                                          pool_kernel_size=None)
    output = layer(inputs)
    self.assertAllEqual([None, 64, 64, 256], output.shape)

  def test_aspp_invalid_shape(self):
    inputs = tf_keras.Input(shape=(64, 64), dtype=tf.float32)
    layer = deeplab.SpatialPyramidPooling(output_channels=256,
                                          dilation_rates=[6, 12, 18])
    with self.assertRaises(ValueError):
      _ = layer(inputs)

  def test_config_with_custom_name(self):
    layer = deeplab.SpatialPyramidPooling(256, [5], name='aspp')
    config = layer.get_config()
    layer_1 = deeplab.SpatialPyramidPooling.from_config(config)
    self.assertEqual(layer_1.name, layer.name)


if __name__ == '__main__':
  tf.test.main()
