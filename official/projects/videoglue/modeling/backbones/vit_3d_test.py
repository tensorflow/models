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

"""Tests for vit_3d."""
from absl.testing import parameterized
import tensorflow as tf

from official.projects.videoglue.modeling.backbones import vit_3d


class Vit3DTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (8, 224, 87718656),
      (16, 256, 88204032),
  )
  def test_network_creation(self, num_frames, input_size, params_count):
    """Test creation of VisionTransformer family models."""
    tf.keras.backend.set_image_data_format('channels_last')
    input_specs = tf.keras.layers.InputSpec(
        shape=[2, num_frames, input_size, input_size, 3])
    network = vit_3d.VisionTransformer3D(input_specs=input_specs)

    inputs = tf.keras.Input(
        shape=(num_frames, input_size, input_size, 3), batch_size=1)
    _ = network(inputs)
    self.assertEqual(network.count_params(), params_count)

  def test_network_none_pooler(self):
    """Tests creation of VisionTransformer family models."""
    tf.keras.backend.set_image_data_format('channels_last')
    num_frames = 8
    input_size = 224
    input_specs = tf.keras.layers.InputSpec(
        shape=[2, num_frames, input_size, input_size, 3])
    network = vit_3d.VisionTransformer3D(
        input_specs=input_specs,
        pooler='none',
        representation_size=128)

    inputs = tf.keras.Input(
        shape=(num_frames, input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)
    self.assertEqual(endpoints['encoded_tokens'].shape, [1, 2, 14, 14, 128])

  @parameterized.parameters('native', 'mae')
  def test_network_convention(self, variant):
    """Tests creation of VisionTransformer family models."""
    tf.keras.backend.set_image_data_format('channels_last')
    num_frames = 8
    input_size = 224
    input_specs = tf.keras.layers.InputSpec(
        shape=[2, num_frames, input_size, input_size, 3])
    network = vit_3d.VisionTransformer3D(
        variant=variant,
        input_specs=input_specs,
        pooler='none',
        representation_size=128)

    inputs = tf.keras.Input(
        shape=(num_frames, input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)
    self.assertEqual(endpoints['encoded_tokens'].shape, [1, 2, 14, 14, 128])

  def test_network_pos_embed_interpolation_mae(self):
    """Tests creation of VisionTransformer family models."""
    tf.keras.backend.set_image_data_format('channels_last')
    variant = 'mae'
    pos_embed_shape = (8, 14, 14)
    num_frames = 8
    input_size = 256
    input_specs = tf.keras.layers.InputSpec(
        shape=[2, num_frames, input_size, input_size, 3])
    network = vit_3d.VisionTransformer3D(
        variant=variant,
        input_specs=input_specs,
        pooler='none',
        representation_size=128,
        pos_embed_shape=pos_embed_shape)

    inputs = tf.keras.Input(
        shape=(num_frames, input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)
    self.assertEqual(endpoints['encoded_tokens'].shape, [1, 2, 16, 16, 128])


if __name__ == '__main__':
  tf.test.main()
