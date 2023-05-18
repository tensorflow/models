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

"""Tests for VIT."""

import math

from absl.testing import parameterized
import tensorflow as tf

from official.vision.modeling.backbones import vit


class VisionTransformerTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (224, 85798656),
      (256, 85844736),
  )
  def test_network_creation(self, input_size, params_count):
    """Test creation of VisionTransformer family models."""
    tf.keras.backend.set_image_data_format('channels_last')
    input_specs = tf.keras.layers.InputSpec(
        shape=[2, input_size, input_size, 3])
    network = vit.VisionTransformer(input_specs=input_specs)

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    _ = network(inputs)
    self.assertEqual(network.count_params(), params_count)

  @parameterized.product(
      patch_size=[6, 4],
      output_2d_feature_maps=[True, False],
      pooler=['none', 'gap', 'token'],
  )
  def test_network_with_diferent_configs(
      self, patch_size, output_2d_feature_maps, pooler):
    tf.keras.backend.set_image_data_format('channels_last')
    input_size = 24
    expected_feat_level = str(round(math.log2(patch_size)))
    num_patch_rows = input_size // patch_size
    input_specs = tf.keras.layers.InputSpec(
        shape=[2, input_size, input_size, 3])
    network = vit.VisionTransformer(
        input_specs=input_specs,
        patch_size=patch_size,
        pooler=pooler,
        hidden_size=8,
        mlp_dim=8,
        num_layers=1,
        num_heads=2,
        representation_size=16,
        output_2d_feature_maps=output_2d_feature_maps)

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    output = network(inputs)
    if pooler == 'none':
      self.assertEqual(
          output['encoded_tokens'].shape, [1, num_patch_rows**2, 16])
    else:
      self.assertEqual(output['pre_logits'].shape, [1, 1, 1, 16])

    if output_2d_feature_maps:
      self.assertIn(expected_feat_level, output)
      self.assertIn(expected_feat_level, network.output_specs)
      self.assertEqual(
          network.output_specs[expected_feat_level][1:],
          [num_patch_rows, num_patch_rows, 8])
    else:
      self.assertNotIn(expected_feat_level, output)

  def test_posembedding_interpolation(self):
    tf.keras.backend.set_image_data_format('channels_last')
    input_size = 256
    input_specs = tf.keras.layers.InputSpec(
        shape=[2, input_size, input_size, 3])
    network = vit.VisionTransformer(
        input_specs=input_specs,
        patch_size=16,
        pooler='gap',
        pos_embed_shape=(14, 14))  # (224 // 16)

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    output = network(inputs)['pre_logits']
    self.assertEqual(output.shape, [1, 1, 1, 768])


if __name__ == '__main__':
  tf.test.main()
