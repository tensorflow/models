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

"""Tests for 3D UNet decoder."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from official.projects.volumetric_models.modeling.backbones import unet_3d
from official.projects.volumetric_models.modeling.decoders import unet_3d_decoder


class UNet3DDecoderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ([128, 64], 4),
      ([256, 128], 6),
  )
  def test_network_creation(self, input_size, model_id):
    """Test creation of UNet3D family models."""
    tf.keras.backend.set_image_data_format('channels_last')

    # `input_size` consists of [spatial size, volume size].
    inputs = tf.keras.Input(
        shape=(input_size[0], input_size[0], input_size[1], 3), batch_size=1)
    backbone = unet_3d.UNet3D(model_id=model_id)
    network = unet_3d_decoder.UNet3DDecoder(
        model_id=model_id, input_specs=backbone.output_specs)

    endpoints = backbone(inputs)
    feats = network(endpoints)

    self.assertIn('1', feats)
    self.assertAllEqual([1, input_size[0], input_size[0], input_size[1], 64],
                        feats['1'].shape.as_list())

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        model_id=4,
        input_specs=unet_3d.UNet3D(model_id=4).output_specs,
        pool_size=(2, 2, 2),
        kernel_size=(3, 3, 3),
        kernel_regularizer=None,
        activation='relu',
        norm_momentum=0.99,
        norm_epsilon=0.001,
        use_sync_bn=False,
        use_batch_normalization=True,
        use_deconvolution=True)
    network = unet_3d_decoder.UNet3DDecoder(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = unet_3d_decoder.UNet3DDecoder.from_config(
        network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == '__main__':
  tf.test.main()
