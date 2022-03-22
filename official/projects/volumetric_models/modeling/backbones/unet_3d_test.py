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

"""Tests for 3D UNet backbone."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from official.projects.volumetric_models.modeling.backbones import unet_3d


class UNet3DTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ([128, 64], 4),
      ([256, 128], 6),
  )
  def test_network_creation(self, input_size, model_id):
    """Test creation of UNet3D family models."""
    tf.keras.backend.set_image_data_format('channels_last')
    network = unet_3d.UNet3D(model_id=model_id)
    inputs = tf.keras.Input(
        shape=(input_size[0], input_size[0], input_size[1], 3), batch_size=1)
    endpoints = network(inputs)

    for layer_depth in range(model_id):
      self.assertAllEqual([
          1, input_size[0] / 2**layer_depth, input_size[0] / 2**layer_depth,
          input_size[1] / 2**layer_depth, 64 * 2**layer_depth
      ], endpoints[str(layer_depth + 1)].shape.as_list())

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        model_id=4,
        pool_size=(2, 2, 2),
        kernel_size=(3, 3, 3),
        activation='relu',
        base_filters=32,
        kernel_regularizer=None,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        use_sync_bn=False,
        use_batch_normalization=True)
    network = unet_3d.UNet3D(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = unet_3d.UNet3D.from_config(network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == '__main__':
  tf.test.main()
