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

"""Tests for resnet."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.vision.modeling.backbones import resnet_unet


class ResNetUNetTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (128, 50, 4),
  )
  def test_network_creation(self, input_size, model_id, endpoint_filter_scale):
    """Test creation of ResNet family models."""
    resnet_unet_params = {
        50: 55_205_440,
    }
    tf_keras.backend.set_image_data_format('channels_last')

    network = resnet_unet.ResNetUNet(
        model_id=model_id,
        upsample_repeats=[18, 1, 1],
        upsample_filters=[384, 384, 384],
        upsample_kernel_sizes=[7, 7, 7],
    )
    self.assertEqual(network.count_params(), resnet_unet_params[model_id])

    inputs = tf_keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)
    print(endpoints)

    self.assertAllEqual(
        [1, input_size / 2**2, input_size / 2**2, 64 * endpoint_filter_scale],
        endpoints['2'].shape.as_list(),
    )
    for i in range(3, 6):
      self.assertAllEqual(
          [1, input_size / 2**i, input_size / 2**i, 384],
          endpoints[f'{i}'].shape.as_list(),
      )

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        model_id=50,
        upsample_repeats=[18, 1, 1],
        upsample_filters=[384, 384, 384],
        upsample_kernel_sizes=[7, 7, 7],
    )
    network = resnet_unet.ResNetUNet(**kwargs)

    # Create another network object from the first object's config.
    new_network = resnet_unet.ResNetUNet.from_config(network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == '__main__':
  tf.test.main()
