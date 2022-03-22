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

"""Tests for RevNet."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.modeling.backbones import revnet


class RevNetTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (128, 56, 4),
      (128, 104, 4),
  )
  def test_network_creation(self, input_size, model_id,
                            endpoint_filter_scale):
    """Test creation of RevNet family models."""
    tf.keras.backend.set_image_data_format('channels_last')

    network = revnet.RevNet(model_id=model_id)
    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)
    network.summary()

    self.assertAllEqual(
        [1, input_size / 2**2, input_size / 2**2, 128 * endpoint_filter_scale],
        endpoints['2'].shape.as_list())
    self.assertAllEqual(
        [1, input_size / 2**3, input_size / 2**3, 256 * endpoint_filter_scale],
        endpoints['3'].shape.as_list())
    self.assertAllEqual(
        [1, input_size / 2**4, input_size / 2**4, 512 * endpoint_filter_scale],
        endpoints['4'].shape.as_list())
    self.assertAllEqual(
        [1, input_size / 2**5, input_size / 2**5, 832 * endpoint_filter_scale],
        endpoints['5'].shape.as_list())

  @parameterized.parameters(1, 3, 4)
  def test_input_specs(self, input_dim):
    """Test different input feature dimensions."""
    tf.keras.backend.set_image_data_format('channels_last')

    input_specs = tf.keras.layers.InputSpec(shape=[None, None, None, input_dim])
    network = revnet.RevNet(model_id=56, input_specs=input_specs)

    inputs = tf.keras.Input(shape=(128, 128, input_dim), batch_size=1)
    _ = network(inputs)

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        model_id=56,
        activation='relu',
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
    )
    network = revnet.RevNet(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = revnet.RevNet.from_config(network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == '__main__':
  tf.test.main()
