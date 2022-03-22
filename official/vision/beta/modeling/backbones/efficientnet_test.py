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

"""Tests for EfficientNet."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.modeling.backbones import efficientnet


class EfficientNetTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(32, 224)
  def test_network_creation(self, input_size):
    """Test creation of EfficientNet family models."""
    tf.keras.backend.set_image_data_format('channels_last')

    network = efficientnet.EfficientNet(model_id='b0')

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)

    self.assertAllEqual([1, input_size / 2**2, input_size / 2**2, 24],
                        endpoints['2'].shape.as_list())
    self.assertAllEqual([1, input_size / 2**3, input_size / 2**3, 40],
                        endpoints['3'].shape.as_list())
    self.assertAllEqual([1, input_size / 2**4, input_size / 2**4, 112],
                        endpoints['4'].shape.as_list())
    self.assertAllEqual([1, input_size / 2**5, input_size / 2**5, 320],
                        endpoints['5'].shape.as_list())

  @parameterized.parameters('b0', 'b3', 'b6')
  def test_network_scaling(self, model_id):
    """Test compound scaling."""
    efficientnet_params = {
        'b0': 4049564,
        'b3': 10783528,
        'b6': 40960136,
    }
    tf.keras.backend.set_image_data_format('channels_last')

    input_size = 32
    network = efficientnet.EfficientNet(model_id=model_id, se_ratio=0.25)
    self.assertEqual(network.count_params(), efficientnet_params[model_id])

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    _ = network(inputs)

  @parameterized.parameters(1, 3)
  def test_input_specs(self, input_dim):
    """Test different input feature dimensions."""
    tf.keras.backend.set_image_data_format('channels_last')

    input_specs = tf.keras.layers.InputSpec(shape=[None, None, None, input_dim])
    network = efficientnet.EfficientNet(model_id='b0', input_specs=input_specs)

    inputs = tf.keras.Input(shape=(128, 128, input_dim), batch_size=1)
    _ = network(inputs)

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        model_id='b0',
        se_ratio=0.25,
        stochastic_depth_drop_rate=None,
        use_sync_bn=False,
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_regularizer=None,
        activation='relu',
        norm_momentum=0.99,
        norm_epsilon=0.001,
    )
    network = efficientnet.EfficientNet(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = efficientnet.EfficientNet.from_config(network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == '__main__':
  tf.test.main()
