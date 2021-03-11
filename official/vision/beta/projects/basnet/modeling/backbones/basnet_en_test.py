# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for resnet_deeplab models."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.vision.beta.modeling.backbones import basnet_en


class ResNetTest(parameterized.TestCase, tf.test.TestCase):

  #@parameterized.parameters(
  #    (256),
  #)
  #def test_network_creation(self, input_size, model_id,
  #                          endpoint_filter_scale, output_stride):
  def test_network_creation(self):
    """Test creation of ResNet models."""

    input_size = 224

    tf.keras.backend.set_image_data_format('channels_last')

    network = basnet_en.BASNet_En()
    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)
    print(endpoints)
    #print(endpoints[str(int(np.math.log2(32)))].shape.as_list())
    self.assertAllEqual([
        1, input_size / 32, input_size / 32,
        512
    ], endpoints[str(int(np.math.log2(32)))].shape.as_list())

  @combinations.generate(
      combinations.combine(
          strategy=[
              #strategy_combinations.tpu_strategy,
              strategy_combinations.one_device_strategy_gpu,
          ],
          use_sync_bn=[False, True],
      ))
  def test_sync_bn_multiple_devices(self, strategy, use_sync_bn):
    #Test for sync bn on TPU and GPU devices.
    inputs = np.random.rand(64, 128, 128, 3)

    tf.keras.backend.set_image_data_format('channels_last')

    with strategy.scope():
      network = basnet_en.BASNet_En(
          use_sync_bn=use_sync_bn)
      _ = network(inputs)

  @parameterized.parameters(1, 3, 4)
  def test_input_specs(self, input_dim):
    #Test different input feature dimensions.
    tf.keras.backend.set_image_data_format('channels_last')

    input_specs = tf.keras.layers.InputSpec(shape=[None, None, None, input_dim])
    network = basnet_en.BASNet_En(
        input_specs=input_specs)

    inputs = tf.keras.Input(shape=(224, 224, input_dim), batch_size=1)
    _ = network(inputs)

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        use_sync_bn=False,
        activation='relu',
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    network = basnet_en.BASNet_En(**kwargs)
    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = basnet_en.BASNet_En.from_config(network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())

if __name__ == '__main__':
  tf.test.main()
