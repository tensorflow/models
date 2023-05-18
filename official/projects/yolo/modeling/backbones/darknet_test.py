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

"""Tests for yolo."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.projects.yolo.modeling.backbones import darknet


class DarknetTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (224, 'darknet53', 2, 1, True),
      (224, 'darknettiny', 1, 2, False),
      (224, 'cspdarknettiny', 1, 1, False),
      (224, 'cspdarknet53', 2, 1, True),
  )
  def test_network_creation(self, input_size, model_id, endpoint_filter_scale,
                            scale_final, dilate):
    """Test creation of ResNet family models."""
    tf.keras.backend.set_image_data_format('channels_last')

    network = darknet.Darknet(
        model_id=model_id, min_level=3, max_level=5, dilate=dilate)
    self.assertEqual(network.model_id, model_id)

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)

    if dilate:
      self.assertAllEqual([
          1, input_size / 2**3, input_size / 2**3, 128 * endpoint_filter_scale
      ], endpoints['3'].shape.as_list())
      self.assertAllEqual([
          1, input_size / 2**3, input_size / 2**3, 256 * endpoint_filter_scale
      ], endpoints['4'].shape.as_list())
      self.assertAllEqual([
          1, input_size / 2**3, input_size / 2**3,
          512 * endpoint_filter_scale * scale_final
      ], endpoints['5'].shape.as_list())
    else:
      self.assertAllEqual([
          1, input_size / 2**3, input_size / 2**3, 128 * endpoint_filter_scale
      ], endpoints['3'].shape.as_list())
      self.assertAllEqual([
          1, input_size / 2**4, input_size / 2**4, 256 * endpoint_filter_scale
      ], endpoints['4'].shape.as_list())
      self.assertAllEqual([
          1, input_size / 2**5, input_size / 2**5,
          512 * endpoint_filter_scale * scale_final
      ], endpoints['5'].shape.as_list())

  @combinations.generate(
      combinations.combine(
          strategy=[
              strategy_combinations.cloud_tpu_strategy,
              strategy_combinations.one_device_strategy_gpu,
          ],
          use_sync_bn=[False, True],
      ))
  def test_sync_bn_multiple_devices(self, strategy, use_sync_bn):
    """Test for sync bn on TPU and GPU devices."""
    inputs = np.random.rand(1, 224, 224, 3)

    tf.keras.backend.set_image_data_format('channels_last')

    with strategy.scope():
      network = darknet.Darknet(
          model_id='darknet53', min_size=3, max_size=5, use_sync_bn=use_sync_bn
      )
      _ = network(inputs)

  @parameterized.parameters(1, 3, 4)
  def test_input_specs(self, input_dim):
    """Test different input feature dimensions."""
    tf.keras.backend.set_image_data_format('channels_last')

    input_specs = tf.keras.layers.InputSpec(shape=[None, None, None, input_dim])
    network = darknet.Darknet(
        model_id='darknet53', min_level=3, max_level=5, input_specs=input_specs)

    inputs = tf.keras.Input(shape=(224, 224, input_dim), batch_size=1)
    _ = network(inputs)

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        model_id='darknet53',
        min_level=3,
        max_level=5,
        use_sync_bn=False,
        activation='relu',
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    network = darknet.Darknet(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = darknet.Darknet.from_config(network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == '__main__':
  tf.test.main()
