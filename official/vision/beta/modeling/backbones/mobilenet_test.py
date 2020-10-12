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
"""Tests for MobileNet."""

import itertools
# Import libraries

from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.modeling.backbones import mobilenet


class MobileNetTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters('MobileNetV1', 'MobileNetV2',
                            'MobileNetV3Large', 'MobileNetV3Small',
                            'MobileNetV3EdgeTPU')
  def test_serialize_deserialize(self, model_id):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        model_id=model_id,
        filter_size_scale=1.0,
        stochastic_depth_drop_rate=None,
        use_sync_bn=False,
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_regularizer=None,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        output_stride=None,
        min_depth=8,
        divisible_by=8,
        regularize_depthwise=False,
        finegrain_classification_mode=True
    )
    network = mobilenet.MobileNet(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = mobilenet.MobileNet.from_config(network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())

  @parameterized.parameters(
      itertools.product((1, 3),
                        ('MobileNetV1', 'MobileNetV2', 'MobileNetV3Large',
                         'MobileNetV3Small', 'MobileNetV3EdgeTPU')))
  def test_input_specs(self, input_dim, model_id):
    """Test different input feature dimensions."""
    tf.keras.backend.set_image_data_format('channels_last')

    input_specs = tf.keras.layers.InputSpec(shape=[None, None, None, input_dim])
    network = mobilenet.MobileNet(model_id=model_id, input_specs=input_specs)

    inputs = tf.keras.Input(shape=(128, 128, input_dim), batch_size=1)
    _ = network(inputs)

  @parameterized.parameters(32, 224)
  def test_mobilenet_v1_creation(self, input_size):
    """Test creation of EfficientNet family models."""
    tf.keras.backend.set_image_data_format('channels_last')

    network = mobilenet.MobileNet(model_id='MobileNetV1',
                                  filter_size_scale=0.75)

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)

    self.assertAllEqual([1, input_size / 2 ** 1, input_size / 2 ** 1, 24],
                        endpoints[1].shape.as_list())
    self.assertAllEqual([1, input_size / 2 ** 1, input_size / 2 ** 1, 48],
                        endpoints[2].shape.as_list())
    self.assertAllEqual([1, input_size / 2 ** 2, input_size / 2 ** 2, 96],
                        endpoints[3].shape.as_list())
    self.assertAllEqual([1, input_size / 2 ** 2, input_size / 2 ** 2, 96],
                        endpoints[4].shape.as_list())

  @parameterized.parameters(32, 224)
  def test_mobilenet_v2_creation(self, input_size):
    """Test creation of EfficientNet family models."""
    tf.keras.backend.set_image_data_format('channels_last')

    network = mobilenet.MobileNet(model_id='MobileNetV2',
                                  filter_size_scale=1.0)

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)

    self.assertAllEqual([1, input_size / 2 ** 1, input_size / 2 ** 1, 32],
                        endpoints[1].shape.as_list())
    self.assertAllEqual([1, input_size / 2 ** 1, input_size / 2 ** 1, 16],
                        endpoints[2].shape.as_list())
    self.assertAllEqual([1, input_size / 2 ** 2, input_size / 2 ** 2, 24],
                        endpoints[3].shape.as_list())
    self.assertAllEqual([1, input_size / 2 ** 2, input_size / 2 ** 2, 24],
                        endpoints[4].shape.as_list())

  @parameterized.parameters(32, 224)
  def test_mobilenet_v3_small_creation(self, input_size):
    """Test creation of EfficientNet family models."""
    tf.keras.backend.set_image_data_format('channels_last')

    network = mobilenet.MobileNet(model_id='MobileNetV3Small',
                                  filter_size_scale=0.75)

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)

    self.assertAllEqual([1, input_size / 2 ** 1, input_size / 2 ** 1, 16],
                        endpoints[1].shape.as_list())
    self.assertAllEqual([1, input_size / 2 ** 2, input_size / 2 ** 2, 16],
                        endpoints[2].shape.as_list())
    self.assertAllEqual([1, input_size / 2 ** 3, input_size / 2 ** 3, 24],
                        endpoints[3].shape.as_list())
    self.assertAllEqual([1, input_size / 2 ** 3, input_size / 2 ** 3, 24],
                        endpoints[4].shape.as_list())

  @parameterized.parameters(32, 224)
  def test_mobilenet_v3_large_creation(self, input_size):
    """Test creation of EfficientNet family models."""
    tf.keras.backend.set_image_data_format('channels_last')

    network = mobilenet.MobileNet(model_id='MobileNetV3Large',
                                  filter_size_scale=0.75)

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)

    self.assertAllEqual([1, input_size / 2 ** 1, input_size / 2 ** 1, 16],
                        endpoints[1].shape.as_list())
    self.assertAllEqual([1, input_size / 2 ** 1, input_size / 2 ** 1, 16],
                        endpoints[2].shape.as_list())
    self.assertAllEqual([1, input_size / 2 ** 2, input_size / 2 ** 2, 24],
                        endpoints[3].shape.as_list())
    self.assertAllEqual([1, input_size / 2 ** 2, input_size / 2 ** 2, 24],
                        endpoints[4].shape.as_list())

  @parameterized.parameters(32, 224)
  def test_mobilenet_v3_edgetpu_creation(self, input_size):
    """Test creation of EfficientNet family models."""
    tf.keras.backend.set_image_data_format('channels_last')

    network = mobilenet.MobileNet(model_id='MobileNetV3EdgeTPU',
                                  filter_size_scale=0.75)

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)

    self.assertAllEqual([1, input_size / 2 ** 1, input_size / 2 ** 1, 24],
                        endpoints[1].shape.as_list())
    self.assertAllEqual([1, input_size / 2 ** 1, input_size / 2 ** 1, 16],
                        endpoints[2].shape.as_list())
    self.assertAllEqual([1, input_size / 2 ** 2, input_size / 2 ** 2, 24],
                        endpoints[3].shape.as_list())
    self.assertAllEqual([1, input_size / 2 ** 2, input_size / 2 ** 2, 24],
                        endpoints[4].shape.as_list())

  @parameterized.parameters(1.0, 0.75)
  def test_mobilenet_v1_scaling(self, filter_size_scale):
    mobilenet_v1_params = {
        1.0: 3228864,
        0.75: 1832976
    }

    input_size = 224
    network = mobilenet.MobileNet(model_id='MobileNetV1',
                                  filter_size_scale=filter_size_scale)
    self.assertEqual(network.count_params(),
                     mobilenet_v1_params[filter_size_scale])

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    _ = network(inputs)

  @parameterized.parameters(1.0, 0.75)
  def test_mobilenet_v2_scaling(self, filter_size_scale):
    mobilenet_v2_params = {
        1.0: 2257984,
        0.75: 1382064
    }

    input_size = 224
    network = mobilenet.MobileNet(model_id='MobileNetV2',
                                  filter_size_scale=filter_size_scale)
    self.assertEqual(network.count_params(),
                     mobilenet_v2_params[filter_size_scale])

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    _ = network(inputs)

  @parameterized.parameters(1.0, 0.75)
  def test_mobilenet_v3_large_scaling(self, filter_size_scale):
    mobilenet_v3_large_params = {
        1.0: 4226432,
        0.75: 2731616
    }

    input_size = 224
    network = mobilenet.MobileNet(model_id='MobileNetV3Large',
                                  filter_size_scale=filter_size_scale)
    self.assertEqual(network.count_params(),
                     mobilenet_v3_large_params[filter_size_scale])

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    _ = network(inputs)

  @parameterized.parameters(1.0, 0.75)
  def test_mobilenet_v3_small_scaling(self, filter_size_scale):
    mobilenet_v3_small_params = {
        1.0: 1529968,
        0.75: 1026552
    }

    input_size = 224
    network = mobilenet.MobileNet(model_id='MobileNetV3Small',
                                  filter_size_scale=filter_size_scale)
    self.assertEqual(network.count_params(),
                     mobilenet_v3_small_params[filter_size_scale])

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    _ = network(inputs)

  @parameterized.parameters(1.0, 0.75)
  def test_mobilenet_v3_edgetpu_scaling(self, filter_size_scale):
    mobilenet_v3_edgetpu_params = {
        1.0: 2849312,
        0.75: 1737288
    }

    input_size = 224
    network = mobilenet.MobileNet(model_id='MobileNetV3EdgeTPU',
                                  filter_size_scale=filter_size_scale)
    self.assertEqual(network.count_params(),
                     mobilenet_v3_edgetpu_params[filter_size_scale])

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    _ = network(inputs)
