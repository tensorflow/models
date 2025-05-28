# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for Mobiledet."""

import itertools

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.vision.modeling.backbones import mobiledet


class MobileDetTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      'MobileDetCPU',
      'MobileDetDSP',
      'MobileDetEdgeTPU',
      'MobileDetGPU',
  )
  def test_serialize_deserialize(self, model_id):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        model_id=model_id,
        filter_size_scale=1.0,
        use_sync_bn=False,
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_regularizer=None,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        min_depth=8,
        divisible_by=8,
        regularize_depthwise=False,
    )
    network = mobiledet.MobileDet(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = mobiledet.MobileDet.from_config(network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())

  @parameterized.parameters(
      itertools.product(
          [1, 3],
          [
              'MobileDetCPU',
              'MobileDetDSP',
              'MobileDetEdgeTPU',
              'MobileDetGPU',
          ],
      ))
  def test_input_specs(self, input_dim, model_id):
    """Test different input feature dimensions."""
    tf_keras.backend.set_image_data_format('channels_last')

    input_specs = tf_keras.layers.InputSpec(shape=[None, None, None, input_dim])
    network = mobiledet.MobileDet(model_id=model_id, input_specs=input_specs)

    inputs = tf_keras.Input(shape=(128, 128, input_dim), batch_size=1)
    _ = network(inputs)

  @parameterized.parameters(
      itertools.product(
          [
              'MobileDetCPU',
              'MobileDetDSP',
              'MobileDetEdgeTPU',
              'MobileDetGPU',
          ],
          [32, 224],
      ))
  def test_mobiledet_creation(self, model_id, input_size):
    """Test creation of MobileDet family models."""
    tf_keras.backend.set_image_data_format('channels_last')

    mobiledet_layers = {
        # The number of filters of layers having outputs been collected
        # for filter_size_scale = 1.0
        'MobileDetCPU': [8, 16, 32, 72, 144],
        'MobileDetDSP': [24, 32, 64, 144, 240],
        'MobileDetEdgeTPU': [16, 16, 40, 96, 384],
        'MobileDetGPU': [16, 32, 64, 128, 384],
    }

    network = mobiledet.MobileDet(model_id=model_id,
                                  filter_size_scale=1.0)

    inputs = tf_keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)

    for idx, num_filter in enumerate(mobiledet_layers[model_id]):
      self.assertAllEqual(
          [1, input_size / 2 ** (idx+1), input_size / 2 ** (idx+1), num_filter],
          endpoints[str(idx+1)].shape.as_list())
