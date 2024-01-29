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

"""Tests for SpineNet."""
# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from official.vision.modeling.backbones import spinenet


class SpineNetTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (128, 0.65, 1, 0.5, 128, 4, 6),
      (256, 1.0, 1, 0.5, 256, 3, 6),
      (384, 1.0, 2, 0.5, 256, 4, 7),
      (512, 1.0, 3, 1.0, 256, 3, 7),
      (640, 1.3, 4, 1.0, 384, 3, 7),
  )
  def test_network_creation(self, input_size, filter_size_scale, block_repeats,
                            resample_alpha, endpoints_num_filters, min_level,
                            max_level):
    """Test creation of SpineNet models."""

    tf.keras.backend.set_image_data_format('channels_last')

    input_specs = tf.keras.layers.InputSpec(
        shape=[None, input_size, input_size, 3])
    model = spinenet.SpineNet(
        input_specs=input_specs,
        min_level=min_level,
        max_level=max_level,
        endpoints_num_filters=endpoints_num_filters,
        resample_alpha=resample_alpha,
        block_repeats=block_repeats,
        filter_size_scale=filter_size_scale,
        init_stochastic_depth_rate=0.2,
    )

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    endpoints = model(inputs)

    for l in range(min_level, max_level + 1):
      self.assertIn(str(l), endpoints.keys())
      self.assertAllEqual(
          [1, input_size / 2**l, input_size / 2**l, endpoints_num_filters],
          endpoints[str(l)].shape.as_list())

  @parameterized.parameters(
      ((128, 128), (128, 128)),
      ((128, 128), (256, 256)),
      ((640, 640), (896, 1664)),
  )
  def test_load_from_different_input_specs(self, input_size_1, input_size_2):
    """Test loading checkpoints with different input size."""

    def build_spinenet(input_size):
      tf.keras.backend.set_image_data_format('channels_last')
      input_specs = tf.keras.layers.InputSpec(
          shape=[None, input_size[0], input_size[1], 3])
      model = spinenet.SpineNet(
          input_specs=input_specs,
          min_level=3,
          max_level=7,
          endpoints_num_filters=384,
          resample_alpha=1.0,
          block_repeats=2,
          filter_size_scale=0.5)
      return model

    model_1 = build_spinenet(input_size_1)
    model_2 = build_spinenet(input_size_2)

    ckpt_1 = tf.train.Checkpoint(backbone=model_1)
    ckpt_2 = tf.train.Checkpoint(backbone=model_2)

    ckpt_path = self.get_temp_dir() + '/ckpt'
    ckpt_1.write(ckpt_path)
    ckpt_2.restore(ckpt_path).expect_partial()

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        min_level=3,
        max_level=7,
        endpoints_num_filters=256,
        resample_alpha=0.5,
        block_repeats=1,
        filter_size_scale=1.0,
        init_stochastic_depth_rate=0.2,
        use_sync_bn=False,
        activation='relu',
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    network = spinenet.SpineNet(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = spinenet.SpineNet.from_config(network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())

  @parameterized.parameters(
      ('relu', tf.nn.relu),
      ('swish', tf.nn.swish)
  )
  def test_activation(self, activation, activation_fn):
    model = spinenet.SpineNet(activation=activation)
    self.assertEqual(model._activation_fn, activation_fn)

  def test_invalid_activation_raises_valurerror(self):
    with self.assertRaises(ValueError):
      spinenet.SpineNet(activation='invalid_activation_name')


if __name__ == '__main__':
  tf.test.main()
