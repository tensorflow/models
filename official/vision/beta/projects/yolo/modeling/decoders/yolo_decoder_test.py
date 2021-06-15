# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

# Lint as: python3
"""Tests for YOLO."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.vision.beta.projects.yolo.modeling.decoders import yolo_decoder as decoders


class YoloDecoderTest(parameterized.TestCase, tf.test.TestCase):

  def _build_yolo_decoder(self, input_specs, name='1'):
    # Builds 4 different arbitrary decoders.
    if name == '1':
      model = decoders.YoloDecoder(
          input_specs=input_specs,
          embed_spp=False,
          use_fpn=False,
          max_level_process_len=2,
          path_process_len=1,
          activation='mish')
    elif name == '6spp':
      model = decoders.YoloDecoder(
          input_specs=input_specs,
          embed_spp=True,
          use_fpn=False,
          max_level_process_len=None,
          path_process_len=6,
          activation='mish')
    elif name == '6sppfpn':
      model = decoders.YoloDecoder(
          input_specs=input_specs,
          embed_spp=True,
          use_fpn=True,
          max_level_process_len=None,
          path_process_len=6,
          activation='mish')
    elif name == '6':
      model = decoders.YoloDecoder(
          input_specs=input_specs,
          embed_spp=False,
          use_fpn=False,
          max_level_process_len=None,
          path_process_len=6,
          activation='mish')
    else:
      raise NotImplementedError(f'YOLO decoder test {type} not implemented.')
    return model

  @parameterized.parameters('1', '6spp', '6sppfpn', '6')
  def test_network_creation(self, version):
    """Test creation of ResNet family models."""
    tf.keras.backend.set_image_data_format('channels_last')
    input_shape = {
        '3': [1, 52, 52, 256],
        '4': [1, 26, 26, 512],
        '5': [1, 13, 13, 1024]
    }
    decoder = self._build_yolo_decoder(input_shape, version)

    inputs = {}
    for key in input_shape:
      inputs[key] = tf.ones(input_shape[key], dtype=tf.float32)

    endpoints = decoder.call(inputs)

    for key in endpoints.keys():
      self.assertAllEqual(endpoints[key].shape.as_list(), input_shape[key])

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

    tf.keras.backend.set_image_data_format('channels_last')

    with strategy.scope():
      input_shape = {
          '3': [1, 52, 52, 256],
          '4': [1, 26, 26, 512],
          '5': [1, 13, 13, 1024]
      }
      decoder = self._build_yolo_decoder(input_shape, '6')

      inputs = {}
      for key in input_shape:
        inputs[key] = tf.ones(input_shape[key], dtype=tf.float32)

      _ = decoder.call(inputs)

  @parameterized.parameters(1, 3, 4)
  def test_input_specs(self, input_dim):
    """Test different input feature dimensions."""
    tf.keras.backend.set_image_data_format('channels_last')

    input_shape = {
        '3': [1, 52, 52, 256],
        '4': [1, 26, 26, 512],
        '5': [1, 13, 13, 1024]
    }
    decoder = self._build_yolo_decoder(input_shape, '6')

    inputs = {}
    for key in input_shape:
      inputs[key] = tf.ones(input_shape[key], dtype=tf.float32)
    _ = decoder(inputs)

  def test_serialize_deserialize(self):
    """Create a network object that sets all of its config options."""
    tf.keras.backend.set_image_data_format('channels_last')

    input_shape = {
        '3': [1, 52, 52, 256],
        '4': [1, 26, 26, 512],
        '5': [1, 13, 13, 1024]
    }
    decoder = self._build_yolo_decoder(input_shape, '6')

    inputs = {}
    for key in input_shape:
      inputs[key] = tf.ones(input_shape[key], dtype=tf.float32)

    _ = decoder(inputs)
    config = decoder.get_config()
    decoder_from_config = decoders.YoloDecoder.from_config(config)
    self.assertAllEqual(decoder.get_config(), decoder_from_config.get_config())


if __name__ == '__main__':
  tf.test.main()
