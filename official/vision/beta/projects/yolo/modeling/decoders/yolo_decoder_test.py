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
"""Tests for resnet."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
# from yolo.modeling.backbones import darknet
from official.vision.beta.projects.yolo.modeling.decoders import yolo_decoder as decoders


class YoloDecoderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters('a', 'b', 'c')
  def test_network_creation(self, version):
    """Test creation of ResNet family models."""
    tf.keras.backend.set_image_data_format('channels_last')
    input_shape = {'3': [1, 52, 52, 256], '4': [1, 26, 26, 512], '5': [1, 13, 13, 1024]}
    decoder = build_yolo_decoder(input_shape, version)

    inputs = {}
    for key in input_shape.keys():
        inputs[key] = tf.ones(input_shape[key], dtype = tf.float32)
    
    endpoints = decoder.call(inputs)
    # print(endpoints)

    for key in endpoints.keys():
        self.assertAllEqual(endpoints[key].shape.as_list(), input_shape[key])


  @combinations.generate(
      combinations.combine(
          strategy=[
              strategy_combinations.tpu_strategy,
              strategy_combinations.one_device_strategy_gpu,
          ],
          use_sync_bn=[False, True],
      ))
  def test_sync_bn_multiple_devices(self, strategy, use_sync_bn):
    """Test for sync bn on TPU and GPU devices."""

    tf.keras.backend.set_image_data_format('channels_last')

    with strategy.scope():
      input_shape = {'3': [1, 52, 52, 256], '4': [1, 26, 26, 512], '5': [1, 13, 13, 1024]}
      decoder = build_yolo_decoder(input_shape, 'c')

      inputs = {}
      for key in input_shape.keys():
        inputs[key] = tf.ones(input_shape[key], dtype = tf.float32)
    
      _ = decoder.call(inputs)

  @parameterized.parameters(1, 3, 4)
  def test_input_specs(self, input_dim):
    """Test different input feature dimensions."""
    tf.keras.backend.set_image_data_format('channels_last')

    input_shape = {'3': [1, 52, 52, 256], '4': [1, 26, 26, 512], '5': [1, 13, 13, 1024]}
    decoder = build_yolo_decoder(input_shape, 'c')

    inputs = {}
    for key in input_shape.keys():
      inputs[key] = tf.ones(input_shape[key], dtype = tf.float32)
    _ = decoder(inputs)

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    tf.keras.backend.set_image_data_format('channels_last')

    input_shape = {'3': [1, 52, 52, 256], '4': [1, 26, 26, 512], '5': [1, 13, 13, 1024]}
    decoder = build_yolo_decoder(input_shape, 'c')

    inputs = {}
    for key in input_shape.keys():
      inputs[key] = tf.ones(input_shape[key], dtype = tf.float32)

    _ = decoder(inputs)

    a = decoder.get_config()


    b = decoders.YoloDecoder.from_config(a)

    print(a)
    self.assertAllEqual(decoder.get_config(), b.get_config())

def build_yolo_decoder(input_specs, type):
  if type == "a":
    model = decoders.YoloDecoder(
        embed_spp=False,
        embed_fpn=False,
        max_level_process_len=2,
        path_process_len=1,
        activation="mish")
  elif type == "b":
    model = decoders.YoloDecoder(
        embed_spp=True,
        embed_fpn=False,
        max_level_process_len=None,
        path_process_len=6,
        activation="mish")
  else:
    model = decoders.YoloDecoder(
          embed_spp=False,
          embed_fpn=False,
          max_level_process_len=None,
          path_process_len=6,
          activation="mish")
  model.build(input_specs)
  return model

if __name__ == "__main__":
  tf.test.main()