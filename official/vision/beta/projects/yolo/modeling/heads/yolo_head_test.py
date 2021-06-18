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
"""Tests for yolo heads."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.projects.yolo.modeling.heads import yolo_head as heads


class YoloDecoderTest(parameterized.TestCase, tf.test.TestCase):

  def test_network_creation(self):
    """Test creation of YOLO family models."""
    tf.keras.backend.set_image_data_format('channels_last')
    input_shape = {
        '3': [1, 52, 52, 256],
        '4': [1, 26, 26, 512],
        '5': [1, 13, 13, 1024]
    }
    classes = 100
    bps = 3
    head = heads.YoloHead(3, 5, classes=classes, boxes_per_level=bps)

    inputs = {}
    for key in input_shape:
      inputs[key] = tf.ones(input_shape[key], dtype=tf.float32)

    endpoints = head(inputs)
    # print(endpoints)

    for key in endpoints.keys():
      expected_input_shape = input_shape[key]
      expected_input_shape[-1] = (classes + 5) * bps
      self.assertAllEqual(endpoints[key].shape.as_list(), expected_input_shape)

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    tf.keras.backend.set_image_data_format('channels_last')
    input_shape = {
        '3': [1, 52, 52, 256],
        '4': [1, 26, 26, 512],
        '5': [1, 13, 13, 1024]
    }
    classes = 100
    bps = 3
    head = heads.YoloHead(3, 5, classes=classes, boxes_per_level=bps)

    inputs = {}
    for key in input_shape:
      inputs[key] = tf.ones(input_shape[key], dtype=tf.float32)

    _ = head(inputs)
    configs = head.get_config()
    head_from_config = heads.YoloHead.from_config(configs)
    self.assertAllEqual(head.get_config(), head_from_config.get_config())


if __name__ == '__main__':
  tf.test.main()
