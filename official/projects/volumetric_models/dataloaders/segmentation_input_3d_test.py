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

"""Tests for segmentation_input_3d.py."""

import os

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.projects.volumetric_models.dataloaders import segmentation_input_3d
from official.vision.dataloaders import tfexample_utils


class InputReaderTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    data_dir = os.path.join(self.get_temp_dir(), 'data')
    tf.io.gfile.makedirs(data_dir)
    self._data_path = os.path.join(data_dir, 'data.tfrecord')
    self._example = tfexample_utils.create_3d_image_test_example(
        image_height=32, image_width=32, image_volume=32, image_channel=2)

  @parameterized.parameters(
      ([32, 32, 32], 2, 2, False),
      ([32, 32, 32], 2, 2, True),
  )
  def testSegmentationInputReader(self, input_size, num_classes, num_channels,
                                  is_training):

    decoder = segmentation_input_3d.Decoder()
    parser = segmentation_input_3d.Parser(
        input_size=input_size,
        num_classes=num_classes,
        num_channels=num_channels)

    decoded_tensor = decoder.decode(self._example.SerializeToString())
    image, labels = parser.parse_fn(is_training=is_training)(decoded_tensor)

    # Checks image shape.
    self.assertEqual(
        list(image.numpy().shape),
        [input_size[0], input_size[1], input_size[2], num_channels])
    self.assertEqual(
        list(labels.numpy().shape),
        [input_size[0], input_size[1], input_size[2], num_classes])


if __name__ == '__main__':
  tf.test.main()
