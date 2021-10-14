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

"""Tests for segmentation_input_3d.py."""

import os

from absl.testing import parameterized
import tensorflow as tf

from official.common import dataset_fn
from official.core import config_definitions as cfg
from official.core import input_reader
from official.projects.volumetric_models.dataloaders import segmentation_input_3d
from official.vision.beta.dataloaders import tfexample_utils


class InputReaderTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    data_dir = os.path.join(self.get_temp_dir(), 'data')
    tf.io.gfile.makedirs(data_dir)
    self._data_path = os.path.join(data_dir, 'data.tfrecord')
    # pylint: disable=g-complex-comprehension
    examples = [
        tfexample_utils.create_3d_image_test_example(
            image_height=32, image_width=32, image_volume=32, image_channel=2)
        for _ in range(20)
    ]
    # pylint: enable=g-complex-comprehension
    tfexample_utils.dump_to_tfrecord(self._data_path, tf_examples=examples)

  @parameterized.parameters(([32, 32, 32], 2, 2))
  def testSegmentationInputReader(self, input_size, num_classes, num_channels):
    params = cfg.DataConfig(
        input_path=self._data_path, global_batch_size=2, is_training=False)

    decoder = segmentation_input_3d.Decoder()
    parser = segmentation_input_3d.Parser(
        input_size=input_size,
        num_classes=num_classes,
        num_channels=num_channels)

    reader = input_reader.InputReader(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn('tfrecord'),
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read()
    iterator = iter(dataset)
    image, labels = next(iterator)

    # Checks image shape.
    self.assertEqual(
        list(image.numpy().shape),
        [2, input_size[0], input_size[1], input_size[2], num_channels])
    self.assertEqual(
        list(labels.numpy().shape),
        [2, input_size[0], input_size[1], input_size[2], num_classes])


if __name__ == '__main__':
  tf.test.main()
