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

"""Tests for tfds factory functions."""

from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.dataloaders import decoder as base_decoder
from official.vision.beta.dataloaders import tfds_factory


class TFDSFactoryTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ('imagenet2012'),
      ('cifar10'),
      ('cifar100'),
  )
  def test_classification_decoder(self, tfds_name):
    decoder = tfds_factory.get_classification_decoder(tfds_name)
    self.assertIsInstance(decoder, base_decoder.Decoder)

  @parameterized.parameters(
      ('flowers'),
      ('coco'),
  )
  def test_doesnt_exit_classification_decoder(self, tfds_name):
    with self.assertRaises(ValueError):
      _ = tfds_factory.get_classification_decoder(tfds_name)

  @parameterized.parameters(
      ('coco'),
      ('coco/2014'),
      ('coco/2017'),
  )
  def test_detection_decoder(self, tfds_name):
    decoder = tfds_factory.get_detection_decoder(tfds_name)
    self.assertIsInstance(decoder, base_decoder.Decoder)

  @parameterized.parameters(
      ('pascal'),
      ('cityscapes'),
  )
  def test_doesnt_exit_detection_decoder(self, tfds_name):
    with self.assertRaises(ValueError):
      _ = tfds_factory.get_detection_decoder(tfds_name)

  @parameterized.parameters(
      ('cityscapes'),
      ('cityscapes/semantic_segmentation'),
      ('cityscapes/semantic_segmentation_extra'),
  )
  def test_segmentation_decoder(self, tfds_name):
    decoder = tfds_factory.get_segmentation_decoder(tfds_name)
    self.assertIsInstance(decoder, base_decoder.Decoder)

  @parameterized.parameters(
      ('coco'),
      ('imagenet'),
  )
  def test_doesnt_exit_segmentation_decoder(self, tfds_name):
    with self.assertRaises(ValueError):
      _ = tfds_factory.get_segmentation_decoder(tfds_name)

if __name__ == '__main__':
  tf.test.main()
