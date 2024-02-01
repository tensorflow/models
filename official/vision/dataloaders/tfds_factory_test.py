# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
import tensorflow as tf, tf_keras

from official.vision.dataloaders import decoder as base_decoder
from official.vision.dataloaders import tfds_factory


class TFDSFactoryTest(tf.test.TestCase, parameterized.TestCase):

  def _create_test_example(self):
    serialized_example = {
        'image': tf.ones(shape=(100, 100, 3), dtype=tf.uint8),
        'label': 1,
        'image/id': 0,
        'objects': {
            'label': 1,
            'is_crowd': 0,
            'area': 0.5,
            'bbox': [0.1, 0.2, 0.3, 0.4]
        },
        'segmentation_label': tf.ones((100, 100, 1), dtype=tf.uint8),
        'image_left': tf.ones(shape=(100, 100, 3), dtype=tf.uint8)
    }
    return serialized_example

  @parameterized.parameters(
      ('imagenet2012'),
      ('cifar10'),
      ('cifar100'),
  )
  def test_classification_decoder(self, tfds_name):
    decoder = tfds_factory.get_classification_decoder(tfds_name)
    self.assertIsInstance(decoder, base_decoder.Decoder)
    decoded_tensor = decoder.decode(self._create_test_example())
    self.assertLen(decoded_tensor, 2)
    self.assertIn('image/encoded', decoded_tensor)
    self.assertIn('image/class/label', decoded_tensor)

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
    decoded_tensor = decoder.decode(self._create_test_example())
    self.assertLen(decoded_tensor, 8)
    self.assertIn('image', decoded_tensor)
    self.assertIn('source_id', decoded_tensor)
    self.assertIn('height', decoded_tensor)
    self.assertIn('width', decoded_tensor)
    self.assertIn('groundtruth_classes', decoded_tensor)
    self.assertIn('groundtruth_is_crowd', decoded_tensor)
    self.assertIn('groundtruth_area', decoded_tensor)
    self.assertIn('groundtruth_boxes', decoded_tensor)

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
    decoded_tensor = decoder.decode(self._create_test_example())
    self.assertLen(decoded_tensor, 4)
    self.assertIn('image/encoded', decoded_tensor)
    self.assertIn('image/segmentation/class/encoded', decoded_tensor)
    self.assertIn('image/height', decoded_tensor)
    self.assertIn('image/width', decoded_tensor)

  @parameterized.parameters(
      ('coco'),
      ('imagenet'),
  )
  def test_doesnt_exit_segmentation_decoder(self, tfds_name):
    with self.assertRaises(ValueError):
      _ = tfds_factory.get_segmentation_decoder(tfds_name)

if __name__ == '__main__':
  tf.test.main()
