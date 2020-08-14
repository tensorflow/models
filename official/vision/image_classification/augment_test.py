# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for autoaugment."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

from absl.testing import parameterized

import tensorflow as tf

from official.vision.image_classification import augment


def get_dtype_test_cases():
  return [
      ('uint8', tf.uint8),
      ('int32', tf.int32),
      ('float16', tf.float16),
      ('float32', tf.float32),
  ]


@parameterized.named_parameters(get_dtype_test_cases())
class TransformsTest(parameterized.TestCase, tf.test.TestCase):
  """Basic tests for fundamental transformations."""

  def test_to_from_4d(self, dtype):
    for shape in [(10, 10), (10, 10, 10), (10, 10, 10, 10)]:
      original_ndims = len(shape)
      image = tf.zeros(shape, dtype=dtype)
      image_4d = augment.to_4d(image)
      self.assertEqual(4, tf.rank(image_4d))
      self.assertAllEqual(image, augment.from_4d(image_4d, original_ndims))

  def test_transform(self, dtype):
    image = tf.constant([[1, 2], [3, 4]], dtype=dtype)
    self.assertAllEqual(
        augment.transform(image, transforms=[1] * 8), [[4, 4], [4, 4]])

  def test_translate(self, dtype):
    image = tf.constant(
        [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=dtype)
    translations = [-1, -1]
    translated = augment.translate(image=image, translations=translations)
    expected = [[1, 0, 1, 1], [0, 1, 0, 0], [1, 0, 1, 1], [1, 0, 1, 1]]
    self.assertAllEqual(translated, expected)

  def test_translate_shapes(self, dtype):
    translation = [0, 0]
    for shape in [(3, 3), (5, 5), (224, 224, 3)]:
      image = tf.zeros(shape, dtype=dtype)
      self.assertAllEqual(image, augment.translate(image, translation))

  def test_translate_invalid_translation(self, dtype):
    image = tf.zeros((1, 1), dtype=dtype)
    invalid_translation = [[[1, 1]]]
    with self.assertRaisesRegex(TypeError, 'rank 1 or 2'):
      _ = augment.translate(image, invalid_translation)

  def test_rotate(self, dtype):
    image = tf.reshape(tf.cast(tf.range(9), dtype), (3, 3))
    rotation = 90.
    transformed = augment.rotate(image=image, degrees=rotation)
    expected = [[2, 5, 8], [1, 4, 7], [0, 3, 6]]
    self.assertAllEqual(transformed, expected)

  def test_rotate_shapes(self, dtype):
    degrees = 0.
    for shape in [(3, 3), (5, 5), (224, 224, 3)]:
      image = tf.zeros(shape, dtype=dtype)
      self.assertAllEqual(image, augment.rotate(image, degrees))


class AutoaugmentTest(tf.test.TestCase):

  def test_autoaugment(self):
    """Smoke test to be sure there are no syntax errors."""
    image = tf.zeros((224, 224, 3), dtype=tf.uint8)

    augmenter = augment.AutoAugment()
    aug_image = augmenter.distort(image)

    self.assertEqual((224, 224, 3), aug_image.shape)

  def test_randaug(self):
    """Smoke test to be sure there are no syntax errors."""
    image = tf.zeros((224, 224, 3), dtype=tf.uint8)

    augmenter = augment.RandAugment()
    aug_image = augmenter.distort(image)

    self.assertEqual((224, 224, 3), aug_image.shape)

  def test_all_policy_ops(self):
    """Smoke test to be sure all augmentation functions can execute."""

    prob = 1
    magnitude = 10
    replace_value = [128] * 3
    cutout_const = 100
    translate_const = 250

    image = tf.ones((224, 224, 3), dtype=tf.uint8)

    for op_name in augment.NAME_TO_FUNC:
      func, _, args = augment._parse_policy_info(op_name, prob, magnitude,
                                                 replace_value, cutout_const,
                                                 translate_const)
      image = func(image, *args)

    self.assertEqual((224, 224, 3), image.shape)


if __name__ == '__main__':
  tf.test.main()
