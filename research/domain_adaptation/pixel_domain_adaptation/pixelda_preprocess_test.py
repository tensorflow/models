# Copyright 2017 Google Inc.
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

"""Tests for domain_adaptation.pixel_domain_adaptation.pixelda_preprocess."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow as tf

from domain_adaptation.pixel_domain_adaptation import pixelda_preprocess


class PixelDAPreprocessTest(tf.test.TestCase):

  def assert_preprocess_classification_is_centered(self, dtype, is_training):
    tf.set_random_seed(0)

    if dtype == tf.uint8:
      image = tf.random_uniform((100, 200, 3), maxval=255, dtype=tf.int64)
      image = tf.cast(image, tf.uint8)
    else:
      image = tf.random_uniform((100, 200, 3), maxval=1.0, dtype=dtype)

    labels = {}
    image, labels = pixelda_preprocess.preprocess_classification(
        image, labels, is_training=is_training)

    with self.test_session() as sess:
      np_image = sess.run(image)

      self.assertTrue(np_image.min() <= -0.95)
      self.assertTrue(np_image.min() >= -1.0)
      self.assertTrue(np_image.max() >= 0.95)
      self.assertTrue(np_image.max() <= 1.0)

  def testPreprocessClassificationZeroCentersUint8DuringTrain(self):
    self.assert_preprocess_classification_is_centered(
        tf.uint8, is_training=True)

  def testPreprocessClassificationZeroCentersUint8DuringTest(self):
    self.assert_preprocess_classification_is_centered(
        tf.uint8, is_training=False)

  def testPreprocessClassificationZeroCentersFloatDuringTrain(self):
    self.assert_preprocess_classification_is_centered(
        tf.float32, is_training=True)

  def testPreprocessClassificationZeroCentersFloatDuringTest(self):
    self.assert_preprocess_classification_is_centered(
        tf.float32, is_training=False)


if __name__ == '__main__':
  tf.test.main()
