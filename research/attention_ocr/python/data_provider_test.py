# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Tests for data_provider."""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import queues

import datasets
import data_provider


class DataProviderTest(tf.test.TestCase):
  def setUp(self):
    tf.test.TestCase.setUp(self)

  def test_preprocessed_image_values_are_in_range(self):
    image_shape = (5, 4, 3)
    fake_image = np.random.randint(low=0, high=255, size=image_shape)
    image_tf = data_provider.preprocess_image(fake_image)

    with self.test_session() as sess:
      image_np = sess.run(image_tf)

    self.assertEqual(image_np.shape, image_shape)
    min_value, max_value = np.min(image_np), np.max(image_np)
    self.assertTrue((-1.28 < min_value) and (min_value < 1.27))
    self.assertTrue((-1.28 < max_value) and (max_value < 1.27))

  def test_provided_data_has_correct_shape(self):
    batch_size = 4
    data = data_provider.get_data(
        dataset=datasets.fsns_test.get_test_split(),
        batch_size=batch_size,
        augment=True,
        central_crop_size=None)

    with self.test_session() as sess, queues.QueueRunners(sess):
      images_np, labels_np = sess.run([data.images, data.labels_one_hot])

    self.assertEqual(images_np.shape, (batch_size, 150, 600, 3))
    self.assertEqual(labels_np.shape, (batch_size, 37, 134))

  def test_optionally_applies_central_crop(self):
    batch_size = 4
    data = data_provider.get_data(
        dataset=datasets.fsns_test.get_test_split(),
        batch_size=batch_size,
        augment=True,
        central_crop_size=(500, 100))

    with self.test_session() as sess, queues.QueueRunners(sess):
      images_np = sess.run(data.images)

    self.assertEqual(images_np.shape, (batch_size, 100, 500, 3))


if __name__ == '__main__':
  tf.test.main()
