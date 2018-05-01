# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

import tensorflow as tf

import data_provider

mock = tf.test.mock


class DataProviderTest(tf.test.TestCase):

  def test_normalize_image(self):
    image = tf.random_uniform(shape=(8, 8, 3), maxval=256, dtype=tf.int32)
    rescaled_image = data_provider.normalize_image(image)
    self.assertEqual(tf.float32, rescaled_image.dtype)
    self.assertListEqual(image.shape.as_list(), rescaled_image.shape.as_list())
    with self.test_session(use_gpu=True) as sess:
      rescaled_image_out = sess.run(rescaled_image)
      self.assertTrue(np.all(np.abs(rescaled_image_out) <= 1.0))

  def test_sample_patch(self):
    image = tf.zeros(shape=(8, 8, 3))
    patch1 = data_provider._sample_patch(image, 7)
    patch2 = data_provider._sample_patch(image, 10)
    image = tf.zeros(shape=(8, 8, 1))
    patch3 = data_provider._sample_patch(image, 10)
    with self.test_session(use_gpu=True) as sess:
      self.assertTupleEqual((7, 7, 3), sess.run(patch1).shape)
      self.assertTupleEqual((10, 10, 3), sess.run(patch2).shape)
      self.assertTupleEqual((10, 10, 3), sess.run(patch3).shape)

  def _get_testdata_dir(self):
    return os.path.join(
        tf.flags.FLAGS.test_srcdir,
        'google3/third_party/tensorflow_models/gan/cyclegan/testdata')

  def test_custom_dataset_provider(self):
    file_pattern = os.path.join(self._get_testdata_dir(), '*.jpg')
    batch_size = 3
    patch_size = 8
    images = data_provider._provide_custom_dataset(
        file_pattern, batch_size=batch_size, patch_size=patch_size)
    self.assertListEqual([batch_size, patch_size, patch_size, 3],
                         images.shape.as_list())
    self.assertEqual(tf.float32, images.dtype)

    with self.test_session(use_gpu=True) as sess:
      sess.run(tf.local_variables_initializer())
      with tf.contrib.slim.queues.QueueRunners(sess):
        images_out = sess.run(images)
        self.assertTupleEqual((batch_size, patch_size, patch_size, 3),
                              images_out.shape)
        self.assertTrue(np.all(np.abs(images_out) <= 1.0))

  def test_custom_datasets_provider(self):
    file_pattern = os.path.join(self._get_testdata_dir(), '*.jpg')
    batch_size = 3
    patch_size = 8
    images_list = data_provider.provide_custom_datasets(
        [file_pattern, file_pattern],
        batch_size=batch_size,
        patch_size=patch_size)
    for images in images_list:
      self.assertListEqual([batch_size, patch_size, patch_size, 3],
                           images.shape.as_list())
      self.assertEqual(tf.float32, images.dtype)

    with self.test_session(use_gpu=True) as sess:
      sess.run(tf.local_variables_initializer())
      with tf.contrib.slim.queues.QueueRunners(sess):
        images_out_list = sess.run(images_list)
        for images_out in images_out_list:
          self.assertTupleEqual((batch_size, patch_size, patch_size, 3),
                                images_out.shape)
          self.assertTrue(np.all(np.abs(images_out) <= 1.0))


if __name__ == '__main__':
  tf.test.main()
