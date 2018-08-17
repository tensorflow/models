# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


from absl import flags
import numpy as np
import tensorflow as tf

import data_provider


class DataProviderTest(tf.test.TestCase):

  def setUp(self):
    super(DataProviderTest, self).setUp()
    self.testdata_dir = os.path.join(
        flags.FLAGS.test_srcdir,
        'google3/third_party/tensorflow_models/gan/progressive_gan/testdata/')

  def test_normalize_image(self):
    image_np = np.asarray([0, 255, 210], dtype=np.uint8)
    normalized_image = data_provider.normalize_image(tf.constant(image_np))
    self.assertEqual(normalized_image.dtype, tf.float32)
    self.assertEqual(normalized_image.shape.as_list(), [3])
    with self.test_session(use_gpu=True) as sess:
      normalized_image_np = sess.run(normalized_image)
    self.assertNDArrayNear(normalized_image_np, [-1, 1, 0.6470588235], 1.0e-6)

  def test_sample_patch_large_patch_returns_upscaled_image(self):
    image_np = np.reshape(np.arange(2 * 2), [2, 2, 1])
    image = tf.constant(image_np, dtype=tf.float32)
    image_patch = data_provider.sample_patch(
        image, patch_height=3, patch_width=3, colors=1)
    with self.test_session(use_gpu=True) as sess:
      image_patch_np = sess.run(image_patch)
    expected_np = np.asarray([[[0.], [0.66666669], [1.]], [[1.33333337], [2.],
                                                           [2.33333349]],
                              [[2.], [2.66666675], [3.]]])
    self.assertNDArrayNear(image_patch_np, expected_np, 1.0e-6)

  def test_sample_patch_small_patch_returns_downscaled_image(self):
    image_np = np.reshape(np.arange(3 * 3), [3, 3, 1])
    image = tf.constant(image_np, dtype=tf.float32)
    image_patch = data_provider.sample_patch(
        image, patch_height=2, patch_width=2, colors=1)
    with self.test_session(use_gpu=True) as sess:
      image_patch_np = sess.run(image_patch)
    expected_np = np.asarray([[[0.], [1.5]], [[4.5], [6.]]])
    self.assertNDArrayNear(image_patch_np, expected_np, 1.0e-6)

  def test_batch_images(self):
    image_np = np.reshape(np.arange(3 * 3), [3, 3, 1])
    image = tf.constant(image_np, dtype=tf.float32)
    images = data_provider.batch_images(
        image,
        patch_height=2,
        patch_width=2,
        colors=1,
        batch_size=2,
        shuffle=False,
        num_threads=1)
    with self.test_session(use_gpu=True) as sess:
      with tf.contrib.slim.queues.QueueRunners(sess):
        images_np = sess.run(images)
    expected_np = np.asarray([[[[0.], [1.5]], [[4.5], [6.]]], [[[0.], [1.5]],
                                                               [[4.5], [6.]]]])
    self.assertNDArrayNear(images_np, expected_np, 1.0e-6)

  def test_provide_data(self):
    images = data_provider.provide_data(
        'mnist',
        'train',
        dataset_dir=self.testdata_dir,
        batch_size=2,
        shuffle=False,
        patch_height=3,
        patch_width=3,
        colors=1)
    self.assertEqual(images.shape.as_list(), [2, 3, 3, 1])
    with self.test_session(use_gpu=True) as sess:
      with tf.contrib.slim.queues.QueueRunners(sess):
        images_np = sess.run(images)
    self.assertEqual(images_np.shape, (2, 3, 3, 1))

  def test_provide_data_from_image_files_a_single_pattern(self):
    file_pattern = os.path.join(self.testdata_dir, '*.jpg')
    images = data_provider.provide_data_from_image_files(
        file_pattern,
        batch_size=2,
        shuffle=False,
        patch_height=3,
        patch_width=3,
        colors=1)
    self.assertEqual(images.shape.as_list(), [2, 3, 3, 1])
    with self.test_session(use_gpu=True) as sess:
      sess.run(tf.local_variables_initializer())
      with tf.contrib.slim.queues.QueueRunners(sess):
        images_np = sess.run(images)
    self.assertEqual(images_np.shape, (2, 3, 3, 1))

  def test_provide_data_from_image_files_a_list_of_patterns(self):
    file_pattern = [os.path.join(self.testdata_dir, '*.jpg')]
    images = data_provider.provide_data_from_image_files(
        file_pattern,
        batch_size=2,
        shuffle=False,
        patch_height=3,
        patch_width=3,
        colors=1)
    self.assertEqual(images.shape.as_list(), [2, 3, 3, 1])
    with self.test_session(use_gpu=True) as sess:
      sess.run(tf.local_variables_initializer())
      with tf.contrib.slim.queues.QueueRunners(sess):
        images_np = sess.run(images)
    self.assertEqual(images_np.shape, (2, 3, 3, 1))


if __name__ == '__main__':
  tf.test.main()
