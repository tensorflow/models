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

from absl import flags
import numpy as np

import tensorflow as tf

import data_provider


class DataProviderTest(tf.test.TestCase):

  def test_cifar10_train_set(self):
    dataset_dir = os.path.join(
        flags.FLAGS.test_srcdir,
        'google3/third_party/tensorflow_models/gan/cifar/testdata')

    batch_size = 4
    images, labels, num_samples, num_classes = data_provider.provide_data(
        batch_size, dataset_dir)
    self.assertEqual(50000, num_samples)
    self.assertEqual(10, num_classes)
    with self.test_session(use_gpu=True) as sess:
      with tf.contrib.slim.queues.QueueRunners(sess):
        images_out, labels_out = sess.run([images, labels])
        self.assertEqual(images_out.shape, (batch_size, 32, 32, 3))
        expected_label_shape = (batch_size, 10)
        self.assertEqual(expected_label_shape, labels_out.shape)
        # Check range.
        self.assertTrue(np.all(np.abs(images_out) <= 1))


if __name__ == '__main__':
  tf.test.main()
