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
import tensorflow as tf

import data_provider


class DataProviderTest(tf.test.TestCase):

  def test_mnist_data_reading(self):
    dataset_dir = os.path.join(
        flags.FLAGS.test_srcdir,
        'google3/third_party/tensorflow_models/gan/mnist/testdata')

    batch_size = 5
    images, labels, num_samples = data_provider.provide_data(
        'test', batch_size, dataset_dir)
    self.assertEqual(num_samples, 10000)

    with self.test_session() as sess:
      with tf.contrib.slim.queues.QueueRunners(sess):
        images, labels = sess.run([images, labels])
        self.assertEqual(images.shape, (batch_size, 28, 28, 1))
        self.assertEqual(labels.shape, (batch_size, 10))

if __name__ == '__main__':
  tf.test.main()
