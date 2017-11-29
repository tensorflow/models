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
"""Tests for mnist.train."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np

import tensorflow as tf
import train

FLAGS = tf.flags.FLAGS
mock = tf.test.mock


class TrainTest(tf.test.TestCase):

  @mock.patch.object(train, 'data_provider', autospec=True)
  def test_run_one_train_step(self, mock_data_provider):
    FLAGS.max_number_of_steps = 1
    FLAGS.gan_type = 'unconditional'
    FLAGS.batch_size = 5
    FLAGS.grid_size = 1
    tf.set_random_seed(1234)

    # Mock input pipeline.
    mock_imgs = np.zeros([FLAGS.batch_size, 28, 28, 1], dtype=np.float32)
    mock_lbls = np.concatenate(
        (np.ones([FLAGS.batch_size, 1], dtype=np.int32),
         np.zeros([FLAGS.batch_size, 9], dtype=np.int32)), axis=1)
    mock_data_provider.provide_data.return_value = (mock_imgs, mock_lbls, None)

    train.main(None)

  def _test_build_graph_helper(self, gan_type):
    FLAGS.max_number_of_steps = 0
    FLAGS.gan_type = gan_type

    # Mock input pipeline.
    mock_imgs = np.zeros([FLAGS.batch_size, 28, 28, 1], dtype=np.float32)
    mock_lbls = np.concatenate(
        (np.ones([FLAGS.batch_size, 1], dtype=np.int32),
         np.zeros([FLAGS.batch_size, 9], dtype=np.int32)), axis=1)
    with mock.patch.object(train, 'data_provider') as mock_data_provider:
      mock_data_provider.provide_data.return_value = (
          mock_imgs, mock_lbls, None)
      train.main(None)

  def test_build_graph_unconditional(self):
    self._test_build_graph_helper('unconditional')

  def test_build_graph_conditional(self):
    self._test_build_graph_helper('conditional')

  def test_build_graph_infogan(self):
    self._test_build_graph_helper('infogan')

if __name__ == '__main__':
  tf.test.main()
