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
"""Tests for image_compression.train."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
import train

FLAGS = tf.flags.FLAGS
mock = tf.test.mock


class TrainTest(tf.test.TestCase):

  def _test_build_graph_helper(self, weight_factor):
    FLAGS.max_number_of_steps = 0
    FLAGS.weight_factor = weight_factor

    batch_size = 3
    patch_size = 16

    FLAGS.batch_size = batch_size
    FLAGS.patch_size = patch_size
    mock_imgs = np.zeros([batch_size, patch_size, patch_size, 3],
                         dtype=np.float32)

    with mock.patch.object(train, 'data_provider') as mock_data_provider:
      mock_data_provider.provide_data.return_value = mock_imgs
      train.main(None)

  def test_build_graph_noadversarialloss(self):
    self._test_build_graph_helper(0.0)

  def test_build_graph_adversarialloss(self):
    self._test_build_graph_helper(1.0)


if __name__ == '__main__':
  tf.test.main()

