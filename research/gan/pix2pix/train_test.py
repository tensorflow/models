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
"""Tests for pix2pix.train."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import train

FLAGS = flags.FLAGS
mock = tf.test.mock


class TrainTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('NoAdversarialLoss', 0.0),
      ('AdversarialLoss', 1.0))
  def test_build_graph(self, weight_factor):
    FLAGS.max_number_of_steps = 0
    FLAGS.weight_factor = weight_factor
    FLAGS.batch_size = 9
    FLAGS.patch_size = 32

    mock_imgs = np.zeros(
        [FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, 3],
        dtype=np.float32)
    with mock.patch.object(train, 'data_provider') as mock_data_provider:
      mock_data_provider.provide_data.return_value = mock_imgs
      train.main(None)

if __name__ == '__main__':
  tf.test.main()

