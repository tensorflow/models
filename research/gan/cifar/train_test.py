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
"""Tests for tfgan.examples.train."""

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
      ('Unconditional', False, False),
      ('Conditional', True, False),
      ('SyncReplicas', False, True))
  def test_build_graph_helper(self, conditional, use_sync_replicas):
    FLAGS.max_number_of_steps = 0
    FLAGS.conditional = conditional
    FLAGS.use_sync_replicas = use_sync_replicas
    FLAGS.batch_size = 16

    # Mock input pipeline.
    mock_imgs = np.zeros([FLAGS.batch_size, 32, 32, 3], dtype=np.float32)
    mock_lbls = np.concatenate(
        (np.ones([FLAGS.batch_size, 1], dtype=np.int32),
         np.zeros([FLAGS.batch_size, 9], dtype=np.int32)), axis=1)
    with mock.patch.object(train, 'data_provider') as mock_data_provider:
      mock_data_provider.provide_data.return_value = (
          mock_imgs, mock_lbls, None, None)
      train.main(None)


if __name__ == '__main__':
  tf.test.main()
