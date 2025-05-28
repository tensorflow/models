# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for ops."""

from unittest import mock

import tensorflow as tf, tf_keras

from official.modeling.privacy import ops


class OpsTest(tf.test.TestCase):

  def test_clip_l2_norm(self):
    x = tf.constant([4.0, 3.0])
    y = tf.constant([[12.0]])
    tensors = [(x, x), (y, y)]
    clipped = ops.clip_l2_norm(tensors, 1.0)
    for a, b in zip(clipped, tensors):
      self.assertAllClose(a[0], b[0] / 13.0)  # sqrt(4^2 + 3^2 + 12 ^3) = 13
      self.assertAllClose(a[1], b[1])

  @mock.patch.object(tf.random,
                     'normal',
                     autospec=True)
  def test_add_noise(self, mock_random):
    x = tf.constant([0.0, 0.0])
    y = tf.constant([[0.0]])
    tensors = [(x, x), (y, y)]
    mock_random.side_effect = [tf.constant([1.0, 1.0]), tf.constant([[1.0]])]
    added = ops.add_noise(tensors, 10.0)
    for a, b in zip(added, tensors):
      self.assertAllClose(a[0], b[0] + 1.0)
      self.assertAllClose(a[1], b[1])
    _, kwargs = mock_random.call_args
    self.assertEqual(kwargs['stddev'], 10.0)


if __name__ == '__main__':
  tf.test.main()
