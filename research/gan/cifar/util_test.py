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
"""Tests for gan.cifar.util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import util

mock = tf.test.mock


class UtilTest(tf.test.TestCase):

  def test_get_generator_conditioning(self):
    conditioning = util.get_generator_conditioning(12, 4)
    self.assertEqual([12, 4], conditioning.shape.as_list())

  def test_get_image_grid(self):
    util.get_image_grid(
        tf.zeros([6, 28, 28, 1]),
        batch_size=6,
        num_classes=3,
        num_images_per_class=1)

  # Mock `inception_score` which is expensive.
  @mock.patch.object(util.tfgan.eval, 'inception_score', autospec=True)
  def test_get_inception_scores(self, mock_inception_score):
    mock_inception_score.return_value = 1.0
    util.get_inception_scores(
        tf.placeholder(tf.float32, shape=[None, 28, 28, 3]),
        batch_size=100,
        num_inception_images=10)

  # Mock `frechet_inception_distance` which is expensive.
  @mock.patch.object(util.tfgan.eval, 'frechet_inception_distance',
                     autospec=True)
  def test_get_frechet_inception_distance(self, mock_fid):
    mock_fid.return_value = 1.0
    util.get_frechet_inception_distance(
        tf.placeholder(tf.float32, shape=[None, 28, 28, 3]),
        tf.placeholder(tf.float32, shape=[None, 28, 28, 3]),
        batch_size=100,
        num_inception_images=10)


if __name__ == '__main__':
  tf.test.main()
