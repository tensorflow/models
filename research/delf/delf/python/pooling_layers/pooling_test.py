# Copyright 2021 The TensorFlow Authors All Rights Reserved.
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
"""Tests for pooling layers."""

import tensorflow as tf

from delf.python.pooling_layers import pooling


class PoolingsTest(tf.test.TestCase):

  def testMac(self):
    x = tf.constant([[[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]]])
    # Run tested function.
    result = pooling.mac(x)
    # Define expected result.
    exp_output = [[6., 7.]]
    # Compare actual and expected.
    self.assertAllClose(exp_output, result)

  def testSpoc(self):
    x = tf.constant([[[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]]])
    # Run tested function.
    result = pooling.spoc(x)
    # Define expected result.
    exp_output = [[3., 4.]]
    # Compare actual and expected.
    self.assertAllClose(exp_output, result)

  def testGem(self):
    x = tf.constant([[[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]]])
    # Run tested function.
    result = pooling.gem(x, power=3., eps=1e-6)
    # Define expected result.
    exp_output = [[4.1601677, 4.9866314]]
    # Compare actual and expected.
    self.assertAllClose(exp_output, result)

  def testGeMPooling2D(self):
    # Create a testing tensor.
    x = tf.constant([[[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]]])
    x = tf.reshape(x, [1, 3, 3, 1])

    # Checking GeMPooling2D relation to MaxPooling2D for the large values of
    # `p`.
    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                               strides=(1, 1), padding='valid')
    out_max = max_pool_2d(x)
    gem_pool_2d = pooling.GeMPooling2D(power=30., pool_size=(2, 2),
                                       strides=(1, 1), padding='valid')
    out_gem_max = gem_pool_2d(x)

    # Check that for large `p` GeMPooling2D is close to MaxPooling2D.
    self.assertAllEqual(out_max, tf.round(out_gem_max))

    # Checking GeMPooling2D relation to AveragePooling2D for the value
    # of `p` = 1.
    avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                                   strides=(1, 1),
                                                   padding='valid')
    out_avg = avg_pool_2d(x)
    gem_pool_2d = pooling.GeMPooling2D(power=1., pool_size=(2, 2),
                                       strides=(1, 1), padding='valid')
    out_gem_avg = gem_pool_2d(x)
    # Check that for `p` equals 1., GeMPooling2D becomes AveragePooling2D.
    self.assertAllEqual(out_avg, out_gem_avg)


if __name__ == '__main__':
  tf.test.main()
