# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for dcgan."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from nets import dcgan


class DCGANTest(tf.test.TestCase):

  def test_generator_run(self):
    tf.compat.v1.set_random_seed(1234)
    noise = tf.random.normal([100, 64])
    image, _ = dcgan.generator(noise)
    with self.test_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      image.eval()

  def test_generator_graph(self):
    tf.compat.v1.set_random_seed(1234)
    # Check graph construction for a number of image size/depths and batch
    # sizes.
    for i, batch_size in zip(xrange(3, 7), xrange(3, 8)):
      tf.compat.v1.reset_default_graph()
      final_size = 2 ** i
      noise = tf.random.normal([batch_size, 64])
      image, end_points = dcgan.generator(
          noise,
          depth=32,
          final_size=final_size)

      self.assertAllEqual([batch_size, final_size, final_size, 3],
                          image.shape.as_list())

      expected_names = ['deconv%i' % j for j in xrange(1, i)] + ['logits']
      self.assertSetEqual(set(expected_names), set(end_points.keys()))

      # Check layer depths.
      for j in range(1, i):
        layer = end_points['deconv%i' % j]
        self.assertEqual(32 * 2**(i-j-1), layer.get_shape().as_list()[-1])

  def test_generator_invalid_input(self):
    wrong_dim_input = tf.zeros([5, 32, 32])
    with self.assertRaises(ValueError):
      dcgan.generator(wrong_dim_input)

    correct_input = tf.zeros([3, 2])
    with self.assertRaisesRegexp(ValueError, 'must be a power of 2'):
      dcgan.generator(correct_input, final_size=30)

    with self.assertRaisesRegexp(ValueError, 'must be greater than 8'):
      dcgan.generator(correct_input, final_size=4)

  def test_discriminator_run(self):
    image = tf.random.uniform([5, 32, 32, 3], -1, 1)
    output, _ = dcgan.discriminator(image)
    with self.test_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      output.eval()

  def test_discriminator_graph(self):
    # Check graph construction for a number of image size/depths and batch
    # sizes.
    for i, batch_size in zip(xrange(1, 6), xrange(3, 8)):
      tf.compat.v1.reset_default_graph()
      img_w = 2 ** i
      image = tf.random.uniform([batch_size, img_w, img_w, 3], -1, 1)
      output, end_points = dcgan.discriminator(
          image,
          depth=32)

      self.assertAllEqual([batch_size, 1], output.get_shape().as_list())

      expected_names = ['conv%i' % j for j in xrange(1, i+1)] + ['logits']
      self.assertSetEqual(set(expected_names), set(end_points.keys()))

      # Check layer depths.
      for j in range(1, i+1):
        layer = end_points['conv%i' % j]
        self.assertEqual(32 * 2**(j-1), layer.get_shape().as_list()[-1])

  def test_discriminator_invalid_input(self):
    wrong_dim_img = tf.zeros([5, 32, 32])
    with self.assertRaises(ValueError):
      dcgan.discriminator(wrong_dim_img)

    spatially_undefined_shape = tf.compat.v1.placeholder(
        tf.float32, [5, 32, None, 3])
    with self.assertRaises(ValueError):
      dcgan.discriminator(spatially_undefined_shape)

    not_square = tf.zeros([5, 32, 16, 3])
    with self.assertRaisesRegexp(ValueError, 'not have equal width and height'):
      dcgan.discriminator(not_square)

    not_power_2 = tf.zeros([5, 30, 30, 3])
    with self.assertRaisesRegexp(ValueError, 'not a power of 2'):
      dcgan.discriminator(not_power_2)


if __name__ == '__main__':
  tf.test.main()
