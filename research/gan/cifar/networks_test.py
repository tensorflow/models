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
"""Tests for tfgan.examples.cifar.networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
import networks


class NetworksTest(tf.test.TestCase):

  def test_generator(self):
    tf.set_random_seed(1234)
    batch_size = 100
    noise = tf.random_normal([batch_size, 64])
    image = networks.generator(noise)
    with self.test_session(use_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())
      image_np = image.eval()

    self.assertAllEqual([batch_size, 32, 32, 3], image_np.shape)
    self.assertTrue(np.all(np.abs(image_np) <= 1))

  def test_generator_conditional(self):
    tf.set_random_seed(1234)
    batch_size = 100
    noise = tf.random_normal([batch_size, 64])
    conditioning = tf.one_hot([0] * batch_size, 10)
    image = networks.conditional_generator((noise, conditioning))
    with self.test_session(use_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())
      image_np = image.eval()

    self.assertAllEqual([batch_size, 32, 32, 3], image_np.shape)
    self.assertTrue(np.all(np.abs(image_np) <= 1))

  def test_discriminator(self):
    batch_size = 5
    image = tf.random_uniform([batch_size, 32, 32, 3], -1, 1)
    dis_output = networks.discriminator(image, None)
    with self.test_session(use_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())
      dis_output_np = dis_output.eval()

    self.assertAllEqual([batch_size, 1], dis_output_np.shape)

  def test_discriminator_conditional(self):
    batch_size = 5
    image = tf.random_uniform([batch_size, 32, 32, 3], -1, 1)
    conditioning = (None, tf.one_hot([0] * batch_size, 10))
    dis_output = networks.conditional_discriminator(image, conditioning)
    with self.test_session(use_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())
      dis_output_np = dis_output.eval()

    self.assertAllEqual([batch_size, 1], dis_output_np.shape)


if __name__ == '__main__':
  tf.test.main()

