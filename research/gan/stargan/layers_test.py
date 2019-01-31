# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf

import layers


class LayersTest(tf.test.TestCase):

  def test_residual_block(self):

    n = 2
    h = 32
    w = h
    c = 256

    input_tensor = tf.random_uniform((n, h, w, c))
    output_tensor = layers._residual_block(
        input_net=input_tensor,
        num_outputs=c,
        kernel_size=3,
        stride=1,
        padding_size=1)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output_tensor)
      self.assertTupleEqual((n, h, w, c), output.shape)

  def test_generator_down_sample(self):

    n = 2
    h = 128
    w = h
    c = 3 + 3

    input_tensor = tf.random_uniform((n, h, w, c))
    output_tensor = layers.generator_down_sample(input_tensor)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output_tensor)
      self.assertTupleEqual((n, h // 4, w // 4, 256), output.shape)

  def test_generator_bottleneck(self):

    n = 2
    h = 32
    w = h
    c = 256

    input_tensor = tf.random_uniform((n, h, w, c))
    output_tensor = layers.generator_bottleneck(input_tensor)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output_tensor)
      self.assertTupleEqual((n, h, w, c), output.shape)

  def test_generator_up_sample(self):

    n = 2
    h = 32
    w = h
    c = 256
    c_out = 3

    input_tensor = tf.random_uniform((n, h, w, c))
    output_tensor = layers.generator_up_sample(input_tensor, c_out)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output_tensor)
      self.assertTupleEqual((n, h * 4, w * 4, c_out), output.shape)

  def test_discriminator_input_hidden(self):

    n = 2
    h = 128
    w = 128
    c = 3

    input_tensor = tf.random_uniform((n, h, w, c))
    output_tensor = layers.discriminator_input_hidden(input_tensor)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output_tensor)
      self.assertTupleEqual((n, 2, 2, 2048), output.shape)

  def test_discriminator_output_source(self):

    n = 2
    h = 2
    w = 2
    c = 2048

    input_tensor = tf.random_uniform((n, h, w, c))
    output_tensor = layers.discriminator_output_source(input_tensor)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output_tensor)
      self.assertTupleEqual((n, h, w, 1), output.shape)

  def test_discriminator_output_class(self):

    n = 2
    h = 2
    w = 2
    c = 2048
    num_domain = 3

    input_tensor = tf.random_uniform((n, h, w, c))
    output_tensor = layers.discriminator_output_class(input_tensor, num_domain)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output_tensor)
      self.assertTupleEqual((n, num_domain), output.shape)


if __name__ == '__main__':
  tf.test.main()
