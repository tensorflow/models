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

import network


class NetworkTest(tf.test.TestCase):

  def test_generator(self):

    n = 2
    h = 128
    w = h
    c = 4
    class_num = 3

    input_tensor = tf.random_uniform((n, h, w, c))
    target_tensor = tf.random_uniform((n, class_num))
    output_tensor = network.generator(input_tensor, target_tensor)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output_tensor)
      self.assertTupleEqual((n, h, w, c), output.shape)

  def test_discriminator(self):

    n = 2
    h = 128
    w = h
    c = 3
    class_num = 3

    input_tensor = tf.random_uniform((n, h, w, c))
    output_src_tensor, output_cls_tensor = network.discriminator(
        input_tensor, class_num)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output_src, output_cls = sess.run([output_src_tensor, output_cls_tensor])
      self.assertEqual(1, len(output_src.shape))
      self.assertEqual(n, output_src.shape[0])
      self.assertTupleEqual((n, class_num), output_cls.shape)


if __name__ == '__main__':
  tf.test.main()
