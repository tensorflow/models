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

import ops


class OpsTest(tf.test.TestCase):

  def test_padding_arg(self):

    pad_h = 2
    pad_w = 3

    self.assertListEqual([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]],
                         ops._padding_arg(pad_h, pad_w, 'NHWC'))

  def test_padding_arg_specify_format(self):

    pad_h = 2
    pad_w = 3

    self.assertListEqual([[pad_h, pad_h], [pad_w, pad_w], [0, 0]],
                         ops._padding_arg(pad_h, pad_w, 'HWC'))

  def test_padding_arg_invalid_format(self):

    pad_h = 2
    pad_w = 3

    with self.assertRaises(ValueError):
      ops._padding_arg(pad_h, pad_w, 'INVALID')

  def test_padding(self):

    n = 2
    h = 128
    w = 64
    c = 3
    pad = 3

    test_input_tensor = tf.random_uniform((n, h, w, c))
    test_output_tensor = ops.pad(test_input_tensor, padding_size=pad)

    with self.test_session() as sess:
      output = sess.run(test_output_tensor)
      self.assertTupleEqual((n, h + pad * 2, w + pad * 2, c), output.shape)

  def test_padding_with_3D_tensor(self):

    h = 128
    w = 64
    c = 3
    pad = 3

    test_input_tensor = tf.random_uniform((h, w, c))
    test_output_tensor = ops.pad(test_input_tensor, padding_size=pad)

    with self.test_session() as sess:
      output = sess.run(test_output_tensor)
      self.assertTupleEqual((h + pad * 2, w + pad * 2, c), output.shape)

  def test_padding_with_tensor_of_invalid_shape(self):

    n = 2
    invalid_rank = 1
    h = 128
    w = 64
    c = 3
    pad = 3

    test_input_tensor = tf.random_uniform((n, invalid_rank, h, w, c))

    with self.assertRaises(ValueError):
      ops.pad(test_input_tensor, padding_size=pad)

  def test_condition_input_with_pixel_padding(self):

    n = 2
    h = 128
    w = h
    c = 3
    num_label = 5

    input_tensor = tf.random_uniform((n, h, w, c))
    label_tensor = tf.random_uniform((n, num_label))
    output_tensor = ops.condition_input_with_pixel_padding(
        input_tensor, label_tensor)

    with self.test_session() as sess:
      labels, outputs = sess.run([label_tensor, output_tensor])
      self.assertTupleEqual((n, h, w, c + num_label), outputs.shape)
      for label, output in zip(labels, outputs):
        for i in range(output.shape[0]):
          for j in range(output.shape[1]):
            self.assertListEqual(label.tolist(), output[i, j, c:].tolist())


if __name__ == '__main__':
  tf.test.main()
