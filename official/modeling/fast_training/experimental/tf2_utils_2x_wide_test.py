# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for tf2_utils_2x_wide."""

import numpy as np
import tensorflow as tf, tf_keras

from official.modeling.fast_training.experimental import tf2_utils_2x_wide


class Tf2Utils2XWideTest(tf.test.TestCase):

  def test_expand_vector(self):
    x = np.array([1, 2])
    self.assertAllClose(tf2_utils_2x_wide.expand_vector(x),
                        np.array([1, 1, 2, 2]))

  def test_expand_matrix(self):
    x = np.array([[1, 2], [3, 4]])
    x = tf2_utils_2x_wide.expand_2_axes(x, epsilon=0.1)
    self.assertAllClose(x[0, :] + x[1, :], np.array([1, 1, 2, 2]))
    self.assertAllClose(x[2, :] + x[3, :], np.array([3, 3, 4, 4]))

  def test_expand_matrix_axis_0(self):
    x = np.array([[1, 2], [3, 4]])
    x = tf2_utils_2x_wide.expand_1_axis(x, axis=0, epsilon=0.1)
    self.assertAllClose(x[0, :] + x[1, :], np.array([1, 2]))
    self.assertAllClose(x[2, :] + x[3, :], np.array([3, 4]))

  def test_expand_matrix_axis_1(self):
    x = np.array([[1, 2], [3, 4]])
    x = tf2_utils_2x_wide.expand_1_axis(x, axis=-1, epsilon=0.1)
    self.assertAllClose(x[:, 0] + x[:, 1], np.array([1, 3]))
    self.assertAllClose(x[:, 2] + x[:, 3], np.array([2, 4]))

  def test_expand_3d_tensor(self):
    x0 = np.array([10, 11])
    x1 = np.array([10, 10, 11, 11])
    w0 = np.random.rand(2, 2)
    w1 = tf2_utils_2x_wide.expand_2_axes(w0, epsilon=0.1)
    o0 = np.matmul(x0, w0)
    o1 = np.matmul(x1, w1)
    self.assertAllClose(np.repeat(o0, 2, axis=-1), o1)

  def test_expand_3d_tensor_axis_0(self):
    x0 = np.array([10, 11])
    x1 = np.array([10, 10, 11, 11])
    w0 = np.random.rand(2, 2)
    w1 = tf2_utils_2x_wide.expand_1_axis(w0, axis=0, epsilon=0.1)
    o0 = np.matmul(x0, w0)
    o1 = np.matmul(x1, w1)
    self.assertAllClose(o0, o1)

  def test_expand_3d_tensor_axis_2(self):
    x = np.array([10, 11])
    w0 = np.random.rand(2, 2)
    w1 = tf2_utils_2x_wide.expand_1_axis(w0, axis=-1, epsilon=0.1)
    o0 = np.matmul(x, w0)
    o1 = np.matmul(x, w1)
    self.assertAllClose(o0, np.sum(o1.reshape(2, 2), axis=-1))

  def test_end_to_end(self):
    """Covers expand_vector, expand_2_axes, and expand_1_axis."""
    model_narrow = tf_keras.Sequential()
    model_narrow.add(tf_keras.Input(shape=(3,)))
    model_narrow.add(tf_keras.layers.Dense(4))
    model_narrow.add(tf_keras.layers.Dense(4))
    model_narrow.add(tf_keras.layers.Dense(1))

    model_wide = tf_keras.Sequential()
    model_wide.add(tf_keras.Input(shape=(6,)))
    model_wide.add(tf_keras.layers.Dense(8))
    model_wide.add(tf_keras.layers.Dense(8))
    model_wide.add(tf_keras.layers.Dense(1))

    x0 = np.array([[1, 2, 3]])
    x1 = np.array([[1, 1, 2, 2, 3, 3]])

    # Call model once to build variables first.
    _, _ = model_narrow(x0), model_wide(x1)
    tf2_utils_2x_wide.model_to_model_2x_wide(
        model_narrow, model_wide, epsilon=0.2)

    self.assertAllClose(model_narrow(x0), model_wide(x1),
                        rtol=1e-05, atol=1e-05)


if __name__ == "__main__":
  tf.test.main()
