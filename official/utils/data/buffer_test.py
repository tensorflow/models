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
"""Ensure that NumPy array to Dataset works as expected."""

import numpy as np
import tensorflow as tf

from official.utils.data import buffer


_NAMESPACE = "test_array"
_NUM_ROWS = 100


class BaseTest(tf.test.TestCase):
  def setUp(self):
    buffer.cleanup(_NAMESPACE)

  def _compare_input_and_output(self, dtype):
    shape = (_NUM_ROWS, 4)
    min_val = np.iinfo(dtype).min
    max_val = np.iinfo(dtype).max
    if dtype.__name__.startswith("float"):
      x = np.random.uniform(low=min_val, high=max_val, size=shape)
    else:
      x = np.random.randint(low=min_val, high=max_val, size=shape)

    g = tf.Graph()

    with g.as_default():
      dataset = buffer.array_to_dataset(source_array=x, namespace=_NAMESPACE)

    with self.test_session(graph=g) as sess:
      row_op = dataset.make_one_shot_iterator().get_next()
      for i in range(_NUM_ROWS):
        row = sess.run(row_op)
        self.assertAllClose(row, x[i, :])

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(row_op)

  def test_uint8(self):
    self._compare_input_and_output(dtype=np.uint8)

  def test_uint16(self):
    self._compare_input_and_output(dtype=np.uint16)

  def test_int8(self):
    self._compare_input_and_output(dtype=np.int8)

  def test_int32(self):
    self._compare_input_and_output(dtype=np.int32)

  def test_int64(self):
    self._compare_input_and_output(dtype=np.int64)


if __name__ == "__main__":
  tf.test.main()
