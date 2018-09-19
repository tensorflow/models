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
"""Tests for binary data file utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import multiprocessing

# pylint: disable=wrong-import-order
import numpy as np
import pandas as pd
import tensorflow as tf
# pylint: enable=wrong-import-order

from official.utils.data import file_io


_RAW_ROW = "raw_row"
_DUMMY_COL = "column_0"
_DUMMY_VEC_COL = "column_1"
_DUMMY_VEC_LEN = 4

_ROWS_PER_CORE = 4
_TEST_CASES = [
    # One batch of one
    dict(row_count=1, cpu_count=1, expected=[
        [[0]]
    ]),

    dict(row_count=10, cpu_count=1, expected=[
        [[0, 1, 2, 3]], [[4, 5, 6, 7]], [[8, 9]]
    ]),

    dict(row_count=21, cpu_count=1, expected=[
        [[0, 1, 2, 3]], [[4, 5, 6, 7]], [[8, 9, 10, 11]],
        [[12, 13, 14, 15]], [[16, 17, 18, 19]], [[20]]
    ]),

    dict(row_count=1, cpu_count=4, expected=[
        [[0]]
    ]),

    dict(row_count=10, cpu_count=4, expected=[
        [[0, 1], [2, 3, 4], [5, 6], [7, 8, 9]]
    ]),

    dict(row_count=21, cpu_count=4, expected=[
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        [[16], [17], [18], [19, 20]]
    ]),

    dict(row_count=10, cpu_count=8, expected=[
        [[0], [1], [2], [3, 4], [5], [6], [7], [8, 9]]
    ]),

    dict(row_count=40, cpu_count=8, expected=[
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
         [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27],
         [28, 29, 30, 31]],
        [[32], [33], [34], [35], [36], [37], [38], [39]]
    ]),
]

_FEATURE_MAP = {
    _RAW_ROW: tf.FixedLenFeature([1], dtype=tf.int64),
    _DUMMY_COL: tf.FixedLenFeature([1], dtype=tf.int64),
    _DUMMY_VEC_COL: tf.FixedLenFeature([_DUMMY_VEC_LEN], dtype=tf.float32)
}


@contextlib.contextmanager
def fixed_core_count(cpu_count):
  """Override CPU count.

  file_io.py uses the cpu_count function to scale to the size of the instance.
  However, this is not desirable for testing because it can make the test flaky.
  Instead, this context manager fixes the count for more robust testing.

  Args:
    cpu_count: How many cores multiprocessing claims to have.

  Yields:
    Nothing. (for context manager only)
  """
  old_count_fn = multiprocessing.cpu_count
  multiprocessing.cpu_count = lambda: cpu_count
  yield
  multiprocessing.cpu_count = old_count_fn


class BaseTest(tf.test.TestCase):

  def _test_sharding(self, row_count, cpu_count, expected):
    df = pd.DataFrame({_DUMMY_COL: list(range(row_count))})
    with fixed_core_count(cpu_count):
      shards = list(file_io.iter_shard_dataframe(df, _ROWS_PER_CORE))
    result = [[j[_DUMMY_COL].tolist() for j in i] for i in shards]
    self.assertAllEqual(expected, result)

  def test_tiny_rows_low_core(self):
    self._test_sharding(**_TEST_CASES[0])

  def test_small_rows_low_core(self):
    self._test_sharding(**_TEST_CASES[1])

  def test_large_rows_low_core(self):
    self._test_sharding(**_TEST_CASES[2])

  def test_tiny_rows_medium_core(self):
    self._test_sharding(**_TEST_CASES[3])

  def test_small_rows_medium_core(self):
    self._test_sharding(**_TEST_CASES[4])

  def test_large_rows_medium_core(self):
    self._test_sharding(**_TEST_CASES[5])

  def test_small_rows_large_core(self):
    self._test_sharding(**_TEST_CASES[6])

  def test_large_rows_large_core(self):
    self._test_sharding(**_TEST_CASES[7])

  def _serialize_deserialize(self, num_cores=1, num_rows=20):
    np.random.seed(1)
    df = pd.DataFrame({
        # Serialization order is only deterministic for num_cores=1. raw_row is
        # used in validation after the deserialization.
        _RAW_ROW: np.array(range(num_rows), dtype=np.int64),
        _DUMMY_COL: np.random.randint(0, 35, size=(num_rows,)),
        _DUMMY_VEC_COL: [
            np.array([np.random.random() for _ in range(_DUMMY_VEC_LEN)])
            for i in range(num_rows)  # pylint: disable=unused-variable
        ]
    })

    with fixed_core_count(num_cores):
      buffer_path = file_io.write_to_temp_buffer(
          df, self.get_temp_dir(), [_RAW_ROW, _DUMMY_COL, _DUMMY_VEC_COL])

    with self.test_session(graph=tf.Graph()) as sess:
      dataset = tf.data.TFRecordDataset(buffer_path)
      dataset = dataset.batch(1).map(
          lambda x: tf.parse_example(x, _FEATURE_MAP))

      data_iter = dataset.make_one_shot_iterator()
      seen_rows = set()
      for i in range(num_rows+5):
        row = data_iter.get_next()
        try:
          row_id, val_0, val_1 = sess.run(
              [row[_RAW_ROW], row[_DUMMY_COL], row[_DUMMY_VEC_COL]])
          row_id, val_0, val_1 = row_id[0][0], val_0[0][0], val_1[0]
          assert row_id not in seen_rows
          seen_rows.add(row_id)

          self.assertEqual(val_0, df[_DUMMY_COL][row_id])
          self.assertAllClose(val_1, df[_DUMMY_VEC_COL][row_id])

          self.assertLess(i, num_rows, msg="Too many rows.")
        except tf.errors.OutOfRangeError:
          self.assertGreaterEqual(i, num_rows, msg="Too few rows.")

    file_io._GARBAGE_COLLECTOR.purge()
    assert not tf.gfile.Exists(buffer_path)

  def test_serialize_deserialize_0(self):
    self._serialize_deserialize(num_cores=1)

  def test_serialize_deserialize_1(self):
    self._serialize_deserialize(num_cores=2)

  def test_serialize_deserialize_2(self):
    self._serialize_deserialize(num_cores=8)


if __name__ == "__main__":
  tf.test.main()
