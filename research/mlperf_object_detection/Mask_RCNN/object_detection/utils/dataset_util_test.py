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

"""Tests for object_detection.utils.dataset_util."""

import os
import numpy as np
import tensorflow as tf

from object_detection.protos import input_reader_pb2
from object_detection.utils import dataset_util


class DatasetUtilTest(tf.test.TestCase):

  def setUp(self):
    self._path_template = os.path.join(self.get_temp_dir(), 'examples_%s.txt')

    for i in range(5):
      path = self._path_template % i
      with tf.gfile.Open(path, 'wb') as f:
        f.write('\n'.join([str(i + 1), str((i + 1) * 10)]))

    self._shuffle_path_template = os.path.join(self.get_temp_dir(),
                                               'shuffle_%s.txt')
    for i in range(2):
      path = self._shuffle_path_template % i
      with tf.gfile.Open(path, 'wb') as f:
        f.write('\n'.join([str(i)] * 5))

  def _get_dataset_next(self, files, config, batch_size):
    def decode_func(value):
      return [tf.string_to_number(value, out_type=tf.int32)]

    dataset = dataset_util.read_dataset(
        tf.data.TextLineDataset, decode_func, files, config)
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

  def test_read_examples_list(self):
    example_list_data = """example1 1\nexample2 2"""
    example_list_path = os.path.join(self.get_temp_dir(), 'examples.txt')
    with tf.gfile.Open(example_list_path, 'wb') as f:
      f.write(example_list_data)

    examples = dataset_util.read_examples_list(example_list_path)
    self.assertListEqual(['example1', 'example2'], examples)

  def test_make_initializable_iterator_with_hashTable(self):
    keys = [1, 0, -1]
    dataset = tf.data.Dataset.from_tensor_slices([[1, 2, -1, 5]])
    table = tf.contrib.lookup.HashTable(
        initializer=tf.contrib.lookup.KeyValueTensorInitializer(
            keys=keys,
            values=list(reversed(keys))),
        default_value=100)
    dataset = dataset.map(table.lookup)
    data = dataset_util.make_initializable_iterator(dataset).get_next()
    init = tf.tables_initializer()

    with self.test_session() as sess:
      sess.run(init)
      self.assertAllEqual(sess.run(data), [-1, 100, 1, 100])

  def test_read_dataset(self):
    config = input_reader_pb2.InputReader()
    config.num_readers = 1
    config.shuffle = False

    data = self._get_dataset_next([self._path_template % '*'], config,
                                  batch_size=20)
    with self.test_session() as sess:
      self.assertAllEqual(sess.run(data),
                          [[1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 1, 10, 2, 20, 3,
                            30, 4, 40, 5, 50]])

  def test_reduce_num_reader(self):
    config = input_reader_pb2.InputReader()
    config.num_readers = 10
    config.shuffle = False

    data = self._get_dataset_next([self._path_template % '*'], config,
                                  batch_size=20)
    with self.test_session() as sess:
      self.assertAllEqual(sess.run(data),
                          [[1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 1, 10, 2, 20, 3,
                            30, 4, 40, 5, 50]])

  def test_enable_shuffle(self):
    config = input_reader_pb2.InputReader()
    config.num_readers = 1
    config.shuffle = True

    data = self._get_dataset_next(
        [self._shuffle_path_template % '*'], config, batch_size=10)
    expected_non_shuffle_output = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    with self.test_session() as sess:
      self.assertTrue(
          np.any(np.not_equal(sess.run(data), expected_non_shuffle_output)))

  def test_disable_shuffle_(self):
    config = input_reader_pb2.InputReader()
    config.num_readers = 1
    config.shuffle = False

    data = self._get_dataset_next(
        [self._shuffle_path_template % '*'], config, batch_size=10)
    expected_non_shuffle_output = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    with self.test_session() as sess:
      self.assertAllEqual(sess.run(data), [expected_non_shuffle_output])

  def test_read_dataset_single_epoch(self):
    config = input_reader_pb2.InputReader()
    config.num_epochs = 1
    config.num_readers = 1
    config.shuffle = False

    data = self._get_dataset_next([self._path_template % '0'], config,
                                  batch_size=30)
    with self.test_session() as sess:
      # First batch will retrieve as much as it can, second batch will fail.
      self.assertAllEqual(sess.run(data), [[1, 10]])
      self.assertRaises(tf.errors.OutOfRangeError, sess.run, data)


if __name__ == '__main__':
  tf.test.main()
