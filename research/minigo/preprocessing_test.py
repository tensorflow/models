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
"""Tests for preprocessing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import tempfile

import tensorflow as tf  # pylint: disable=g-bad-import-order

import coords
import features
import go
import model_params
import numpy as np
import preprocessing
import utils_test

tf.logging.set_verbosity(tf.logging.ERROR)

TEST_SGF = '''(;CA[UTF-8]SZ[9]PB[Murakawa Daisuke]PW[Iyama Yuta]KM[6.5]
           HA[0]RE[W+1.5]GM[1];B[fd];W[cf])'''


class TestPreprocessing(utils_test.MiniGoUnitTest):

  def create_random_data(self, num_examples):
    raw_data = []
    for _ in range(num_examples):
      feature = np.random.random([
          utils_test.BOARD_SIZE, utils_test.BOARD_SIZE,
          features.NEW_FEATURES_PLANES]).astype(np.uint8)
      pi = np.random.random([utils_test.BOARD_SIZE * utils_test.BOARD_SIZE
                             + 1]).astype(np.float32)
      value = np.random.random()
      raw_data.append((feature, pi, value))
    return raw_data

  def extract_data(self, tf_record, filter_amount=1):
    pos_tensor, label_tensors = preprocessing.get_input_tensors(
        model_params.DummyMiniGoParams(), 1, [tf_record], num_repeats=1,
        shuffle_records=False, shuffle_examples=False,
        filter_amount=filter_amount)
    recovered_data = []
    with tf.Session() as sess:
      while True:
        try:
          pos_value, label_values = sess.run([pos_tensor, label_tensors])
          recovered_data.append((
              pos_value,
              label_values['pi_tensor'],
              label_values['value_tensor']))
        except tf.errors.OutOfRangeError:
          break
    return recovered_data

  def assertEqualData(self, data1, data2):
    # Assert that two data are equal, where both are of form:
    # data = List<Tuple<feature_array, pi_array, value>>
    self.assertEqual(len(data1), len(data2))
    for datum1, datum2 in zip(data1, data2):
      # feature
      self.assertEqualNPArray(datum1[0], datum2[0])
      # pi
      self.assertEqualNPArray(datum1[1], datum2[1])
      # value
      self.assertEqual(datum1[2], datum2[2])

  def test_serialize_round_trip(self):
    np.random.seed(1)
    raw_data = self.create_random_data(10)
    tfexamples = list(map(preprocessing.make_tf_example, *zip(*raw_data)))

    with tempfile.NamedTemporaryFile() as f:
      preprocessing.write_tf_examples(f.name, tfexamples)
      recovered_data = self.extract_data(f.name)

    self.assertEqualData(raw_data, recovered_data)

  def test_filter(self):
    raw_data = self.create_random_data(100)
    tfexamples = list(map(preprocessing.make_tf_example, *zip(*raw_data)))

    with tempfile.NamedTemporaryFile() as f:
      preprocessing.write_tf_examples(f.name, tfexamples)
      recovered_data = self.extract_data(f.name, filter_amount=.05)

    self.assertLess(len(recovered_data), 50)

  def test_serialize_round_trip_no_parse(self):
    np.random.seed(1)
    raw_data = self.create_random_data(10)
    tfexamples = list(map(preprocessing.make_tf_example, *zip(*raw_data)))

    with tempfile.NamedTemporaryFile() as start_file, \
        tempfile.NamedTemporaryFile() as rewritten_file:
      preprocessing.write_tf_examples(start_file.name, tfexamples)
      # We want to test that the rewritten, shuffled file contains correctly
      # serialized tf.Examples.
      batch_size = 4
      batches = list(preprocessing.shuffle_tf_examples(
          1000, batch_size, [start_file.name]))
      # 2 batches of 4, 1 incomplete batch of 2.
      self.assertEqual(len(batches), 3)

      # concatenate list of lists into one list
      all_batches = list(itertools.chain.from_iterable(batches))

      for _ in batches:
        preprocessing.write_tf_examples(
            rewritten_file.name, all_batches, serialize=False)

      original_data = self.extract_data(start_file.name)
      recovered_data = self.extract_data(rewritten_file.name)

    # stuff is shuffled, so sort before checking equality
    def sort_key(nparray_tuple):
      return nparray_tuple[2]
    original_data = sorted(original_data, key=sort_key)
    recovered_data = sorted(recovered_data, key=sort_key)

    self.assertEqualData(original_data, recovered_data)

  def test_make_dataset_from_sgf(self):
    with tempfile.NamedTemporaryFile() as sgf_file, \
        tempfile.NamedTemporaryFile() as record_file:
      sgf_file.write(TEST_SGF.encode('utf8'))
      sgf_file.seek(0)
      preprocessing.make_dataset_from_sgf(
          utils_test.BOARD_SIZE, sgf_file.name, record_file.name)
      recovered_data = self.extract_data(record_file.name)
    start_pos = go.Position(utils_test.BOARD_SIZE)
    first_move = coords.from_sgf('fd')
    next_pos = start_pos.play_move(first_move)
    second_move = coords.from_sgf('cf')
    expected_data = [
        (
            features.extract_features(utils_test.BOARD_SIZE, start_pos),
            preprocessing._one_hot(utils_test.BOARD_SIZE, coords.to_flat(
                utils_test.BOARD_SIZE, first_move)), -1
        ),
        (
            features.extract_features(utils_test.BOARD_SIZE, next_pos),
            preprocessing._one_hot(utils_test.BOARD_SIZE, coords.to_flat(
                utils_test.BOARD_SIZE, second_move)), -1
        )
    ]
    self.assertEqualData(expected_data, recovered_data)


if __name__ == '__main__':
  tf.test.main()
