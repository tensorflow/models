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
"""Unit tests for dataset.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.recommendation import dataset

_TRAIN_FNAME = os.path.join(
    os.path.dirname(__file__), "unittest_data/test_train_ratings.csv")
_TEST_FNAME = os.path.join(
    os.path.dirname(__file__), "unittest_data/test_eval_ratings.csv")
_TEST_NEG_FNAME = os.path.join(
    os.path.dirname(__file__), "unittest_data/test_eval_negative.csv")
_NUM_NEG = 4


class DatasetTest(tf.test.TestCase):

  def test_load_data(self):
    data = dataset.load_data(_TEST_FNAME)
    self.assertEqual(len(data), 2)

    self.assertEqual(data[0][0], 0)
    self.assertEqual(data[0][2], 1)

    self.assertEqual(data[-1][0], 1)
    self.assertEqual(data[-1][2], 1)

  def test_data_preprocessing(self):
    ncf_dataset = dataset.data_preprocessing(
        _TRAIN_FNAME, _TEST_FNAME, _TEST_NEG_FNAME, _NUM_NEG)

    # Check train data preprocessing
    self.assertAllEqual(np.array(ncf_dataset.train_data)[:, 2],
                        np.full(len(ncf_dataset.train_data), 1))
    self.assertEqual(ncf_dataset.num_users, 2)
    self.assertEqual(ncf_dataset.num_items, 175)

    # Check test dataset
    test_dataset = ncf_dataset.all_eval_data
    first_true_item = test_dataset[100]
    self.assertEqual(first_true_item[1], ncf_dataset.eval_true_items[0])
    self.assertEqual(first_true_item[1], ncf_dataset.eval_all_items[0][-1])

    last_gt_item = test_dataset[-1]
    self.assertEqual(last_gt_item[1], ncf_dataset.eval_true_items[-1])
    self.assertEqual(last_gt_item[1], ncf_dataset.eval_all_items[-1][-1])

    test_list = test_dataset.tolist()

    first_test_items = [x[1] for x in test_list if x[0] == 0]
    self.assertAllEqual(first_test_items, ncf_dataset.eval_all_items[0])

    last_test_items = [x[1] for x in test_list if x[0] == 1]
    self.assertAllEqual(last_test_items, ncf_dataset.eval_all_items[-1])

  def test_generate_train_dataset(self):
    # Check train dataset
    ncf_dataset = dataset.data_preprocessing(
        _TRAIN_FNAME, _TEST_FNAME, _TEST_NEG_FNAME, _NUM_NEG)

    train_dataset = dataset.generate_train_dataset(
        ncf_dataset.train_data, ncf_dataset.num_items, _NUM_NEG)

    # Each user has 1 positive instance followed by _NUM_NEG negative instances
    train_data_0 = train_dataset[0]
    self.assertEqual(train_data_0[2], 1)
    for i in range(1, _NUM_NEG + 1):
      train_data = train_dataset[i]
      self.assertEqual(train_data_0[0], train_data[0])
      self.assertNotEqual(train_data_0[1], train_data[1])
      self.assertEqual(0, train_data[2])

    train_data_last = train_dataset[-1 - _NUM_NEG]
    self.assertEqual(train_data_last[2], 1)
    for i in range(-1, -_NUM_NEG):
      train_data = train_dataset[i]
      self.assertEqual(train_data_last[0], train_data[0])
      self.assertNotEqual(train_data_last[1], train_data[1])
      self.assertEqual(0, train_data[2])


if __name__ == "__main__":
  tf.test.main()
