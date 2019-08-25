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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.datasets import movielens
from official.utils.misc import keras_utils
from official.utils.testing import integration
from official.r1.wide_deep import movielens_dataset
from official.r1.wide_deep import movielens_main

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


TEST_INPUT_VALUES = {
    "genres": np.array(
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "user_id": [3],
    "item_id": [4],
}

TEST_ITEM_DATA = """item_id,titles,genres
1,Movie_1,Comedy|Romance
2,Movie_2,Adventure|Children's
3,Movie_3,Comedy|Drama
4,Movie_4,Comedy
5,Movie_5,Action|Crime|Thriller
6,Movie_6,Action
7,Movie_7,Action|Adventure|Thriller"""

TEST_RATING_DATA = """user_id,item_id,rating,timestamp
1,2,5,978300760
1,3,3,978302109
1,6,3,978301968
2,1,4,978300275
2,7,5,978824291
3,1,3,978302268
3,4,5,978302039
3,5,5,978300719
"""


class BaseTest(tf.test.TestCase):
  """Tests for Wide Deep model."""

  @classmethod
  def setUpClass(cls):  # pylint: disable=invalid-name
    super(BaseTest, cls).setUpClass()
    movielens_main.define_movie_flags()

  def setUp(self):
    # Create temporary CSV file
    self.temp_dir = self.get_temp_dir()
    tf.io.gfile.makedirs(os.path.join(self.temp_dir, movielens.ML_1M))

    self.ratings_csv = os.path.join(
        self.temp_dir, movielens.ML_1M, movielens.RATINGS_FILE)
    self.item_csv = os.path.join(
        self.temp_dir, movielens.ML_1M, movielens.MOVIES_FILE)

    with tf.io.gfile.GFile(self.ratings_csv, "w") as f:
      f.write(TEST_RATING_DATA)

    with tf.io.gfile.GFile(self.item_csv, "w") as f:
      f.write(TEST_ITEM_DATA)

  @unittest.skipIf(keras_utils.is_v2_0(), "TF 1.0 only test.")
  def test_input_fn(self):
    train_input_fn, _, _ = movielens_dataset.construct_input_fns(
        dataset=movielens.ML_1M, data_dir=self.temp_dir, batch_size=8, repeat=1)

    dataset = train_input_fn()
    features, labels = dataset.make_one_shot_iterator().get_next()

    with self.session() as sess:
      features, labels = sess.run((features, labels))

      # Compare the two features dictionaries.
      for key in TEST_INPUT_VALUES:
        self.assertTrue(key in features)
        self.assertAllClose(TEST_INPUT_VALUES[key], features[key][0])

      self.assertAllClose(labels[0], [1.0])

  @unittest.skipIf(keras_utils.is_v2_0(), "TF 1.0 only test.")
  def test_end_to_end_deep(self):
    integration.run_synthetic(
        main=movielens_main.main, tmp_root=self.temp_dir,
        extra_flags=[
            "--data_dir", self.temp_dir,
            "--download_if_missing=false",
            "--train_epochs", "1",
            "--epochs_between_evals", "1"
        ],
        synth=False)


if __name__ == "__main__":
  tf.test.main()
