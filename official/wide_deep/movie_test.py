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

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.testing import integration
from official.wide_deep import movie_dataset
from official.wide_deep import movie_main
from official.wide_deep import wide_deep_run_loop

tf.logging.set_verbosity(tf.logging.ERROR)

TEST_RATINGS_CSV = os.path.join(
    os.path.dirname(__file__), "movie_test_ratings.csv")
TEST_METADATA_CSV = os.path.join(
    os.path.dirname(__file__), "movie_test_metadata.csv")

TEST_INPUT_VALUES = {
  "movieId": [15602],
  "rating": [1.],
  "budget": [0.],
  "genres_1": b"Comedy",
  "genres_0": b"Romance",
  "original_language": b"en",
  "userId": [7],
  "year": [1995]
}


class BaseTest(tf.test.TestCase):
  """Tests for Wide Deep model."""

  @classmethod
  def setUpClass(cls):  # pylint: disable=invalid-name
    super(BaseTest, cls).setUpClass()
    movie_main.define_movie_flags()

  def setUp(self):
    # Create temporary CSV file
    self.temp_dir = self.get_temp_dir()

    self.ratings_csv = os.path.join(self.temp_dir, "ratings.csv")
    self.metadata_csv = os.path.join(self.temp_dir, "movies_metadata.csv")

    tf.gfile.Copy(TEST_RATINGS_CSV, self.ratings_csv)
    tf.gfile.Copy(TEST_METADATA_CSV, self.metadata_csv)

  def test_input_fn(self):
    train_input_fn, _, _ = movie_dataset.get_input_fns(
        self.temp_dir, repeat=1,
        batch_size=8, small=False
    )
    dataset = train_input_fn()
    features, labels = dataset.make_one_shot_iterator().get_next()

    with self.test_session() as sess:
      features, labels = sess.run((features, labels))

      # Compare the two features dictionaries.
      for key in TEST_INPUT_VALUES:
        self.assertTrue(key in features)
        self.assertEqual(TEST_INPUT_VALUES[key], features[key][0])

      self.assertEqual(labels[0], [1.])

  def test_end_to_end_deep(self):
    integration.run_synthetic(
        main=movie_main.main, tmp_root=self.get_temp_dir(),
        extra_flags=[
          "--data_dir", self.get_temp_dir(),
          "--download_if_missing=false",
          "--train_epochs", "1"
        ],
        synth=False, max_train=None)


if __name__ == "__main__":
  tf.test.main()

