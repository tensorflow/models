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
"""Test NCF data pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from official.datasets import movielens
from official.recommendation import constants as rconst
from official.recommendation import data_preprocessing


DATASET = "ml-test"
NUM_USERS = 1000
NUM_ITEMS = 2000
NUM_PTS = 50000
BATCH_SIZE = 2048
NUM_NEG = 4


def mock_download(*args, **kwargs):
  return


class BaseTest(tf.test.TestCase):
  def setUp(self):
    self.temp_data_dir = self.get_temp_dir()
    ratings_folder = os.path.join(self.temp_data_dir, DATASET)
    tf.gfile.MakeDirs(ratings_folder)
    np.random.seed(0)
    raw_user_ids = np.arange(NUM_USERS * 3)
    np.random.shuffle(raw_user_ids)
    raw_user_ids = raw_user_ids[:NUM_USERS]

    raw_item_ids = np.arange(NUM_ITEMS * 3)
    np.random.shuffle(raw_item_ids)
    raw_item_ids = raw_item_ids[:NUM_ITEMS]

    users = np.random.choice(raw_user_ids, NUM_PTS)
    items = np.random.choice(raw_item_ids, NUM_PTS)
    scores = np.random.randint(low=0, high=5, size=NUM_PTS)
    times = np.random.randint(low=1000000000, high=1200000000, size=NUM_PTS)

    rating_file = os.path.join(ratings_folder, movielens.RATINGS_FILE)
    self.seen_pairs = set()
    self.holdout = {}
    with tf.gfile.Open(rating_file, "w") as f:
      f.write("user_id,item_id,rating,timestamp\n")
      for usr, itm, scr, ts in zip(users, items, scores, times):
        pair = (usr, itm)
        if pair in self.seen_pairs:
          continue
        self.seen_pairs.add(pair)
        if usr not in self.holdout or (ts, itm) > self.holdout[usr]:
          self.holdout[usr] = (ts, itm)

        f.write("{},{},{},{}\n".format(usr, itm, scr, ts))

    movielens.download = mock_download
    movielens.NUM_RATINGS[DATASET] = NUM_PTS

  def test_preprocessing(self):
    # For the most part the necessary checks are performed within
    # construct_cache()
    ncf_dataset = data_preprocessing.construct_cache(
        dataset=DATASET, data_dir=self.temp_data_dir, num_data_readers=2,
        match_mlperf=False, deterministic=False)
    assert ncf_dataset.num_users == NUM_USERS
    assert ncf_dataset.num_items == NUM_ITEMS

    time.sleep(1)  # Ensure we create the next cache in a new directory.
    ncf_dataset = data_preprocessing.construct_cache(
        dataset=DATASET, data_dir=self.temp_data_dir, num_data_readers=2,
        match_mlperf=True, deterministic=False)
    assert ncf_dataset.num_users == NUM_USERS
    assert ncf_dataset.num_items == NUM_ITEMS

  def drain_dataset(self, dataset, g):
    # type: (tf.data.Dataset, tf.Graph) -> list
    with self.test_session(graph=g) as sess:
      with g.as_default():
        batch = dataset.make_one_shot_iterator().get_next()
      output = []
      while True:
        try:
          output.append(sess.run(batch))
        except tf.errors.OutOfRangeError:
          break
    return output

  def test_end_to_end(self):
    ncf_dataset, _ = data_preprocessing.instantiate_pipeline(
        dataset=DATASET, data_dir=self.temp_data_dir,
        batch_size=BATCH_SIZE, eval_batch_size=BATCH_SIZE, num_data_readers=2,
        num_neg=NUM_NEG)

    g = tf.Graph()
    with g.as_default():
      input_fn, record_dir, batch_count = \
        data_preprocessing.make_train_input_fn(ncf_dataset)
      dataset = input_fn({"batch_size": BATCH_SIZE, "use_tpu": False})
    first_epoch = self.drain_dataset(dataset=dataset, g=g)
    user_inv_map = {v: k for k, v in ncf_dataset.user_map.items()}
    item_inv_map = {v: k for k, v in ncf_dataset.item_map.items()}

    train_examples = {
        True: set(),
        False: set(),
    }
    for features, labels in first_epoch:
      for u, i, l in zip(features[movielens.USER_COLUMN],
                         features[movielens.ITEM_COLUMN], labels):
        u_raw = user_inv_map[u]
        i_raw = item_inv_map[i]
        if ((u_raw, i_raw) in self.seen_pairs) != l:
          # The evaluation item is not considered during false negative
          # generation, so it will occasionally appear as a negative example
          # during training.
          assert not l
          assert i_raw == self.holdout[u_raw][1]
        train_examples[l].add((u_raw, i_raw))
    num_positives_seen = len(train_examples[True])

    # The numbers don't match exactly because the last batch spills over into
    # the next epoch
    assert ncf_dataset.num_train_positives - num_positives_seen < BATCH_SIZE

    # This check is more heuristic because negatives are sampled with
    # replacement. It only checks that negative generation is reasonably random.
    assert len(train_examples[False]) / NUM_NEG / num_positives_seen > 0.9

  def test_shard_randomness(self):
    users = [0, 0, 0, 0, 1, 1, 1, 1]
    items = [0, 2, 4, 6, 0, 2, 4, 6]
    times = [1, 2, 3, 4, 1, 2, 3, 4]
    df = pd.DataFrame({movielens.USER_COLUMN: users,
                       movielens.ITEM_COLUMN: items,
                       movielens.TIMESTAMP_COLUMN: times})
    cache_paths = rconst.Paths(data_dir=self.temp_data_dir)
    np.random.seed(1)
    data_preprocessing.generate_train_eval_data(df, approx_num_shards=2,
                                                num_items=10,
                                                cache_paths=cache_paths,
                                                match_mlperf=True)
    with tf.gfile.Open(cache_paths.eval_raw_file, "rb") as f:
      eval_data = pickle.load(f)
    eval_items_per_user = rconst.NUM_EVAL_NEGATIVES + 1
    self.assertAllClose(eval_data[0][movielens.USER_COLUMN],
                        [0] * eval_items_per_user + [1] * eval_items_per_user)

    # Each shard process should generate different random items.
    self.assertNotAllClose(
        eval_data[0][movielens.ITEM_COLUMN][:eval_items_per_user],
        eval_data[0][movielens.ITEM_COLUMN][eval_items_per_user:])


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
