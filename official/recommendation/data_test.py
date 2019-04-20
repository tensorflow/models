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

from collections import defaultdict
import hashlib
import os

import mock
import numpy as np
import scipy.stats
import tensorflow as tf

from official.datasets import movielens
from official.recommendation import constants as rconst
from official.recommendation import data_preprocessing
from official.recommendation import popen_helper


DATASET = "ml-test"
NUM_USERS = 1000
NUM_ITEMS = 2000
NUM_PTS = 50000
BATCH_SIZE = 2048
EVAL_BATCH_SIZE = 4000
NUM_NEG = 4


END_TO_END_TRAIN_MD5 = "b218738e915e825d03939c5e305a2698"
END_TO_END_EVAL_MD5 = "d753d0f3186831466d6e218163a9501e"
FRESH_RANDOMNESS_MD5 = "63d0dff73c0e5f1048fbdc8c65021e22"


def mock_download(*args, **kwargs):
  return

# The forkpool used by data producers interacts badly with the threading
# used by TestCase. Without this patch tests will hang, and no amount
# of diligent closing and joining within the producer will prevent it.
@mock.patch.object(popen_helper, "get_forkpool", popen_helper.get_fauxpool)
class BaseTest(tf.test.TestCase):
  def setUp(self):
    self.temp_data_dir = self.get_temp_dir()
    ratings_folder = os.path.join(self.temp_data_dir, DATASET)
    tf.io.gfile.makedirs(ratings_folder)
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

    self.rating_file = os.path.join(ratings_folder, movielens.RATINGS_FILE)
    self.seen_pairs = set()
    self.holdout = {}
    with tf.io.gfile.GFile(self.rating_file, "w") as f:
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
    data_preprocessing.DATASET_TO_NUM_USERS_AND_ITEMS[DATASET] = (NUM_USERS,
                                                                  NUM_ITEMS)

  def make_params(self, train_epochs=1):
    return {
        "train_epochs": train_epochs,
        "batches_per_step": 1,
        "use_seed": False,
        "batch_size": BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "num_neg": NUM_NEG,
        "match_mlperf": True,
        "use_tpu": False,
        "use_xla_for_gpu": False,
    }

  def test_preprocessing(self):
    # For the most part the necessary checks are performed within
    # _filter_index_sort()

    cache_path = os.path.join(self.temp_data_dir, "test_cache.pickle")
    data, valid_cache = data_preprocessing._filter_index_sort(
        self.rating_file, cache_path=cache_path)

    assert len(data[rconst.USER_MAP]) == NUM_USERS
    assert len(data[rconst.ITEM_MAP]) == NUM_ITEMS

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

  def _test_end_to_end(self, constructor_type):
    params = self.make_params(train_epochs=1)
    _, _, producer = data_preprocessing.instantiate_pipeline(
        dataset=DATASET, data_dir=self.temp_data_dir, params=params,
        constructor_type=constructor_type, deterministic=True)

    producer.start()
    producer.join()
    assert producer._fatal_exception is None

    user_inv_map = {v: k for k, v in producer.user_map.items()}
    item_inv_map = {v: k for k, v in producer.item_map.items()}

    # ==========================================================================
    # == Training Data =========================================================
    # ==========================================================================
    g = tf.Graph()
    with g.as_default():
      input_fn = producer.make_input_fn(is_training=True)
      dataset = input_fn(params)

    first_epoch = self.drain_dataset(dataset=dataset, g=g)

    counts = defaultdict(int)
    train_examples = {
        True: set(),
        False: set(),
    }

    md5 = hashlib.md5()
    for features, labels in first_epoch:
      data_list = [
          features[movielens.USER_COLUMN], features[movielens.ITEM_COLUMN],
          features[rconst.VALID_POINT_MASK], labels]
      for i in data_list:
        md5.update(i.tobytes())

      for u, i, v, l in zip(*data_list):
        if not v:
          continue  # ignore padding

        u_raw = user_inv_map[u]
        i_raw = item_inv_map[i]
        if ((u_raw, i_raw) in self.seen_pairs) != l:
          # The evaluation item is not considered during false negative
          # generation, so it will occasionally appear as a negative example
          # during training.
          assert not l
          self.assertEqual(i_raw, self.holdout[u_raw][1])
        train_examples[l].add((u_raw, i_raw))
        counts[(u_raw, i_raw)] += 1

    self.assertRegexpMatches(md5.hexdigest(), END_TO_END_TRAIN_MD5)

    num_positives_seen = len(train_examples[True])
    self.assertEqual(producer._train_pos_users.shape[0], num_positives_seen)

    # This check is more heuristic because negatives are sampled with
    # replacement. It only checks that negative generation is reasonably random.
    self.assertGreater(
        len(train_examples[False]) / NUM_NEG / num_positives_seen, 0.9)

    # This checks that the samples produced are independent by checking the
    # number of duplicate entries. If workers are not properly independent there
    # will be lots of repeated pairs.
    self.assertLess(np.mean(list(counts.values())), 1.1)

    # ==========================================================================
    # == Eval Data =============================================================
    # ==========================================================================
    with g.as_default():
      input_fn = producer.make_input_fn(is_training=False)
      dataset = input_fn(params)

    eval_data = self.drain_dataset(dataset=dataset, g=g)

    current_user = None
    md5 = hashlib.md5()
    for features in eval_data:
      data_list = [
          features[movielens.USER_COLUMN], features[movielens.ITEM_COLUMN],
          features[rconst.DUPLICATE_MASK]]
      for i in data_list:
        md5.update(i.tobytes())

      for idx, (u, i, d) in enumerate(zip(*data_list)):
        u_raw = user_inv_map[u]
        i_raw = item_inv_map[i]
        if current_user is None:
          current_user = u

        # Ensure that users appear in blocks, as the evaluation logic expects
        # this structure.
        self.assertEqual(u, current_user)

        # The structure of evaluation data is 999 negative examples followed
        # by the holdout positive.
        if not (idx + 1) % (rconst.NUM_EVAL_NEGATIVES + 1):
          # Check that the last element in each chunk is the holdout item.
          self.assertEqual(i_raw, self.holdout[u_raw][1])
          current_user = None

        elif i_raw == self.holdout[u_raw][1]:
          # Because the holdout item is not given to the negative generation
          # process, it can appear as a negative. In that case, it should be
          # masked out as a duplicate. (Since the true positive is placed at
          # the end and would therefore lose the tie.)
          assert d

        else:
          # Otherwise check that the other 999 points for a user are selected
          # from the negatives.
          assert (u_raw, i_raw) not in self.seen_pairs

    self.assertRegexpMatches(md5.hexdigest(), END_TO_END_EVAL_MD5)

  def _test_fresh_randomness(self, constructor_type):
    train_epochs = 5
    params = self.make_params(train_epochs=train_epochs)
    _, _, producer = data_preprocessing.instantiate_pipeline(
        dataset=DATASET, data_dir=self.temp_data_dir, params=params,
        constructor_type=constructor_type, deterministic=True)

    producer.start()

    results = []
    g = tf.Graph()
    with g.as_default():
      for _ in range(train_epochs):
        input_fn = producer.make_input_fn(is_training=True)
        dataset = input_fn(params)
        results.extend(self.drain_dataset(dataset=dataset, g=g))

    producer.join()
    assert producer._fatal_exception is None

    positive_counts, negative_counts = defaultdict(int), defaultdict(int)
    md5 = hashlib.md5()
    for features, labels in results:
      data_list = [
          features[movielens.USER_COLUMN], features[movielens.ITEM_COLUMN],
          features[rconst.VALID_POINT_MASK], labels]
      for i in data_list:
        md5.update(i.tobytes())

      for u, i, v, l in zip(*data_list):
        if not v:
          continue  # ignore padding

        if l:
          positive_counts[(u, i)] += 1
        else:
          negative_counts[(u, i)] += 1

    self.assertRegexpMatches(md5.hexdigest(), FRESH_RANDOMNESS_MD5)

    # The positive examples should appear exactly once each epoch
    self.assertAllEqual(list(positive_counts.values()),
                        [train_epochs for _ in positive_counts])

    # The threshold for the negatives is heuristic, but in general repeats are
    # expected, but should not appear too frequently.

    pair_cardinality = NUM_USERS * NUM_ITEMS
    neg_pair_cardinality = pair_cardinality - len(self.seen_pairs)

    # Approximation for the expectation number of times that a particular
    # negative will appear in a given epoch. Implicit in this calculation is the
    # treatment of all negative pairs as equally likely. Normally is not
    # necessarily reasonable; however the generation in self.setUp() will
    # approximate this behavior sufficiently for heuristic testing.
    e_sample = len(self.seen_pairs) * NUM_NEG / neg_pair_cardinality

    # The frequency of occurance of a given negative pair should follow an
    # approximately binomial distribution in the limit that the cardinality of
    # the negative pair set >> number of samples per epoch.
    approx_pdf = scipy.stats.binom.pmf(k=np.arange(train_epochs+1),
                                       n=train_epochs, p=e_sample)

    # Tally the actual observed counts.
    count_distribution = [0 for _ in range(train_epochs + 1)]
    for i in negative_counts.values():
      i = min([i, train_epochs])  # round down tail for simplicity.
      count_distribution[i] += 1
    count_distribution[0] = neg_pair_cardinality - sum(count_distribution[1:])

    # Check that the frequency of negative pairs is approximately binomial.
    for i in range(train_epochs + 1):
      if approx_pdf[i] < 0.05:
        continue  # Variance will be high at the tails.

      observed_fraction = count_distribution[i] / neg_pair_cardinality
      deviation = (2 * abs(observed_fraction - approx_pdf[i]) /
                   (observed_fraction + approx_pdf[i]))

      self.assertLess(deviation, 0.2)

  def test_end_to_end_materialized(self):
    self._test_end_to_end("materialized")

  def test_end_to_end_bisection(self):
    self._test_end_to_end("bisection")

  def test_fresh_randomness_materialized(self):
    self._test_fresh_randomness("materialized")

  def test_fresh_randomness_bisection(self):
    self._test_fresh_randomness("bisection")


if __name__ == "__main__":
  tf.test.main()
