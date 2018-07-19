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
"""End-to-end test of the training data GRPC pipeline.

This module holds onto the Dataframe generated in prepare.py, and checks that
what comes out of the GRPC data server pipeline matches that Dataframe.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from absl import app as absl_app
from absl import flags
import numpy as np
import tensorflow as tf

from official.datasets import movielens
from official.recommendation.data_server import pipeline
from official.utils.flags import core as flags_core


BATCH_SIZE = 16384


def define_flags():
  movielens.define_data_download_flags()
  flags.adopt_module_key_flags(movielens)
  flags_core.set_defaults(dataset="ml-1m",
                          data_dir="/tmp/ncf_pipeline_test_debug")


def main(_):
  dataset = flags.FLAGS.dataset  # type: str
  data_dir = flags.FLAGS.data_dir  # type: str

  ncf_dataset = pipeline.initialize(
      dataset=dataset, data_dir=data_dir, num_neg=4, num_data_readers=4,
      debug=True)

  positives = []
  import collections

  positives_by_user = collections.defaultdict(list)
  n = ncf_dataset.train_data[movielens.USER_COLUMN].shape[0]
  for i in range(n):
    user = ncf_dataset.train_data[movielens.USER_COLUMN][i]
    item = ncf_dataset.train_data[movielens.ITEM_COLUMN][i]
    positives.append((user, item))
    positives_by_user[user].append(item)
  positive_set = set(positives)
  assert len(positives) == len(positive_set)

  dataset = pipeline.get_input_fn(
      training=True, ncf_dataset=ncf_dataset, batch_size=BATCH_SIZE,
      num_epochs=1, shuffle=True)() # type: tf.data.Dataset
  batch_tensor = dataset.make_one_shot_iterator().get_next()

  mislabels = 0
  processed_points = 0

  # The training server does not return a partial batch in order to guarantee
  # shape information of the batches.
  expected_points = int(len(positives) * (1 + ncf_dataset.num_train_neg)
                        // BATCH_SIZE * BATCH_SIZE)

  current_index = 0
  point_index = collections.defaultdict(list)
  with tf.Session().as_default() as sess:
    while True:
      try:
        batch = sess.run(batch_tensor)
      except tf.errors.OutOfRangeError:
        break

      features, labels = batch
      users = features[movielens.USER_COLUMN][:, 0]  # type: np.ndarray
      items = features[movielens.ITEM_COLUMN][:, 0]  # type: np.ndarray
      labels = labels[:, 0]  # type: np.ndarray

      assert users.shape == items.shape == labels.shape
      n = users.shape[0]
      for i in range(n):
        if bool((users[i], items[i]) in positive_set) != labels[i]:
          mislabels += 1
          print("Mislabel:", users[i], items[i], labels[i])
        point_index[users[i]].append(current_index)
        current_index += 1
      processed_points += n

  if mislabels:
    raise ValueError("{} Points were mislabeled. There is an incorrect "
                     "transformation in the pipeline.".format(mislabels))

  if expected_points != processed_points:
    raise ValueError("{} points should have been present, but pipeline "
                     "returned {}.".format(expected_points, processed_points))

  point_spreads = np.concatenate([np.array(i)[1:] - np.array(i)[:-1]
                                  for i in point_index.values()])
  print("Point index spreads:")
  print("Mean:   {:.1f}".format(np.mean(point_spreads)))
  print("Median: {:.1f}".format(np.median(point_spreads)))
  print("Std:    {:.1f}".format(np.std(point_spreads)))


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  define_flags()
  absl_app.run(main)