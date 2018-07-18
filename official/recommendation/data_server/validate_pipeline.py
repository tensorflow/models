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


def define_flags():
  flags.DEFINE_enum(
      name="dataset", default=movielens.ML_1M,
      enum_values=movielens.DATASETS, case_sensitive=False,
      help=flags_core.help_wrap("Dataset to be trained and evaluated."))


def main(_):
  dataset = flags.FLAGS.dataset  # type: str
  data_dir = "/tmp/ncf_pipeline_test_debug"
  # data_dir = tempfile.mkdtemp(prefix="ncf_pipeline_test")

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

  dataset = pipeline.get_input_fn(training=True, ncf_dataset=ncf_dataset,
                                  batch_size=16384, num_epochs=1, shuffle=True)() # type: tf.data.Dataset
  batch_tensor = dataset.make_one_shot_iterator().get_next()

  print(n)
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
      mislabels = 0
      for i in range(n):
        if bool((users[i], items[i]) in positive_set) != labels[i]:
          mislabels += 1
          print("Mislabel:", users[i], items[i], labels[i])

      if mislabels:
        print(mislabels, n)


  # try:
  #   print(data_dir)
  # finally:
  #   tf.gfile.DeleteRecursively(data_dir)



if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  define_flags()
  absl_app.run(main)