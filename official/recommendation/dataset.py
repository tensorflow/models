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
"""Prepare dataset for NCF.

Load the training dataset and evaluation dataset from csv file into memory.
Prepare input for model training and evaluation.
"""
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from official.recommendation import constants  # pylint: disable=g-bad-import-order

# The buffer size for shuffling train dataset.
_SHUFFLE_BUFFER_SIZE = 1024


class NCFDataSet(object):
  """A class containing data information for model training and evaluation."""

  def __init__(self, train_data, num_users, num_items, num_negatives,
               true_items, all_items, all_eval_data):
    """Initialize NCFDataset class.

    Args:
      train_data: A list containing the positive training instances.
      num_users: An integer, the number of users in training dataset.
      num_items: An integer, the number of items in training dataset.
      num_negatives: An integer, the number of negative instances for each user
        in train dataset.
      true_items: A list, the ground truth (positive) items of users for
        evaluation. Each entry is a latest positive instance for one user.
      all_items: A nested list, all items for evaluation, and each entry is the
        evaluation items for one user.
      all_eval_data: A numpy array of eval/test dataset.
    """
    self.train_data = train_data
    self.num_users = num_users
    self.num_items = num_items
    self.num_negatives = num_negatives
    self.eval_true_items = true_items
    self.eval_all_items = all_items
    self.all_eval_data = all_eval_data


def load_data(file_name):
  """Load data from a csv file which splits on tab key."""
  lines = tf.gfile.Open(file_name, "r").readlines()

  # Process the file line by line
  def _process_line(line):
    return [int(col) for col in line.split("\t")]

  data = [_process_line(line) for line in lines]
  return data


def data_preprocessing(train_fname, test_fname, test_neg_fname, num_negatives):
  """Preprocess the train and test dataset.

  In data preprocessing, the training positive instances are loaded into memory
  for random negative instance generation in each training epoch. The test
  dataset are generated from test positive and negative instances.

  Args:
    train_fname: A string, the file name of training positive dataset.
    test_fname: A string, the file name of test positive dataset. Each user has
      one positive instance.
    test_neg_fname: A string, the file name of test negative dataset. Each user
      has 100 negative instances by default.
    num_negatives: An integer, the number of negative instances for each user
      in train dataset.

  Returns:
    ncf_dataset: A NCFDataset object containing information about training and
      evaluation/test dataset.
  """
  # Load training positive instances into memory for later train data generation
  train_data = load_data(train_fname)
  # Get total number of users in the dataset
  num_users = len(np.unique(np.array(train_data)[:, 0]))

  # Process test dataset to csv file
  test_ratings = load_data(test_fname)
  test_negatives = load_data(test_neg_fname)
  # Get the total number of items in both train dataset and test dataset (the
  # whole dataset)
  num_items = len(
      set(np.array(train_data)[:, 1]) | set(np.array(test_ratings)[:, 1]))

  # Generate test instances for each user
  true_items, all_items = [], []
  all_test_data = []
  for idx in range(num_users):
    items = test_negatives[idx]
    rating = test_ratings[idx]
    user = rating[0]  # User
    true_item = rating[1]  # Positive item as ground truth

    # All items with first 100 as negative and last one positive
    items.append(true_item)
    users = np.full(len(items), user, dtype=np.int32)

    users_items = list(zip(users, items))  # User-item list
    true_items.append(true_item)  # all ground truth items
    all_items.append(items)  # All items (including positive and negative items)
    all_test_data.extend(users_items)  # Generate test dataset

  # Create NCFDataset object
  ncf_dataset = NCFDataSet(
      train_data, num_users, num_items, num_negatives, true_items, all_items,
      np.asarray(all_test_data)
  )

  return ncf_dataset


def generate_train_dataset(train_data, num_items, num_negatives):
  """Generate train dataset for each epoch.

  Given positive training instances, randomly generate negative instances to
  form the training dataset.

  Args:
    train_data: A list of positive training instances.
    num_items: An integer, the number of items in positive training instances.
    num_negatives: An integer, the number of negative training instances
      following positive training instances. It is 4 by default.

  Returns:
    A numpy array of training dataset.
  """
  all_train_data = []
  # A set with user-item tuples
  train_data_set = set((u, i) for u, i, _ in train_data)
  for u, i, _ in train_data:
    # Positive instance
    all_train_data.append([u, i, 1])
    # Negative instances, randomly generated
    for _ in xrange(num_negatives):
      j = np.random.randint(num_items)
      while (u, j) in train_data_set:
        j = np.random.randint(num_items)
      all_train_data.append([u, j, 0])

  return np.asarray(all_train_data)


def input_fn(training, batch_size, ncf_dataset, repeat=1):
  """Input function for model training and evaluation.

  The train input consists of 1 positive instance (user and item have
  interactions) followed by some number of negative instances in which the items
  are randomly chosen. The number of negative instances is "num_negatives" which
  is 4 by default. Note that for each epoch, we need to re-generate the negative
  instances. Together with positive instances, they form a new train dataset.

  Args:
    training: A boolean flag for training mode.
    batch_size: An integer, batch size for training and evaluation.
    ncf_dataset: An NCFDataSet object, which contains the information about
      training and test data.
    repeat: An integer, how many times to repeat the dataset.

  Returns:
    dataset: A tf.data.Dataset object containing examples loaded from the files.
  """
  # Generate random negative instances for training in each epoch
  if training:
    train_data = generate_train_dataset(
        ncf_dataset.train_data, ncf_dataset.num_items,
        ncf_dataset.num_negatives)
    # Get train features and labels
    train_features = [
        (constants.USER, np.expand_dims(train_data[:, 0], axis=1)),
        (constants.ITEM, np.expand_dims(train_data[:, 1], axis=1))
    ]
    train_labels = [
        (constants.RATING, np.expand_dims(train_data[:, 2], axis=1))]

    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(train_features), dict(train_labels))
    )
    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER_SIZE)
  else:
    # Create eval/test dataset
    test_user = ncf_dataset.all_eval_data[:, 0]
    test_item = ncf_dataset.all_eval_data[:, 1]
    test_features = [
        (constants.USER, np.expand_dims(test_user, axis=1)),
        (constants.ITEM, np.expand_dims(test_item, axis=1))]

    dataset = tf.data.Dataset.from_tensor_slices(dict(test_features))

  # Repeat and batch the dataset
  dataset = dataset.repeat(repeat)
  dataset = dataset.batch(batch_size)

  # Prefetch to improve speed of input pipeline.
  dataset = dataset.prefetch(1)
  return dataset
