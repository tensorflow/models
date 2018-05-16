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
import scipy.sparse as sp
from six.moves import xrange  # pylint: disable=redefined-builtin


def load_train_matrix(train_fname):
  """Load the train matrix from a given file.

  Args:
    train_fname: The csv file name of training dataset.

  Returns:
    mat: The sparse matrix of training data.
  """
  def process_line(line):
    tmp = line.split('\t')
    # user, item, rating
    return [int(tmp[0]), int(tmp[1]), float(tmp[2]) > 0]

  lines = open(train_fname, 'r').readlines()[1:]
  data = list(map(process_line, lines))
  nb_users = max(data, key=lambda x: x[0])[0] + 1
  nb_items = max(data, key=lambda x: x[1])[1] + 1

  # Generate training matrix
  # data = list(filter(lambda x: x[2], data))
  data = list(x for x in data if x[2])
  mat = sp.dok_matrix(
      (nb_users, nb_items), dtype=np.float32)
  for user, item, _ in data:
    mat[user, item] = 1.
  return mat


def load_test_ratings(test_fname):
  """Load the test rating data from a given file.

  Args:
    test_fname: The csv file name of test rating dataset.

  Returns:
    ratings: The list of test ratings data.
  """
  def process_line(line):
    tmp = map(int, line.split('\t')[0:2])
    return list(tmp)
  lines = open(test_fname, 'r').readlines()
  ratings = map(process_line, lines)
  return list(ratings)


def load_test_negs(test_neg_fname):
  """Load the test negative data from a given file.

  Args:
    test_neg_fname: The csv file name of test negative dataset.

  Returns:
    negs: The list of test negatives data.
  """
  def process_line(line):
    tmp = map(int, line.split('\t')[1:])
    return list(tmp)
  lines = open(test_neg_fname, 'r').readlines()
  negs = map(process_line, lines)
  return list(negs)


def get_train_input(train_matrix, num_negatives):
  """Prepare input data for model training.

  The train input consists of 1 positive instance (user and item have
  interactions) followed by some number of negative instances in which the items
  are randomly chosen. The number of negative instances is 'num_negatives'.

  Args:
    train_matrix: The training matrix dataset.
    num_negatives: The number of negatives for a user. It is 4 by default.

  Returns:
    train_user_input: The user input data for training.
    train_item_input: The item input data for training.
    train_labels: The interaction labels, either 0 or 1.
  """
  user_input, item_input, labels = [], [], []
  _, num_items = train_matrix.shape

  for (u, i) in train_matrix.keys():
    # Positive instance
    user_input.append(u)
    item_input.append(i)
    labels.append(1)
    # Negative instances
    for _ in xrange(num_negatives):
      j = np.random.randint(num_items)
      while (u, j) in train_matrix:
        j = np.random.randint(num_items)
      user_input.append(u)
      item_input.append(j)
      labels.append(0)

  # Generate training instances
  input_dim = np.array(user_input).shape[0]
  train_user_input = np.array(user_input).reshape(input_dim, 1)
  train_item_input = np.array(item_input).reshape(input_dim, 1)
  train_labels = np.array(labels).reshape(input_dim, 1)

  return train_user_input, train_item_input, train_labels


def get_eval_input(test_ratings, test_negatives):
  """Prepare input data for model evaluation.

  The evaluation input consists of 100 test negative instances followed by
  1 positive instance (which is the latest interaction for each user).

  Args:
    test_ratings: The latest positive interaction left out for evaluation.
    test_negatives: Negative test data for model evaluation.

  Returns:
    eval_user_input: The user input data for evaluation.
    eval_item_input: The item input data for evaluation.
    gt_items: The ground truth item for each user to be used in evaluation.
  """
  all_items, all_users = [], []
  gt_items = []

  # 1 is for the positive item
  single_user_input_dim = len(test_negatives[0]) + 1
  num_users = len(test_ratings)
  input_dim = single_user_input_dim * num_users

  for idx in range(num_users):
    items = test_negatives[idx]
    rating = test_ratings[idx]
    user = rating[0]  # User
    gt_item = rating[1]  # Positive item as ground truth

    # All items with first 100 as negative and last one positive
    items.append(gt_item)
    users = np.full(len(items), user, dtype='int32')

    all_items.append(items)
    all_users.append(users)
    gt_items.append(gt_item)

  eval_user_input = np.array(all_users).reshape(input_dim, 1)
  eval_item_input = np.array(all_items).reshape(input_dim, 1)

  return eval_user_input, eval_item_input, gt_items
