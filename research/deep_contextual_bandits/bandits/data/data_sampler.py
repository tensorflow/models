# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Functions to create bandit problems from datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf


def one_hot(df, cols):
  """Returns one-hot encoding of DataFrame df including columns in cols."""
  for col in cols:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(col, axis=1)
  return df


def sample_mushroom_data(file_name,
                         num_contexts,
                         r_noeat=0,
                         r_eat_safe=5,
                         r_eat_poison_bad=-35,
                         r_eat_poison_good=5,
                         prob_poison_bad=0.5):
  """Samples bandit game from Mushroom UCI Dataset.

  Args:
    file_name: Route of file containing the original Mushroom UCI dataset.
    num_contexts: Number of points to sample, i.e. (context, action rewards).
    r_noeat: Reward for not eating a mushroom.
    r_eat_safe: Reward for eating a non-poisonous mushroom.
    r_eat_poison_bad: Reward for eating a poisonous mushroom if harmed.
    r_eat_poison_good: Reward for eating a poisonous mushroom if not harmed.
    prob_poison_bad: Probability of being harmed by eating a poisonous mushroom.

  Returns:
    dataset: Sampled matrix with n rows: (context, eat_reward, no_eat_reward).
    opt_vals: Vector of expected optimal (reward, action) for each context.

  We assume r_eat_safe > r_noeat, and r_eat_poison_good > r_eat_poison_bad.
  """

  # first two cols of df encode whether mushroom is edible or poisonous
  df = pd.read_csv(file_name, header=None)
  df = one_hot(df, df.columns)
  ind = np.random.choice(range(df.shape[0]), num_contexts, replace=True)

  contexts = df.iloc[ind, 2:]
  no_eat_reward = r_noeat * np.ones((num_contexts, 1))
  random_poison = np.random.choice(
      [r_eat_poison_bad, r_eat_poison_good],
      p=[prob_poison_bad, 1 - prob_poison_bad],
      size=num_contexts)
  eat_reward = r_eat_safe * df.iloc[ind, 0]
  eat_reward += np.multiply(random_poison, df.iloc[ind, 1])
  eat_reward = eat_reward.reshape((num_contexts, 1))

  # compute optimal expected reward and optimal actions
  exp_eat_poison_reward = r_eat_poison_bad * prob_poison_bad
  exp_eat_poison_reward += r_eat_poison_good * (1 - prob_poison_bad)
  opt_exp_reward = r_eat_safe * df.iloc[ind, 0] + max(
      r_noeat, exp_eat_poison_reward) * df.iloc[ind, 1]

  if r_noeat > exp_eat_poison_reward:
    # actions: no eat = 0 ; eat = 1
    opt_actions = df.iloc[ind, 0]  # indicator of edible
  else:
    # should always eat (higher expected reward)
    opt_actions = np.ones((num_contexts, 1))

  opt_vals = (opt_exp_reward.values, opt_actions.values)

  return np.hstack((contexts, no_eat_reward, eat_reward)), opt_vals


def sample_stock_data(file_name, context_dim, num_actions, num_contexts,
                      sigma, shuffle_rows=True):
  """Samples linear bandit game from stock prices dataset.

  Args:
    file_name: Route of file containing the stock prices dataset.
    context_dim: Context dimension (i.e. vector with the price of each stock).
    num_actions: Number of actions (different linear portfolio strategies).
    num_contexts: Number of contexts to sample.
    sigma: Vector with additive noise levels for each action.
    shuffle_rows: If True, rows from original dataset are shuffled.

  Returns:
    dataset: Sampled matrix with rows: (context, reward_1, ..., reward_k).
    opt_vals: Vector of expected optimal (reward, action) for each context.
  """

  with tf.gfile.Open(file_name, 'r') as f:
    contexts = np.loadtxt(f, skiprows=1)

  if shuffle_rows:
    np.random.shuffle(contexts)
  contexts = contexts[:num_contexts, :]

  betas = np.random.uniform(-1, 1, (context_dim, num_actions))
  betas /= np.linalg.norm(betas, axis=0)

  mean_rewards = np.dot(contexts, betas)
  noise = np.random.normal(scale=sigma, size=mean_rewards.shape)
  rewards = mean_rewards + noise

  opt_actions = np.argmax(mean_rewards, axis=1)
  opt_rewards = [mean_rewards[i, a] for i, a in enumerate(opt_actions)]
  return np.hstack((contexts, rewards)), (np.array(opt_rewards), opt_actions)


def sample_jester_data(file_name, context_dim, num_actions, num_contexts,
                       shuffle_rows=True, shuffle_cols=False):
  """Samples bandit game from (user, joke) dense subset of Jester dataset.

  Args:
    file_name: Route of file containing the modified Jester dataset.
    context_dim: Context dimension (i.e. vector with some ratings from a user).
    num_actions: Number of actions (number of joke ratings to predict).
    num_contexts: Number of contexts to sample.
    shuffle_rows: If True, rows from original dataset are shuffled.
    shuffle_cols: Whether or not context/action jokes are randomly shuffled.

  Returns:
    dataset: Sampled matrix with rows: (context, rating_1, ..., rating_k).
    opt_vals: Vector of deterministic optimal (reward, action) for each context.
  """

  with tf.gfile.Open(file_name, 'rb') as f:
    dataset = np.load(f)

  if shuffle_cols:
    dataset = dataset[:, np.random.permutation(dataset.shape[1])]
  if shuffle_rows:
    np.random.shuffle(dataset)
  dataset = dataset[:num_contexts, :]

  assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'

  opt_actions = np.argmax(dataset[:, context_dim:], axis=1)
  opt_rewards = np.array([dataset[i, context_dim + a]
                          for i, a in enumerate(opt_actions)])

  return dataset, (opt_rewards, opt_actions)


def sample_statlog_data(file_name, num_contexts, shuffle_rows=True,
                        remove_underrepresented=False):
  """Returns bandit problem dataset based on the UCI statlog data.

  Args:
    file_name: Route of file containing the Statlog dataset.
    num_contexts: Number of contexts to sample.
    shuffle_rows: If True, rows from original dataset are shuffled.
    remove_underrepresented: If True, removes arms with very few rewards.

  Returns:
    dataset: Sampled matrix with rows: (context, action rewards).
    opt_vals: Vector of deterministic optimal (reward, action) for each context.

  https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)
  """

  with tf.gfile.Open(file_name, 'r') as f:
    data = np.loadtxt(f)

  num_actions = 7  # some of the actions are very rarely optimal.

  # Shuffle data
  if shuffle_rows:
    np.random.shuffle(data)
  data = data[:num_contexts, :]

  # Last column is label, rest are features
  contexts = data[:, :-1]
  labels = data[:, -1].astype(int) - 1  # convert to 0 based index

  if remove_underrepresented:
    contexts, labels = remove_underrepresented_classes(contexts, labels)

  return classification_to_bandit_problem(contexts, labels, num_actions)


def sample_adult_data(file_name, num_contexts, shuffle_rows=True,
                      remove_underrepresented=False):
  """Returns bandit problem dataset based on the UCI adult data.

  Args:
    file_name: Route of file containing the Adult dataset.
    num_contexts: Number of contexts to sample.
    shuffle_rows: If True, rows from original dataset are shuffled.
    remove_underrepresented: If True, removes arms with very few rewards.

  Returns:
    dataset: Sampled matrix with rows: (context, action rewards).
    opt_vals: Vector of deterministic optimal (reward, action) for each context.

  Preprocessing:
    * drop rows with missing values
    * convert categorical variables to 1 hot encoding

  https://archive.ics.uci.edu/ml/datasets/census+income
  """
  with tf.gfile.Open(file_name, 'r') as f:
    df = pd.read_csv(f, header=None,
                     na_values=[' ?']).dropna()

  num_actions = 14

  if shuffle_rows:
    df = df.sample(frac=1)
  df = df.iloc[:num_contexts, :]

  labels = df[6].astype('category').cat.codes.as_matrix()
  df = df.drop([6], axis=1)

  # Convert categorical variables to 1 hot encoding
  cols_to_transform = [1, 3, 5, 7, 8, 9, 13, 14]
  df = pd.get_dummies(df, columns=cols_to_transform)

  if remove_underrepresented:
    df, labels = remove_underrepresented_classes(df, labels)
  contexts = df.as_matrix()

  return classification_to_bandit_problem(contexts, labels, num_actions)


def sample_census_data(file_name, num_contexts, shuffle_rows=True,
                       remove_underrepresented=False):
  """Returns bandit problem dataset based on the UCI census data.

  Args:
    file_name: Route of file containing the Census dataset.
    num_contexts: Number of contexts to sample.
    shuffle_rows: If True, rows from original dataset are shuffled.
    remove_underrepresented: If True, removes arms with very few rewards.

  Returns:
    dataset: Sampled matrix with rows: (context, action rewards).
    opt_vals: Vector of deterministic optimal (reward, action) for each context.

  Preprocessing:
    * drop rows with missing labels
    * convert categorical variables to 1 hot encoding

  Note: this is the processed (not the 'raw') dataset. It contains a subset
  of the raw features and they've all been discretized.

  https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29
  """
  # Note: this dataset is quite large. It will be slow to load and preprocess.
  with tf.gfile.Open(file_name, 'r') as f:
    df = (pd.read_csv(f, header=0, na_values=['?'])
          .dropna())

  num_actions = 9

  if shuffle_rows:
    df = df.sample(frac=1)
  df = df.iloc[:num_contexts, :]

  # Assuming what the paper calls response variable is the label?
  labels = df['dOccup'].astype('category').cat.codes.as_matrix()
  # In addition to label, also drop the (unique?) key.
  df = df.drop(['dOccup', 'caseid'], axis=1)

  # All columns are categorical. Convert to 1 hot encoding.
  df = pd.get_dummies(df, columns=df.columns)

  if remove_underrepresented:
    df, labels = remove_underrepresented_classes(df, labels)
  contexts = df.as_matrix()

  return classification_to_bandit_problem(contexts, labels, num_actions)


def sample_covertype_data(file_name, num_contexts, shuffle_rows=True,
                          remove_underrepresented=False):
  """Returns bandit problem dataset based on the UCI Cover_Type data.

  Args:
    file_name: Route of file containing the Covertype dataset.
    num_contexts: Number of contexts to sample.
    shuffle_rows: If True, rows from original dataset are shuffled.
    remove_underrepresented: If True, removes arms with very few rewards.

  Returns:
    dataset: Sampled matrix with rows: (context, action rewards).
    opt_vals: Vector of deterministic optimal (reward, action) for each context.

  Preprocessing:
    * drop rows with missing labels
    * convert categorical variables to 1 hot encoding

  https://archive.ics.uci.edu/ml/datasets/Covertype
  """
  with tf.gfile.Open(file_name, 'r') as f:
    df = (pd.read_csv(f, header=0, na_values=['?'])
          .dropna())

  num_actions = 7

  if shuffle_rows:
    df = df.sample(frac=1)
  df = df.iloc[:num_contexts, :]

  # Assuming what the paper calls response variable is the label?
  # Last column is label.
  labels = df[df.columns[-1]].astype('category').cat.codes.as_matrix()
  df = df.drop([df.columns[-1]], axis=1)

  # All columns are either quantitative or already converted to 1 hot.
  if remove_underrepresented:
    df, labels = remove_underrepresented_classes(df, labels)
  contexts = df.as_matrix()

  return classification_to_bandit_problem(contexts, labels, num_actions)


def classification_to_bandit_problem(contexts, labels, num_actions=None):
  """Normalize contexts and encode deterministic rewards."""

  if num_actions is None:
    num_actions = np.max(labels) + 1
  num_contexts = contexts.shape[0]

  # Due to random subsampling in small problems, some features may be constant
  sstd = safe_std(np.std(contexts, axis=0, keepdims=True)[0, :])

  # Normalize features
  contexts = ((contexts - np.mean(contexts, axis=0, keepdims=True)) / sstd)

  # One hot encode labels as rewards
  rewards = np.zeros((num_contexts, num_actions))
  rewards[np.arange(num_contexts), labels] = 1.0

  return contexts, rewards, (np.ones(num_contexts), labels)


def safe_std(values):
  """Remove zero std values for ones."""
  return np.array([val if val != 0.0 else 1.0 for val in values])


def remove_underrepresented_classes(features, labels, thresh=0.0005):
  """Removes classes when number of datapoints fraction is below a threshold."""

  # Threshold doesn't seem to agree with https://arxiv.org/pdf/1706.04687.pdf
  # Example: for Covertype, they report 4 classes after filtering, we get 7?
  total_count = labels.shape[0]
  unique, counts = np.unique(labels, return_counts=True)
  ratios = counts.astype('float') / total_count
  vals_and_ratios = dict(zip(unique, ratios))
  print('Unique classes and their ratio of total: %s' % vals_and_ratios)
  keep = [vals_and_ratios[v] >= thresh for v in labels]
  return features[keep], labels[np.array(keep)]
