# Copyright 2017 Google, Inc. All Rights Reserved.
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

"""Functions to generate or load datasets for supervised learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import numpy as np
from sklearn.datasets import make_classification

MAX_SEED = 4294967295


class Dataset(namedtuple("Dataset", "data labels")):
  """Helper class for managing a supervised learning dataset.

  Args:
    data: an array of type float32 with N samples, each of which is the set
      of features for that sample. (Shape (N, D_i), where N is the number of
      samples and D_i is the number of features for that sample.)
    labels: an array of type int32 or int64 with N elements, indicating the
      class label for the corresponding set of features in data.
  """
  # Since this is an immutable object, we don't need to reserve slots.
  __slots__ = ()

  @property
  def size(self):
    """Dataset size (number of samples)."""
    return len(self.data)

  def batch_indices(self, num_batches, batch_size):
    """Creates indices of shuffled minibatches.

    Args:
      num_batches: the number of batches to generate
      batch_size: the size of each batch

    Returns:
      batch_indices: a list of minibatch indices, arranged so that the dataset
          is randomly shuffled.

    Raises:
      ValueError: if the data and labels have different lengths
    """
    if len(self.data) != len(self.labels):
      raise ValueError("Labels and data must have the same number of samples.")

    batch_indices = []

    # Follows logic in mnist.py to ensure we cover the entire dataset.
    index_in_epoch = 0
    dataset_size = len(self.data)
    dataset_indices = np.arange(dataset_size)
    np.random.shuffle(dataset_indices)

    for _ in range(num_batches):
      start = index_in_epoch
      index_in_epoch += batch_size
      if index_in_epoch > dataset_size:

        # Finished epoch, reshuffle.
        np.random.shuffle(dataset_indices)

        # Start next epoch.
        start = 0
        index_in_epoch = batch_size

      end = index_in_epoch
      batch_indices.append(dataset_indices[start:end].tolist())

    return batch_indices


def noisy_parity_class(n_samples,
                       n_classes=2,
                       n_context_ids=5,
                       noise_prob=0.25,
                       random_seed=None):
  """Returns a randomly generated sparse-to-sparse dataset.

  The label is a parity class of a set of context classes.

  Args:
    n_samples: number of samples (data points)
    n_classes: number of class labels (default: 2)
    n_context_ids: how many classes to take the parity of (default: 5).
    noise_prob: how often to corrupt the label (default: 0.25)
    random_seed: seed used for drawing the random data (default: None)
  Returns:
    dataset: A Dataset namedtuple containing the generated data and labels
  """
  np.random.seed(random_seed)
  x = np.random.randint(0, n_classes, [n_samples, n_context_ids])
  noise = np.random.binomial(1, noise_prob, [n_samples])
  y = (np.sum(x, 1) + noise) % n_classes
  return Dataset(x.astype("float32"), y.astype("int32"))


def random(n_features, n_samples, n_classes=2, sep=1.0, random_seed=None):
  """Returns a randomly generated classification dataset.

  Args:
    n_features: number of features (dependent variables)
    n_samples: number of samples (data points)
    n_classes: number of class labels (default: 2)
    sep: separation of the two classes, a higher value corresponds to
      an easier classification problem (default: 1.0)
    random_seed: seed used for drawing the random data (default: None)

  Returns:
    dataset: A Dataset namedtuple containing the generated data and labels
  """
  # Generate the problem data.
  x, y = make_classification(n_samples=n_samples,
                             n_features=n_features,
                             n_informative=n_features,
                             n_redundant=0,
                             n_classes=n_classes,
                             class_sep=sep,
                             random_state=random_seed)

  return Dataset(x.astype("float32"), y.astype("int32"))


def random_binary(n_features, n_samples, random_seed=None):
  """Returns a randomly generated dataset of binary values.

  Args:
    n_features: number of features (dependent variables)
    n_samples: number of samples (data points)
    random_seed: seed used for drawing the random data (default: None)

  Returns:
    dataset: A Dataset namedtuple containing the generated data and labels
  """
  random_seed = (np.random.randint(MAX_SEED) if random_seed is None
                 else random_seed)
  np.random.seed(random_seed)

  x = np.random.randint(2, size=(n_samples, n_features))
  y = np.zeros((n_samples, 1))

  return Dataset(x.astype("float32"), y.astype("int32"))


def random_symmetric(n_features, n_samples, random_seed=None):
  """Returns a randomly generated dataset of values and their negatives.

  Args:
    n_features: number of features (dependent variables)
    n_samples: number of samples (data points)
    random_seed: seed used for drawing the random data (default: None)

  Returns:
    dataset: A Dataset namedtuple containing the generated data and labels
  """
  random_seed = (np.random.randint(MAX_SEED) if random_seed is None
                 else random_seed)
  np.random.seed(random_seed)

  x1 = np.random.normal(size=(int(n_samples/2), n_features))
  x = np.concatenate((x1, -x1), axis=0)
  y = np.zeros((n_samples, 1))

  return Dataset(x.astype("float32"), y.astype("int32"))


def random_mlp(n_features, n_samples, random_seed=None, n_layers=6, width=20):
  """Returns a generated output of an MLP with random weights.

  Args:
    n_features: number of features (dependent variables)
    n_samples: number of samples (data points)
    random_seed: seed used for drawing the random data (default: None)
    n_layers: number of layers in random MLP
    width: width of the layers in random MLP

  Returns:
    dataset: A Dataset namedtuple containing the generated data and labels
  """
  random_seed = (np.random.randint(MAX_SEED) if random_seed is None
                 else random_seed)
  np.random.seed(random_seed)

  x = np.random.normal(size=(n_samples, n_features))
  y = x
  n_in = n_features
  scale_factor = np.sqrt(2.) / np.sqrt(n_features)
  for _ in range(n_layers):
    weights = np.random.normal(size=(n_in, width)) * scale_factor
    y = np.dot(y, weights).clip(min=0)
    n_in = width

  y = y[:, 0]
  y[y > 0] = 1

  return Dataset(x.astype("float32"), y.astype("int32"))


EMPTY_DATASET = Dataset(np.array([], dtype="float32"),
                        np.array([], dtype="int32"))
