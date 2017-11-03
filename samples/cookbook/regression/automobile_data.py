# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions for loading the automobile data set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import pandas as pd
import tensorflow as tf

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

# Order is important for the csv-readers, so we use an OrderedDict here.
COLUMN_TYPES = collections.OrderedDict([
    ("symboling", int),
    ("normalized-losses", float),
    ("make", str),
    ("fuel-type", str),
    ("aspiration", str),
    ("num-of-doors", str),
    ("body-style", str),
    ("drive-wheels", str),
    ("engine-location", str),
    ("wheel-base", float),
    ("length", float),
    ("width", float),
    ("height", float),
    ("curb-weight", float),
    ("engine-type", str),
    ("num-of-cylinders", str),
    ("engine-size", float),
    ("fuel-system", str),
    ("bore", float),
    ("stroke", float),
    ("compression-ratio", float),
    ("horsepower", float),
    ("peak-rpm", float),
    ("city-mpg", float),
    ("highway-mpg", float),
    ("price", float)
])


def raw_dataframe():
  """Load the automobile data set as a pd.DataFrame."""
  # Download and cache the data
  path = tf.keras.utils.get_file(URL.split("/")[-1], URL)

  # Load it into a pandas DataFrame
  df = pd.read_csv(path, names=COLUMN_TYPES.keys(),
                   dtype=COLUMN_TYPES, na_values="?")

  return df


def load_data(y_name="price", train_fraction=0.7, seed=None):
  """Load the automobile data set and split it train/test and features/label.

  A description of the data is available at:
    https://archive.ics.uci.edu/ml/datasets/automobile

  The data itself can be found at:
    https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data

  Args:
    y_name: the column to return as the label.
    train_fraction: the fraction of the data set to use for training.
    seed: The random seed to use when shuffling the data. `None` generates a
      unique shuffle every run.
  Returns:
    a pair of pairs where the first pair is the training data, and the second
    is the test data:
    `(x_train, y_train), (x_test, y_test) = load_data(...)`
    `x` contains a pandas DataFrame of features, while `y` contains the label
    array.
  """
  # Load the raw data columns.
  data = raw_dataframe()

  # Delete rows with unknowns
  data = data.dropna()

  # Shuffle the data
  np.random.seed(seed)

  # Split the data into train/test subsets.
  x_train = data.sample(frac=train_fraction, random_state=seed)
  x_test = data.drop(x_train.index)

  # Extract the label from the features DataFrame.
  y_train = x_train.pop(y_name)
  y_test = x_test.pop(y_name)

  return (x_train, y_train), (x_test, y_test)

def make_dataset(x, y=None):
    """Create a slice Dataset from a pandas DataFrame and labels"""
    # TODO(markdaooust): simplify this after the 1.4 cut.
    # Convert the DataFrame to a dict
    x = dict(x)

    # Convert the pd.Series to np.arrays
    for key in x:
        x[key] = np.array(x[key])

    items = [x]
    if y is not None:
        items.append(np.array(y, dtype=np.float32))

    # Create a Dataset of slices
    return tf.data.Dataset.from_tensor_slices(tuple(items))
