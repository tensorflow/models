# Copyright 2018 The TensorFlow Authors.
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

"""TensorFlow utilities for unit tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def get_variable_by_name(name, scope=""):
  """Gets a tf.Variable by name.

  Args:
    name: Name of the Variable within the specified scope.
    scope: Variable scope; use the empty string for top-level scope.

  Returns:
    The matching tf.Variable object.
  """
  with tf.variable_scope(scope, reuse=True):
    return tf.get_variable(name)


def fake_features(feature_spec, batch_size):
  """Creates random numpy arrays representing input features for unit testing.

  Args:
    feature_spec: Dictionary containing the feature specifications.
    batch_size: Integer batch size.

  Returns:
    Dictionary containing "time_series_features" and "aux_features". Each is a
        dictionary of named numpy arrays of shape [batch_size, length].
  """
  features = {}
  features["time_series_features"] = {
      name: np.random.random([batch_size, spec["length"]])
      for name, spec in feature_spec.items() if spec["is_time_series"]
  }
  features["aux_features"] = {
      name: np.random.random([batch_size, spec["length"]])
      for name, spec in feature_spec.items() if not spec["is_time_series"]
  }
  return features


def fake_labels(output_dim, batch_size):
  """Creates a radom numpy array representing labels for unit testing.

  Args:
    output_dim: Number of output units in the classification model.
    batch_size: Integer batch size.

  Returns:
    Numpy array of shape [batch_size].
  """
  # Binary classification is denoted by output_dim == 1. In that case there are
  # 2 label classes even though there is only 1 output prediction by the model.
  # Otherwise, the classification task is multi-labeled with output_dim classes.
  num_labels = 2 if output_dim == 1 else output_dim
  return np.random.randint(num_labels, size=batch_size)
