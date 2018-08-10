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

"""Operations for feeding input data using TensorFlow placeholders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def prepare_feed_dict(model, features, labels=None, is_training=None):
  """Prepares a feed_dict for sess.run() given a batch of features and labels.

  Args:
    model: An instance of AstroModel.
    features: Dictionary containing "time_series_features" and "aux_features".
        Each is a dictionary of named numpy arrays of shape
        [batch_size, length].
    labels: (Optional). Numpy array of shape [batch_size].
    is_training: (Optional). Python boolean to feed to the model.is_training
        Tensor (if None, no value is fed).

  Returns:
    feed_dict: A dictionary of input Tensor to numpy array.
  """
  feed_dict = {}
  for feature, tensor in model.time_series_features.items():
    feed_dict[tensor] = features["time_series_features"][feature]
  for feature, tensor in model.aux_features.items():
    feed_dict[tensor] = features["aux_features"][feature]

  if labels is not None:
    feed_dict[model.labels] = labels

  if is_training is not None:
    feed_dict[model.is_training] = is_training

  return feed_dict


def build_feature_placeholders(config):
  """Builds tf.Placeholder ops for feeding model features and labels.

  Args:
    config: ConfigDict containing the feature configurations.

  Returns:
    features: A dictionary containing "time_series_features" and "aux_features",
        each of which is a dictionary of tf.Placeholders of features from the
        input configuration. All features have dtype float32 and shape
        [batch_size, length].
  """
  batch_size = None  # Batch size will be dynamically specified.
  features = {"time_series_features": {}, "aux_features": {}}
  for feature_name, feature_spec in config.items():
    placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=[batch_size, feature_spec.length],
        name=feature_name)

    if feature_spec.is_time_series:
      features["time_series_features"][feature_name] = placeholder
    else:
      features["aux_features"][feature_name] = placeholder

  return features


def build_labels_placeholder():
  """Builds a tf.Placeholder op for feeding model labels.

  Returns:
    labels: An int64 tf.Placeholder with shape [batch_size].
  """
  batch_size = None  # Batch size will be dynamically specified.
  return tf.placeholder(dtype=tf.int64, shape=[batch_size], name="labels")
