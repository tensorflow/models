# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Sparse categorical cross-entropy losses."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import tensorflow as tf


def _adjust_labels(labels, predictions):
  """Adjust the 'labels' tensor by squeezing it if needed."""
  labels = tf.cast(labels, tf.int32)
  if len(predictions.shape) == len(labels.shape):
    labels = tf.squeeze(labels, [-1])
  return labels, predictions


def _validate_rank(labels, predictions, weights):
  if weights is not None and len(weights.shape) != len(labels.shape):
    raise RuntimeError(
        ("Weight and label tensors were not of the same rank. weights.shape "
         "was %s, and labels.shape was %s.") %
        (predictions.shape, labels.shape))
  if (len(predictions.shape) - 1) != len(labels.shape):
    raise RuntimeError(
        ("Weighted sparse categorical crossentropy expects `labels` to have a "
         "rank of one less than `predictions`. labels.shape was %s, and "
         "predictions.shape was %s.") % (labels.shape, predictions.shape))


def per_example_loss(labels, predictions, weights=None):
  """Calculate a per-example sparse categorical crossentropy loss.

  This loss function assumes that the predictions are post-softmax.
  Args:
    labels: The labels to evaluate against. Should be a set of integer indices
      ranging from 0 to (vocab_size-1).
    predictions: The network predictions. Should have softmax already applied.
    weights: An optional weight array of the same shape as the 'labels' array.
      If None, all examples will be used.

  Returns:
    A tensor of shape predictions.shape[:-1] containing the per-example
      loss.
  """
  # When using these functions with the Keras core API, we will need to squeeze
  # the labels tensor - Keras adds a spurious inner dimension.
  labels, predictions = _adjust_labels(labels, predictions)
  _validate_rank(labels, predictions, weights)

  labels_one_hot = tf.one_hot(labels, predictions.shape[-1])
  labels_one_hot = tf.cast(labels_one_hot, predictions.dtype)
  per_example_loss_data = -tf.reduce_sum(
      predictions * labels_one_hot, axis=[-1])
  if weights is not None:
    weights = tf.cast(weights, per_example_loss_data.dtype)
    per_example_loss_data = weights * per_example_loss_data
  return per_example_loss_data


def loss(labels, predictions, weights=None):
  """Calculate a per-batch sparse categorical crossentropy loss.

  This loss function assumes that the predictions are post-softmax.
  Args:
    labels: The labels to evaluate against. Should be a set of integer indices
      ranging from 0 to (vocab_size-1).
    predictions: The network predictions. Should have softmax already applied.
    weights: An optional weight array of the same shape as the 'labels' array.
      If None, all examples will be used.

  Returns:
    A loss scalar.

  Raises:
    RuntimeError if the passed tensors do not have the same rank.
  """
  # When using these functions with the Keras core API, we will need to squeeze
  # the labels tensor - Keras adds a spurious inner dimension.
  labels, predictions = _adjust_labels(labels, predictions)
  _validate_rank(labels, predictions, weights)

  per_example_loss_data = per_example_loss(labels, predictions, weights)

  if weights is None:
    return tf.reduce_mean(per_example_loss_data)
  else:
    numerator = tf.reduce_sum(per_example_loss_data)
    weights = tf.cast(weights, predictions.dtype)
    denominator = tf.reduce_sum(weights) + 1e-5
    return numerator / denominator
