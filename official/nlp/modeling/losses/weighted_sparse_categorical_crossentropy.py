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
"""Weighted sparse categorical cross-entropy losses."""

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


def loss(labels, predictions, weights=None, from_logits=False):
  """Calculate a per-batch sparse categorical crossentropy loss.

  This loss function assumes that the predictions are post-softmax.
  Args:
    labels: The labels to evaluate against. Should be a set of integer indices
      ranging from 0 to (vocab_size-1).
    predictions: The network predictions. Should have softmax already applied.
    weights: An optional weight array of the same shape as the 'labels' array.
      If None, all examples will be used.
    from_logits: Whether the input predictions are logits.

  Returns:
    A loss scalar.

  Raises:
    RuntimeError if the passed tensors do not have the same rank.
  """
  # When using these functions with the Keras core API, we will need to squeeze
  # the labels tensor - Keras adds a spurious inner dimension.
  labels, predictions = _adjust_labels(labels, predictions)
  _validate_rank(labels, predictions, weights)

  example_losses = tf.keras.losses.sparse_categorical_crossentropy(
      labels, predictions, from_logits=from_logits)

  if weights is None:
    return tf.reduce_mean(example_losses)
  weights = tf.cast(weights, predictions.dtype)
  return tf.math.divide_no_nan(
      tf.reduce_sum(example_losses * weights), tf.reduce_sum(weights))
