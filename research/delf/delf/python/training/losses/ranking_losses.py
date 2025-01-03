# Copyright 2021 The TensorFlow Authors All Rights Reserved.
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
"""Ranking loss definitions."""

import tensorflow as tf


class ContrastiveLoss(tf.keras.losses.Loss):
  """Contrastive Loss layer.

  Contrastive Loss layer allows to compute contrastive loss for a batch of
  images. Implementation based on: https://arxiv.org/abs/1604.02426.
  """

  def __init__(self, margin=0.7, reduction=tf.keras.losses.Reduction.NONE):
    """Initialization of Contrastive Loss layer.

    Args:
      margin: Float contrastive loss margin.
      reduction: Type of loss reduction.
    """
    super(ContrastiveLoss, self).__init__(reduction)
    self.margin = margin
    # Parameter for numerical stability.
    self.eps = 1e-6

  def __call__(self, queries, positives, negatives):
    """Invokes the Contrastive Loss instance.

    Args:
      queries: [batch_size, dim] Anchor input tensor.
      positives: [batch_size, dim] Positive sample input tensor.
      negatives: [batch_size, num_neg, dim] Negative sample input tensor.

    Returns:
      loss: Scalar tensor.
    """
    return contrastive_loss(
        queries, positives, negatives, margin=self.margin, eps=self.eps)


class TripletLoss(tf.keras.losses.Loss):
  """Triplet Loss layer.

  Triplet Loss layer computes triplet loss for a batch of images.  Triplet
  loss tries to keep all queries closer to positives than to any negatives.
  Margin is used to specify when a triplet has become too "easy" and we no
  longer want to adjust the weights from it. Differently from the Contrastive
  Loss, Triplet Loss uses squared distances when computing the loss.
  Implementation based on: https://arxiv.org/abs/1511.07247.
  """

  def __init__(self, margin=0.1, reduction=tf.keras.losses.Reduction.NONE):
    """Initialization of Triplet Loss layer.

    Args:
      margin: Triplet loss margin.
      reduction: Type of loss reduction.
    """
    super(TripletLoss, self).__init__(reduction)
    self.margin = margin

  def __call__(self, queries, positives, negatives):
    """Invokes the Triplet Loss instance.

    Args:
      queries: [batch_size, dim] Anchor input tensor.
      positives: [batch_size, dim] Positive sample input tensor.
      negatives: [batch_size, num_neg, dim] Negative sample input tensor.

    Returns:
      loss: Scalar tensor.
    """
    return triplet_loss(queries, positives, negatives, margin=self.margin)


def contrastive_loss(queries, positives, negatives, margin=0.7, eps=1e-6):
  """Calculates Contrastive Loss.

  We expect the `queries`, `positives` and `negatives` to be normalized with
  unit length for training stability. The contrastive loss directly
  optimizes this distance by encouraging all positive distances to
  approach 0, while keeping negative distances above a certain threshold.

  Args:
    queries: [batch_size, dim] Anchor input tensor.
    positives: [batch_size, dim] Positive sample input tensor.
    negatives: [batch_size, num_neg, dim] Negative sample input tensor.
    margin: Float contrastive loss loss margin.
    eps: Float parameter for numerical stability.

  Returns:
    loss: Scalar tensor.
  """
  dim = tf.shape(queries)[1]
  # Number of `queries`.
  batch_size = tf.shape(queries)[0]
  # Number of `positives`.
  np = tf.shape(positives)[0]
  # Number of `negatives`.
  num_neg = tf.shape(negatives)[1]

  # Preparing negatives.
  stacked_negatives = tf.reshape(negatives, [num_neg * batch_size, dim])

  # Preparing queries for further loss calculation.
  stacked_queries = tf.repeat(queries, num_neg + 1, axis=0)
  positives_and_negatives = tf.concat([positives, stacked_negatives], axis=0)

  # Calculate an Euclidean norm for each pair of points. For any positive
  # pair of data points this distance should be small, and for
  # negative pair it should be large.
  distances = tf.norm(stacked_queries - positives_and_negatives + eps, axis=1)

  positives_part = 0.5 * tf.pow(distances[:np], 2.0)
  negatives_part = 0.5 * tf.pow(
      tf.math.maximum(margin - distances[np:], 0), 2.0)

  # Final contrastive loss calculation.
  loss = tf.reduce_sum(tf.concat([positives_part, negatives_part], 0))
  return loss


def triplet_loss(queries, positives, negatives, margin=0.1):
  """Calculates Triplet Loss.

  Triplet loss tries to keep all queries closer to positives than to any
  negatives. Differently from the Contrastive Loss, Triplet Loss uses squared
  distances when computing the loss.

  Args:
    queries: [batch_size, dim] Anchor input tensor.
    positives: [batch_size, dim] Positive sample input tensor.
    negatives: [batch_size, num_neg, dim] Negative sample input tensor.
    margin: Float triplet loss loss margin.

  Returns:
    loss: Scalar tensor.
  """
  dim = tf.shape(queries)[1]
  # Number of `queries`.
  batch_size = tf.shape(queries)[0]
  # Number of `negatives`.
  num_neg = tf.shape(negatives)[1]

  # Preparing negatives.
  stacked_negatives = tf.reshape(negatives, [num_neg * batch_size, dim])

  # Preparing queries for further loss calculation.
  stacked_queries = tf.repeat(queries, num_neg, axis=0)

  # Preparing positives for further loss calculation.
  stacked_positives = tf.repeat(positives, num_neg, axis=0)

  # Computes *squared* distances.
  distance_positives = tf.reduce_sum(
      tf.square(stacked_queries - stacked_positives), axis=1)
  distance_negatives = tf.reduce_sum(
      tf.square(stacked_queries - stacked_negatives), axis=1)
  # Final triplet loss calculation.
  loss = tf.reduce_sum(
      tf.maximum(distance_positives - distance_negatives + margin, 0.0))
  return loss
