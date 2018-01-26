# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""This implements single view TCN triplet loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def pairwise_squared_distance(feature):
  """Computes the squared pairwise distance matrix.

  output[i, j] = || feature[i, :] - feature[j, :] ||_2^2

  Args:
    feature: 2-D Tensor of size [number of data, feature dimension]

  Returns:
    pairwise_squared_distances: 2-D Tensor of size
      [number of data, number of data]
  """
  pairwise_squared_distances = tf.add(
      tf.reduce_sum(
          tf.square(feature), axis=1, keep_dims=True),
      tf.reduce_sum(
          tf.square(tf.transpose(feature)), axis=0,
          keep_dims=True)) - 2.0 * tf.matmul(feature, tf.transpose(feature))

  # Deal with numerical inaccuracies. Set small negatives to zero.
  pairwise_squared_distances = tf.maximum(pairwise_squared_distances, 0.0)
  return pairwise_squared_distances


def masked_maximum(data, mask, dim=1):
  """Computes the axis wise maximum over chosen elements.

  Args:
    data: N-D Tensor.
    mask: N-D Tensor of zeros or ones.
    dim: The dimension over which to compute the maximum.

  Returns:
    masked_maximums: N-D Tensor.
      The maximized dimension is of size 1 after the operation.
  """
  axis_minimums = tf.reduce_min(data, dim, keep_dims=True)
  masked_maximums = tf.reduce_max(
      tf.multiply(
          data - axis_minimums, mask), dim, keep_dims=True) + axis_minimums
  return masked_maximums


def masked_minimum(data, mask, dim=1):
  """Computes the axis wise minimum over chosen elements.

  Args:
    data: 2-D Tensor of size [n, m].
    mask: 2-D Boolean Tensor of size [n, m].
    dim: The dimension over which to compute the minimum.

  Returns:
    masked_minimums: N-D Tensor.
      The minimized dimension is of size 1 after the operation.
  """
  axis_maximums = tf.reduce_max(data, dim, keep_dims=True)
  masked_minimums = tf.reduce_min(
      tf.multiply(
          data - axis_maximums, mask), dim, keep_dims=True) + axis_maximums
  return masked_minimums


def singleview_tcn_loss(
    embeddings, timesteps, pos_radius, neg_radius, margin=1.0,
    sequence_ids=None, multiseq=False):
  """Computes the single view triplet loss with semi-hard negative mining.

  The loss encourages the positive distances (between a pair of embeddings with
  the same labels) to be smaller than the minimum negative distance among
  which are at least greater than the positive distance plus the margin constant
  (called semi-hard negative) in the mini-batch. If no such negative exists,
  uses the largest negative distance instead.

  Anchor, positive, negative selection is as follow:
  Anchors: We consider every embedding timestep as an anchor.
  Positives: pos_radius defines a radius (in timesteps) around each anchor from
    which positives can be drawn. E.g. An anchor with t=10 and a pos_radius of
    2 produces a set of 4 (anchor,pos) pairs [(a=10, p=8), ... (a=10, p=12)].
  Negatives: neg_radius defines a boundary (in timesteps) around each anchor,
    outside of which negatives can be drawn. E.g. An anchor with t=10 and a
    neg_radius of 4 means negatives can be any t_neg where t_neg < 6 and
    t_neg > 14.

  Args:
    embeddings: 2-D Tensor of embedding vectors.
    timesteps: 1-D Tensor with shape [batch_size, 1] of sequence timesteps.
    pos_radius: int32; the size of the window (in timesteps) around each anchor
      timestep that a positive can be drawn from.
    neg_radius: int32; the size of the window (in timesteps) around each anchor
      timestep that defines a negative boundary. Negatives can only be chosen
      where negative timestep t is < negative boundary min or > negative
      boundary max.
    margin: Float; the triplet loss margin hyperparameter.
    sequence_ids: (Optional) 1-D Tensor with shape [batch_size, 1] of sequence
      ids. Together (sequence_id, sequence_timestep) give us a unique index for
      each image if we have multiple sequences in a batch.
    multiseq: Boolean, whether or not the batch is composed of multiple
      sequences (with possibly colliding timesteps).

  Returns:
    triplet_loss: tf.float32 scalar.
  """
  assert neg_radius > pos_radius

  # If timesteps shape isn't [batchsize, 1], reshape to [batch_size, 1].
  tshape = tf.shape(timesteps)
  assert tshape.shape == 2 or tshape.shape == 1
  if tshape.shape == 1:
    timesteps = tf.reshape(timesteps, [tshape[0], 1])

  # Build pairwise squared distance matrix.
  pdist_matrix = pairwise_squared_distance(embeddings)

  # Build pairwise binary adjacency matrix, where adjacency[i,j] is True
  # if timestep j is inside the positive range for timestep i and both
  # timesteps come from the same sequence.
  pos_radius = tf.cast(pos_radius, tf.int32)

  if multiseq:
    # If sequence_ids shape isn't [batchsize, 1], reshape to [batch_size, 1].
    tshape = tf.shape(sequence_ids)
    assert tshape.shape == 2 or tshape.shape == 1
    if tshape.shape == 1:
      sequence_ids = tf.reshape(sequence_ids, [tshape[0], 1])

    # Build pairwise binary adjacency matrix based on sequence_ids
    sequence_adjacency = tf.equal(sequence_ids, tf.transpose(sequence_ids))

    # Invert so we can select negatives only.
    sequence_adjacency_not = tf.logical_not(sequence_adjacency)

    in_pos_range = tf.logical_and(
        tf.less_equal(
            tf.abs(timesteps - tf.transpose(timesteps)), pos_radius),
        sequence_adjacency)
    # Build pairwise binary discordance matrix, where discordance[i,j] is True
    # if timestep j is inside the negative range for timestep i or if the
    # timesteps come from different sequences.
    in_neg_range = tf.logical_or(
        tf.greater(tf.abs(timesteps - tf.transpose(timesteps)), neg_radius),
        sequence_adjacency_not
    )
  else:
    in_pos_range = tf.less_equal(
        tf.abs(timesteps - tf.transpose(timesteps)), pos_radius)
    in_neg_range = tf.greater(tf.abs(timesteps - tf.transpose(timesteps)),
                              neg_radius)

  batch_size = tf.size(timesteps)

  # compute the mask
  pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])
  mask = tf.logical_and(
      tf.tile(in_neg_range, [batch_size, 1]),
      tf.greater(pdist_matrix_tile,
                 tf.reshape(tf.transpose(pdist_matrix), [-1, 1])))
  mask_final = tf.reshape(
      tf.greater(
          tf.reduce_sum(
              tf.cast(
                  mask, dtype=tf.float32), 1, keep_dims=True),
          0.0), [batch_size, batch_size])
  mask_final = tf.transpose(mask_final)

  in_neg_range = tf.cast(in_neg_range, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)

  # negatives_outside: smallest D_an where D_an > D_ap
  negatives_outside = tf.reshape(
      masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
  negatives_outside = tf.transpose(negatives_outside)

  # negatives_inside: largest D_an
  negatives_inside = tf.tile(
      masked_maximum(pdist_matrix, in_neg_range), [1, batch_size])
  semi_hard_negatives = tf.where(
      mask_final, negatives_outside, negatives_inside)

  loss_mat = tf.add(margin, pdist_matrix - semi_hard_negatives)

  mask_positives = tf.cast(
      in_pos_range, dtype=tf.float32) - tf.diag(tf.ones([batch_size]))

  # In lifted-struct, the authors multiply 0.5 for upper triangular
  #   in semihard, they take all positive pairs except the diagonal.
  num_positives = tf.reduce_sum(mask_positives)

  triplet_loss = tf.truediv(
      tf.reduce_sum(tf.maximum(tf.multiply(loss_mat, mask_positives), 0.0)),
      num_positives,
      name='triplet_svtcn_loss')

  return triplet_loss
