# Copyright 2018 Google Inc. All Rights Reserved.
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

"""TensorFlow ops for maximum spanning tree problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import dragnn.python.load_mst_cc_impl
from dragnn.mst.ops import gen_mst_ops
from dragnn.python import digraph_ops
from syntaxnet.util import check

# Re-export the generated MST op.
maximum_spanning_tree = gen_mst_ops.maximum_spanning_tree


@tf.RegisterGradient("MaximumSpanningTree")
def maximum_spanning_tree_gradient(mst_op, d_loss_d_max_scores, *_):
  """Returns a subgradient of the MaximumSpanningTree op.

  Note that MaximumSpanningTree is only differentiable w.r.t. its |scores| input
  and its |max_scores| output.

  Args:
    mst_op: The MaximumSpanningTree op being differentiated.
    d_loss_d_max_scores: [B] vector where entry b is the gradient of the network
                         loss w.r.t. entry b of the |max_scores| output of the
                         |mst_op|.
    *_: The gradients w.r.t. the other outputs; ignored.

  Returns:
    1. None, since the op is not differentiable w.r.t. its |num_nodes| input.
    2. [B,M,M] tensor where entry b,t,s is a subgradient of the network loss
       w.r.t. entry b,t,s of the |scores| input, with the same dtype as
       |d_loss_d_max_scores|.
  """
  dtype = d_loss_d_max_scores.dtype.base_dtype
  check.NotNone(dtype)

  argmax_sources_bxm = mst_op.outputs[1]
  input_dim = tf.shape(argmax_sources_bxm)[1]  # M in the docstring

  # The one-hot argmax is a subgradient of max.  Convert the batch of maximal
  # spanning trees into 0/1 indicators, then scale them by the relevant output
  # gradients from |d_loss_d_max_scores|.  Note that |d_loss_d_max_scores| must
  # be reshaped in order for it to broadcast across the batch dimension.
  indicators_bxmxm = tf.one_hot(argmax_sources_bxm, input_dim, dtype=dtype)
  d_loss_d_max_scores_bx1 = tf.expand_dims(d_loss_d_max_scores, -1)
  d_loss_d_max_scores_bx1x1 = tf.expand_dims(d_loss_d_max_scores_bx1, -1)
  d_loss_d_scores_bxmxm = indicators_bxmxm * d_loss_d_max_scores_bx1x1
  return None, d_loss_d_scores_bxmxm


def log_partition_function(num_nodes,
                           scores,
                           forest=False,
                           max_dynamic_range=None):
  r"""Returns the log of the sum-of-product of spanning trees or forests.

  Computing the sum-of-product in the log domain reduces the chance of overflow
  or underflow, and ML techniques (e.g., CRF loss functions) typically require
  the log partition function anyways.  For similar reasons, the scores input is
  assumed to be specified in the log domain.

  The partition function is caluclated via application of the Matrix-Tree
  theorem; see the following for details:
    https://en.wikipedia.org/wiki/Kirchhoff%27s_theorem
    http://www.aclweb.org/anthology/D/D07/D07-1015.pdf

  Computing the gradient of the log partition function requires inverting the
  Laplacian matrix.  Numerical issues may occur if the Laplacian is singular or
  nearly-so.  (Intuitively, the Laplacian will be close to singular when the
  input scores strongly favor invalid structures such as cycles).  In the EMNLP
  paper, we alleviated the numerical issues by clipping the difference between
  the minimum and maximum score for each node to 20 (in the log domain).  The
  |max_dynamic_range| argument can be used for this purpose.

  TODO(googleuser): Try improving the condition number of the Laplacian matrix
  directly, instead of using the indirect approach above.  For example, one
  could add c*I to the Laplacian (i.e., Tikhonov regularization).

  Args:
    num_nodes: [B] vector of graph sizes per batch item.
    scores: [B,M,M] tensor of padded batched arc and root scores, in the format
      used by the maximum_spanning_tree() op.  Padding values must be finite.
    forest: If true, sum over spanning forests instead of trees.
    max_dynamic_range: If specified, incoming scores for each node are clipped
      to at most this far from the maximum such score (in the log domain).

  Returns:
    [B] vector Z of log partition function values, where
      Z[b] = log(
          \sum_{tree spanning batch item b}
              score(root_of(tree)) \prod_{arc in tree} score(arc))
  """
  orig_dtype = scores.dtype.base_dtype
  scores_bxmxm = tf.to_double(scores)  # use doubles to reduce under/overflow
  shape_bxmxm = tf.shape(scores_bxmxm)
  batch_size = shape_bxmxm[0]
  max_nodes = shape_bxmxm[1]
  total_nodes = batch_size * max_nodes

  # To eliminate overflow, we locally normalize the scores.  Specifically, for
  # each node we divide its incoming arc scores and root selection score by the
  # maximum such score.  Since each node in a tree must select exactly one of
  # these scores (i.e., it is either a root or has exactly one incoming arc),
  # the local normalization factors are identical for all trees and can thus be
  # factored out of the sum over trees.
  #
  # More concretely, we find the maximum per node, divide all scores for that
  # node by the maximum, and then find the partition function of the normalized
  # scores.  Then we recover the un-normalized partition function by multiplying
  # the per-node maxima back in.  This final step is performed in the log domain
  # to avoid overflow.
  #
  # Note that underflow is still possible, but unlikely as long as the scores
  # are close to feasible (i.e., there is not too much mass on non-trees).  The
  # |max_dynamic_range| argument can be used to mitigate this.

  # Finding the maximum incoming score is difficult, because the batch padding
  # may contain arbitrary values.  We restrict the maximization to valid arcs
  # using tf.unsorted_segment_max() with a specially-constructed set of IDs.
  _, valid_tokens_bxm = digraph_ops.ValidArcAndTokenMasks(
      num_nodes, max_nodes, dtype=tf.int32)

  # Create a tensor of "target IDs".  In each row of each sub-matrix, the
  # positions of valid source tokens are filled with the 1-origin index of that
  # row in the entire batch, and zero elsewhere.  For example, given a batch
  # with num_nodes=[2, 3] we might have
  #   [[[1, 1, 0],
  #     [2, 2, 0],
  #     [3, 3, 0]],
  #    [[4, 4, 4],
  #     [5, 5, 5],
  #     [6, 6, 6]]]
  #
  # TODO(googleuser): The dynamic masking is pretty awkward.  Find an op that does
  # this (I looked, but maybe not hard enough), or write a custom op for this.
  valid_tokens_bx1xm = tf.expand_dims(valid_tokens_bxm, 1)
  valid_sources_bxmxm = tf.tile(valid_tokens_bx1xm, [1, max_nodes, 1])
  sequence_bm = 1 + tf.range(total_nodes, dtype=tf.int32)
  sequence_bxmx1 = tf.reshape(sequence_bm, [batch_size, max_nodes, 1])
  target_ids_bxmxm = valid_sources_bxmxm * sequence_bxmx1

  max_scores_bm1 = tf.unsorted_segment_max(scores_bxmxm, target_ids_bxmxm,
                                           total_nodes + 1)
  max_scores_bm = max_scores_bm1[1:]  # ID 0 corresponds to padding

  # Similar to above, we need to sum over the valid tokens.  We analogously use
  # tf.unsorted_segment_sum() with a specially-constructed set of "batch IDs".
  sequence_b = 1 + tf.range(batch_size, dtype=tf.int32)
  sequence_bx1 = tf.expand_dims(sequence_b, 1)
  batch_ids_bxm = valid_tokens_bxm * sequence_bx1
  batch_ids_bm = tf.reshape(batch_ids_bxm, [-1])

  log_normalization_factor_b1 = tf.unsorted_segment_sum(
      max_scores_bm, batch_ids_bm, batch_size + 1)
  log_normalization_factor_b = log_normalization_factor_b1[1:]

  # Locally-normalize and optionally clip the scores.
  max_scores_bxmx1 = tf.reshape(max_scores_bm, [batch_size, max_nodes, 1])
  scores_bxmxm -= max_scores_bxmx1
  if max_dynamic_range is not None:
    # After normalization, the scores are non-positive with max=0, so the
    # |max_dynamic_range| can be applied directly.
    #
    # PyLint thinks "-max_dynamic_range" is invalid because it defaults to None.

    scores_bxmxm = tf.maximum(scores_bxmxm, -max_dynamic_range)
  scores_bxmxm = tf.exp(scores_bxmxm)

  # Apply the Matrix-Tree theorem.
  exp_normalized_laplacian_bxmxm = digraph_ops.LaplacianMatrix(
      num_nodes, scores_bxmxm, forest=forest)
  log_normalized_partition_function_b = tf.log(
      tf.matrix_determinant(exp_normalized_laplacian_bxmxm))

  # Reapply the normalization factor that was divided out.
  log_partition_function_b = (
      log_normalized_partition_function_b + log_normalization_factor_b)
  return tf.cast(log_partition_function_b, orig_dtype)
