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
"""DRAGNN wrappers for the MST solver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from dragnn.python import mst_ops
from dragnn.python import network_units
from syntaxnet.util import check


class MstSolverNetwork(network_units.NetworkUnitInterface):
  """Network unit that performs MST prediction with structured loss.

  Parameters:
    forest: If true, solve for a spanning forest instead of a spanning tree.
    loss: The loss function for training.  Select from
      softmax: Default unstructured softmax (prediction is still structured).
      m3n: Max-Margin Markov Networks loss.
    crf_max_dynamic_range: Max dynamic range for the log partition function.

  Links:
    lengths: [B, 1] sequence lengths per batch item.
    scores: [B * N, N] matrix of padded batched arc scores.

  Layers:
    lengths: [B] sequence lengths per batch item.
    scores: [B, N, N] tensor of padded batched arc scores.
    logits: [B * N, N] matrix of padded batched arc scores.
    arcs: [B * N, N] matrix of padded batched 0/1 indicators for MST arcs.
  """

  def __init__(self, component):
    """Initializes layers.

    Args:
      component: Parent ComponentBuilderBase object.
    """
    layers = [
        network_units.Layer(self, 'lengths', -1),
        network_units.Layer(self, 'scores', -1),
        network_units.Layer(self, 'logits', -1),
        network_units.Layer(self, 'arcs', -1),
    ]
    super(MstSolverNetwork, self).__init__(component, init_layers=layers)

    self._attrs = network_units.get_attrs_with_defaults(
        component.spec.network_unit.parameters,
        defaults={
            'forest': False,
            'loss': 'softmax',
            'crf_max_dynamic_range': 20,
        })

    check.Eq(
        len(self._fixed_feature_dims.items()), 0, 'Expected no fixed features')
    check.Eq(
        len(self._linked_feature_dims.items()), 2,
        'Expected two linked features')

    check.In('lengths', self._linked_feature_dims,
             'Missing required linked feature')
    check.In('scores', self._linked_feature_dims,
             'Missing required linked feature')

  def create(self,
             fixed_embeddings,
             linked_embeddings,
             context_tensor_arrays,
             attention_tensor,
             during_training,
             stride=None):
    """Forwards the lengths and scores."""
    check.NotNone(stride, 'MstSolverNetwork requires stride')

    lengths = network_units.lookup_named_tensor('lengths', linked_embeddings)
    lengths_b = tf.to_int32(tf.squeeze(lengths.tensor, [1]))

    scores = network_units.lookup_named_tensor('scores', linked_embeddings)
    scores_bnxn = scores.tensor
    max_length = tf.shape(scores_bnxn)[1]
    scores_bxnxn = tf.reshape(scores_bnxn, [stride, max_length, max_length])

    _, argmax_sources_bxn = mst_ops.maximum_spanning_tree(
        forest=self._attrs['forest'], num_nodes=lengths_b, scores=scores_bxnxn)
    argmax_sources_bn = tf.reshape(argmax_sources_bxn, [-1])
    arcs_bnxn = tf.one_hot(argmax_sources_bn, max_length, dtype=tf.float32)

    return [lengths_b, scores_bxnxn, scores_bnxn, arcs_bnxn]

  def get_logits(self, network_tensors):
    return network_tensors[self.get_layer_index('logits')]

  def get_bulk_predictions(self, stride, network_tensors):
    return network_tensors[self.get_layer_index('arcs')]

  def compute_bulk_loss(self, stride, network_tensors, gold):
    """See base class."""
    if self._attrs['loss'] == 'softmax':
      return (None, None, None)  # fall back to default bulk softmax

    lengths_b, scores_bxnxn, _, arcs_bnxn = network_tensors
    max_length = tf.shape(scores_bxnxn)[2]
    arcs_bxnxn = tf.reshape(arcs_bnxn, [stride, max_length, max_length])
    gold_bxn = tf.reshape(gold, [stride, max_length])
    gold_bxnxn = tf.one_hot(gold_bxn, max_length, dtype=tf.float32)

    loss = self._compute_loss(lengths_b, scores_bxnxn, gold_bxnxn)
    correct = tf.reduce_sum(tf.to_int32(arcs_bxnxn * gold_bxnxn))
    total = tf.reduce_sum(lengths_b)
    return loss, correct, total

  def _compute_loss(self, lengths, scores, gold):
    """Computes the configured structured loss for a batch.

    Args:
      lengths: [B] sequence lengths per batch item.
      scores: [B, N, N] tensor of padded batched arc scores.
      gold: [B, N, N] tensor of 0/1 indicators for gold arcs.

    Returns:
      Scalar sum of losses across the batch.
    """
    # Dispatch to one of the _compute_*_loss() methods.
    method_name = '_compute_%s_loss' % self._attrs['loss']
    loss_b = getattr(self, method_name)(lengths, scores, gold)
    return tf.reduce_sum(loss_b)

  def _compute_m3n_loss(self, lengths, scores, gold):
    """Computes the M3N-style structured hinge loss for a batch."""
    # Perform hamming-loss-augmented inference.
    gold_scores_b = tf.reduce_sum(scores * gold, axis=[1, 2])
    hamming_loss_bxnxn = 1 - gold
    scores_bxnxn = scores + hamming_loss_bxnxn
    max_scores_b, _ = mst_ops.maximum_spanning_tree(
        num_nodes=lengths, scores=scores_bxnxn, forest=self._attrs['forest'])
    return max_scores_b - gold_scores_b

  def _compute_crf_loss(self, lengths, scores, gold):
    """Computes the negative CRF log-probability for a batch."""
    # The |scores| are assumed to be in the log domain.
    log_gold_scores_b = tf.reduce_sum(scores * gold, axis=[1, 2])
    log_partition_functions_b = mst_ops.log_partition_function(
        num_nodes=lengths,
        scores=scores,
        forest=self._attrs['forest'],
        max_dynamic_range=self._attrs['crf_max_dynamic_range'])
    return log_partition_functions_b - log_gold_scores_b  # negative log-prob
