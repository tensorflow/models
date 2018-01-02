# Copyright 2017 Google Inc. All Rights Reserved.
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

"""Network units used in the Dozat and Manning (2017) biaffine parser."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from dragnn.python import digraph_ops
from dragnn.python import network_units
from syntaxnet.util import check


class BiaffineDigraphNetwork(network_units.NetworkUnitInterface):
  """Network unit that computes biaffine digraph scores.

  The D&M parser uses two MLPs to create two activation vectors for each token,
  which represent the token when it it used as the source or target of an arc.
  Arcs are scored using a "biaffine" function that includes a bilinear and
  linear term:

    sources[s] * arc_weights * targets[t] + sources[s] * source_weights

  The digraph is "unlabeled" in that there is at most one arc between any pair
  of tokens.  If labels are required, the BiaffineLabelNetwork can be used to
  label a set of selected arcs.

  Note that in the typical use case where the source and target activations are
  the same dimension and are produced by single-layer MLPs, it is arithmetically
  equivalent to produce the source and target activations using a single MLP of
  twice the size, and then split those activations in half.  The |SplitNetwork|
  can be used for this purpose.

  Parameters:
    None.

  Features:
    sources: [B * N, S] matrix of batched activations for source tokens.
    targets: [B * N, T] matrix of batched activations for target tokens.

  Layers:
    adjacency: [B * N, N] matrix where entry b*N+s,t is the score of the arc
               from s to t in batch b, if s != t, or the score for selecting t
               as a root, if s == t.
  """

  def __init__(self, component):
    """Initializes weights and layers.

    Args:
      component: Parent ComponentBuilderBase object.
    """
    super(BiaffineDigraphNetwork, self).__init__(component)

    check.Eq(len(self._fixed_feature_dims.items()), 0,
             'Expected no fixed features')
    check.Eq(len(self._linked_feature_dims.items()), 2,
             'Expected two linked features')

    check.In('sources', self._linked_feature_dims,
             'Missing required linked feature')
    check.In('targets', self._linked_feature_dims,
             'Missing required linked feature')
    self._source_dim = self._linked_feature_dims['sources']
    self._target_dim = self._linked_feature_dims['targets']

    # TODO(googleuser): Make parameter initialization configurable.
    self._weights = []
    self._weights.append(tf.get_variable(
        'weights_arc', [self._source_dim, self._target_dim], tf.float32,
        tf.random_normal_initializer(stddev=1e-4)))
    self._weights.append(tf.get_variable(
        'weights_source', [self._source_dim], tf.float32,
        tf.random_normal_initializer(stddev=1e-4)))
    self._weights.append(tf.get_variable(
        'root', [self._source_dim], tf.float32,
        tf.random_normal_initializer(stddev=1e-4)))

    self._params.extend(self._weights)
    self._regularized_weights.extend(self._weights)

    # Negative Layer.dim indicates that the dimension is dynamic.
    self._layers.append(network_units.Layer(component, 'adjacency', -1))

  def create(self,
             fixed_embeddings,
             linked_embeddings,
             context_tensor_arrays,
             attention_tensor,
             during_training,
             stride=None):
    """Requires |stride|; otherwise see base class."""
    check.NotNone(stride,
                  'BiaffineDigraphNetwork requires "stride" and must be called '
                  'in the bulk feature extractor component.')

    # TODO(googleuser): Add dropout during training.
    del during_training

    # Retrieve (possibly averaged) weights.
    weights_arc = self._component.get_variable('weights_arc')
    weights_source = self._component.get_variable('weights_source')
    root = self._component.get_variable('root')

    # Extract the source and target token activations.  Use |stride| to collapse
    # batch and beam into a single dimension.
    sources = network_units.lookup_named_tensor('sources', linked_embeddings)
    targets = network_units.lookup_named_tensor('targets', linked_embeddings)
    source_tokens_bxnxs = tf.reshape(sources.tensor,
                                     [stride, -1, self._source_dim])
    target_tokens_bxnxt = tf.reshape(targets.tensor,
                                     [stride, -1, self._target_dim])
    num_tokens = tf.shape(source_tokens_bxnxs)[1]

    # Compute the arc, source, and root potentials.
    arcs_bxnxn = digraph_ops.ArcPotentialsFromTokens(
        source_tokens_bxnxs, target_tokens_bxnxt, weights_arc)
    sources_bxnxn = digraph_ops.ArcSourcePotentialsFromTokens(
        source_tokens_bxnxs, weights_source)
    roots_bxn = digraph_ops.RootPotentialsFromTokens(
        root, target_tokens_bxnxt, weights_arc)

    # Combine them into a single matrix with the roots on the diagonal.
    adjacency_bxnxn = digraph_ops.CombineArcAndRootPotentials(
        arcs_bxnxn + sources_bxnxn, roots_bxn)

    return [tf.reshape(adjacency_bxnxn, [-1, num_tokens])]


class BiaffineLabelNetwork(network_units.NetworkUnitInterface):
  """Network unit that computes biaffine label scores.

  D&M parser uses a slightly modified version of the arc scoring function to
  score labels.  The differences are:

    1. Each label has its own source and target MLPs and biaffine weights.
    2. A linear term for the target token is added.
    3. A bias term is added.

  Parameters:
    num_labels: The number of dependency labels, L.

  Features:
    sources: [B * N, S] matrix of batched activations for source tokens.
    targets: [B * N, T] matrix of batched activations for target tokens.

  Layers:
    labels: [B * N, L] matrix where entry b*N+t,l is the score of the label of
            the inbound arc for token t in batch b.
  """

  def __init__(self, component):
    """Initializes weights and layers.

    Args:
      component: Parent ComponentBuilderBase object.
    """
    super(BiaffineLabelNetwork, self).__init__(component)

    parameters = component.spec.network_unit.parameters
    self._num_labels = int(parameters['num_labels'])

    check.Gt(self._num_labels, 0, 'Expected some labels')
    check.Eq(len(self._fixed_feature_dims.items()), 0,
             'Expected no fixed features')
    check.Eq(len(self._linked_feature_dims.items()), 2,
             'Expected two linked features')

    check.In('sources', self._linked_feature_dims,
             'Missing required linked feature')
    check.In('targets', self._linked_feature_dims,
             'Missing required linked feature')

    self._source_dim = self._linked_feature_dims['sources']
    self._target_dim = self._linked_feature_dims['targets']

    # TODO(googleuser): Make parameter initialization configurable.
    self._weights = []
    self._weights.append(tf.get_variable(
        'weights_pair', [self._num_labels, self._source_dim, self._target_dim],
        tf.float32, tf.random_normal_initializer(stddev=1e-4)))
    self._weights.append(tf.get_variable(
        'weights_source', [self._num_labels, self._source_dim], tf.float32,
        tf.random_normal_initializer(stddev=1e-4)))
    self._weights.append(tf.get_variable(
        'weights_target', [self._num_labels, self._target_dim], tf.float32,
        tf.random_normal_initializer(stddev=1e-4)))

    self._biases = []
    self._biases.append(tf.get_variable(
        'biases', [self._num_labels], tf.float32,
        tf.random_normal_initializer(stddev=1e-4)))

    self._params.extend(self._weights + self._biases)
    self._regularized_weights.extend(self._weights)

    self._layers.append(
        network_units.Layer(component, 'labels', self._num_labels))

  def create(self,
             fixed_embeddings,
             linked_embeddings,
             context_tensor_arrays,
             attention_tensor,
             during_training,
             stride=None):
    """Requires |stride|; otherwise see base class."""
    check.NotNone(stride,
                  'BiaffineLabelNetwork requires "stride" and must be called '
                  'in the bulk feature extractor component.')

    # TODO(googleuser): Add dropout during training.
    del during_training

    # Retrieve (possibly averaged) weights.
    weights_pair = self._component.get_variable('weights_pair')
    weights_source = self._component.get_variable('weights_source')
    weights_target = self._component.get_variable('weights_target')
    biases = self._component.get_variable('biases')

    # Extract and shape the source and target token activations.  Use |stride|
    # to collapse batch and beam into a single dimension.
    sources = network_units.lookup_named_tensor('sources', linked_embeddings)
    targets = network_units.lookup_named_tensor('targets', linked_embeddings)
    sources_bxnxs = tf.reshape(sources.tensor, [stride, -1, self._source_dim])
    targets_bxnxt = tf.reshape(targets.tensor, [stride, -1, self._target_dim])

    # Compute the pair, source, and target potentials.
    pairs_bxnxl = digraph_ops.LabelPotentialsFromTokenPairs(sources_bxnxs,
                                                            targets_bxnxt,
                                                            weights_pair)
    sources_bxnxl = digraph_ops.LabelPotentialsFromTokens(sources_bxnxs,
                                                          weights_source)
    targets_bxnxl = digraph_ops.LabelPotentialsFromTokens(targets_bxnxt,
                                                          weights_target)

    # Combine them with the biases.
    labels_bxnxl = pairs_bxnxl + sources_bxnxl + targets_bxnxl + biases

    # Flatten out the batch dimension.
    return [tf.reshape(labels_bxnxl, [-1, self._num_labels])]
