"""Component builders for non-recurrent networks in DRAGNN."""


import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

from dragnn.python import component
from dragnn.python import dragnn_ops
from dragnn.python import network_units
from syntaxnet.util import check


def fetch_linked_embedding(comp, network_states, feature_spec):
  """Looks up linked embeddings in other components.

  Args:
    comp: ComponentBuilder object with respect to which the feature is to be
        fetched
    network_states: dictionary of NetworkState objects
    feature_spec: FeatureSpec proto for the linked feature to be looked up

  Returns:
    NamedTensor containing the linked feature tensor

  Raises:
    NotImplementedError: if a linked feature with source translator other than
        'identity' is configured.
    RuntimeError: if a recurrent linked feature is configured.
  """
  if feature_spec.source_translator != 'identity':
    raise NotImplementedError(feature_spec.source_translator)
  if feature_spec.source_component == comp.name:
    raise RuntimeError(
        'Recurrent linked features are not supported in bulk extraction.')
  tf.logging.info('[%s] Adding linked feature "%s"', comp.name,
                  feature_spec.name)
  source = comp.master.lookup_component[feature_spec.source_component]

  return network_units.NamedTensor(
      network_states[source.name].activations[
          feature_spec.source_layer].bulk_tensor,
      feature_spec.name)


def _validate_embedded_fixed_features(comp):
  """Checks that the embedded fixed features of |comp| are set up properly."""
  for feature in comp.spec.fixed_feature:
    check.Gt(feature.embedding_dim, 0,
             'Embeddings requested for non-embedded feature: %s' % feature)
    if feature.is_constant:
      check.IsTrue(feature.HasField('pretrained_embedding_matrix'),
                   'Constant embeddings must be pretrained: %s' % feature)


def fetch_differentiable_fixed_embeddings(comp, state, stride):
  """Looks up fixed features with separate, differentiable, embedding lookup.

  Args:
    comp: Component whose fixed features we wish to look up.
    state: live MasterState object for the component.
    stride: Tensor containing current batch * beam size.

  Returns:
    state handle: updated state handle to be used after this call
    fixed_embeddings: list of NamedTensor objects
  """
  _validate_embedded_fixed_features(comp)
  num_channels = len(comp.spec.fixed_feature)
  if not num_channels:
    return state.handle, []

  state.handle, indices, ids, weights, num_steps = (
      dragnn_ops.bulk_fixed_features(
          state.handle, component=comp.name, num_channels=num_channels))
  fixed_embeddings = []
  for channel, feature_spec in enumerate(comp.spec.fixed_feature):
    differentiable_or_constant = ('constant' if feature_spec.is_constant else
                                  'differentiable')
    tf.logging.info('[%s] Adding %s fixed feature "%s"', comp.name,
                    differentiable_or_constant, feature_spec.name)
    size = stride * num_steps * feature_spec.size
    fixed_embedding = network_units.embedding_lookup(
        comp.get_variable(network_units.fixed_embeddings_name(channel)),
        indices[channel], ids[channel], weights[channel], size)
    if feature_spec.is_constant:
      fixed_embedding = tf.stop_gradient(fixed_embedding)
    fixed_embeddings.append(
        network_units.NamedTensor(fixed_embedding, feature_spec.name))

  return state.handle, fixed_embeddings


def fetch_fast_fixed_embeddings(comp, state):
  """Looks up fixed features with fast, non-differentiable, op.

  Since BulkFixedEmbeddings is non-differentiable with respect to the
  embeddings, the idea is to call this function only when the graph is
  not being used for training.

  Args:
    comp: Component whose fixed features we wish to look up.
    state: live MasterState object for the component.

  Returns:
    state handle: updated state handle to be used after this call
    fixed_embeddings: list of NamedTensor objects
  """
  _validate_embedded_fixed_features(comp)
  num_channels = len(comp.spec.fixed_feature)
  if not num_channels:
    return state.handle, []
  tf.logging.info('[%s] Adding %d fast fixed features', comp.name, num_channels)

  state.handle, bulk_embeddings, _ = dragnn_ops.bulk_fixed_embeddings(
      state.handle, [
          comp.get_variable(network_units.fixed_embeddings_name(c))
          for c in range(num_channels)
      ],
      component=comp.name)

  bulk_embeddings = network_units.NamedTensor(bulk_embeddings,
                                              'bulk-%s-fixed-features' %
                                              comp.name)
  return state.handle, [bulk_embeddings]


def extract_fixed_feature_ids(comp, state, stride):
  """Extracts fixed feature IDs.

  Args:
    comp: Component whose fixed feature IDs we wish to extract.
    state: Live MasterState object for the component.
    stride: Tensor containing current batch * beam size.

  Returns:
    state handle: Updated state handle to be used after this call.
    ids: List of [stride * num_steps, 1] feature IDs per channel.  Missing IDs
         (e.g., due to batch padding) are set to -1.
  """
  num_channels = len(comp.spec.fixed_feature)
  if not num_channels:
    return state.handle, []

  for feature_spec in comp.spec.fixed_feature:
    check.Eq(feature_spec.size, 1, 'All features must have size=1')
    check.Lt(feature_spec.embedding_dim, 0, 'All features must be non-embedded')

  state.handle, indices, ids, _, num_steps = dragnn_ops.bulk_fixed_features(
      state.handle, component=comp.name, num_channels=num_channels)
  size = stride * num_steps

  fixed_ids = []
  for channel, feature_spec in enumerate(comp.spec.fixed_feature):
    tf.logging.info('[%s] Adding fixed feature IDs "%s"', comp.name,
                    feature_spec.name)

    # The +1 and -1 increments ensure that missing IDs default to -1.
    #
    # TODO(googleuser): This formula breaks if multiple IDs are extracted at some
    # step.  Try using tf.unique() to enforce the unique-IDS precondition.
    sums = tf.unsorted_segment_sum(ids[channel] + 1, indices[channel], size) - 1
    sums = tf.expand_dims(sums, axis=1)
    fixed_ids.append(network_units.NamedTensor(sums, feature_spec.name, dim=1))
  return state.handle, fixed_ids


def update_network_states(comp, tensors, network_states, stride):
  """Stores Tensor objects corresponding to layer outputs.

  For use in subsequent tasks.

  Args:
    comp: Component for which the tensor handles are being stored.
    tensors: list of Tensors to store
    network_states: dictionary of component NetworkState objects
    stride: stride of the stored tensor.
  """
  network_state = network_states[comp.name]
  with tf.name_scope(comp.name + '/stored_act'):
    for index, network_tensor in enumerate(tensors):
      network_state.activations[comp.network.layers[index].name] = (
          network_units.StoredActivations(tensor=network_tensor, stride=stride,
                                          dim=comp.network.layers[index].dim))


def build_cross_entropy_loss(logits, gold):
  """Constructs a cross entropy from logits and one-hot encoded gold labels.

  Supports skipping rows where the gold label is the magic -1 value.

  Args:
    logits: float Tensor of scores.
    gold: int Tensor of one-hot labels.

  Returns:
    cost, correct, total: the total cost, the total number of correctly
        predicted labels, and the total number of valid labels.
  """
  valid = tf.reshape(tf.where(tf.greater(gold, -1)), [-1])
  gold = tf.gather(gold, valid)
  logits = tf.gather(logits, valid)
  correct = tf.reduce_sum(tf.to_int32(tf.nn.in_top_k(logits, gold, 1)))
  total = tf.size(gold)
  cost = tf.reduce_sum(
      tf.contrib.nn.deprecated_flipped_sparse_softmax_cross_entropy_with_logits(
          logits, tf.cast(gold, tf.int64))) / tf.cast(total, tf.float32)
  return cost, correct, total


class BulkFeatureExtractorComponentBuilder(component.ComponentBuilderBase):
  """A component builder to bulk extract features.

  Both fixed and linked features are supported, with some restrictions:
  1. Fixed features may not be recurrent. Fixed features are extracted along the
     gold path, which does not work during inference.
  2. Linked features may not be recurrent and are 'untranslated'. For now,
     linked features are extracted without passing them through any transition
     system or source translator.
  """

  def build_greedy_training(self, state, network_states):
    """Extracts features and advances a batch using the oracle path.

    Args:
      state: MasterState from the 'AdvanceMaster' op that advances the
          underlying master to this component.
      network_states: dictionary of component NetworkState objects

    Returns:
      state handle: final state after advancing
      cost: regularization cost, possibly associated with embedding matrices
      correct: since no gold path is available, 0.
      total: since no gold path is available, 0.
    """
    logging.info('Building component: %s', self.spec.name)
    stride = state.current_batch_size * self.training_beam_size
    with tf.variable_scope(self.name, reuse=True):
      state.handle, fixed_embeddings = fetch_differentiable_fixed_embeddings(
          self, state, stride)

    linked_embeddings = [
        fetch_linked_embedding(self, network_states, spec)
        for spec in self.spec.linked_feature
    ]

    with tf.variable_scope(self.name, reuse=True):
      tensors = self.network.create(
          fixed_embeddings, linked_embeddings, None, None, True, stride=stride)
    update_network_states(self, tensors, network_states, stride)
    cost = self.add_regularizer(tf.constant(0.))

    return state.handle, cost, 0, 0

  def build_greedy_inference(self, state, network_states,
                             during_training=False):
    """Extracts features and advances a batch using the oracle path.

    NOTE(danielandor) For now this method cannot be called during training.
    That is to say, unroll_using_oracle for this component must be set to true.
    This will be fixed by separating train_with_oracle and train_with_inference.

    Args:
      state: MasterState from the 'AdvanceMaster' op that advances the
          underlying master to this component.
      network_states: dictionary of component NetworkState objects
      during_training: whether the graph is being constructed during training

    Returns:
      state handle: final state after advancing
    """
    logging.info('Building component: %s', self.spec.name)
    if during_training:
      stride = state.current_batch_size * self.training_beam_size
    else:
      stride = state.current_batch_size * self.inference_beam_size

    with tf.variable_scope(self.name, reuse=True):
      if during_training:
        state.handle, fixed_embeddings = fetch_differentiable_fixed_embeddings(
            self, state, stride)
      else:
        state.handle, fixed_embeddings = fetch_fast_fixed_embeddings(self,
                                                                     state)

    linked_embeddings = [
        fetch_linked_embedding(self, network_states, spec)
        for spec in self.spec.linked_feature
    ]

    with tf.variable_scope(self.name, reuse=True):
      tensors = self.network.create(
          fixed_embeddings,
          linked_embeddings,
          None,
          None,
          during_training=during_training,
          stride=stride)

    update_network_states(self, tensors, network_states, stride)
    return state.handle


class BulkFeatureIdExtractorComponentBuilder(component.ComponentBuilderBase):
  """A component builder to bulk extract feature IDs.

  This is a variant of BulkFeatureExtractorComponentBuilder that only supports
  fixed features, and extracts raw feature IDs instead of feature embeddings.
  Since the extracted feature IDs are integers, the results produced by this
  component are in general not differentiable.
  """

  def __init__(self, master, component_spec):
    """Initializes the feature ID extractor component.

    Args:
      master: dragnn.MasterBuilder object.
      component_spec: dragnn.ComponentSpec proto to be built.
    """
    super(BulkFeatureIdExtractorComponentBuilder, self).__init__(
        master, component_spec)
    check.Eq(len(self.spec.linked_feature), 0, 'Linked features are forbidden')
    for feature_spec in self.spec.fixed_feature:
      check.Lt(feature_spec.embedding_dim, 0,
               'Features must be non-embedded: %s' % feature_spec)

  def build_greedy_training(self, state, network_states):
    """See base class."""
    state.handle = self._extract_feature_ids(state, network_states, True)
    cost = self.add_regularizer(tf.constant(0.))
    return state.handle, cost, 0, 0

  def build_greedy_inference(self, state, network_states,
                             during_training=False):
    """See base class."""
    return self._extract_feature_ids(state, network_states, during_training)

  def _extract_feature_ids(self, state, network_states, during_training):
    """Extracts feature IDs and advances a batch using the oracle path.

    Args:
      state: MasterState from the 'AdvanceMaster' op that advances the
          underlying master to this component.
      network_states: Dictionary of component NetworkState objects.
      during_training: Whether the graph is being constructed during training.

    Returns:
      state handle: Final state after advancing.
    """
    logging.info('Building component: %s', self.spec.name)

    if during_training:
      stride = state.current_batch_size * self.training_beam_size
    else:
      stride = state.current_batch_size * self.inference_beam_size

    with tf.variable_scope(self.name, reuse=True):
      state.handle, ids = extract_fixed_feature_ids(self, state, stride)

    with tf.variable_scope(self.name, reuse=True):
      tensors = self.network.create(
          ids, [], None, None, during_training, stride=stride)
    update_network_states(self, tensors, network_states, stride)
    return state.handle


class BulkAnnotatorComponentBuilder(component.ComponentBuilderBase):
  """A component builder to bulk annotate or compute the cost of a gold path.

  This component can be used with features that don't depend on the
  transition system state.

  Since no feature extraction is performed, only non-recurrent
  'identity' linked features are supported.

  If a FeedForwardNetwork is configured with no hidden units, this component
  acts as a 'bulk softmax' component.
  """

  def build_greedy_training(self, state, network_states):
    """Advances a batch using oracle paths, returning the overall CE cost.

    Args:
      state: MasterState from the 'AdvanceMaster' op that advances the
          underlying master to this component.
      network_states: dictionary of component NetworkState objects

    Returns:
      (state handle, cost, correct, total): TF ops corresponding to the final
          state after unrolling, the total cost, the total number of correctly
          predicted actions, and the total number of actions.

    Raises:
      RuntimeError: if fixed features are configured.
    """
    logging.info('Building component: %s', self.spec.name)
    if self.spec.fixed_feature:
      raise RuntimeError(
          'Fixed features are not compatible with bulk annotation. '
          'Use the "bulk-features" component instead.')
    linked_embeddings = [
        fetch_linked_embedding(self, network_states, spec)
        for spec in self.spec.linked_feature
    ]

    stride = state.current_batch_size * self.training_beam_size
    with tf.variable_scope(self.name, reuse=True):
      network_tensors = self.network.create([], linked_embeddings, None, None,
                                            True, stride)

    update_network_states(self, network_tensors, network_states, stride)

    logits = self.network.get_logits(network_tensors)
    state.handle, gold = dragnn_ops.bulk_advance_from_oracle(
        state.handle, component=self.name)

    cost, correct, total = build_cross_entropy_loss(logits, gold)
    cost = self.add_regularizer(cost)

    return state.handle, cost, correct, total

  def build_greedy_inference(self, state, network_states,
                             during_training=False):
    """Annotates a batch of documents using network scores.

    Args:
      state: MasterState from the 'AdvanceMaster' op that advances the
          underlying master to this component.
      network_states: dictionary of component NetworkState objects
      during_training: whether the graph is being constructed during training

    Returns:
      Handle to the state once inference is complete for this Component.

    Raises:
      RuntimeError: if fixed features are configured
    """
    logging.info('Building component: %s', self.spec.name)
    if self.spec.fixed_feature:
      raise RuntimeError(
          'Fixed features are not compatible with bulk annotation. '
          'Use the "bulk-features" component instead.')
    linked_embeddings = [
        fetch_linked_embedding(self, network_states, spec)
        for spec in self.spec.linked_feature
    ]

    if during_training:
      stride = state.current_batch_size * self.training_beam_size
    else:
      stride = state.current_batch_size * self.inference_beam_size

    with tf.variable_scope(self.name, reuse=True):
      network_tensors = self.network.create(
          [], linked_embeddings, None, None, during_training, stride)

    update_network_states(self, network_tensors, network_states, stride)

    logits = self.network.get_logits(network_tensors)
    return dragnn_ops.bulk_advance_from_prediction(
        state.handle, logits, component=self.name)
