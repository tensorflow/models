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
"""Utils for building DRAGNN specs."""


from six.moves import xrange
import tensorflow as tf

from dragnn.protos import spec_pb2
from dragnn.python import lexicon
from syntaxnet.ops import gen_parser_ops
from syntaxnet.util import check


class ComponentSpecBuilder(object):
  """Wrapper to help construct SyntaxNetComponent specifications.

  This class will help make sure that ComponentSpec's are consistent with the
  expectations of the SyntaxNet Component backend. It contains defaults used to
  create LinkFeatureChannel specifications according to the network_unit and
  transition_system of the source compnent.  It also encapsulates common recipes
  for hooking up FML and translators.

  Attributes:
    spec: The dragnn.ComponentSpec proto.
  """

  def __init__(self,
               name,
               builder='DynamicComponentBuilder',
               backend='SyntaxNetComponent'):
    """Initializes the ComponentSpec with some defaults for SyntaxNet.

    Args:
      name: The name of this Component in the pipeline.
      builder: The component builder type.
      backend: The component backend type.
    """
    self.spec = spec_pb2.ComponentSpec(
        name=name,
        backend=self.make_module(backend),
        component_builder=self.make_module(builder))

  def make_module(self, name, **kwargs):
    """Forwards kwargs to easily created a RegisteredModuleSpec.

    Note: all kwargs should be string-valued.

    Args:
      name: The registered name of the module.
      **kwargs: Proto fields to be specified in the module.

    Returns:
      Newly created RegisteredModuleSpec.
    """
    return spec_pb2.RegisteredModuleSpec(
        registered_name=name, parameters=kwargs)

  def default_source_layer(self):
    """Returns the default source_layer setting for this ComponentSpec.

    Usually links are intended for a specific layer in the network unit.
    For common network units, this returns the hidden layer intended
    to be read by recurrent and cross-component connections.

    Returns:
      String name of default network layer.

    Raises:
      ValueError: if no default is known for the given setup.
    """
    for network, default_layer in [('FeedForwardNetwork', 'layer_0'),
                                   ('LayerNormBasicLSTMNetwork', 'state_h_0'),
                                   ('LSTMNetwork', 'layer_0'),
                                   ('IdentityNetwork', 'input_embeddings')]:
      if self.spec.network_unit.registered_name.endswith(network):
        return default_layer

    raise ValueError('No default source for network unit: %s' %
                     self.spec.network_unit)

  def default_token_translator(self):
    """Returns the default source_translator setting for token representations.

    Most links are token-based: given a target token index, retrieve a learned
    representation for that token from this component. This depends on the
    transition system; e.g. we should make sure that left-to-right sequence
    models reverse the incoming token index when looking up representations from
    a right-to-left model.

    Returns:
      String name of default translator for this transition system.

    Raises:
      ValueError: if no default is known for the given setup.
    """
    transition_spec = self.spec.transition_system
    if transition_spec.registered_name == 'arc-standard':
      return 'shift-reduce-step'

    if transition_spec.registered_name in ('shift-only', 'tagger'):
      if 'left_to_right' in transition_spec.parameters:
        if transition_spec.parameters['left_to_right'] == 'false':
          return 'reverse-token'
      return 'identity'

    raise ValueError('Invalid transition spec: %s' % str(transition_spec))

  def add_token_link(self, source=None, source_layer=None, **kwargs):
    """Adds a link to source's token representations using default settings.

    Constructs a LinkedFeatureChannel proto and adds it to the spec, using
    defaults to assign the name, component, translator, and layer of the
    channel.  The user must provide fml and embedding_dim.

    Args:
      source: SyntaxComponentBuilder object to pull representations from.
      source_layer: Optional override for a source layer instead of the default.
      **kwargs: Forwarded arguments to the LinkedFeatureChannel proto.
    """
    if source_layer is None:
      source_layer = source.default_source_layer()

    self.spec.linked_feature.add(
        name=source.spec.name,
        source_component=source.spec.name,
        source_layer=source_layer,
        source_translator=source.default_token_translator(),
        **kwargs)

  def add_rnn_link(self, source_layer=None, **kwargs):
    """Adds a recurrent link to this component using default settings.

    This adds the connection to the previous time step only to the network.  It
    constructs a LinkedFeatureChannel proto and adds it to the spec, using
    defaults to assign the name, component, translator, and layer of the
    channel.  The user must provide the embedding_dim only.

    Args:
      source_layer: Optional override for a source layer instead of the default.
      **kwargs: Forwarded arguments to the LinkedFeatureChannel proto.
    """
    if source_layer is None:
      source_layer = self.default_source_layer()

    self.spec.linked_feature.add(
        name='rnn',
        source_layer=source_layer,
        source_component=self.spec.name,
        source_translator='history',
        fml='constant',
        **kwargs)

  def set_transition_system(self, *args, **kwargs):
    """Shorthand to set transition_system using kwargs."""
    self.spec.transition_system.CopyFrom(self.make_module(*args, **kwargs))

  def set_network_unit(self, *args, **kwargs):
    """Shorthand to set network_unit using kwargs."""
    self.spec.network_unit.CopyFrom(self.make_module(*args, **kwargs))

  def add_fixed_feature(self, **kwargs):
    """Shorthand to add a fixed_feature using kwargs."""
    self.spec.fixed_feature.add(**kwargs)

  def add_link(self,
               source,
               source_layer=None,
               source_translator='identity',
               name=None,
               **kwargs):
    """Add a link using default naming and layers only."""
    if source_layer is None:
      source_layer = source.default_source_layer()
    if name is None:
      name = source.spec.name
    self.spec.linked_feature.add(
        source_component=source.spec.name,
        source_layer=source_layer,
        name=name,
        source_translator=source_translator,
        **kwargs)

  def fill_from_resources(self, resource_path, tf_master=''):
    """Fills in feature sizes and vocabularies using SyntaxNet lexicon.

    Must be called before the spec is ready to be used to build TensorFlow
    graphs. Requires a SyntaxNet lexicon built at the resource_path. Using the
    lexicon, this will call the SyntaxNet custom ops to return the number of
    features and vocabulary sizes based on the FML specifications and the
    lexicons. It will also compute the number of actions of the transition
    system.

    This will often CHECK-fail if the spec doesn't correspond to a valid
    transition system or feature setup.

    Args:
      resource_path: Path to the lexicon.
      tf_master: TensorFlow master executor (string, defaults to '' to use the
        local instance).
    """
    check.IsTrue(
        self.spec.transition_system.registered_name,
        'Set a transition system before calling fill_from_resources().')

    context = lexicon.create_lexicon_context(resource_path)

    # If there are any transition system-specific params or resources,
    # copy them over into the context.
    for resource in self.spec.resource:
      context.input.add(name=resource.name).part.add(
          file_pattern=resource.part[0].file_pattern)
    for key, value in self.spec.transition_system.parameters.iteritems():
      context.parameter.add(name=key, value=value)

    context.parameter.add(
        name='brain_parser_embedding_dims',
        value=';'.join([str(x.embedding_dim) for x in self.spec.fixed_feature]))
    context.parameter.add(
        name='brain_parser_features',
        value=';'.join([x.fml for x in self.spec.fixed_feature]))
    context.parameter.add(
        name='brain_parser_predicate_maps',
        value=';'.join(['' for x in self.spec.fixed_feature]))
    context.parameter.add(
        name='brain_parser_embedding_names',
        value=';'.join([x.name for x in self.spec.fixed_feature]))
    context.parameter.add(
        name='brain_parser_transition_system',
        value=self.spec.transition_system.registered_name)

    # Propagate information from SyntaxNet C++ backends into the DRAGNN
    # self.spec.
    with tf.Session(tf_master) as sess:
      feature_sizes, domain_sizes, _, num_actions = sess.run(
          gen_parser_ops.feature_size(task_context_str=str(context)))
      self.spec.num_actions = int(num_actions)
      for i in xrange(len(feature_sizes)):
        self.spec.fixed_feature[i].size = int(feature_sizes[i])
        self.spec.fixed_feature[i].vocabulary_size = int(domain_sizes[i])

    for i in xrange(len(self.spec.linked_feature)):
      self.spec.linked_feature[i].size = len(
          self.spec.linked_feature[i].fml.split(' '))

    del self.spec.resource[:]
    for resource in context.input:
      self.spec.resource.add(name=resource.name).part.add(
          file_pattern=resource.part[0].file_pattern)


def complete_master_spec(master_spec, lexicon_corpus, output_path,
                         tf_master=''):
  """Finishes a MasterSpec that defines the network config.

  Given a MasterSpec that defines the DRAGNN architecture, completes the spec so
  that it can be used to build a DRAGNN graph and run training/inference.

  Args:
    master_spec: MasterSpec.
    lexicon_corpus: the corpus to be used with the LexiconBuilder.
    output_path: directory to save resources to.
    tf_master: TensorFlow master executor (string, defaults to '' to use the
      local instance).

  Returns:
    None, since the spec is changed in-place.
  """
  if lexicon_corpus:
    lexicon.build_lexicon(output_path, lexicon_corpus)

  # Use Syntaxnet builder to fill out specs.
  for i, spec in enumerate(master_spec.component):
    builder = ComponentSpecBuilder(spec.name)
    builder.spec = spec
    builder.fill_from_resources(output_path, tf_master=tf_master)
    master_spec.component[i].CopyFrom(builder.spec)


def default_targets_from_spec(spec):
  """Constructs a default set of TrainTarget protos from a DRAGNN spec.

  For each component in the DRAGNN spec, it adds a training target for that
  component's oracle. It also stops unrolling the graph with that component.  It
  skips any 'shift-only' transition systems which have no oracle. E.g.: if there
  are three components, a 'shift-only', a 'tagger', and a 'arc-standard', it
  will construct two training targets, one for the tagger and one for the
  arc-standard parser.

  Arguments:
    spec: DRAGNN spec.

  Returns:
    List of TrainTarget protos.
  """
  component_targets = [
      spec_pb2.TrainTarget(
          name=component.name,
          max_index=idx + 1,
          unroll_using_oracle=[False] * idx + [True])
      for idx, component in enumerate(spec.component)
      if not component.transition_system.registered_name.endswith('shift-only')
  ]
  return component_targets
