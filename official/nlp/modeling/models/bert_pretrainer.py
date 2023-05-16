# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""BERT Pre-training model."""
# pylint: disable=g-classes-have-attributes
import collections
import copy
from typing import List, Optional

from absl import logging
import gin
import tensorflow as tf

from official.modeling import tf_utils
from official.nlp.modeling import layers
from official.nlp.modeling import networks


@tf.keras.utils.register_keras_serializable(package='Text')
class BertPretrainer(tf.keras.Model):
  """BERT pretraining model.

  [Note] Please use the new `BertPretrainerV2` for your projects.

  The BertPretrainer allows a user to pass in a transformer stack, and
  instantiates the masked language model and classification networks that are
  used to create the training objectives.

  *Note* that the model is constructed by
  [Keras Functional API](https://keras.io/guides/functional_api/).

  Args:
    network: A transformer network. This network should output a sequence output
      and a classification output.
    num_classes: Number of classes to predict from the classification network.
    num_token_predictions: Number of tokens to predict from the masked LM.
    embedding_table: Embedding table of a network. If None, the
      "network.get_embedding_table()" is used.
    activation: The activation (if any) to use in the masked LM network. If
      None, no activation will be used.
    initializer: The initializer (if any) to use in the masked LM and
      classification networks. Defaults to a Glorot uniform initializer.
    output: The output style for this network. Can be either `logits` or
      `predictions`.
  """

  def __init__(self,
               network,
               num_classes,
               num_token_predictions,
               embedding_table=None,
               activation=None,
               initializer='glorot_uniform',
               output='logits',
               **kwargs):

    # We want to use the inputs of the passed network as the inputs to this
    # Model. To do this, we need to keep a copy of the network inputs for use
    # when we construct the Model object at the end of init. (We keep a copy
    # because we'll be adding another tensor to the copy later.)
    network_inputs = network.inputs
    inputs = copy.copy(network_inputs)

    # Because we have a copy of inputs to create this Model object, we can
    # invoke the Network object with its own input tensors to start the Model.
    # Note that, because of how deferred construction happens, we can't use
    # the copy of the list here - by the time the network is invoked, the list
    # object contains the additional input added below.
    sequence_output, cls_output = network(network_inputs)

    # The encoder network may get outputs from all layers.
    if isinstance(sequence_output, list):
      sequence_output = sequence_output[-1]
    if isinstance(cls_output, list):
      cls_output = cls_output[-1]
    sequence_output_length = sequence_output.shape.as_list()[1]
    if sequence_output_length is not None and (sequence_output_length <
                                               num_token_predictions):
      raise ValueError(
          "The passed network's output length is %s, which is less than the "
          'requested num_token_predictions %s.' %
          (sequence_output_length, num_token_predictions))

    masked_lm_positions = tf.keras.layers.Input(
        shape=(num_token_predictions,),
        name='masked_lm_positions',
        dtype=tf.int32)
    inputs.append(masked_lm_positions)

    if embedding_table is None:
      embedding_table = network.get_embedding_table()
    masked_lm = layers.MaskedLM(
        embedding_table=embedding_table,
        activation=activation,
        initializer=tf_utils.clone_initializer(initializer),
        output=output,
        name='cls/predictions')
    lm_outputs = masked_lm(
        sequence_output, masked_positions=masked_lm_positions)

    classification = networks.Classification(
        input_width=cls_output.shape[-1],
        num_classes=num_classes,
        initializer=tf_utils.clone_initializer(initializer),
        output=output,
        name='classification')
    sentence_outputs = classification(cls_output)

    super(BertPretrainer, self).__init__(
        inputs=inputs,
        outputs=dict(masked_lm=lm_outputs, classification=sentence_outputs),
        **kwargs)

    # b/164516224
    # Once we've created the network using the Functional API, we call
    # super().__init__ as though we were invoking the Functional API Model
    # constructor, resulting in this object having all the properties of a model
    # created using the Functional API. Once super().__init__ is called, we
    # can assign attributes to `self` - note that all `self` assignments are
    # below this line.
    config_dict = {
        'network': network,
        'num_classes': num_classes,
        'num_token_predictions': num_token_predictions,
        'activation': activation,
        'initializer': initializer,
        'output': output,
    }

    # We are storing the config dict as a namedtuple here to ensure checkpoint
    # compatibility with an earlier version of this model which did not track
    # the config dict attribute. TF does not track immutable attrs which
    # do not contain Trackables, so by creating a config namedtuple instead of
    # a dict we avoid tracking it.
    config_cls = collections.namedtuple('Config', config_dict.keys())
    self._config = config_cls(**config_dict)

    self.encoder = network
    self.classification = classification
    self.masked_lm = masked_lm

  def get_config(self):
    return dict(self._config._asdict())

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


@tf.keras.utils.register_keras_serializable(package='Text')
@gin.configurable
class BertPretrainerV2(tf.keras.Model):
  """BERT pretraining model V2.

  Adds the masked language model head and optional classification heads upon the
  transformer encoder.

  Args:
    encoder_network: A transformer network. This network should output a
      sequence output and a classification output.
    mlm_activation: The activation (if any) to use in the masked LM network. If
      None, no activation will be used.
    mlm_initializer: The initializer (if any) to use in the masked LM. Default
      to a Glorot uniform initializer.
    classification_heads: A list of optional head layers to transform on encoder
      sequence outputs.
    customized_masked_lm: A customized masked_lm layer. If None, will create
      a standard layer from `layers.MaskedLM`; if not None, will use the
      specified masked_lm layer. Above arguments `mlm_activation` and
      `mlm_initializer` will be ignored.
    name: The name of the model.
  Inputs: Inputs defined by the encoder network, plus `masked_lm_positions` as a
    dictionary.
  Outputs: A dictionary of `lm_output`, classification head outputs keyed by
    head names, and also outputs from `encoder_network`, keyed by
    `sequence_output` and `encoder_outputs` (if any).
  """

  def __init__(
      self,
      encoder_network: tf.keras.Model,
      mlm_activation=None,
      mlm_initializer='glorot_uniform',
      classification_heads: Optional[List[tf.keras.layers.Layer]] = None,
      customized_masked_lm: Optional[tf.keras.layers.Layer] = None,
      name: str = 'bert',
      **kwargs):
    super().__init__(self, name=name, **kwargs)
    self._config = {
        'encoder_network': encoder_network,
        'mlm_initializer': mlm_initializer,
        'mlm_activation': mlm_activation,
        'classification_heads': classification_heads,
        'name': name,
    }
    self.encoder_network = encoder_network
    # Makes sure the weights are built.
    _ = self.encoder_network(self.encoder_network.inputs)
    inputs = copy.copy(self.encoder_network.inputs)
    self.classification_heads = classification_heads or []
    if len(set([cls.name for cls in self.classification_heads])) != len(
        self.classification_heads):
      raise ValueError('Classification heads should have unique names.')

    self.masked_lm = customized_masked_lm or layers.MaskedLM(
        embedding_table=self.encoder_network.get_embedding_table(),
        activation=mlm_activation,
        initializer=mlm_initializer,
        name='cls/predictions')
    masked_lm_positions = tf.keras.layers.Input(
        shape=(None,), name='masked_lm_positions', dtype=tf.int32)
    if isinstance(inputs, dict):
      inputs['masked_lm_positions'] = masked_lm_positions
    else:
      inputs.append(masked_lm_positions)
    self.inputs = inputs

  def call(self, inputs):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    if isinstance(inputs, list):
      logging.warning('List inputs to BertPretrainer are discouraged.')
      inputs = dict([
          (ref.name, tensor) for ref, tensor in zip(self.inputs, inputs)
      ])

    outputs = dict()
    encoder_network_outputs = self.encoder_network(inputs)
    if isinstance(encoder_network_outputs, list):
      outputs['pooled_output'] = encoder_network_outputs[1]
      # When `encoder_network` was instantiated with return_all_encoder_outputs
      # set to True, `encoder_network_outputs[0]` is a list containing
      # all transformer layers' output.
      if isinstance(encoder_network_outputs[0], list):
        outputs['encoder_outputs'] = encoder_network_outputs[0]
        outputs['sequence_output'] = encoder_network_outputs[0][-1]
      else:
        outputs['sequence_output'] = encoder_network_outputs[0]
    elif isinstance(encoder_network_outputs, dict):
      outputs = encoder_network_outputs
    else:
      raise ValueError('encoder_network\'s output should be either a list '
                       'or a dict, but got %s' % encoder_network_outputs)
    sequence_output = outputs['sequence_output']
    # Inference may not have masked_lm_positions and mlm_logits is not needed.
    if 'masked_lm_positions' in inputs:
      masked_lm_positions = inputs['masked_lm_positions']
      outputs['mlm_logits'] = self.masked_lm(
          sequence_output, masked_positions=masked_lm_positions)
    for cls_head in self.classification_heads:
      cls_outputs = cls_head(sequence_output)
      if isinstance(cls_outputs, dict):
        outputs.update(cls_outputs)
      else:
        outputs[cls_head.name] = cls_outputs
    return outputs

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(encoder=self.encoder_network, masked_lm=self.masked_lm)
    for head in self.classification_heads:
      for key, item in head.checkpoint_items.items():
        items['.'.join([head.name, key])] = item
    return items

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
