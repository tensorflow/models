# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
import copy
from typing import List, Optional

import tensorflow as tf

from official.nlp.modeling import layers


@tf.keras.utils.register_keras_serializable(package='Text')
class MobileBERTEdgeTPUPretrainer(tf.keras.Model):
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

    inputs = copy.copy(encoder_network.inputs)
    outputs = {}
    encoder_network_outputs = encoder_network(inputs)
    if isinstance(encoder_network_outputs, list):
      outputs['pooled_output'] = encoder_network_outputs[1]
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

    masked_lm_positions = tf.keras.layers.Input(
        shape=(None,), name='masked_lm_positions', dtype=tf.int32)
    inputs.append(masked_lm_positions)
    masked_lm_layer = customized_masked_lm or layers.MaskedLM(
        embedding_table=encoder_network.get_embedding_table(),
        activation=mlm_activation,
        initializer=mlm_initializer,
        name='cls/predictions')
    sequence_output = outputs['sequence_output']
    outputs['mlm_logits'] = masked_lm_layer(
        sequence_output, masked_positions=masked_lm_positions)

    classification_head_layers = classification_heads or []
    for cls_head in classification_head_layers:
      cls_outputs = cls_head(sequence_output)
      if isinstance(cls_outputs, dict):
        outputs.update(cls_outputs)
      else:
        outputs[cls_head.name] = cls_outputs

    super(MobileBERTEdgeTPUPretrainer, self).__init__(
        inputs=inputs,
        outputs=outputs,
        name=name,
        **kwargs)

    self._config = {
        'encoder_network': encoder_network,
        'mlm_activation': mlm_activation,
        'mlm_initializer': mlm_initializer,
        'classification_heads': classification_heads,
        'customized_masked_lm': customized_masked_lm,
        'name': name,
    }

    self.encoder_network = encoder_network
    self.masked_lm = masked_lm_layer
    self.classification_heads = classification_head_layers

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
