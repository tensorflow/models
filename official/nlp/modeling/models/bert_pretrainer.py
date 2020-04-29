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
"""Trainer network for BERT-style models."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import copy
import tensorflow as tf

from official.nlp.modeling import networks


@tf.keras.utils.register_keras_serializable(package='Text')
class BertPretrainer(tf.keras.Model):
  """BERT network training model.

  This is an implementation of the network structure surrounding a transformer
  encoder as described in "BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding" (https://arxiv.org/abs/1810.04805).

  The BertPretrainer allows a user to pass in a transformer stack, and
  instantiates the masked language model and classification networks that are
  used to create the training objectives.

  Arguments:
    network: A transformer network. This network should output a sequence output
      and a classification output.
    num_classes: Number of classes to predict from the classification network.
    num_token_predictions: Number of tokens to predict from the masked LM.
    embedding_table: Embedding table of a network. If None, the
      "network.get_embedding_table()" is used.
    activation: The activation (if any) to use in the masked LM and
      classification networks. If None, no activation will be used.
    initializer: The initializer (if any) to use in the masked LM and
      classification networks. Defaults to a Glorot uniform initializer.
    output: The output style for this network. Can be either 'logits' or
      'predictions'.
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
    self._self_setattr_tracking = False
    self._config = {
        'network': network,
        'num_classes': num_classes,
        'num_token_predictions': num_token_predictions,
        'activation': activation,
        'initializer': initializer,
        'output': output,
    }

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

    sequence_output_length = sequence_output.shape.as_list()[1]
    if sequence_output_length < num_token_predictions:
      raise ValueError(
          "The passed network's output length is %s, which is less than the "
          'requested num_token_predictions %s.' %
          (sequence_output_length, num_token_predictions))

    masked_lm_positions = tf.keras.layers.Input(
        shape=(num_token_predictions,),
        name='masked_lm_positions',
        dtype=tf.int32)
    inputs.append(masked_lm_positions)

    self.masked_lm = networks.MaskedLM(
        num_predictions=num_token_predictions,
        input_width=sequence_output.shape[-1],
        source_network=network,
        embedding_table=embedding_table,
        activation=activation,
        initializer=initializer,
        output=output,
        name='masked_lm')
    lm_outputs = self.masked_lm([sequence_output, masked_lm_positions])

    self.classification = networks.Classification(
        input_width=cls_output.shape[-1],
        num_classes=num_classes,
        initializer=initializer,
        output=output,
        name='classification')
    sentence_outputs = self.classification(cls_output)

    super(BertPretrainer, self).__init__(
        inputs=inputs, outputs=[lm_outputs, sentence_outputs], **kwargs)

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
