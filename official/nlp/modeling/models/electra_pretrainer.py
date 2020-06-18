# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import sys
import copy
import tensorflow as tf
from official.nlp.modeling import networks
from official.modeling import tf_utils

@tf.keras.utils.register_keras_serializable(package='Text')
class ElectraPretrainer(tf.keras.Model):
  """BERT network training model.

  This is an implementation of the network structure surrounding a transformer
  encoder as described in "BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding" (https://arxiv.org/abs/1810.04805).

  The BertPretrainer allows a user to pass in a transformer stack, and
  instantiates the masked language model and classification networks that are
  used to create the training objectives.

  Arguments:
    network: A transformer network. This network should output a sequence output
      and a classification output. Furthermore, it should expose its embedding
      table via a "get_embedding_table" method.
    num_classes: Number of classes to predict from the classification network.
    num_token_predictions: Number of tokens to predict from the masked LM.
    activation: The activation (if any) to use in the masked LM and
      classification networks. If None, no activation will be used.
    initializer: The initializer (if any) to use in the masked LM and
      classification networks. Defaults to a Glorot uniform initializer.
    output: The output style for this network. Can be either 'logits' or
      'predictions'.
  """

  def __init__(self,
               network,
               discriminator,
               num_classes,
               num_token_predictions,
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
        activation=activation,
        initializer=initializer,
        output='logits',
        name='generator')
    lm_outputs = tf.identity(
        self.masked_lm([sequence_output, masked_lm_positions]),
        "mlm_logits")

    #Generates fake data to train discriminator on
    fake_data, labels = _get_fake_data(inputs[0],
                                       lm_outputs, masked_lm_positions)

    other_output, other_cls_output = discriminator([fake_data,
                                                    inputs[1], inputs[2]])
    self.discrimnator = networks.Discriminator(
        input_width=other_output.shape[-1],
        source_network=discriminator,
        activation=activation,
        initializer=initializer,
        name='discriminator'
    )

    discrim_outputs = self.discrimnator(other_output)

    super(ElectraPretrainer, self).__init__(
        inputs=inputs, outputs=[lm_outputs, discrim_outputs, labels], **kwargs)

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

def _get_fake_data(orig_sent, predictions, maskedlm_ids):
  """Gets the corrupted sentence from generator outputs

  Args:
    orig_sent: A [batch_size, seq_length] or
        [batch_size, seq_length, depth] tensor of values
    predictions: A [batch_size, seq_length, vocab_size]
        or [batch_size, seq_length, depth, vocab_size] tensor from the generator
    maskedlm_ids: A [batch_size, n_positions] tensor of indices

  Returns:
     updatedids: A [batch_size, seq_length] or
        [batch_size, seq_length, depth] tensor that is the corrupted sentence
    labels: A tensor that is the same size as updatedids

  """
  tokids = tf.stop_gradient(tf.math.argmax(predictions, axis=-1,
                                           output_type=tf.dtypes.int32))
  updated_ids, mask = _scatter_update(orig_sent, maskedlm_ids, tokids)
  labels = mask * (1 - tf.cast(tf.equal(orig_sent, updated_ids), tf.int32))
  return updated_ids, labels

def _scatter_update(orig_sent, maskedlm_ids, tokids):
    """Returns a sentence that replaces certain masked words
        with words that generator deems likely

    Args:
      orig_sent: A [batch_size, seq_length] or
          [batch_size, seq_length, depth] tensor of values
      maskedlm_ids: A [batch_size, n_positions] tensor of indices
      tokids: A [batch_size, seq_length]
          or [batch_size, seq_length, depth] tensor
          that represents the most likely word at each position

    Returns:
       updated_sequence: a [batch_size, seq_len, depth] that
           represents the corrupted sentence
       updates_mask: a [batch_size, seq_len] tensor of
           which tokens have been masked

    """

    sequence_shape = tf_utils.get_shape_list(
        orig_sent, name='input_word_ids')
    batch, seq_length = sequence_shape
    positions = maskedlm_ids.shape[1]
    shift = tf.reshape(tf.range(0, batch, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(maskedlm_ids+shift, [-1, 1])
    flat_updates = tf.reshape(tokids, [-1])
    updates = tf.scatter_nd(flat_positions, flat_updates, [batch * seq_length])
    updates = tf.reshape(updates, [batch, seq_length])

    flat_updates_mask = tf.ones([batch * positions], tf.int32)
    updates_mask = tf.scatter_nd(flat_positions, flat_updates_mask,
                                 [batch * seq_length])
    updates_mask = tf.reshape(updates_mask, [batch, seq_length])
    not_first_token = tf.concat([tf.zeros((batch, 1), tf.int32),
                                 tf.ones((batch, seq_length - 1),
                                         tf.int32, name="otherones")], -1)
    updates_mask *= not_first_token
    updates = tf.math.floordiv(updates, tf.maximum(1, updates_mask))
    updates_mask = tf.minimum(updates_mask, 1, name='updates_mask')
    updated_sequence = (((1 - updates_mask) * orig_sent) +
                        (updates_mask * updates))
    return updated_sequence, updates_mask
