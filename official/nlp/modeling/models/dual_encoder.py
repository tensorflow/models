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
"""Trainer network for dual encoder style models."""
# pylint: disable=g-classes-have-attributes

import tensorflow as tf

from official.nlp.modeling import layers


@tf.keras.utils.register_keras_serializable(package='Text')
class DualEncoder(tf.keras.Model):
  """A dual encoder model based on a transformer-based encoder.

  This is an implementation of the dual encoder network structure based on the
  transfomer stack, as described in ["Language-agnostic BERT Sentence
  Embedding"](https://arxiv.org/abs/2007.01852)

  The DualEncoder allows a user to pass in a transformer stack, and build a dual
  encoder model based on the transformer stack.

  Arguments:
    network: A transformer network which should output an encoding output.
    max_seq_length: The maximum allowed sequence length for transformer.
    normalize: If set to True, normalize the encoding produced by transfomer.
    logit_scale: The scaling factor of dot products when doing training.
    logit_margin: The margin between positive and negative when doing training.
    output: The output style for this network. Can be either 'logits' or
      'predictions'. If set to 'predictions', it will output the embedding
      producted by transformer network.
  """

  def __init__(self,
               network: tf.keras.Model,
               max_seq_length: int = 32,
               normalize: bool = True,
               logit_scale: float = 1.0,
               logit_margin: float = 0.0,
               output: str = 'logits',
               **kwargs) -> None:
    self._self_setattr_tracking = False
    self._config = {
        'network': network,
        'max_seq_length': max_seq_length,
        'normalize': normalize,
        'logit_scale': logit_scale,
        'logit_margin': logit_margin,
        'output': output,
    }

    self.network = network

    if output == 'logits':
      left_word_ids = tf.keras.layers.Input(
          shape=(max_seq_length,), dtype=tf.int32, name='left_word_ids')
      left_mask = tf.keras.layers.Input(
          shape=(max_seq_length,), dtype=tf.int32, name='left_mask')
      left_type_ids = tf.keras.layers.Input(
          shape=(max_seq_length,), dtype=tf.int32, name='left_type_ids')
    else:
      # Keep the consistant with legacy BERT hub module input names.
      left_word_ids = tf.keras.layers.Input(
          shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
      left_mask = tf.keras.layers.Input(
          shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
      left_type_ids = tf.keras.layers.Input(
          shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

    left_inputs = [left_word_ids, left_mask, left_type_ids]
    left_outputs = network(left_inputs)
    if isinstance(left_outputs, list):
      left_sequence_output, left_encoded = left_outputs
    else:
      left_sequence_output = left_outputs['sequence_output']
      left_encoded = left_outputs['pooled_output']
    if normalize:
      left_encoded = tf.keras.layers.Lambda(
          lambda x: tf.nn.l2_normalize(x, axis=1))(
              left_encoded)

    if output == 'logits':
      right_word_ids = tf.keras.layers.Input(
          shape=(max_seq_length,), dtype=tf.int32, name='right_word_ids')
      right_mask = tf.keras.layers.Input(
          shape=(max_seq_length,), dtype=tf.int32, name='right_mask')
      right_type_ids = tf.keras.layers.Input(
          shape=(max_seq_length,), dtype=tf.int32, name='right_type_ids')

      right_inputs = [right_word_ids, right_mask, right_type_ids]
      right_outputs = network(right_inputs)
      if isinstance(right_outputs, list):
        _, right_encoded = right_outputs
      else:
        right_encoded = right_outputs['pooled_output']
      if normalize:
        right_encoded = tf.keras.layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1))(
                right_encoded)

      dot_products = layers.MatMulWithMargin(
          logit_scale=logit_scale,
          logit_margin=logit_margin,
          name='dot_product')

      inputs = [
          left_word_ids, left_mask, left_type_ids, right_word_ids, right_mask,
          right_type_ids
      ]
      left_logits, right_logits = dot_products(left_encoded, right_encoded)

      outputs = dict(left_logits=left_logits, right_logits=right_logits)

    elif output == 'predictions':
      inputs = [left_word_ids, left_mask, left_type_ids]

      # To keep consistent with legacy BERT hub modules, the outputs are
      # "pooled_output" and "sequence_output".
      outputs = dict(
          sequence_output=left_sequence_output, pooled_output=left_encoded)
    else:
      raise ValueError('output type %s is not supported' % output)

    super(DualEncoder, self).__init__(inputs=inputs, outputs=outputs, **kwargs)

    # Set _self_setattr_tracking to True so it can be exported with assets.
    self._self_setattr_tracking = True

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(encoder=self.network)
    return items
