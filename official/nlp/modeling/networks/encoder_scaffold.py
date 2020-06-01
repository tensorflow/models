# Lint as: python3
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
"""Transformer-based text encoder network."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import inspect

import gin
import tensorflow as tf

from official.nlp.modeling import layers


@tf.keras.utils.register_keras_serializable(package='Text')
@gin.configurable
class EncoderScaffold(tf.keras.Model):
  """Bi-directional Transformer-based encoder network scaffold.

  This network allows users to flexibly implement an encoder similar to the one
  described in "BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding" (https://arxiv.org/abs/1810.04805).

  In this network, users can choose to provide a custom embedding subnetwork
  (which will replace the standard embedding logic) and/or a custom hidden layer
  class (which will replace the Transformer instantiation in the encoder). For
  each of these custom injection points, users can pass either a class or a
  class instance. If a class is passed, that class will be instantiated using
  the 'embedding_cfg' or 'hidden_cfg' argument, respectively; if an instance
  is passed, that instance will be invoked. (In the case of hidden_cls, the
  instance will be invoked 'num_hidden_instances' times.

  If the hidden_cls is not overridden, a default transformer layer will be
  instantiated.

  Arguments:
    pooled_output_dim: The dimension of pooled output.
    pooler_layer_initializer: The initializer for the classification
      layer.
    embedding_cls: The class or instance to use to embed the input data. This
      class or instance defines the inputs to this encoder and outputs
      (1) embeddings tensor with shape [batch_size, seq_length, hidden_size] and
      (2) attention masking with tensor [batch_size, seq_length, seq_length].
      If embedding_cls is not set, a default embedding network
      (from the original BERT paper) will be created.
    embedding_cfg: A dict of kwargs to pass to the embedding_cls, if it needs to
      be instantiated. If embedding_cls is not set, a config dict must be
      passed to 'embedding_cfg' with the following values:
      "vocab_size": The size of the token vocabulary.
      "type_vocab_size": The size of the type vocabulary.
      "hidden_size": The hidden size for this encoder.
      "max_seq_length": The maximum sequence length for this encoder.
      "seq_length": The sequence length for this encoder.
      "initializer": The initializer for the embedding portion of this encoder.
      "dropout_rate": The dropout rate to apply before the encoding layers.
    embedding_data: A reference to the embedding weights that will be used to
      train the masked language model, if necessary. This is optional, and only
      needed if (1) you are overriding embedding_cls and (2) are doing standard
      pretraining.
    num_hidden_instances: The number of times to instantiate and/or invoke the
      hidden_cls.
    hidden_cls: The class or instance to encode the input data. If hidden_cls is
      not set, a KerasBERT transformer layer will be used as the encoder class.
    hidden_cfg: A dict of kwargs to pass to the hidden_cls, if it needs to be
      instantiated. If hidden_cls is not set, a config dict must be passed to
      'hidden_cfg' with the following values:
        "num_attention_heads": The number of attention heads. The hidden size
          must be divisible by num_attention_heads.
        "intermediate_size": The intermediate size of the transformer.
        "intermediate_activation": The activation to apply in the transfomer.
        "dropout_rate": The overall dropout rate for the transformer layers.
        "attention_dropout_rate": The dropout rate for the attention layers.
        "kernel_initializer": The initializer for the transformer layers.
    return_all_layer_outputs: Whether to output sequence embedding outputs of
      all encoder transformer layers.
  """

  def __init__(
      self,
      pooled_output_dim,
      pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(
          stddev=0.02),
      embedding_cls=None,
      embedding_cfg=None,
      embedding_data=None,
      num_hidden_instances=1,
      hidden_cls=layers.Transformer,
      hidden_cfg=None,
      return_all_layer_outputs=False,
      **kwargs):
    self._self_setattr_tracking = False
    self._hidden_cls = hidden_cls
    self._hidden_cfg = hidden_cfg
    self._num_hidden_instances = num_hidden_instances
    self._pooled_output_dim = pooled_output_dim
    self._pooler_layer_initializer = pooler_layer_initializer
    self._embedding_cls = embedding_cls
    self._embedding_cfg = embedding_cfg
    self._embedding_data = embedding_data
    self._return_all_layer_outputs = return_all_layer_outputs
    self._kwargs = kwargs

    if embedding_cls:
      if inspect.isclass(embedding_cls):
        self._embedding_network = embedding_cls(
            **embedding_cfg) if embedding_cfg else embedding_cls()
      else:
        self._embedding_network = embedding_cls
      inputs = self._embedding_network.inputs
      embeddings, attention_mask = self._embedding_network(inputs)
    else:
      self._embedding_network = None
      word_ids = tf.keras.layers.Input(
          shape=(embedding_cfg['seq_length'],),
          dtype=tf.int32,
          name='input_word_ids')
      mask = tf.keras.layers.Input(
          shape=(embedding_cfg['seq_length'],),
          dtype=tf.int32,
          name='input_mask')
      type_ids = tf.keras.layers.Input(
          shape=(embedding_cfg['seq_length'],),
          dtype=tf.int32,
          name='input_type_ids')
      inputs = [word_ids, mask, type_ids]

      self._embedding_layer = layers.OnDeviceEmbedding(
          vocab_size=embedding_cfg['vocab_size'],
          embedding_width=embedding_cfg['hidden_size'],
          initializer=embedding_cfg['initializer'],
          name='word_embeddings')

      word_embeddings = self._embedding_layer(word_ids)

      # Always uses dynamic slicing for simplicity.
      self._position_embedding_layer = layers.PositionEmbedding(
          initializer=embedding_cfg['initializer'],
          use_dynamic_slicing=True,
          max_sequence_length=embedding_cfg['max_seq_length'],
          name='position_embedding')
      position_embeddings = self._position_embedding_layer(word_embeddings)

      type_embeddings = (
          layers.OnDeviceEmbedding(
              vocab_size=embedding_cfg['type_vocab_size'],
              embedding_width=embedding_cfg['hidden_size'],
              initializer=embedding_cfg['initializer'],
              use_one_hot=True,
              name='type_embeddings')(type_ids))

      embeddings = tf.keras.layers.Add()(
          [word_embeddings, position_embeddings, type_embeddings])
      embeddings = (
          tf.keras.layers.LayerNormalization(
              name='embeddings/layer_norm',
              axis=-1,
              epsilon=1e-12,
              dtype=tf.float32)(embeddings))
      embeddings = (
          tf.keras.layers.Dropout(
              rate=embedding_cfg['dropout_rate'])(embeddings))

      attention_mask = layers.SelfAttentionMask()([embeddings, mask])

    data = embeddings

    layer_output_data = []
    self._hidden_layers = []
    for _ in range(num_hidden_instances):
      if inspect.isclass(hidden_cls):
        layer = hidden_cls(**hidden_cfg) if hidden_cfg else hidden_cls()
      else:
        layer = hidden_cls
      data = layer([data, attention_mask])
      layer_output_data.append(data)
      self._hidden_layers.append(layer)

    first_token_tensor = (
        tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(
            layer_output_data[-1]))
    self._pooler_layer = tf.keras.layers.Dense(
        units=pooled_output_dim,
        activation='tanh',
        kernel_initializer=pooler_layer_initializer,
        name='cls_transform')
    cls_output = self._pooler_layer(first_token_tensor)

    if return_all_layer_outputs:
      outputs = [layer_output_data, cls_output]
    else:
      outputs = [layer_output_data[-1], cls_output]

    super(EncoderScaffold, self).__init__(
        inputs=inputs, outputs=outputs, **kwargs)

  def get_config(self):
    config_dict = {
        'num_hidden_instances':
            self._num_hidden_instances,
        'pooled_output_dim':
            self._pooled_output_dim,
        'pooler_layer_initializer':
            self._pooler_layer_initializer,
        'embedding_cls':
            self._embedding_network,
        'embedding_cfg':
            self._embedding_cfg,
        'hidden_cfg':
            self._hidden_cfg,
        'return_all_layer_outputs':
            self._return_all_layer_outputs,
    }
    if inspect.isclass(self._hidden_cls):
      config_dict['hidden_cls_string'] = tf.keras.utils.get_registered_name(
          self._hidden_cls)
    else:
      config_dict['hidden_cls'] = self._hidden_cls

    config_dict.update(self._kwargs)
    return config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    if 'hidden_cls_string' in config:
      config['hidden_cls'] = tf.keras.utils.get_registered_object(
          config['hidden_cls_string'], custom_objects=custom_objects)
      del config['hidden_cls_string']
    return cls(**config)

  def get_embedding_table(self):
    if self._embedding_network is None:
      # In this case, we don't have a custom embedding network and can return
      # the standard embedding data.
      return self._embedding_layer.embeddings

    if self._embedding_data is None:
      raise RuntimeError(('The EncoderScaffold %s does not have a reference '
                          'to the embedding data. This is required when you '
                          'pass a custom embedding network to the scaffold. '
                          'It is also possible that you are trying to get '
                          'embedding data from an embedding scaffold with a '
                          'custom embedding network where the scaffold has '
                          'been serialized and deserialized. Unfortunately, '
                          'accessing custom embedding references after '
                          'serialization is not yet supported.') % self.name)
    else:
      return self._embedding_data

  @property
  def hidden_layers(self):
    """List of hidden layers in the encoder."""
    return self._hidden_layers

  @property
  def pooler_layer(self):
    """The pooler dense layer after the transformer layers."""
    return self._pooler_layer
