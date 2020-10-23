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
"""ALBERT (https://arxiv.org/abs/1810.04805) text encoder network."""
# pylint: disable=g-classes-have-attributes
import collections
import tensorflow as tf

from official.modeling import activations
from official.nlp import keras_nlp
from official.nlp.modeling import layers


@tf.keras.utils.register_keras_serializable(package='Text')
class AlbertEncoder(tf.keras.Model):
  """ALBERT (https://arxiv.org/abs/1810.04805) text encoder network.

  This network implements the encoder described in the paper "ALBERT: A Lite
  BERT for Self-supervised Learning of Language Representations"
  (https://arxiv.org/abs/1909.11942).

  Compared with BERT (https://arxiv.org/abs/1810.04805), ALBERT refactorizes
  embedding parameters into two smaller matrices and shares parameters
  across layers.

  The default values for this object are taken from the ALBERT-Base
  implementation described in the paper.

  *Note* that the network is constructed by Keras Functional API.

  Arguments:
    vocab_size: The size of the token vocabulary.
    embedding_width: The width of the word embeddings. If the embedding width is
      not equal to hidden size, embedding parameters will be factorized into two
      matrices in the shape of ['vocab_size', 'embedding_width'] and
      ['embedding_width', 'hidden_size'] ('embedding_width' is usually much
      smaller than 'hidden_size').
    hidden_size: The size of the transformer hidden layers.
    num_layers: The number of transformer layers.
    num_attention_heads: The number of attention heads for each transformer. The
      hidden size must be divisible by the number of attention heads.
    max_sequence_length: The maximum sequence length that this encoder can
      consume. If None, max_sequence_length uses the value from sequence length.
      This determines the variable shape for positional embeddings.
    type_vocab_size: The number of types that the 'type_ids' input can take.
    intermediate_size: The intermediate size for the transformer layers.
    activation: The activation to use for the transformer layers.
    dropout_rate: The dropout rate to use for the transformer layers.
    attention_dropout_rate: The dropout rate to use for the attention layers
      within the transformer layers.
    initializer: The initialzer to use for all weights in this encoder.
    dict_outputs: Whether to use a dictionary as the model outputs.
  """

  def __init__(self,
               vocab_size,
               embedding_width=128,
               hidden_size=768,
               num_layers=12,
               num_attention_heads=12,
               max_sequence_length=512,
               type_vocab_size=16,
               intermediate_size=3072,
               activation=activations.gelu,
               dropout_rate=0.1,
               attention_dropout_rate=0.1,
               initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
               dict_outputs=False,
               **kwargs):
    activation = tf.keras.activations.get(activation)
    initializer = tf.keras.initializers.get(initializer)

    word_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_word_ids')
    mask = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_mask')
    type_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_type_ids')

    if embedding_width is None:
      embedding_width = hidden_size
    embedding_layer = layers.OnDeviceEmbedding(
        vocab_size=vocab_size,
        embedding_width=embedding_width,
        initializer=initializer,
        name='word_embeddings')
    word_embeddings = embedding_layer(word_ids)

    # Always uses dynamic slicing for simplicity.
    position_embedding_layer = keras_nlp.layers.PositionEmbedding(
        initializer=initializer,
        max_length=max_sequence_length,
        name='position_embedding')
    position_embeddings = position_embedding_layer(word_embeddings)

    type_embeddings = (
        layers.OnDeviceEmbedding(
            vocab_size=type_vocab_size,
            embedding_width=embedding_width,
            initializer=initializer,
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
    embeddings = (tf.keras.layers.Dropout(rate=dropout_rate)(embeddings))
    # We project the 'embedding' output to 'hidden_size' if it is not already
    # 'hidden_size'.
    if embedding_width != hidden_size:
      embeddings = tf.keras.layers.experimental.EinsumDense(
          '...x,xy->...y',
          output_shape=hidden_size,
          bias_axes='y',
          kernel_initializer=initializer,
          name='embedding_projection')(
              embeddings)

    data = embeddings
    attention_mask = layers.SelfAttentionMask()([data, mask])
    shared_layer = keras_nlp.layers.TransformerEncoderBlock(
        num_attention_heads=num_attention_heads,
        inner_dim=intermediate_size,
        inner_activation=activation,
        output_dropout=dropout_rate,
        attention_dropout=attention_dropout_rate,
        kernel_initializer=initializer,
        name='transformer')
    encoder_outputs = []
    for _ in range(num_layers):
      data = shared_layer([data, attention_mask])
      encoder_outputs.append(data)

    # Applying a tf.slice op (through subscript notation) to a Keras tensor
    # like this will create a SliceOpLambda layer. This is better than a Lambda
    # layer with Python code, because that is fundamentally less portable.
    first_token_tensor = data[:, 0, :]
    cls_output = tf.keras.layers.Dense(
        units=hidden_size,
        activation='tanh',
        kernel_initializer=initializer,
        name='pooler_transform')(
            first_token_tensor)
    if dict_outputs:
      outputs = dict(
          sequence_output=data,
          encoder_outputs=encoder_outputs,
          pooled_output=cls_output,
      )
    else:
      outputs = [data, cls_output]

    # b/164516224
    # Once we've created the network using the Functional API, we call
    # super().__init__ as though we were invoking the Functional API Model
    # constructor, resulting in this object having all the properties of a model
    # created using the Functional API. Once super().__init__ is called, we
    # can assign attributes to `self` - note that all `self` assignments are
    # below this line.
    super(AlbertEncoder, self).__init__(
        inputs=[word_ids, mask, type_ids], outputs=outputs, **kwargs)
    config_dict = {
        'vocab_size': vocab_size,
        'embedding_width': embedding_width,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_attention_heads': num_attention_heads,
        'max_sequence_length': max_sequence_length,
        'type_vocab_size': type_vocab_size,
        'intermediate_size': intermediate_size,
        'activation': tf.keras.activations.serialize(activation),
        'dropout_rate': dropout_rate,
        'attention_dropout_rate': attention_dropout_rate,
        'initializer': tf.keras.initializers.serialize(initializer),
    }

    # We are storing the config dict as a namedtuple here to ensure checkpoint
    # compatibility with an earlier version of this model which did not track
    # the config dict attribute. TF does not track immutable attrs which
    # do not contain Trackables, so by creating a config namedtuple instead of
    # a dict we avoid tracking it.
    config_cls = collections.namedtuple('Config', config_dict.keys())
    self._config = config_cls(**config_dict)
    self._embedding_layer = embedding_layer
    self._position_embedding_layer = position_embedding_layer

  def get_embedding_table(self):
    return self._embedding_layer.embeddings

  def get_config(self):
    return dict(self._config._asdict())

  @classmethod
  def from_config(cls, config):
    return cls(**config)
