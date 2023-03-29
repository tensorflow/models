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

"""Transformer-based BERT encoder network."""
# pylint: disable=g-classes-have-attributes

from typing import Any, Callable, Optional, Union
from absl import logging
import tensorflow as tf

from official.modeling import tf_utils
from official.nlp.modeling import layers

_Initializer = Union[str, tf.keras.initializers.Initializer]
_Activation = Union[str, Callable[..., Any]]

_approx_gelu = lambda x: tf.keras.activations.gelu(x, approximate=True)


class BertEncoderV2(tf.keras.layers.Layer):
  """Bi-directional Transformer-based encoder network.

  This network implements a bi-directional Transformer-based encoder as
  described in "BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding" (https://arxiv.org/abs/1810.04805). It includes the
  embedding lookups and transformer layers, but not the masked language model
  or classification task networks.

  The default values for this object are taken from the BERT-Base implementation
  in "BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding".

  Args:
    vocab_size: The size of the token vocabulary.
    hidden_size: The size of the transformer hidden layers.
    num_layers: The number of transformer layers.
    num_attention_heads: The number of attention heads for each transformer. The
      hidden size must be divisible by the number of attention heads.
    max_sequence_length: The maximum sequence length that this encoder can
      consume. This determines the variable shape for positional embeddings.
    type_vocab_size: The number of types that the 'type_ids' input can take.
    inner_dim: The output dimension of the first Dense layer in a two-layer
      feedforward network for each transformer.
    inner_activation: The activation for the first Dense layer in a two-layer
      feedforward network for each transformer.
    output_dropout: Dropout probability for the post-attention and output
      dropout.
    attention_dropout: The dropout rate to use for the attention layers within
      the transformer layers.
    initializer: The initialzer to use for all weights in this encoder.
    output_range: The sequence output range, [0, output_range), by slicing the
      target sequence of the last transformer layer. `None` means the entire
      target sequence will attend to the source sequence, which yields the full
      output.
    embedding_width: The width of the word embeddings. If the embedding width is
      not equal to hidden size, embedding parameters will be factorized into two
      matrices in the shape of ['vocab_size', 'embedding_width'] and
      ['embedding_width', 'hidden_size'] ('embedding_width' is usually much
      smaller than 'hidden_size').
    embedding_layer: An optional Layer instance which will be called to generate
      embeddings for the input word IDs.
    norm_first: Whether to normalize inputs to attention and intermediate dense
      layers. If set False, output of attention and intermediate dense layers is
      normalized.
    with_dense_inputs: Whether to accept dense embeddings as the input.
    return_attention_scores: Whether to add an additional output containing the
      attention scores of all transformer layers. This will be a list of length
      `num_layers`, and each element will be in the shape [batch_size,
      num_attention_heads, seq_dim, seq_dim].
  """

  def __init__(
      self,
      vocab_size: int,
      hidden_size: int = 768,
      num_layers: int = 12,
      num_attention_heads: int = 12,
      max_sequence_length: int = 512,
      type_vocab_size: int = 16,
      inner_dim: int = 3072,
      inner_activation: _Activation = _approx_gelu,
      output_dropout: float = 0.1,
      attention_dropout: float = 0.1,
      initializer: _Initializer = tf.keras.initializers.TruncatedNormal(
          stddev=0.02),
      output_range: Optional[int] = None,
      embedding_width: Optional[int] = None,
      embedding_layer: Optional[tf.keras.layers.Layer] = None,
      norm_first: bool = False,
      with_dense_inputs: bool = False,
      return_attention_scores: bool = False,
      **kwargs):
    # Pops kwargs that are used in V1 implementation.
    if 'dict_outputs' in kwargs:
      kwargs.pop('dict_outputs')
    if 'return_all_encoder_outputs' in kwargs:
      kwargs.pop('return_all_encoder_outputs')
    if 'intermediate_size' in kwargs:
      inner_dim = kwargs.pop('intermediate_size')
    if 'activation' in kwargs:
      inner_activation = kwargs.pop('activation')
    if 'dropout_rate' in kwargs:
      output_dropout = kwargs.pop('dropout_rate')
    if 'attention_dropout_rate' in kwargs:
      attention_dropout = kwargs.pop('attention_dropout_rate')
    super().__init__(**kwargs)

    self._output_range = output_range

    activation = tf.keras.activations.get(inner_activation)
    initializer = tf.keras.initializers.get(initializer)

    if embedding_width is None:
      embedding_width = hidden_size

    if embedding_layer is None:
      self._embedding_layer = layers.OnDeviceEmbedding(
          vocab_size=vocab_size,
          embedding_width=embedding_width,
          initializer=tf_utils.clone_initializer(initializer),
          name='word_embeddings')
    else:
      self._embedding_layer = embedding_layer

    self._position_embedding_layer = layers.PositionEmbedding(
        initializer=tf_utils.clone_initializer(initializer),
        max_length=max_sequence_length,
        name='position_embedding')

    self._type_embedding_layer = layers.OnDeviceEmbedding(
        vocab_size=type_vocab_size,
        embedding_width=embedding_width,
        initializer=tf_utils.clone_initializer(initializer),
        use_one_hot=True,
        name='type_embeddings')

    self._embedding_norm_layer = tf.keras.layers.LayerNormalization(
        name='embeddings/layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32)

    self._embedding_dropout = tf.keras.layers.Dropout(
        rate=output_dropout, name='embedding_dropout')

    # We project the 'embedding' output to 'hidden_size' if it is not already
    # 'hidden_size'.
    self._embedding_projection = None
    if embedding_width != hidden_size:
      self._embedding_projection = tf.keras.layers.EinsumDense(
          '...x,xy->...y',
          output_shape=hidden_size,
          bias_axes='y',
          kernel_initializer=tf_utils.clone_initializer(initializer),
          name='embedding_projection')

    self._transformer_layers = []
    self._attention_mask_layer = layers.SelfAttentionMask(
        name='self_attention_mask')
    self._num_layers = num_layers
    for i in range(num_layers):
      layer = layers.TransformerEncoderBlock(
          num_attention_heads=num_attention_heads,
          inner_dim=inner_dim,
          inner_activation=inner_activation,
          output_dropout=output_dropout,
          attention_dropout=attention_dropout,
          norm_first=norm_first,
          return_attention_scores=return_attention_scores,
          kernel_initializer=tf_utils.clone_initializer(initializer),
          name='transformer/layer_%d' % i)
      self._transformer_layers.append(layer)

    self._pooler_layer = tf.keras.layers.Dense(
        units=hidden_size,
        activation='tanh',
        kernel_initializer=tf_utils.clone_initializer(initializer),
        name='pooler_transform')

    self._config = {
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_attention_heads': num_attention_heads,
        'max_sequence_length': max_sequence_length,
        'type_vocab_size': type_vocab_size,
        'inner_dim': inner_dim,
        'inner_activation': tf_utils.serialize_activation(
            activation, use_legacy_format=True
        ),
        'output_dropout': output_dropout,
        'attention_dropout': attention_dropout,
        'initializer': tf_utils.serialize_initializer(
            initializer, use_legacy_format=True
        ),
        'output_range': output_range,
        'embedding_width': embedding_width,
        'embedding_layer': embedding_layer,
        'norm_first': norm_first,
        'with_dense_inputs': with_dense_inputs,
        'return_attention_scores': return_attention_scores,
    }
    if with_dense_inputs:
      self.inputs = dict(
          input_word_ids=tf.keras.Input(shape=(None,), dtype=tf.int32),
          input_mask=tf.keras.Input(shape=(None,), dtype=tf.int32),
          input_type_ids=tf.keras.Input(shape=(None,), dtype=tf.int32),
          dense_inputs=tf.keras.Input(
              shape=(None, embedding_width), dtype=tf.float32),
          dense_mask=tf.keras.Input(shape=(None,), dtype=tf.int32),
          dense_type_ids=tf.keras.Input(shape=(None,), dtype=tf.int32),
      )
    else:
      self.inputs = dict(
          input_word_ids=tf.keras.Input(shape=(None,), dtype=tf.int32),
          input_mask=tf.keras.Input(shape=(None,), dtype=tf.int32),
          input_type_ids=tf.keras.Input(shape=(None,), dtype=tf.int32))

  def call(self, inputs):
    word_embeddings = None
    if isinstance(inputs, dict):
      word_ids = inputs.get('input_word_ids')
      mask = inputs.get('input_mask')
      type_ids = inputs.get('input_type_ids')
      word_embeddings = inputs.get('input_word_embeddings', None)

      dense_inputs = inputs.get('dense_inputs', None)
      dense_mask = inputs.get('dense_mask', None)
      dense_type_ids = inputs.get('dense_type_ids', None)
    else:
      raise ValueError('Unexpected inputs type to %s.' % self.__class__)

    if word_embeddings is None:
      word_embeddings = self._embedding_layer(word_ids)

    if dense_inputs is not None:
      mask = tf.concat([mask, dense_mask], axis=1)

    embeddings = self._get_embeddings(word_ids, type_ids, word_embeddings,
                                      dense_inputs, dense_type_ids)
    embeddings = self._embedding_norm_layer(embeddings)
    embeddings = self._embedding_dropout(embeddings)

    if self._embedding_projection is not None:
      embeddings = self._embedding_projection(embeddings)

    attention_mask = self._attention_mask_layer(embeddings, mask)

    encoder_outputs = []
    attention_outputs = []
    x = embeddings
    for i, layer in enumerate(self._transformer_layers):
      transformer_output_range = None
      if i == self._num_layers - 1:
        transformer_output_range = self._output_range
      x = layer([x, attention_mask], output_range=transformer_output_range)
      if self._config['return_attention_scores']:
        x, attention_scores = x
        attention_outputs.append(attention_scores)
      encoder_outputs.append(x)

    last_encoder_output = encoder_outputs[-1]
    first_token_tensor = last_encoder_output[:, 0, :]
    pooled_output = self._pooler_layer(first_token_tensor)

    output = dict(
        sequence_output=encoder_outputs[-1],
        pooled_output=pooled_output,
        encoder_outputs=encoder_outputs)
    if self._config['return_attention_scores']:
      output['attention_scores'] = attention_outputs
    return output

  def get_embedding_table(self):
    return self._embedding_layer.embeddings

  def get_embedding_layer(self):
    return self._embedding_layer

  def get_config(self):
    return dict(self._config)

  @property
  def transformer_layers(self):
    """List of Transformer layers in the encoder."""
    return self._transformer_layers

  @property
  def pooler_layer(self):
    """The pooler dense layer after the transformer layers."""
    return self._pooler_layer

  @classmethod
  def from_config(cls, config, custom_objects=None):
    if 'embedding_layer' in config and config['embedding_layer'] is not None:
      warn_string = (
          'You are reloading a model that was saved with a '
          'potentially-shared embedding layer object. If you contine to '
          'train this model, the embedding layer will no longer be shared. '
          'To work around this, load the model outside of the Keras API.')
      print('WARNING: ' + warn_string)
      logging.warn(warn_string)

    return cls(**config)

  def _get_embeddings(self, word_ids: tf.Tensor, type_ids: tf.Tensor,
                      word_embeddings: Optional[tf.Tensor],
                      dense_inputs: Optional[tf.Tensor],
                      dense_type_ids: Optional[tf.Tensor]) -> tf.Tensor:
    if word_embeddings is None:
      word_embeddings = self._embedding_layer(word_ids)

    if dense_inputs is not None:
      # Concat the dense embeddings at sequence end.
      word_embeddings = tf.concat([word_embeddings, dense_inputs], axis=1)
      type_ids = tf.concat([type_ids, dense_type_ids], axis=1)

    type_embeddings = self._type_embedding_layer(type_ids)

    # absolute position embeddings.
    position_embeddings = self._position_embedding_layer(word_embeddings)
    return word_embeddings + position_embeddings + type_embeddings


@tf.keras.utils.register_keras_serializable(package='Text')
class BertEncoder(tf.keras.Model):
  """Bi-directional Transformer-based encoder network.

  This network implements a bi-directional Transformer-based encoder as
  described in "BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding" (https://arxiv.org/abs/1810.04805). It includes the
  embedding lookups and transformer layers, but not the masked language model
  or classification task networks.

  The default values for this object are taken from the BERT-Base implementation
  in "BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding".

  *Note* that the network is constructed by
  [Keras Functional API](https://keras.io/guides/functional_api/).

  Args:
    vocab_size: The size of the token vocabulary.
    hidden_size: The size of the transformer hidden layers.
    num_layers: The number of transformer layers.
    num_attention_heads: The number of attention heads for each transformer. The
      hidden size must be divisible by the number of attention heads.
    max_sequence_length: The maximum sequence length that this encoder can
      consume. If None, max_sequence_length uses the value from sequence length.
      This determines the variable shape for positional embeddings.
    type_vocab_size: The number of types that the 'type_ids' input can take.
    inner_dim: The output dimension of the first Dense layer in a two-layer
      feedforward network for each transformer.
    inner_activation: The activation for the first Dense layer in a two-layer
      feedforward network for each transformer.
    output_dropout: Dropout probability for the post-attention and output
      dropout.
    attention_dropout: The dropout rate to use for the attention layers within
      the transformer layers.
    initializer: The initialzer to use for all weights in this encoder.
    output_range: The sequence output range, [0, output_range), by slicing the
      target sequence of the last transformer layer. `None` means the entire
      target sequence will attend to the source sequence, which yields the full
      output.
    embedding_width: The width of the word embeddings. If the embedding width is
      not equal to hidden size, embedding parameters will be factorized into two
      matrices in the shape of ['vocab_size', 'embedding_width'] and
      ['embedding_width', 'hidden_size'] ('embedding_width' is usually much
      smaller than 'hidden_size').
    embedding_layer: An optional Layer instance which will be called to generate
      embeddings for the input word IDs.
    norm_first: Whether to normalize inputs to attention and intermediate dense
      layers. If set False, output of attention and intermediate dense layers is
      normalized.
    dict_outputs: Whether to use a dictionary as the model outputs.
    return_all_encoder_outputs: Whether to output sequence embedding outputs of
      all encoder transformer layers. Note: when the following `dict_outputs`
      argument is True, all encoder outputs are always returned in the dict,
      keyed by `encoder_outputs`.
    return_attention_scores: Whether to add an additional output containing the
      attention scores of all transformer layers. This will be a list of length
      `num_layers`, and each element will be in the shape [batch_size,
      num_attention_heads, seq_dim, seq_dim].
  """

  def __init__(
      self,
      vocab_size,
      hidden_size=768,
      num_layers=12,
      num_attention_heads=12,
      max_sequence_length=512,
      type_vocab_size=16,
      inner_dim=3072,
      inner_activation=lambda x: tf.keras.activations.gelu(x, approximate=True),
      output_dropout=0.1,
      attention_dropout=0.1,
      initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
      output_range=None,
      embedding_width=None,
      embedding_layer=None,
      norm_first=False,
      dict_outputs=False,
      return_all_encoder_outputs=False,
      return_attention_scores: bool = False,
      **kwargs):
    if 'sequence_length' in kwargs:
      kwargs.pop('sequence_length')
      logging.warning('`sequence_length` is a deprecated argument to '
                      '`BertEncoder`, which has no effect for a while. Please '
                      'remove `sequence_length` argument.')

    # Handles backward compatible kwargs.
    if 'intermediate_size' in kwargs:
      inner_dim = kwargs.pop('intermediate_size')

    if 'activation' in kwargs:
      inner_activation = kwargs.pop('activation')

    if 'dropout_rate' in kwargs:
      output_dropout = kwargs.pop('dropout_rate')

    if 'attention_dropout_rate' in kwargs:
      attention_dropout = kwargs.pop('attention_dropout_rate')

    activation = tf.keras.activations.get(inner_activation)
    initializer = tf.keras.initializers.get(initializer)

    word_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_word_ids')
    mask = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_mask')
    type_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_type_ids')

    if embedding_width is None:
      embedding_width = hidden_size

    if embedding_layer is None:
      embedding_layer_inst = layers.OnDeviceEmbedding(
          vocab_size=vocab_size,
          embedding_width=embedding_width,
          initializer=tf_utils.clone_initializer(initializer),
          name='word_embeddings')
    else:
      embedding_layer_inst = embedding_layer
    word_embeddings = embedding_layer_inst(word_ids)

    # Always uses dynamic slicing for simplicity.
    position_embedding_layer = layers.PositionEmbedding(
        initializer=tf_utils.clone_initializer(initializer),
        max_length=max_sequence_length,
        name='position_embedding')
    position_embeddings = position_embedding_layer(word_embeddings)
    type_embedding_layer = layers.OnDeviceEmbedding(
        vocab_size=type_vocab_size,
        embedding_width=embedding_width,
        initializer=tf_utils.clone_initializer(initializer),
        use_one_hot=True,
        name='type_embeddings')
    type_embeddings = type_embedding_layer(type_ids)

    embeddings = tf.keras.layers.Add()(
        [word_embeddings, position_embeddings, type_embeddings])

    embedding_norm_layer = tf.keras.layers.LayerNormalization(
        name='embeddings/layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32)

    embeddings = embedding_norm_layer(embeddings)
    embeddings = (tf.keras.layers.Dropout(rate=output_dropout)(embeddings))

    # We project the 'embedding' output to 'hidden_size' if it is not already
    # 'hidden_size'.
    if embedding_width != hidden_size:
      embedding_projection = tf.keras.layers.EinsumDense(
          '...x,xy->...y',
          output_shape=hidden_size,
          bias_axes='y',
          kernel_initializer=tf_utils.clone_initializer(initializer),
          name='embedding_projection')
      embeddings = embedding_projection(embeddings)
    else:
      embedding_projection = None

    transformer_layers = []
    data = embeddings
    attention_mask = layers.SelfAttentionMask()(data, mask)
    encoder_outputs = []
    attention_outputs = []
    for i in range(num_layers):
      transformer_output_range = None
      if i == num_layers - 1:
        transformer_output_range = output_range
      layer = layers.TransformerEncoderBlock(
          num_attention_heads=num_attention_heads,
          inner_dim=inner_dim,
          inner_activation=inner_activation,
          output_dropout=output_dropout,
          attention_dropout=attention_dropout,
          norm_first=norm_first,
          return_attention_scores=return_attention_scores,
          kernel_initializer=tf_utils.clone_initializer(initializer),
          name='transformer/layer_%d' % i)
      transformer_layers.append(layer)
      data = layer([data, attention_mask],
                   output_range=transformer_output_range)
      if return_attention_scores:
        data, attention_scores = data
        attention_outputs.append(attention_scores)
      encoder_outputs.append(data)

    last_encoder_output = encoder_outputs[-1]
    # Applying a tf.slice op (through subscript notation) to a Keras tensor
    # like this will create a SliceOpLambda layer. This is better than a Lambda
    # layer with Python code, because that is fundamentally less portable.
    first_token_tensor = last_encoder_output[:, 0, :]
    pooler_layer = tf.keras.layers.Dense(
        units=hidden_size,
        activation='tanh',
        kernel_initializer=tf_utils.clone_initializer(initializer),
        name='pooler_transform')
    cls_output = pooler_layer(first_token_tensor)

    outputs = dict(
        sequence_output=encoder_outputs[-1],
        pooled_output=cls_output,
        encoder_outputs=encoder_outputs,
    )
    if return_attention_scores:
      outputs['attention_scores'] = attention_outputs

    if dict_outputs:
      super().__init__(
          inputs=[word_ids, mask, type_ids], outputs=outputs, **kwargs)
    else:
      cls_output = outputs['pooled_output']
      if return_all_encoder_outputs:
        encoder_outputs = outputs['encoder_outputs']
        outputs = [encoder_outputs, cls_output]
      else:
        sequence_output = outputs['sequence_output']
        outputs = [sequence_output, cls_output]
      if return_attention_scores:
        outputs.append(attention_outputs)
      super().__init__(  # pylint: disable=bad-super-call
          inputs=[word_ids, mask, type_ids],
          outputs=outputs,
          **kwargs)

    self._pooler_layer = pooler_layer
    self._transformer_layers = transformer_layers
    self._embedding_norm_layer = embedding_norm_layer
    self._embedding_layer = embedding_layer_inst
    self._position_embedding_layer = position_embedding_layer
    self._type_embedding_layer = type_embedding_layer
    if embedding_projection is not None:
      self._embedding_projection = embedding_projection

    config_dict = {
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_attention_heads': num_attention_heads,
        'max_sequence_length': max_sequence_length,
        'type_vocab_size': type_vocab_size,
        'inner_dim': inner_dim,
        'inner_activation': tf_utils.serialize_activation(
            activation, use_legacy_format=True
        ),
        'output_dropout': output_dropout,
        'attention_dropout': attention_dropout,
        'initializer': tf_utils.serialize_initializer(
            initializer, use_legacy_format=True
        ),
        'output_range': output_range,
        'embedding_width': embedding_width,
        'embedding_layer': embedding_layer,
        'norm_first': norm_first,
        'dict_outputs': dict_outputs,
        'return_attention_scores': return_attention_scores,
    }
    # pylint: disable=protected-access
    self._setattr_tracking = False
    self._config = config_dict
    self._setattr_tracking = True
    # pylint: enable=protected-access

  def get_embedding_table(self):
    return self._embedding_layer.embeddings

  def get_embedding_layer(self):
    return self._embedding_layer

  def get_config(self):
    return self._config

  @property
  def transformer_layers(self):
    """List of Transformer layers in the encoder."""
    return self._transformer_layers

  @property
  def pooler_layer(self):
    """The pooler dense layer after the transformer layers."""
    return self._pooler_layer

  @classmethod
  def from_config(cls, config, custom_objects=None):
    if 'embedding_layer' in config and config['embedding_layer'] is not None:
      warn_string = (
          'You are reloading a model that was saved with a '
          'potentially-shared embedding layer object. If you contine to '
          'train this model, the embedding layer will no longer be shared. '
          'To work around this, load the model outside of the Keras API.')
      print('WARNING: ' + warn_string)
      logging.warn(warn_string)

    return cls(**config)
