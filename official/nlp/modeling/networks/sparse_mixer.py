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

"""Sparse Mixer encoder network.

Based on ["Sparse Mixers: Combining MoE and Mixing to build a more efficient
BERT"](https://arxiv.org/abs/2205.12399).
"""
# pylint: disable=g-classes-have-attributes

from typing import Any, Callable, Optional, Sequence, Union
from absl import logging
import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.nlp.modeling import layers

_Activation = Union[str, Callable[..., Any]]
_Initializer = Union[str, tf_keras.initializers.Initializer]

_approx_gelu = lambda x: tf_keras.activations.gelu(x, approximate=True)


class SparseMixer(tf_keras.layers.Layer):
  """Sparse Mixer encoder network.

  Based on ["Sparse Mixers: Combining MoE and Mixing to build a more efficient
  BERT"](https://arxiv.org/abs/2205.12399). Sparse Mixer is an efficient
  encoder network that replaces typical Transformer encoder blocks with a
  combination of linear mixing and sparsely activated Mixture-of-Experts (MoE)
  sublayers.

  This implementation defaults to the canonical Sparse Mixer Base model. To use
  the "Fast Sparse Mixer" configuration, set `*_capacity_factor`=0.5. This
  yields a sparser and faster variant of the canonical Sparse Mixer model, in
  which each expert processes roughly 50% less tokens.

  Notes:
  - The underlying MoeLayer uses the Keras add_loss() and add_metric() APIs to
    propagate auxiliary MoE losses and metrics. Any model using this network,
    should collect these losses and, if desired, metrics.
  - The input length is fixed to 'max_sequence_length' to accomodate the mixing
    mechanisms.

  Args:
    vocab_size: The size of the token vocabulary.
    hidden_size: The size of the transformer hidden layers.
    num_layers: The number of transformer layers.
    moe_layers: Specifies which layers, if any, should be sparsely activated
      Mixture-of-Experts (MoE) layers. The remaining [0, num_layers) setminus
      moe_layers will use the vanilla MLP sublayers. Defaults to placing MoE
      layers in the middle of the model.
    attention_layers: Specifies which layers, if any, should be attention layers
      in the encoder. The remaining [0, num_layers) setminus attention_layers
      will use the specified `mixing_mechanism`. If using attention layers, a
      good rule of thumb is to place them in the final few layers.
    num_experts: Number of experts. Experts are themselves MLP modules, with the
      same `inner_dim` and `inner_activation` as the vanilla MLP sublayers.
    train_capacity_factor: Scaling factor to increase the expert token capacity
      during training. See layers.MoeLayer for further details. The "Fast Sparse
      Mixer" increases model sparsity (and speed) by using a capacity factor of
      0.5.
    eval_capacity_factor: As above, but used during evaluation.
    max_group_size: The total number of tokens on each device is subdivided into
      groups of this size. Router computations are then performed on a per-group
      basis. See layers.MoeLayer for further details.
    mixing_mechanism: Type of mixing mechanism used in place of self-attention
      layers. Defaults to 'Linear' mixing.
    use_fft: Only used for spectral mixing mechanisms. Determines whether to use
      Fast Fourier Transform (True) or the Discrete Fourier Transform (DFT)
      matrix (False; default) to compute the Fourier Transform. See
      layers.FourierTransformLayer or layers.HartleyTransformLayer for advice.
    num_attention_heads: The number of attention heads for each transformer. The
      hidden size must be divisible by the number of attention heads.
    max_sequence_length: The only sequence length that this encoder can consume.
      This determines the variable shape for positional embeddings and the size
      of the mixing matrices.
    type_vocab_size: The number of types that the 'type_ids' input can take.
    inner_dim: The output dimension of the first Dense layer in a two-layer
      feedforward network for each transformer.
    inner_activation: The activation for the first Dense layer in a two-layer
      feedforward network for each transformer.
    output_dropout: Dropout probability for the post-attention and output
      dropout.
    attention_dropout: The dropout rate to use for the attention layers within
      the transformer layers.
    initializer: The initializer to use for all weights in this encoder.
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
    export_metrics: Whether to export metrics using Keras add_metric API.
  """

  def __init__(
      self,
      vocab_size: int,
      hidden_size: int = 512,
      num_layers: int = 14,
      moe_layers: Sequence[int] = (5, 6, 7, 8),
      attention_layers: Sequence[int] = (10, 11, 12, 13),
      num_experts: int = 16,
      train_capacity_factor: float = 1.,
      eval_capacity_factor: float = 1.,
      examples_per_group: float = 1.,
      mixing_mechanism: layers.MixingMechanism = layers.MixingMechanism.LINEAR,
      use_fft: bool = False,
      num_attention_heads: int = 8,
      max_sequence_length: int = 512,
      type_vocab_size: int = 16,
      inner_dim: int = 2048,
      inner_activation: _Activation = _approx_gelu,
      output_dropout: float = 0.1,
      attention_dropout: float = 0.1,
      initializer: _Initializer = tf_keras.initializers.TruncatedNormal(
          stddev=0.02),
      output_range: Optional[int] = None,
      embedding_width: Optional[int] = None,
      embedding_layer: Optional[tf_keras.layers.Layer] = None,
      norm_first: bool = False,
      with_dense_inputs: bool = False,
      export_metrics: bool = True,
      **kwargs):
    super().__init__(**kwargs)

    activation = tf_keras.activations.get(inner_activation)
    initializer = tf_keras.initializers.get(initializer)

    if embedding_width is None:
      embedding_width = hidden_size

    self._config = {
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'moe_layers': moe_layers,
        'num_experts': num_experts,
        'train_capacity_factor': train_capacity_factor,
        'eval_capacity_factor': eval_capacity_factor,
        'examples_per_group': examples_per_group,
        'mixing_mechanism': mixing_mechanism,
        'use_fft': use_fft,
        'attention_layers': attention_layers,
        'num_attention_heads': num_attention_heads,
        'max_sequence_length': max_sequence_length,
        'type_vocab_size': type_vocab_size,
        'inner_dim': inner_dim,
        'inner_activation': tf_keras.activations.serialize(activation),
        'output_dropout': output_dropout,
        'attention_dropout': attention_dropout,
        'initializer': tf_keras.initializers.serialize(initializer),
        'output_range': output_range,
        'embedding_width': embedding_width,
        'embedding_layer': embedding_layer,
        'norm_first': norm_first,
        'with_dense_inputs': with_dense_inputs,
        'export_metrics': export_metrics,
    }

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

    self._embedding_norm_layer = tf_keras.layers.LayerNormalization(
        name='embeddings/layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32)

    self._embedding_dropout = tf_keras.layers.Dropout(
        rate=output_dropout, name='embedding_dropout')

    # We project the 'embedding' output to 'hidden_size' if it is not already
    # 'hidden_size'.
    self._embedding_projection = None
    if embedding_width != hidden_size:
      self._embedding_projection = tf_keras.layers.EinsumDense(
          '...x,xy->...y',
          output_shape=hidden_size,
          bias_axes='y',
          kernel_initializer=tf_utils.clone_initializer(initializer),
          name='embedding_projection')

    self._transformer_layers = []
    for layer in range(num_layers):
      if layer in attention_layers:
        mixing_layer = layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=int(hidden_size // num_attention_heads),
            dropout=attention_dropout,
            use_bias=True,
            kernel_initializer=tf_utils.clone_initializer(initializer),
            name='self_attention',
        )
      else:
        mixing_layer = self._init_mixing_sublayer(layer)

      if layer in moe_layers:
        feedforward_layer = layers.MoeLayer(
            experts=layers.FeedForwardExperts(
                num_experts=num_experts,
                d_ff=inner_dim,
                output_dropout=output_dropout,
                activation=inner_activation,
                kernel_initializer=tf_utils.clone_initializer(initializer),
                name='experts'),
            router=layers.ExpertsChooseMaskedRouter(
                num_experts=num_experts,
                kernel_initializer=tf_utils.clone_initializer(initializer),
                export_metrics=export_metrics,
                name='router'),
            train_capacity_factor=train_capacity_factor,
            eval_capacity_factor=eval_capacity_factor,
            examples_per_group=examples_per_group,
            name='moe')
      else:
        feedforward_layer = None  # Fallback to default (dense) MLP class

      block = layers.TransformerScaffold(
          num_attention_heads=num_attention_heads,
          inner_dim=inner_dim,
          inner_activation=inner_activation,
          attention_cls=mixing_layer,
          feedforward_cls=feedforward_layer,
          output_dropout=output_dropout,
          attention_dropout=attention_dropout,
          norm_first=norm_first,
          output_range=output_range if layer == num_layers - 1 else None,
          kernel_initializer=tf_utils.clone_initializer(initializer),
          name='transformer/layer_%d' % layer)
      self._transformer_layers.append(block)

    self._attention_mask_layer = layers.SelfAttentionMask(
        name='self_attention_mask')

    self._pooler_layer = tf_keras.layers.Dense(
        units=hidden_size,
        activation='tanh',
        kernel_initializer=tf_utils.clone_initializer(initializer),
        name='pooler_transform')

    if with_dense_inputs:
      self.inputs = dict(
          # The total length of token ids and dense inputs still has to be
          # max_sequence_length. It is checked in call().
          input_word_ids=tf_keras.Input(shape=(None,), dtype=tf.int32),
          input_mask=tf_keras.Input(shape=(None,), dtype=tf.int32),
          input_type_ids=tf_keras.Input(shape=(None,), dtype=tf.int32),
          dense_inputs=tf_keras.Input(
              shape=(None, embedding_width), dtype=tf.float32),
          dense_mask=tf_keras.Input(shape=(None,), dtype=tf.int32),
          dense_type_ids=tf_keras.Input(shape=(None,), dtype=tf.int32),
      )
    else:
      self.inputs = dict(
          input_word_ids=tf_keras.Input(
              shape=(max_sequence_length,), dtype=tf.int32),
          input_mask=tf_keras.Input(
              shape=(max_sequence_length,), dtype=tf.int32),
          input_type_ids=tf_keras.Input(
              shape=(max_sequence_length,), dtype=tf.int32))
    self._max_sequence_length = max_sequence_length

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
      raise ValueError('Unexpected inputs type (%s) to %s.' %
                       (type(inputs), self.__class__))

    if word_embeddings is None:
      word_embeddings = self._embedding_layer(word_ids)

    if dense_inputs is not None:
      # Concat the dense embeddings at sequence end.
      word_embeddings = tf.concat([word_embeddings, dense_inputs], axis=1)
      type_ids = tf.concat([type_ids, dense_type_ids], axis=1)
      mask = tf.concat([mask, dense_mask], axis=1)

    # SparseMixer: Sequence length must be the same as `max_sequence_length`.
    word_embeddings = tf.ensure_shape(word_embeddings,
                                      [None, self._max_sequence_length, None])

    # Absolute position embeddings.
    position_embeddings = self._position_embedding_layer(word_embeddings)
    type_embeddings = self._type_embedding_layer(type_ids)

    embeddings = word_embeddings + position_embeddings + type_embeddings
    embeddings = self._embedding_norm_layer(embeddings)
    embeddings = self._embedding_dropout(embeddings)

    if self._embedding_projection is not None:
      embeddings = self._embedding_projection(embeddings)

    attention_mask = self._attention_mask_layer(embeddings, mask)

    encoder_outputs = []
    x = embeddings
    for layer in self._transformer_layers:
      x = layer([x, attention_mask])
      encoder_outputs.append(x)

    last_encoder_output = encoder_outputs[-1]
    first_token_tensor = last_encoder_output[:, 0, :]
    pooled_output = self._pooler_layer(first_token_tensor)

    output = dict(
        sequence_output=encoder_outputs[-1],
        pooled_output=pooled_output,
        encoder_outputs=encoder_outputs)
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

  def _init_mixing_sublayer(self, layer: int):
    """Initializes config-dependent mixing sublayer."""
    if self._config['mixing_mechanism'] == layers.MixingMechanism.FOURIER:
      mixing_sublayer = layers.FourierTransformLayer(
          use_fft=self._config['use_fft'], name='fourier_transform')
    elif self._config['mixing_mechanism'] == layers.MixingMechanism.HARTLEY:
      mixing_sublayer = layers.HartleyTransformLayer(
          use_fft=self._config['use_fft'], name='hartley_transform')
    elif self._config['mixing_mechanism'] == layers.MixingMechanism.LINEAR:
      mixing_sublayer = layers.LinearTransformLayer(
          kernel_initializer=tf_utils.clone_initializer(
              self._config['initializer']),
          name='linear_transform')
    else:
      raise ValueError('Unsupported mixing mechanism: %s' %
                       self._config['mixing_mechanism'])

    return mixing_sublayer
