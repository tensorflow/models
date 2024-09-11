# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""VisionTransformer models."""

import math
from typing import Optional, Tuple

from absl import logging
import tensorflow as tf, tf_keras

from official.modeling import activations
from official.vision.modeling.backbones import factory
from official.vision.modeling.backbones.vit_specs import VIT_SPECS
from official.vision.modeling.layers import nn_blocks
from official.vision.modeling.layers import nn_layers


layers = tf_keras.layers


class AddPositionEmbs(layers.Layer):
  """Adds (optionally learned) positional embeddings to the inputs."""

  def __init__(self,
               posemb_init: Optional[tf_keras.initializers.Initializer] = None,
               posemb_origin_shape: Optional[Tuple[int, int]] = None,
               posemb_target_shape: Optional[Tuple[int, int]] = None,
               **kwargs):
    """Constructs Positional Embedding module.

    The logic of this module is: the learnable positional embeddings length will
    be determined by the inputs_shape or posemb_origin_shape (if provided)
    during the construction. If the posemb_target_shape is provided and is
    different from the positional embeddings length, the embeddings will be
    interpolated during the forward call.

    Args:
      posemb_init: The positional embedding initializer.
      posemb_origin_shape: The intended positional embedding shape.
      posemb_target_shape: The potential target shape positional embedding may
        be interpolated to.
      **kwargs: other args.
    """
    super().__init__(**kwargs)
    self.posemb_init = posemb_init
    self.posemb_origin_shape = posemb_origin_shape
    self.posemb_target_shape = posemb_target_shape

  def build(self, inputs_shape):
    if self.posemb_origin_shape is not None:
      pos_emb_length = self.posemb_origin_shape[0] * self.posemb_origin_shape[1]
    else:
      pos_emb_length = inputs_shape[1]
    pos_emb_shape = (1, pos_emb_length, inputs_shape[2])
    self.pos_embedding = self.add_weight(
        'pos_embedding', pos_emb_shape, initializer=self.posemb_init)

  def _interpolate(self, pos_embedding: tf.Tensor, from_shape: Tuple[int, int],
                   to_shape: Tuple[int, int]) -> tf.Tensor:
    """Interpolates the positional embeddings."""
    logging.info('Interpolating postional embedding from length: %s to %s',
                 from_shape, to_shape)
    grid_emb = tf.reshape(pos_embedding, [1] + list(from_shape) + [-1])
    # NOTE: Using BILINEAR interpolation by default.
    grid_emb = tf.image.resize(grid_emb, to_shape)
    return tf.reshape(grid_emb, [1, to_shape[0] * to_shape[1], -1])

  def call(self, inputs, inputs_positions=None):
    del inputs_positions
    pos_embedding = self.pos_embedding
    # inputs.shape is (batch_size, seq_len, emb_dim).
    if inputs.shape[1] != pos_embedding.shape[1]:
      pos_embedding = self._interpolate(
          pos_embedding,
          from_shape=self.posemb_origin_shape,
          to_shape=self.posemb_target_shape)
    pos_embedding = tf.cast(pos_embedding, inputs.dtype)

    return inputs + pos_embedding


class TokenLayer(layers.Layer):
  """A simple layer to wrap token parameters."""

  def build(self, inputs_shape):
    self.cls = self.add_weight(
        'cls', (1, 1, inputs_shape[-1]), initializer='zeros')

  def call(self, inputs):
    cls = tf.cast(self.cls, inputs.dtype)
    cls = cls + tf.zeros_like(inputs[:, 0:1])  # A hacky way to tile.
    x = tf.concat([cls, inputs], axis=1)
    return x


class Encoder(layers.Layer):
  """Transformer Encoder."""

  def __init__(
      self,
      num_layers,
      mlp_dim,
      num_heads,
      dropout_rate=0.1,
      attention_dropout_rate=0.1,
      kernel_regularizer=None,
      inputs_positions=None,
      init_stochastic_depth_rate=0.0,
      kernel_initializer='glorot_uniform',
      add_pos_embed=True,
      pos_embed_origin_shape=None,
      pos_embed_target_shape=None,
      layer_scale_init_value=0.0,
      transformer_partition_dims=None,
      output_attention_scores=False,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self._num_layers = num_layers
    self._mlp_dim = mlp_dim
    self._num_heads = num_heads
    self._dropout_rate = dropout_rate
    self._attention_dropout_rate = attention_dropout_rate
    self._kernel_regularizer = kernel_regularizer
    self._inputs_positions = inputs_positions
    self._init_stochastic_depth_rate = init_stochastic_depth_rate
    self._kernel_initializer = kernel_initializer
    self._add_pos_embed = add_pos_embed
    self._pos_embed_origin_shape = pos_embed_origin_shape
    self._pos_embed_target_shape = pos_embed_target_shape
    self._layer_scale_init_value = layer_scale_init_value
    self._transformer_partition_dims = transformer_partition_dims
    self._output_attention_scores = output_attention_scores

  def build(self, input_shape):
    if self._add_pos_embed:
      self._pos_embed = AddPositionEmbs(
          posemb_init=tf_keras.initializers.RandomNormal(stddev=0.02),
          posemb_origin_shape=self._pos_embed_origin_shape,
          posemb_target_shape=self._pos_embed_target_shape,
          name='posembed_input')
    self._dropout = layers.Dropout(rate=self._dropout_rate)

    self._encoder_layers = []
    # Set layer norm epsilons to 1e-6 to be consistent with JAX implementation.
    # https://flax.readthedocs.io/en/latest/_autosummary/flax.deprecated.nn.LayerNorm.html
    for i in range(self._num_layers):
      encoder_layer = nn_blocks.TransformerEncoderBlock(
          inner_activation=activations.gelu,
          num_attention_heads=self._num_heads,
          inner_dim=self._mlp_dim,
          output_dropout=self._dropout_rate,
          attention_dropout=self._attention_dropout_rate,
          kernel_regularizer=self._kernel_regularizer,
          kernel_initializer=self._kernel_initializer,
          norm_first=True,
          stochastic_depth_drop_rate=nn_layers.get_stochastic_depth_rate(
              self._init_stochastic_depth_rate, i + 1, self._num_layers
          ),
          norm_epsilon=1e-6,
          layer_scale_init_value=self._layer_scale_init_value,
          transformer_partition_dims=self._transformer_partition_dims,
          return_attention_scores=self._output_attention_scores,
      )
      self._encoder_layers.append(encoder_layer)
    self._norm = layers.LayerNormalization(epsilon=1e-6)
    super().build(input_shape)

  def call(self, inputs, training=None):
    x = inputs
    if self._add_pos_embed:
      x = self._pos_embed(x, inputs_positions=self._inputs_positions)
    x = self._dropout(x, training=training)

    attention_scores = None  # Needed to suppress undefined-variable warning.
    for encoder_layer in self._encoder_layers:
      if self._output_attention_scores:
        x, attention_scores = encoder_layer(x, training=training)
      else:
        x = encoder_layer(x, training=training)
    x = self._norm(x)

    if self._output_attention_scores:
      return x, attention_scores
    return x

  def get_config(self):
    config = super().get_config()
    updates = {
        'num_layers': self._num_layers,
        'mlp_dim': self._mlp_dim,
        'num_heads': self._num_heads,
        'dropout_rate': self._dropout_rate,
        'attention_dropout_rate': self._attention_dropout_rate,
        'kernel_regularizer': self._kernel_regularizer,
        'inputs_positions': self._inputs_positions,
        'init_stochastic_depth_rate': self._init_stochastic_depth_rate,
        'kernel_initializer': self._kernel_initializer,
        'add_pos_embed': self._add_pos_embed,
        'pos_embed_origin_shape': self._pos_embed_origin_shape,
        'pos_embed_target_shape': self._pos_embed_target_shape,
        'layer_scale_init_value': self._layer_scale_init_value,
        'transformer_partition_dims': self._transformer_partition_dims,
        'output_attention_scores': self._output_attention_scores,
    }
    config.update(updates)
    return config


class VisionTransformer(tf_keras.Model):
  """Class to build VisionTransformer family model."""

  def __init__(
      self,
      mlp_dim=3072,
      num_heads=12,
      num_layers=12,
      attention_dropout_rate=0.0,
      dropout_rate=0.1,
      init_stochastic_depth_rate=0.0,
      input_specs=layers.InputSpec(shape=[None, None, None, 3]),
      patch_size=16,
      hidden_size=768,
      representation_size=0,
      pooler='token',
      kernel_regularizer=None,
      original_init: bool = True,
      output_encoded_tokens: bool = True,
      output_2d_feature_maps: bool = False,
      pos_embed_shape: Optional[Tuple[int, int]] = None,
      layer_scale_init_value: float = 0.0,
      transformer_partition_dims: Optional[Tuple[int, int, int, int]] = None,
      output_attention_scores: bool = False,
  ):
    """VisionTransformer initialization function."""
    self._mlp_dim = mlp_dim
    self._num_heads = num_heads
    self._num_layers = num_layers
    self._hidden_size = hidden_size
    self._patch_size = patch_size

    inputs = tf_keras.Input(shape=input_specs.shape[1:])

    x = layers.Conv2D(
        filters=hidden_size,
        kernel_size=patch_size,
        strides=patch_size,
        padding='valid',
        kernel_regularizer=kernel_regularizer,
        kernel_initializer='lecun_normal' if original_init else 'he_uniform')(
            inputs)
    if tf_keras.backend.image_data_format() == 'channels_last':
      rows_axis, cols_axis = (1, 2)
    else:
      rows_axis, cols_axis = (2, 3)
      # The reshape below assumes the data_format is 'channels_last,' so
      # transpose to that. Once the data is flattened by the reshape, the
      # data_format is irrelevant, so no need to update
      # tf_keras.backend.image_data_format.
      x = tf.transpose(x, perm=[0, 2, 3, 1])

    pos_embed_target_shape = (x.shape[rows_axis], x.shape[cols_axis])
    feat_h = input_specs.shape[rows_axis] // patch_size
    feat_w = input_specs.shape[cols_axis] // patch_size
    seq_len = feat_h * feat_w
    x = tf.reshape(x, [-1, seq_len, hidden_size])

    # If we want to add a class token, add it here.
    if pooler == 'token':
      x = TokenLayer(name='cls')(x)

    encoder_output = Encoder(
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer='glorot_uniform'
        if original_init
        else dict(class_name='TruncatedNormal', config=dict(stddev=0.02)),
        init_stochastic_depth_rate=init_stochastic_depth_rate,
        pos_embed_origin_shape=pos_embed_shape,
        pos_embed_target_shape=pos_embed_target_shape,
        layer_scale_init_value=layer_scale_init_value,
        output_attention_scores=output_attention_scores,
    )(x)

    endpoints = {}
    if output_attention_scores:
      x, attention_scores = encoder_output
      endpoints['attention_scores'] = attention_scores
    else:
      x = encoder_output

    if pooler == 'token':
      output_feature = x[:, 1:]
      x = x[:, 0]
    elif pooler == 'gap':
      output_feature = x
      x = tf.reduce_mean(x, axis=1)
    elif pooler == 'none':
      output_feature = x
      x = tf.identity(x, name='encoded_tokens')
    else:
      raise ValueError(f'unrecognized pooler type: {pooler}')

    if output_2d_feature_maps:
      # Use the closest feature level.
      feat_level = round(math.log2(patch_size))
      logging.info(
          'VisionTransformer patch size %d and feature level: %d',
          patch_size,
          feat_level,
      )
      endpoints[str(feat_level)] = tf.reshape(
          output_feature, [-1, feat_h, feat_w, x.shape.as_list()[-1]])

      # Don"t include `pre_logits` or `encoded_tokens` to support decoders.
      self._output_specs = {k: v.shape for k, v in endpoints.items()}

    if representation_size:
      x = layers.Dense(
          representation_size,
          kernel_regularizer=kernel_regularizer,
          name='pre_logits',
          kernel_initializer='lecun_normal' if original_init else 'he_uniform',
      )(x)
      x = tf.nn.tanh(x)
    else:
      x = tf.identity(x, name='pre_logits')

    if pooler == 'none':
      if output_encoded_tokens:
        endpoints['encoded_tokens'] = x
    else:
      endpoints['pre_logits'] = tf.reshape(
          x, [-1, 1, 1, representation_size or hidden_size])

    super().__init__(inputs=inputs, outputs=endpoints)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_backbone_builder('vit')
def build_vit(input_specs,
              backbone_config,
              norm_activation_config,
              l2_regularizer=None):
  """Build ViT model."""
  del norm_activation_config
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'vit', (f'Inconsistent backbone type '
                                  f'{backbone_type}')
  backbone_cfg.override(VIT_SPECS[backbone_cfg.model_name])
  logging.info(
      (
          'ViT specs: mlp_dim=%d, num_heads=%d, num_layers=%d,'
          'patch_size=%d, hidden_size=%d, representation_size=%d.'
      ),
      backbone_cfg.transformer.mlp_dim,
      backbone_cfg.transformer.num_heads,
      backbone_cfg.transformer.num_layers,
      backbone_cfg.patch_size,
      backbone_cfg.hidden_size,
      backbone_cfg.representation_size,
  )

  return VisionTransformer(
      mlp_dim=backbone_cfg.transformer.mlp_dim,
      num_heads=backbone_cfg.transformer.num_heads,
      num_layers=backbone_cfg.transformer.num_layers,
      attention_dropout_rate=backbone_cfg.transformer.attention_dropout_rate,
      dropout_rate=backbone_cfg.transformer.dropout_rate,
      init_stochastic_depth_rate=backbone_cfg.init_stochastic_depth_rate,
      input_specs=input_specs,
      patch_size=backbone_cfg.patch_size,
      hidden_size=backbone_cfg.hidden_size,
      representation_size=backbone_cfg.representation_size,
      pooler=backbone_cfg.pooler,
      kernel_regularizer=l2_regularizer,
      original_init=backbone_cfg.original_init,
      output_encoded_tokens=backbone_cfg.output_encoded_tokens,
      output_2d_feature_maps=backbone_cfg.output_2d_feature_maps,
      layer_scale_init_value=backbone_cfg.layer_scale_init_value,
      pos_embed_shape=backbone_cfg.pos_embed_shape,
      transformer_partition_dims=backbone_cfg.transformer_partition_dims,
      output_attention_scores=backbone_cfg.output_attention_scores,
  )
