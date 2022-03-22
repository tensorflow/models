# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
import tensorflow as tf

from official.modeling import activations
from official.projects.vit.modeling import nn_blocks
from official.vision.modeling.backbones import factory
from official.vision.modeling.layers import nn_layers


layers = tf.keras.layers

VIT_SPECS = {
    'vit-ti16':
        dict(
            hidden_size=192,
            patch_size=16,
            transformer=dict(mlp_dim=768, num_heads=3, num_layers=12),
        ),
    'vit-s16':
        dict(
            hidden_size=384,
            patch_size=16,
            transformer=dict(mlp_dim=1536, num_heads=6, num_layers=12),
        ),
    'vit-b16':
        dict(
            hidden_size=768,
            patch_size=16,
            transformer=dict(mlp_dim=3072, num_heads=12, num_layers=12),
        ),
    'vit-b32':
        dict(
            hidden_size=768,
            patch_size=32,
            transformer=dict(mlp_dim=3072, num_heads=12, num_layers=12),
        ),
    'vit-l16':
        dict(
            hidden_size=1024,
            patch_size=16,
            transformer=dict(mlp_dim=4096, num_heads=16, num_layers=24),
        ),
    'vit-l32':
        dict(
            hidden_size=1024,
            patch_size=32,
            transformer=dict(mlp_dim=4096, num_heads=16, num_layers=24),
        ),
    'vit-h14':
        dict(
            hidden_size=1280,
            patch_size=14,
            transformer=dict(mlp_dim=5120, num_heads=16, num_layers=32),
        ),
    'vit-g14':
        dict(
            hidden_size=1664,
            patch_size=14,
            transformer=dict(mlp_dim=8192, num_heads=16, num_layers=48),
        ),
}


class AddPositionEmbs(tf.keras.layers.Layer):
  """Adds (optionally learned) positional embeddings to the inputs."""

  def __init__(self, posemb_init=None, **kwargs):
    super().__init__(**kwargs)
    self.posemb_init = posemb_init

  def build(self, inputs_shape):
    pos_emb_shape = (1, inputs_shape[1], inputs_shape[2])
    self.pos_embedding = self.add_weight(
        'pos_embedding', pos_emb_shape, initializer=self.posemb_init)

  def call(self, inputs, inputs_positions=None):
    # inputs.shape is (batch_size, seq_len, emb_dim).
    pos_embedding = tf.cast(self.pos_embedding, inputs.dtype)

    return inputs + pos_embedding


class TokenLayer(tf.keras.layers.Layer):
  """A simple layer to wrap token parameters."""

  def build(self, inputs_shape):
    self.cls = self.add_weight(
        'cls', (1, 1, inputs_shape[-1]), initializer='zeros')

  def call(self, inputs):
    cls = tf.cast(self.cls, inputs.dtype)
    cls = cls + tf.zeros_like(inputs[:, 0:1])  # A hacky way to tile.
    x = tf.concat([cls, inputs], axis=1)
    return x


class Encoder(tf.keras.layers.Layer):
  """Transformer Encoder."""

  def __init__(self,
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
               **kwargs):
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

  def build(self, input_shape):
    if self._add_pos_embed:
      self._pos_embed = AddPositionEmbs(
          posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02),
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
              self._init_stochastic_depth_rate, i + 1, self._num_layers),
          norm_epsilon=1e-6)
      self._encoder_layers.append(encoder_layer)
    self._norm = layers.LayerNormalization(epsilon=1e-6)
    super().build(input_shape)

  def call(self, inputs, training=None):
    x = inputs
    if self._add_pos_embed:
      x = self._pos_embed(x, inputs_positions=self._inputs_positions)
    x = self._dropout(x, training=training)

    for encoder_layer in self._encoder_layers:
      x = encoder_layer(x, training=training)
    x = self._norm(x)
    return x


class VisionTransformer(tf.keras.Model):
  """Class to build VisionTransformer family model."""

  def __init__(self,
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
               classifier='token',
               kernel_regularizer=None,
               original_init=True):
    """VisionTransformer initialization function."""
    inputs = tf.keras.Input(shape=input_specs.shape[1:])

    x = layers.Conv2D(
        filters=hidden_size,
        kernel_size=patch_size,
        strides=patch_size,
        padding='valid',
        kernel_regularizer=kernel_regularizer,
        kernel_initializer='lecun_normal' if original_init else 'he_uniform')(
            inputs)
    if tf.keras.backend.image_data_format() == 'channels_last':
      rows_axis, cols_axis = (1, 2)
    else:
      rows_axis, cols_axis = (2, 3)
      # The reshape below assumes the data_format is 'channels_last,' so
      # transpose to that. Once the data is flattened by the reshape, the
      # data_format is irrelevant, so no need to update
      # tf.keras.backend.image_data_format.
      x = tf.transpose(x, perm=[0, 2, 3, 1])
    seq_len = (input_specs.shape[rows_axis] // patch_size) * (
        input_specs.shape[cols_axis] // patch_size)
    x = tf.reshape(x, [-1, seq_len, hidden_size])

    # If we want to add a class token, add it here.
    if classifier == 'token':
      x = TokenLayer(name='cls')(x)

    x = Encoder(
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer='glorot_uniform' if original_init else dict(
            class_name='TruncatedNormal', config=dict(stddev=.02)),
        init_stochastic_depth_rate=init_stochastic_depth_rate)(
            x)

    if classifier == 'token':
      x = x[:, 0]
    elif classifier == 'gap':
      x = tf.reduce_mean(x, axis=1)

    if representation_size:
      x = tf.keras.layers.Dense(
          representation_size,
          kernel_regularizer=kernel_regularizer,
          name='pre_logits',
          kernel_initializer='lecun_normal' if original_init else 'he_uniform')(
              x)
      x = tf.nn.tanh(x)
    else:
      x = tf.identity(x, name='pre_logits')
    endpoints = {
        'pre_logits':
            tf.reshape(x, [-1, 1, 1, representation_size or hidden_size])
    }

    super(VisionTransformer, self).__init__(inputs=inputs, outputs=endpoints)


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
      classifier=backbone_cfg.classifier,
      kernel_regularizer=l2_regularizer,
      original_init=backbone_cfg.original_init)
