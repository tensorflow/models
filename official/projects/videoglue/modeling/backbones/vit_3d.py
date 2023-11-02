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

"""The vision transformer using 3D projection for video inputs."""

from typing import Any, Optional, Tuple, Union

from absl import logging
import tensorflow as tf, tf_keras

from official.projects.videoglue.configs import backbones_3d as cfg
from official.vision.modeling.backbones import factory
from official.vision.modeling.backbones import vit

Encoder = vit.Encoder
TokenLayer = vit.TokenLayer
layers = tf_keras.layers


class AddSeparablePositionEmbs(tf_keras.layers.Layer):
  """Adds (optionally learned) positional embeddings to the inputs."""

  def __init__(self,
               posemb_init: Optional[tf_keras.initializers.Initializer] = None,
               posemb_origin_shape: Optional[Tuple[int, int]] = None,
               posemb_target_shape: Optional[Tuple[int, int]] = None,
               **kwargs):
    """Constructs Postional Embedding module.

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
    """Builds the separable positional embedding layer."""
    if self.posemb_origin_shape is not None:
      nt = self.posemb_origin_shape[0]
      nl = self.posemb_origin_shape[1]
      nc = inputs_shape[-1]
    else:
      _, nt, nl, nc = inputs_shape

    self._pos_embedding_time = self.add_weight(
        'pos_embedding_time',
        (1, nt, nc),
        dtype=tf.float32,
        initializer=tf_keras.initializers.TruncatedNormal(0.02))
    self._pos_embedding_space = self.add_weight(
        'pos_embedding_space',
        (1, nl, nc),
        dtype=tf.float32,
        initializer=tf_keras.initializers.TruncatedNormal(0.02))

  def _interpolate(self, pos_embedding: tf.Tensor,
                   from_shape: Tuple[int, int],
                   to_shape: Tuple[int, int]) -> tf.Tensor:
    """Interpolates the positional embeddings."""
    logging.info('Interpolating postional embedding from length: %s to %s',
                 from_shape, to_shape)
    grid_emb = tf.reshape(pos_embedding, [1] + list(from_shape) + [-1])
    # NOTE: Using BILINEAR interpolation by default.
    grid_emb = tf.image.resize(grid_emb, to_shape)
    return tf.reshape(grid_emb, [1, to_shape[0] * to_shape[1], -1])

  def call(self, inputs: tf.Tensor, inputs_positions: Any = None) -> tf.Tensor:
    # inputs.shape is (batch_size, time_len, seq_len, emb_dim).
    del inputs_positions
    pos_embedding_time = self._pos_embedding_time
    if inputs.shape[1] != pos_embedding_time.shape[1]:
      pos_embedding_time = self._interpolate(
          pos_embedding_time,
          from_shape=(1, self.posemb_origin_shape[0]),
          to_shape=(1, self.posemb_target_shape[0]))

    pos_embedding_space = self._pos_embedding_space
    if inputs.shape[2] != pos_embedding_space.shape[1]:
      pos_embedding_space = self._interpolate(
          pos_embedding_space,
          from_shape=(1, self.posemb_origin_shape[1]),
          to_shape=(1, self.posemb_target_shape[1]))

    pos_embedding_time = tf.cast(pos_embedding_time[:, :, None, :],
                                 inputs.dtype)
    pos_embedding_space = tf.cast(pos_embedding_space[:, None, :, :],
                                  inputs.dtype)
    return inputs + pos_embedding_time + pos_embedding_space


class VisionTransformer3D(tf_keras.Model):
  """Class to build VisionTransformer-3D family model.

  The Vision Transformer architecture with the modification on the first
    patch2token layer in order to process video inputs.
  Reference: https://arxiv.org/abs/2010.11929
  """

  def __init__(
      self,
      variant: str = 'native',
      mlp_dim: int = 3072,
      num_heads: int = 12,
      num_layers: int = 12,
      attention_dropout_rate: float = 0.0,
      dropout_rate: float = 0.1,
      init_stochastic_depth_rate: float = 0.0,
      input_specs: layers.InputSpec = layers.InputSpec(
          shape=[None, None, None, None, 3]),
      temporal_patch_size: int = 4,
      spatial_patch_size: int = 16,
      hidden_size: int = 768,
      representation_size: int = 0,
      pooler: str = 'token',
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      original_init: bool = True,
      pos_embed_shape: Optional[
          Union[Tuple[int, int], Tuple[int, int, int]]] = None):
    """VisionTransformer initialization function.

    Args:
      variant: the implementation variant to use. Currently supporting
        ['native', 'mae'].
      mlp_dim: the mlp dimension in the transformer encoder.
      num_heads: number of heads in the transformer encoder.
      num_layers: number of layers in the transformer encoder.
      attention_dropout_rate: dropout probability within the attention layer.
      dropout_rate: the output layer dropout rate.
      init_stochastic_depth_rate: the initial stochastic depth rate.
      input_specs: the input shape.
      temporal_patch_size: the patch size for the temporal dimension.
      spatial_patch_size: the patch size for the spatial dimension.
      hidden_size: the projection hidden size for the first layer.
      representation_size: the feature size of representation.
      pooler: type of pooler to use. Accept 'none', 'token' or 'gap'.
      kernel_regularizer: kernel regularizer.
      original_init: whether to use the original init described in the paper.
      pos_embed_shape: the original positional embedding shape to use. If None,
        the positional embedding shape will be inferred from the inputs.
    """
    self._variant = variant
    self._mlp_dim = mlp_dim
    self._num_heads = num_heads
    self._num_layers = num_layers
    self._hidden_size = hidden_size
    self._representation_size = representation_size
    self._pooler = pooler
    self._input_specs = input_specs
    self._temporal_patch_size = temporal_patch_size
    self._spatial_patch_size = spatial_patch_size
    self._kernel_regularizer = kernel_regularizer
    self._original_init = original_init
    self._pos_embed_shape = pos_embed_shape

    self._patch_size = (
        self._temporal_patch_size,
        self._spatial_patch_size,
        self._spatial_patch_size,
    )
    nt = self._input_specs.shape[1] // self._temporal_patch_size
    nh = self._input_specs.shape[2] // self._spatial_patch_size
    nw = self._input_specs.shape[3] // self._spatial_patch_size

    inputs = tf_keras.Input(shape=input_specs.shape[1:])
    add_pos_embed = True
    if self._variant == 'native':
      x = self._tokenize(inputs)
    elif self._variant == 'mae':
      x = self._mae_tokenize(inputs)
      # NOTE: MAE variant adds pos_embed in the tokenizer.
      add_pos_embed = False
    else:
      raise ValueError(
          'Unrecognized ViT-3D implementation variant choice: %s' %
          variant)

    # If we want to add a class token, add it here.
    if pooler == 'token':
      x = TokenLayer(name='cls')(x)

    x = vit.Encoder(
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer='glorot_uniform' if original_init else dict(
            class_name='TruncatedNormal', config=dict(stddev=.02)),
        init_stochastic_depth_rate=init_stochastic_depth_rate,
        pos_embed_origin_shape=pos_embed_shape,
        pos_embed_target_shape=None,
        add_pos_embed=add_pos_embed)(x)

    if pooler == 'token':
      x = x[:, 0]
    elif pooler == 'gap':
      x = tf.reduce_mean(x, axis=1)
    elif pooler == 'none':
      x = tf.reshape(x, [-1, nt, nh, nw, x.shape[-1]], name='encoded_tokens')
    else:
      raise ValueError(f'unrecognized pooler type: {pooler}')

    if representation_size:
      x = tf_keras.layers.Dense(
          representation_size,
          kernel_regularizer=kernel_regularizer,
          name='pre_logits',
          kernel_initializer='lecun_normal' if original_init else 'he_uniform')(
              x)
      x = tf.nn.tanh(x)
    else:
      x = tf.identity(x, name='pre_logits')

    if pooler == 'none':
      endpoints = {'encoded_tokens': x}
    else:
      endpoints = {
          'pre_logits':
              tf.reshape(x, [-1, 1, 1, 1, representation_size or hidden_size])
      }

    super().__init__(inputs=inputs, outputs=endpoints)

  def _tokenize(self, inputs: tf.Tensor):
    """The first layer to tokenize and project the input tensor."""
    x = tf_keras.layers.Conv3D(
        filters=self._hidden_size,
        kernel_size=self._patch_size,
        strides=self._patch_size,
        padding='valid',
        kernel_regularizer=self._kernel_regularizer,
        kernel_initializer=('lecun_normal'
                            if self._original_init else 'he_uniform'))(inputs)
    if tf_keras.backend.image_data_format() == 'channels_last':
      time_axis, rows_axis, cols_axis = (1, 2, 3)
    else:
      time_axis, rows_axis, cols_axis = (2, 3, 4)
      # The reshape below assumes the data_format is 'channels_last,' so
      # transpose to that. Once the data is flattened by the reshape, the
      # data_format is irrelevant, so no need to update
      # tf_keras.backend.image_data_format.
      x = tf.transpose(x, perm=[0, 2, 3, 4, 1])

    nt = self._input_specs.shape[time_axis] // self._temporal_patch_size
    nh = self._input_specs.shape[rows_axis] // self._spatial_patch_size
    nw = self._input_specs.shape[cols_axis] // self._spatial_patch_size
    seq_len = nt * nh * nw
    x = tf.reshape(x, [-1, seq_len, self._hidden_size])
    return x

  def _mae_tokenize(self, inputs: tf.Tensor):
    """The first layer to tokenize and project the input tensor."""
    # Follow the same normalization setting as the original implementation:
    # https://github.com/facebookresearch/mae_st/blob/d752324a4a59aab6454236f33b0cd5849f1e600a/util/kinetics.py#L48-L49
    # The inputs are supposed to be normalized to [0, 1] before applying the
    # following mean/std.
    mean = tf.constant((0.45, 0.45, 0.45), dtype=inputs.dtype)
    std = tf.constant((0.225, 0.225, 0.225), dtype=inputs.dtype)
    inputs = (inputs - mean) / std
    x = tf_keras.layers.Conv3D(
        filters=self._hidden_size,
        kernel_size=self._patch_size,
        strides=self._patch_size,
        padding='valid',
        kernel_regularizer=self._kernel_regularizer,
        kernel_initializer=('lecun_normal'
                            if self._original_init else 'he_uniform'))(inputs)
    if tf_keras.backend.image_data_format() == 'channels_last':
      time_axis, rows_axis, cols_axis = (1, 2, 3)
    else:
      time_axis, rows_axis, cols_axis = (2, 3, 4)
      # The reshape below assumes the data_format is 'channels_last,' so
      # transpose to that. Once the data is flattened by the reshape, the
      # data_format is irrelevant, so no need to update
      # tf_keras.backend.image_data_format.
      x = tf.transpose(x, perm=[0, 2, 3, 4, 1])

    nc = x.shape[-1]
    nt = self._input_specs.shape[time_axis] // self._temporal_patch_size
    nh = self._input_specs.shape[rows_axis] // self._spatial_patch_size
    nw = self._input_specs.shape[cols_axis] // self._spatial_patch_size

    x = tf.reshape(x, [-1, nt, nh * nw, nc])
    pos_embed_target_shape = (nt, nh * nw)
    x = AddSeparablePositionEmbs(
        posemb_init=self._original_init,
        posemb_origin_shape=self._pos_embed_shape,
        posemb_target_shape=pos_embed_target_shape)(x)
    x = tf.reshape(x, [-1, nt * nh * nw, nc])
    return x


@factory.register_backbone_builder('vit_3d')
def build_vit_3d(
    input_specs: tf_keras.layers.InputSpec,
    backbone_config: cfg.Backbone3D,
    norm_activation_config: Any,
    l2_regularizer: Optional[tf_keras.regularizers.Regularizer] = None):
  """Builds ViT-3D model.

  Args:
    input_specs: the input shape specs.
    backbone_config: the config for the backbone.
    norm_activation_config: deprecated. norm and activation config.
    l2_regularizer: the l2 regularizer.

  Returns:
    A VisionTransformer3D backbone.
  """
  del norm_activation_config
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'vit_3d', (f'Inconsistent backbone type '
                                     f'{backbone_type}')
  backbone_cfg.override(vit.VIT_SPECS[backbone_cfg.model_name])

  return VisionTransformer3D(
      variant=backbone_cfg.variant,
      mlp_dim=backbone_cfg.transformer.mlp_dim,
      num_heads=backbone_cfg.transformer.num_heads,
      num_layers=backbone_cfg.transformer.num_layers,
      attention_dropout_rate=backbone_cfg.transformer.attention_dropout_rate,
      dropout_rate=backbone_cfg.transformer.dropout_rate,
      init_stochastic_depth_rate=backbone_cfg.init_stochastic_depth_rate,
      input_specs=input_specs,
      temporal_patch_size=backbone_cfg.temporal_patch_size,
      spatial_patch_size=backbone_cfg.patch_size,
      hidden_size=backbone_cfg.hidden_size,
      representation_size=backbone_cfg.representation_size,
      pooler=backbone_cfg.pooler,
      kernel_regularizer=l2_regularizer,
      original_init=backbone_cfg.original_init,
      pos_embed_shape=backbone_cfg.pos_embed_shape)
