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

"""Specialized Transformers for Pix2Seq.

the position embeddings are added to the query and key for every self- and
cross-attention layer.
"""

import tensorflow as tf, tf_keras


class TransformerEncoder(tf_keras.layers.Layer):
  """Transformer encoder."""

  def __init__(
      self,
      num_layers,
      dim,
      mlp_ratio,
      num_heads,
      drop_path=0.1,
      drop_units=0.1,
      drop_att=0.0,
      self_attention=True,
      use_ffn_ln=False,
      ln_scale_shift=True,
      **kwargs
  ):
    super().__init__(**kwargs)
    self._num_layers = num_layers
    self._dim = dim
    self._mlp_ratio = mlp_ratio
    self._num_heads = num_heads
    self._drop_path = drop_path
    self._drop_units = drop_units
    self._drop_att = drop_att
    self._self_attention = self_attention
    self._use_ffn_ln = use_ffn_ln
    self._ln_scale_shift = ln_scale_shift

    self.enc_layers = [
        TransformerEncoderLayer(  # pylint: disable=g-complex-comprehension
            dim,
            mlp_ratio,
            num_heads,
            drop_path,
            drop_units,
            drop_att,
            self_attention=self_attention,
            use_ffn_ln=use_ffn_ln,
            ln_scale_shift=ln_scale_shift,
            name='transformer_encoder' + suffix_id(i),
        )
        for i in range(num_layers)
    ]

  def call(self, x, mask, training, ret_list=False):
    x_list = [x]
    for i in range(self._num_layers):
      x = self.enc_layers[i](x, mask, training)
      x_list.append(x)
    return (x, x_list) if ret_list else x

  def get_config(self):
    config = super().get_config()
    updates = {
        'num_layers': self._num_layers,
        'dim': self._dim,
        'mlp_ratio': self._mlp_ratio,
        'num_heads': self._num_heads,
        'drop_path': self._drop_path,
        'drop_units': self._drop_units,
        'drop_att': self._drop_att,
        'self_attention': self._self_attention,
        'use_ffn_ln': self._use_ffn_ln,
        'ln_scale_shift': self._ln_scale_shift,
    }
    config.update(updates)
    return config


class TransformerEncoderLayer(tf_keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(
      self,
      dim,
      mlp_ratio,
      num_heads,
      drop_path=0.1,
      drop_units=0.1,
      drop_att=0.0,
      self_attention=True,
      use_ffn_ln=False,
      ln_scale_shift=True,
      **kwargs
  ):
    super().__init__(**kwargs)
    self._dim = dim
    self._mlp_ratio = mlp_ratio
    self._num_heads = num_heads
    self._drop_path = drop_path
    self._drop_units = drop_units
    self._drop_att = drop_att
    self.self_attention = self_attention
    self._use_ffn_ln = use_ffn_ln
    self._ln_scale_shift = ln_scale_shift

    if self_attention:
      self.mha_ln = tf_keras.layers.LayerNormalization(
          epsilon=1e-6,
          center=ln_scale_shift,
          scale=ln_scale_shift,
          name='mha/ln',
      )
      self.mha = tf_keras.layers.MultiHeadAttention(
          num_heads, dim // num_heads, dropout=drop_att, name='mha'
      )
    self.mlp = MLP(
        1,
        dim,
        mlp_ratio,
        drop_path,
        drop_units,
        use_ffn_ln=use_ffn_ln,
        ln_scale_shift=ln_scale_shift,
        name='mlp',
    )
    self.dropp = DropPath(drop_path)

  def call(self, x, mask, training):
    # x shape (bsz, seq_len, dim_att), mask shape (bsz, seq_len, seq_len).
    if self.self_attention:
      x_ln = self.mha_ln(x)
      x_residual = self.mha(x_ln, x_ln, x_ln, mask, training=training)
      x = x + self.dropp(x_residual, training)
    x = self.mlp(x, training)
    return x

  def get_config(self):
    config = super().get_config()
    updates = {
        'dim': self._dim,
        'mlp_ratio': self._mlp_ratio,
        'num_heads': self._num_heads,
        'drop_path': self._drop_path,
        'drop_units': self._drop_units,
        'drop_att': self._drop_att,
        'self_attention': self._self_attention,
        'use_ffn_ln': self._use_ffn_ln,
        'ln_scale_shift': self._ln_scale_shift,
    }
    config.update(updates)
    return config


def suffix_id(i):
  """Return suffix id for layer/variable name."""
  return '' if i == 0 else '_%d' % i


class DropPath(tf_keras.layers.Layer):
  """For stochastic depth."""

  def __init__(self, drop_rate=0.0, **kwargs):
    """Initializes a drop path layer."""
    super().__init__(**kwargs)
    self._drop_rate = drop_rate
    if self._drop_rate < 0 or self._drop_rate >= 1.0:
      raise ValueError('drop_rate {} is outside [0, 1)'.format(self._drop_rate))

  def call(self, x, training=False):
    """Performs a forward pass.

    Args:
      x: An input tensor of type tf.Tensor with shape [batch, height, width,
        channels].
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      The output tensor.
    """
    if self._drop_rate == 0.0 or not training:
      return x

    keep_rate = 1.0 - self._drop_rate
    xshape = tf.shape(x)
    drop_mask_shape = [xshape[0]] + [1] * (len(xshape) - 1)
    drop_mask = keep_rate + tf.random.uniform(drop_mask_shape, dtype=x.dtype)
    drop_mask = tf.math.divide(tf.floor(drop_mask), keep_rate)
    return x * drop_mask

  def get_config(self):
    config = super().get_config()
    updates = {
        'drop_rate': self._drop_rate,
    }
    config.update(updates)
    return config


class FeedForwardLayer(tf_keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(
      self,
      dim_att=256,
      dim_mlp=1024,
      drop_units=0.1,
      use_ln=False,
      ln_scale_shift=False,
      **kwargs
  ):
    super().__init__(**kwargs)
    self._dim_att = dim_att
    self._dim_mlp = dim_mlp
    self._drop_units = drop_units
    self._use_ln = use_ln
    self._ln_scale_shift = ln_scale_shift

    self.dense1 = tf_keras.layers.Dense(
        dim_mlp, activation=tf.nn.gelu, name='dense1'
    )
    self.dropout = tf_keras.layers.Dropout(drop_units)
    self.dense2 = tf_keras.layers.Dense(dim_att, name='dense2')
    if use_ln:
      self.ln = tf_keras.layers.LayerNormalization(
          epsilon=1e-6,
          center=ln_scale_shift,
          scale=ln_scale_shift,
          name='mlp_ln',
      )
    else:
      self.ln = lambda x: x

  def call(self, x, training):
    return self.dense2(self.dropout(self.ln(self.dense1(x)), training=training))

  def get_config(self):
    config = super().get_config()
    updates = {
        'dim_att': self._dim_att,
        'dim_mlp': self._dim_mlp,
        'drop_units': self._drop_units,
        'use_ln': self._use_ln,
        'ln_scale_shift': self._ln_scale_shift,
    }
    config.update(updates)
    return config


class MLP(tf_keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(
      self,
      num_layers,
      dim,
      mlp_ratio,
      drop_path=0.1,
      drop_units=0.0,
      use_ffn_ln=False,
      ln_scale_shift=True,
      **kwargs
  ):
    super().__init__(**kwargs)
    self._num_layers = num_layers
    self._dim = dim
    self._mlp_ratio = mlp_ratio
    self._drop_path = drop_path
    self._drop_units = drop_units
    self._use_ffn_ln = use_ffn_ln
    self._ln_scale_shift = ln_scale_shift

    self.mlp_layers = []
    self.layernorms = []
    for i in range(num_layers):
      self.mlp_layers.append(
          FeedForwardLayer(
              dim,
              dim * mlp_ratio,
              drop_units,
              use_ln=use_ffn_ln,
              ln_scale_shift=ln_scale_shift,
              name='ffn' + suffix_id(i),
          )
      )
      self.layernorms.append(
          tf_keras.layers.LayerNormalization(
              epsilon=1e-6,
              center=ln_scale_shift,
              scale=ln_scale_shift,
              name='ffn/ln' + suffix_id(i),
          )
      )
    self.dropp = DropPath(drop_path)

  def call(self, x, training, ret_list=False):
    x_list = [x]
    for i in range(self._num_layers):
      x_residual = self.mlp_layers[i](self.layernorms[i](x), training)
      x = x + self.dropp(x_residual, training)
      x_list.append(x)
    return (x, x_list) if ret_list else x

  def get_config(self):
    config = super().get_config()
    updates = {
        'num_layers': self._num_layers,
        'dim': self._dim,
        'mlp_ratio': self._mlp_ratio,
        'drop_path': self._drop_path,
        'drop_units': self._drop_units,
        'use_ffn_ln': self._use_ffn_ln,
        'ln_scale_shift': self._ln_scale_shift,
    }
    config.update(updates)
    return config


class TransformerDecoderLayer(tf_keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(
      self,
      dim,
      mlp_ratio,
      num_heads,
      drop_path=0.1,
      drop_units=0.1,
      drop_att=0.0,
      dim_x_att=None,
      self_attention=True,
      cross_attention=True,
      use_mlp=True,
      use_enc_ln=False,
      use_ffn_ln=False,
      ln_scale_shift=True,
      **kwargs
  ):
    super().__init__(**kwargs)
    self._dim = dim
    self._mlp_ratio = mlp_ratio
    self._num_heads = num_heads
    self._drop_path = drop_path
    self._drop_units = drop_units
    self._drop_att = drop_att
    self._dim_x_att = dim_x_att
    self._self_attention = self_attention
    self._cross_attention = cross_attention
    self._use_mlp = use_mlp
    self._use_enc_ln = use_enc_ln
    self._use_ffn_ln = use_ffn_ln
    self._ln_scale_shift = ln_scale_shift

    if self_attention:
      self.self_ln = tf_keras.layers.LayerNormalization(
          epsilon=1e-6,
          center=ln_scale_shift,
          scale=ln_scale_shift,
          name='self_mha/ln',
      )
      self.self_mha = tf_keras.layers.MultiHeadAttention(
          num_heads, dim // num_heads, dropout=drop_att, name='self_mha'
      )
    if cross_attention:
      self.cross_ln = tf_keras.layers.LayerNormalization(
          epsilon=1e-6,
          center=ln_scale_shift,
          scale=ln_scale_shift,
          name='cross_mha/ln',
      )
      if use_enc_ln:
        self.enc_ln = tf_keras.layers.LayerNormalization(
            epsilon=1e-6,
            center=ln_scale_shift,
            scale=ln_scale_shift,
            name='cross_mha/enc_ln',
        )
      else:
        self.enc_ln = lambda x: x
      dim_x_att = dim if dim_x_att is None else dim_x_att
      self.cross_mha = tf_keras.layers.MultiHeadAttention(
          num_heads, dim_x_att // num_heads, dropout=drop_att, name='cross_mha'
      )
    if use_mlp:
      self.mlp = MLP(
          1,
          dim,
          mlp_ratio,
          drop_path,
          drop_units,
          use_ffn_ln=use_ffn_ln,
          ln_scale_shift=ln_scale_shift,
          name='mlp',
      )
    self.dropp = DropPath(drop_path)

  def call(self, x, enc, cache, mask_self, mask_cross, training):
    """x in (bsz, seq, d), enc in (bsz, seq', d)."""
    x_for_cache = []
    if self._self_attention:
      x_for_cache = x_ln = kv_ln = self.self_ln(x)
      if cache is not None:  # Augment kv_ln with cache in (bsz, c_size, d).
        q_size, k_size = tf.shape(x)[1], tf.shape(cache)[1]
        mask_self = tf.concat([tf.ones([1, 1, q_size, k_size]), mask_self], -1)
        kv_ln = tf.concat([cache, x_ln], axis=1)
      x_res = self.self_mha(x_ln, kv_ln, kv_ln, mask_self, training=training)
      x = x + self.dropp(x_res, training)
    if self._cross_attention:
      x_ln = self.cross_ln(x)
      enc = self.enc_ln(enc)
      x_res = self.cross_mha(x_ln, enc, enc, mask_cross, training=training)
      x = x + self.dropp(x_res, training)
    if self._use_mlp:
      x = self.mlp(x, training)
    return x, x_for_cache

  def get_config(self):
    config = super().get_config()
    updates = {
        'dim': self._dim,
        'mlp_ratio': self._mlp_ratio,
        'num_heads': self._num_heads,
        'drop_path': self._drop_path,
        'drop_units': self._drop_units,
        'drop_att': self._drop_att,
        'dim_x_att': self._dim_x_att,
        'self_attention': self._self_attention,
        'cross_attention': self._cross_attention,
        'use_mlp': self._use_mlp,
        'use_enc_ln': self._use_enc_ln,
        'use_ffn_ln': self._use_ffn_ln,
        'ln_scale_shift': self._ln_scale_shift,
    }
    config.update(updates)
    return config


class TransformerDecoder(tf_keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(
      self,
      num_layers,
      dim,
      mlp_ratio,
      num_heads,
      drop_path=0.1,
      drop_units=0.1,
      drop_att=0.0,
      dim_x_att=None,
      self_attention=True,
      cross_attention=True,
      use_mlp=True,
      use_enc_ln=False,
      use_ffn_ln=False,
      ln_scale_shift=True,
      **kwargs
  ):
    super().__init__(**kwargs)
    self._num_layers = num_layers
    self._dim = dim
    self._mlp_ratio = mlp_ratio
    self._num_heads = num_heads
    self._drop_path = drop_path
    self._drop_units = drop_units
    self._drop_att = drop_att
    self._dim_x_att = dim_x_att
    self._self_attention = self_attention
    self._cross_attention = cross_attention
    self._use_mlp = use_mlp
    self._use_enc_ln = use_enc_ln
    self._use_ffn_ln = use_ffn_ln
    self._ln_scale_shift = ln_scale_shift

    self.dec_layers = [
        TransformerDecoderLayer(  # pylint: disable=g-complex-comprehension
            dim,
            mlp_ratio,
            num_heads,
            drop_path,
            drop_units,
            drop_att,
            dim_x_att=dim_x_att,
            self_attention=self_attention,
            cross_attention=cross_attention,
            use_mlp=use_mlp,
            use_enc_ln=use_enc_ln,
            use_ffn_ln=use_ffn_ln,
            ln_scale_shift=ln_scale_shift,
            name='transformer_decoder_layer' + suffix_id(i),
        )
        for i in range(num_layers)
    ]

  def call(self, x, enc, caches, mask_self, mask_cross, training):
    """x in (bsz, seq, d), enc in (bsz, seq', d)."""
    presents = []
    for i in range(self._num_layers):
      cache = None if caches is None else caches[i]
      x, x_for_cache = self.dec_layers[i](
          x, enc, cache, mask_self, mask_cross, training
      )
      presents.append(x_for_cache)

    return x, tf.stack(presents)

  def get_config(self):
    config = super().get_config()
    updates = {
        'num_layers': self._num_layers,
        'dim': self._dim,
        'mlp_ratio': self._mlp_ratio,
        'num_heads': self._num_heads,
        'drop_path': self._drop_path,
        'drop_units': self._drop_units,
        'drop_att': self._drop_att,
        'dim_x_att': self._dim_x_att,
        'self_attention': self._self_attention,
        'cross_attention': self._cross_attention,
        'use_mlp': self._use_mlp,
        'use_enc_ln': self._use_enc_ln,
        'use_ffn_ln': self._use_ffn_ln,
        'ln_scale_shift': self._ln_scale_shift,
    }
    config.update(updates)
    return config
