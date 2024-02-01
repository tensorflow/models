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

"""Implements A Language Modeling Framework for Object Detection.

Model paper: https://arxiv.org/abs/2109.10852
This module does not support Keras de/serialization. Please use
tf.train.Checkpoint for object based saving and loading and tf.saved_model.save
for graph serializaiton.
"""
import math
from typing import Any, List, Mapping, Optional, Union

import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.projects.pix2seq.modeling import transformer


def get_shape(x):
  static = x.shape.as_list()
  dynamic = tf.shape(x)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def get_variable_initializer(name=None):
  if name is None:
    return tf_keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)


def add_seq_pos_emb(
    self, pos_encoding, max_seq_len, dim, name_prefix=None, initializer=None
):
  """Add seq_pos_emb variable/tensor to model instance referenced by `self`."""
  if name_prefix is None:
    name_prefix = self.name
  if initializer is None:
    initializer = get_variable_initializer()
  if pos_encoding == "learned":
    self.seq_pos_emb = self.add_weight(
        shape=(max_seq_len + 1, dim),
        initializer=initializer,
        name="%s/seq_pos_embedding" % name_prefix,
    )
  # (gunho) currently only 'learned' positional encoding is supported
  elif pos_encoding == "sin_cos":
    self.seq_pos_emb = None
  else:
    raise ValueError("Unknown pos encoding %s" % pos_encoding)


def add_vocab_token_emb(
    self,
    vocab_size,
    dim,
    shared_embedding,
    output_bias,
    name_prefix=None,
    initializer=None,
):
  """Add token_embedding variable to model instance referenced by `self`."""
  if name_prefix is None:
    name_prefix = self.name
  if initializer is None:
    initializer = get_variable_initializer()
  if shared_embedding:
    self.token_embedding = self.add_weight(
        shape=[vocab_size, dim],
        initializer=initializer,
        name="%s/token_embedding" % name_prefix,
    )
  else:
    self.inp_token_embedding = self.add_weight(
        shape=[vocab_size, dim],
        initializer=initializer,
        name="%s/inp_token_embedding" % name_prefix,
    )
    self.outp_token_embedding = self.add_weight(
        shape=[vocab_size, dim],
        initializer=initializer,
        name="%s/outp_token_embedding" % name_prefix,
    )
  if output_bias:
    self.outp_bias = self.add_weight(
        shape=[vocab_size],
        initializer=initializer,
        name="%s/outp_bias" % name_prefix,
    )


def get_ar_mask(seq_len, dtype=tf.float32):
  """Get autoregressive causal mask so the model cannot attends to the future.

  Args:
    seq_len: a `int` or `int` tensor specifying the sequence length.
    dtype: tf data type for the return tensor.

  Returns:
    tensor of shape [1, 1, seq_len, seq_len] with ones for locations to be
    masked out.
  """
  valid_locs = tf.linalg.band_part(
      tf.ones([seq_len, seq_len], dtype=dtype), -1, 0
  )
  valid_locs = tf.reshape(valid_locs, [1, 1, seq_len, seq_len])
  return 1.0 - valid_locs


def position_embedding_sine(
    attention_mask,
    num_pos_features=256,
    temperature=10000.0,
    normalize=True,
    scale=2 * math.pi,
):
  """Sine-based positional embeddings for 2D images.

  Args:
    attention_mask: a `bool` Tensor specifying the size of the input image to
      the Transformer and which elements are padded, of size [batch_size,
      height, width]
    num_pos_features: a `int` specifying the number of positional features,
      should be equal to the hidden size of the Transformer network
    temperature: a `float` specifying the temperature of the positional
      embedding. Any type that is converted to a `float` can also be accepted.
    normalize: a `bool` determining whether the positional embeddings should be
      normalized between [0, scale] before application of the sine and cos
      functions.
    scale: a `float` if normalize is True specifying the scale embeddings before
      application of the embedding function.

  Returns:
    embeddings: a `float` tensor of the same shape as input_tensor specifying
      the positional embeddings based on sine features.
  """
  if num_pos_features % 2 != 0:
    raise ValueError(
        "Number of embedding features (num_pos_features) must be even when "
        "column and row embeddings are concatenated."
    )
  num_pos_features = num_pos_features // 2

  # Produce row and column embeddings based on total size of the image
  # <tf.float>[batch_size, height, width]
  attention_mask = tf.cast(attention_mask, tf.float32)
  row_embedding = tf.cumsum(attention_mask, 1)
  col_embedding = tf.cumsum(attention_mask, 2)

  if normalize:
    eps = 1e-6
    row_embedding = row_embedding / (row_embedding[:, -1:, :] + eps) * scale
    col_embedding = col_embedding / (col_embedding[:, :, -1:] + eps) * scale

  dim_t = tf.range(num_pos_features, dtype=row_embedding.dtype)
  dim_t = tf.pow(temperature, 2 * (dim_t // 2) / num_pos_features)

  # Creates positional embeddings for each row and column position
  # <tf.float>[batch_size, height, width, num_pos_features]
  pos_row = tf.expand_dims(row_embedding, -1) / dim_t
  pos_col = tf.expand_dims(col_embedding, -1) / dim_t
  pos_row = tf.stack(
      [tf.sin(pos_row[:, :, :, 0::2]), tf.cos(pos_row[:, :, :, 1::2])], axis=4
  )
  pos_col = tf.stack(
      [tf.sin(pos_col[:, :, :, 0::2]), tf.cos(pos_col[:, :, :, 1::2])], axis=4
  )

  final_shape = tf_utils.get_shape_list(pos_row)[:3] + [-1]
  pos_row = tf.reshape(pos_row, final_shape)
  pos_col = tf.reshape(pos_col, final_shape)
  output = tf.concat([pos_row, pos_col], -1)

  embeddings = tf.cast(output, tf.float32)
  return embeddings


def top_logits(
    logits: tf.Tensor, k: int = 0, p: float = 1.0, mask: float = -1e10
) -> tf.Tensor:
  """Remove low probability logits via masking.

  Args:
    logits: class logits in shape of (batch size, total_classes).
    k: specifying top k largest logits to keep.
    p: specifying a probability for finding a minimum set of largest logits to
      keep, where their cumulative probability is no less than p (actually in
      the following version, it is "...cumulative probability is the largest but
      no more than p").
    mask: an value that's used to replace logits that don't satisfy the keep
      conditions.

  Returns:
    logits where low probability ones are replaced with mask.
  """
  mask = tf.ones_like(logits) * mask
  if k > 0:
    min_logits = tf.nn.top_k(logits, k=k)[0][:, -1:]
    logits = tf.where(logits < min_logits, mask, logits)
  if p < 1.0:
    sorted_logits = tf.sort(logits, direction="DESCENDING", axis=-1)
    cum_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    min_logits = -tf.reduce_max(
        tf.where(cum_probs <= p, -sorted_logits, mask), -1, keepdims=True
    )
    min_logits = tf.minimum(min_logits, sorted_logits[:, :1])
    logits = tf.where(logits < min_logits, mask, logits)
  return logits


class Pix2Seq(tf_keras.Model):
  """Pix2Seq model with Keras.

  Pix2Seq consists of backbone, input token embedding, Pix2SeqTransformer.
  """

  def __init__(
      self,
      backbone,
      backbone_endpoint_name,
      max_seq_len,
      vocab_size,
      hidden_size,
      num_heads,
      num_encoder_layers=6,
      num_decoder_layers=6,
      drop_path=0.1,
      drop_units=0.1,
      drop_att=0.0,
      temperature=1.0,
      top_k=0,
      top_p=0.4,
      eos_token: int | None = None,
      **kwargs
  ):
    super().__init__(**kwargs)
    self._backbone = backbone
    self._backbone_endpoint_name = backbone_endpoint_name
    self._max_seq_len = max_seq_len
    self._vocab_size = vocab_size
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._num_encoder_layers = num_encoder_layers
    self._num_decoder_layers = num_decoder_layers
    self._drop_path = drop_path
    self._drop_units = drop_units
    self._drop_att = drop_att
    if hidden_size % 2 != 0:
      raise ValueError("hidden_size must be a multiple of 2.")

    self._dropout = tf_keras.layers.Dropout(self._drop_units)
    self._stem_projection = tf_keras.layers.Dense(
        self._hidden_size, name="stem_projection"
    )
    self._stem_ln = tf_keras.layers.LayerNormalization(
        epsilon=1e-6, name="stem_ln"
    )

    self._transformer = Pix2SeqTransformer(
        max_seq_len=self._max_seq_len,
        vocab_size=self._vocab_size,
        hidden_size=self._hidden_size,
        pos_encoding="learned",
        num_encoder_layers=self._num_encoder_layers,
        num_decoder_layers=self._num_decoder_layers,
        drop_path=self._drop_path,
        drop_units=self._drop_units,
        drop_att=self._drop_att,
        num_heads=self._num_heads,
    )
    self._temperature = temperature
    self._top_k = top_k
    self._top_p = top_p
    self._eos_token = eos_token

  @property
  def backbone(self) -> tf_keras.Model:
    return self._backbone

  @property
  def transformer(self) -> tf_keras.Model:
    return self._transformer

  def get_config(self):
    return {
        "backbone": self._backbone,
        "backbone_endpoint_name": self._backbone_endpoint_name,
        "max_seq_len": self._max_seq_len,
        "vocab_size": self._vocab_size,
        "hidden_size": self._hidden_size,
        "num_encoder_layers": self._num_encoder_layers,
        "num_decoder_layers": self._num_decoder_layers,
        "drop_path": self._drop_path,
        "drop_units": self._drop_units,
        "drop_att": self._drop_att,
        "temperature": self._temperature,
        "top_k": self._top_k,
        "top_p": self._top_p,
        "eos_token": self._eos_token,
        "num_heads": self._num_heads,
    }

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  @property
  def checkpoint_items(
      self) -> Mapping[str, Union[tf_keras.Model, tf_keras.layers.Layer]]:
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(backbone=self.backbone, transformer=self.transformer)
    return items

  def _generate_image_mask(
      self, inputs: tf.Tensor, target_shape: tf.Tensor
  ) -> tf.Tensor:
    """Generates image mask from input image."""
    mask = tf.expand_dims(
        tf.cast(
            tf.not_equal(tf.reduce_sum(inputs, axis=-1), 0.3), inputs.dtype
        ),
        axis=-1,
    )
    mask = tf.image.resize(
        mask, target_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return mask

  def call(
      self,
      inputs: tf.Tensor,
      targets: Optional[tf.Tensor] = None,
      training: bool = None,
      use_teacher_forcing_for_eval: bool = False,
      use_input_as_backbone_features=False,
  ) -> List[Any]:
    if use_input_as_backbone_features:
      features = inputs
    else:
      features = self._backbone(inputs)[self._backbone_endpoint_name]
    mask = tf.ones_like(features)
    batch_size, h, w, num_channels = get_shape(features)
    features = tf.reshape(features, [batch_size, h * w, num_channels])
    features = self._stem_ln(
        self._stem_projection(self._dropout(features, training)))

    pos_emb = position_embedding_sine(
        mask[:, :, :, 0], num_pos_features=self._hidden_size
    )
    pos_emb = tf.reshape(pos_emb, [batch_size, -1, self._hidden_size])
    pos_emb = tf.cast(pos_emb, features.dtype)

    tokens = None
    inputs = {
        "inputs": features,
        "tokens": targets,
        "pos_emb": pos_emb,
    }
    if training:
      logits = self._transformer(inputs, training=True)
    elif use_teacher_forcing_for_eval:
      logits = self._transformer(inputs, training=False)
    else:
      tokens, logits = self._transformer.infer(
          inputs,
          temperature=self._temperature,
          top_k=self._top_k,
          top_p=self._top_p,
          eos_token=self._eos_token,
      )

    return [tokens, logits]


def _create_cond_fn(seq_len: int, eos_token: int | None, prompt_len: int):
  """Returns a loop condition for decoder.

  Args:
    seq_len: the maximum sequence length.
    eos_token: if not None, enable early termination based on end-of-sequence
      token.
    prompt_len: the length of prompt sequence.
  """

  def cond(step, caches, tokens, logits):
    del caches
    del logits
    within_seq_len = (seq_len > prompt_len) & (step < seq_len - 1)
    if eos_token is None:
      return within_seq_len
    else:
      tokens = tokens[prompt_len:step]
      reached_eos = tf.reduce_all(tf.reduce_any(tokens == eos_token, axis=0))
      return within_seq_len & tf.logical_not(reached_eos)

  return cond


class Pix2SeqTransformer(tf_keras.layers.Layer):
  """Encoder and Decoder of Pix2Seq."""

  def __init__(
      self,
      max_seq_len,
      vocab_size,
      hidden_size,
      pos_encoding="learned",
      num_encoder_layers=6,
      num_decoder_layers=6,
      drop_path=0.1,
      drop_units=0.1,
      drop_att=0.0,
      shared_embedding=True,
      output_bias=True,
      num_heads=8,
      **kwargs
  ):
    super().__init__(**kwargs)
    self._max_seq_len = max_seq_len
    self._vocab_size = vocab_size
    self._hidden_size = hidden_size
    self._pos_encoding = pos_encoding
    self._num_encoder_layers = num_encoder_layers
    self._num_decoder_layers = num_decoder_layers
    self._drop_path = drop_path
    self._drop_units = drop_units
    self._drop_att = drop_att
    self._shared_embedding = shared_embedding
    self._output_bias = output_bias
    self._num_heads = num_heads

    add_seq_pos_emb(
        self, self._pos_encoding, self._max_seq_len, self._hidden_size
    )
    add_vocab_token_emb(
        self,
        self._vocab_size,
        self._hidden_size,
        self._shared_embedding,
        self._output_bias,
    )

    if self._num_encoder_layers > 0:
      self._encoder = transformer.TransformerEncoder(
          num_layers=self._num_encoder_layers,
          dim=self._hidden_size,
          mlp_ratio=4,
          num_heads=self._num_heads,
          drop_path=self._drop_path,
          drop_units=self._drop_units,
          drop_att=self._drop_att,
      )
    else:
      self._encoder = None

    self._output_ln_enc = tf_keras.layers.LayerNormalization(
        epsilon=1e-6, name="output_ln_enc"
    )

    self._proj = tf_keras.layers.Dense(self._hidden_size, name="proj/linear")
    self._proj_ln = tf_keras.layers.LayerNormalization(
        epsilon=1e-6, name="proj/ln"
    )
    self._proj_mlp = transformer.MLP(
        num_layers=1,
        dim=self._hidden_size,
        mlp_ratio=4,
        drop_path=self._drop_path,
        drop_units=self._drop_units,
        name="proj/mlp",
    )

    self._decoder = transformer.TransformerDecoder(
        num_layers=self._num_decoder_layers,
        dim=self._hidden_size,
        mlp_ratio=4,
        num_heads=self._num_heads,
        drop_path=self._drop_path,
        drop_units=self._drop_units,
        drop_att=self._drop_att,
    )
    self._output_ln_dec = tf_keras.layers.LayerNormalization(
        epsilon=1e-6, name="output_ln_dec"
    )

  def get_config(self):
    return {
        "max_seq_len": self._max_seq_len,
        "vocab_size": self._vocab_size,
        "hidden_size": self._hidden_size,
        "pos_encoding": self._pos_encoding,
        "num_encoder_layers": self._num_encoder_layers,
        "num_decoder_layers": self._num_decoder_layers,
        "drop_path": self._drop_path,
        "drop_units": self._drop_units,
        "drop_att": self._drop_att,
        "shared_embedding": self._shared_embedding,
        "output_bias": self._output_bias,
        "num_heads": self._num_heads,
    }

  def call(self, inputs: tf.Tensor, training: bool = None):
    sources = inputs["inputs"]
    targets = inputs["tokens"]
    mem_pos_embed = inputs["pos_emb"]

    sources = sources + mem_pos_embed
    if self._encoder is not None:
      encoded = self._encoder(sources, None, training=training, ret_list=False)
    else:
      encoded = sources
    encoded = self._output_ln_enc(encoded)

    encoded = self._proj_ln(self._proj(encoded))
    encoded = encoded + mem_pos_embed
    encoded = self._proj_mlp(encoded, training=training)

    seq_len = tf.shape(targets)[1]
    seq_pos_emb = tf.expand_dims(self.seq_pos_emb[:seq_len], 0)
    inp_embedding = outp_embedding = self.token_embedding
    target_emb = tf.gather(inp_embedding, targets) + seq_pos_emb

    self_attention_mask = 1.0 - get_ar_mask(seq_len, target_emb.dtype)

    decoded, _ = self._decoder(
        target_emb, encoded, None, self_attention_mask, None, training)
    decoded = self._output_ln_dec(decoded)

    decoded = tf.cast(decoded, seq_pos_emb.dtype)
    outp_embedding = tf.cast(outp_embedding, seq_pos_emb.dtype)

    logits = tf.matmul(decoded, outp_embedding, transpose_b=True)
    if self._output_bias:
      logits = tf.nn.bias_add(logits, self.outp_bias)

    return logits

  def infer(
      self,
      inputs: tf.Tensor,
      max_seq_len=None,
      temperature=1.0,
      top_k=0,
      top_p=0.4,
      sampling_callback=None,
      eos_token: int | None = None,
  ):
    """Autoregressive (without teacher-forcing) prediction.

    Note: the autoregressive sampling/inference time can be further optimized by
    caching *transformed* key / value inside multi-head attention for the
    `encoded` and previously generated tokens, but this may make the code less
    readable.

    Args:
      inputs: prompt - `int` tokens with shape of (bsz, prompt_len). encoded -
        `float` encoded representations for conditioning with shape of (bsz,
        size, dim). This can be optional in case of pure decoder.
      max_seq_len: `int` of max generated sequence length (including prompt).
      temperature: `float` scalar for scaling the logits before sampling.
      top_k: `int` scalar for truncating top-k tokens according to logits before
        token sampling.
      top_p: `float` scalar specifying the threshold of cumulative probablity
        for truncating tokens before token sampling.
      sampling_callback: a callbak `function` that take `next_logits`, and
        return `next_token`. This is used when users need a specific logic for
        sampling. Default to `None` with standard free-form sampling.
      eos_token: if not None, stop inference early based on this end-of-sequence
        (EOS) token. This won't change sequence length. However, for each
        sequence, the tokens and logit values after the EOS token will have
        undefined behavior based on implementation detail.

    Returns:
      sampled tokens with shape of (bsz, max_seq_len-prompt_len).
      logits (temperature-scaled) associated with sampled token, in shape of
        (bsz, max_seq_len-prompt_len, vocab_size).
    """

    sources = inputs["inputs"]
    prompt = inputs["tokens"]
    mem_pos_embed = inputs["pos_emb"]

    sources = sources + mem_pos_embed
    if self._encoder is not None:
      encoded = self._encoder(sources, None, training=False, ret_list=False)
    else:
      encoded = sources
    encoded = self._output_ln_enc(encoded)

    encoded = self._proj_ln(self._proj(encoded))
    encoded = encoded + mem_pos_embed
    encoded = self._proj_mlp(encoded, training=False)

    bsz = tf.shape(prompt)[0]
    prompt_len = tf.shape(prompt)[1]

    seq_len = self._max_seq_len if max_seq_len is None else max_seq_len
    # (gunho) 500 (self._max_seq_len) -> 501 for prompt seq
    seq_len = seq_len + 1
    seq_pos_emb = tf.expand_dims(self.seq_pos_emb, 0)
    inp_embedding = self.token_embedding
    outp_embedding = inp_embedding

    # Each step reads caches[:step] and tokens[step:next_step] and updates
    # tokens[next_step], logits[next_step] and caches[step:next_step].
    # On the first step, step=0, next_step=prompt_len. On subsequent steps
    # next_step = step + 1.
    def loop_body(step, caches, tokens, logits, is_prompt=False):
      if is_prompt:
        assert step == 0
        x = tf.gather(inp_embedding, tf.transpose(tokens[:prompt_len]))
        input_pos_embed = seq_pos_emb[:, :prompt_len]
        x += input_pos_embed
        self_attention_mask = 1.0 - get_ar_mask(prompt_len, x.dtype)
        caches_in = None
      else:
        x = tf.gather(inp_embedding, tf.transpose(tokens[step]))
        input_pos_embed = seq_pos_emb[:, step]
        x += input_pos_embed
        x = tf.expand_dims(x, 1)  # (bsz, 1, d)
        self_attention_mask = tf.ones([1, 1, 1, 1])
        caches_in = tf.transpose(caches[:step], [1, 2, 0, 3])
      decoded, caches_out = self._decoder(
          x, encoded, caches_in, self_attention_mask, None, training=False)
      decoded = self._output_ln_dec(decoded)

      # (gunho) transformer.py uses tf.float32 for numeric stability.
      decoded = tf.cast(decoded, seq_pos_emb.dtype)

      next_logits = tf.matmul(  # only take the last for sampling next token.
          decoded, outp_embedding, transpose_b=True
      )[:, -1]
      if self._output_bias:
        next_logits = tf.nn.bias_add(next_logits, self.outp_bias)

      # Scale and trunctate logits and sample next token.
      if sampling_callback:
        next_token = sampling_callback(
            next_logits, step, temperature, top_k, top_p
        )
      else:
        sampling_logits = next_logits / tf.cast(temperature, tf.float32)
        sampling_logits = top_logits(sampling_logits, k=top_k, p=top_p)
        next_token = tf.random.categorical(
            sampling_logits, num_samples=1, dtype=tf.int32
        )[:, 0]

      # Update internal states.
      next_step = step + (prompt_len if is_prompt else 1)
      caches_out = tf.transpose(caches_out, [2, 0, 1, 3])

      caches = tf.tensor_scatter_nd_update(caches, [[step]], caches_out)
      tokens = tf.tensor_scatter_nd_update(tokens, [[next_step]], [next_token])
      logits = tf.tensor_scatter_nd_update(logits, [[next_step]], [next_logits])
      return (next_step, caches, tokens, logits)

    caches_var = tf.zeros(
        [seq_len-1, self._num_decoder_layers, bsz, self._hidden_size])
    tokens_var = tf.zeros([seq_len, bsz], dtype=tf.int64)
    logits_var = tf.zeros([seq_len, bsz, self._vocab_size], dtype=tf.float32)
    indices = tf.expand_dims(tf.range(prompt_len), -1)
    tokens_var = tf.tensor_scatter_nd_update(
        tokens_var, indices, tf.transpose(prompt, [1, 0])
    )

    step = 0
    step, caches_var, tokens_var, logits_var = loop_body(
        step, caches_var, tokens_var, logits_var, is_prompt=True
    )
    _, _, tokens_var, logits_var = tf.while_loop(
        cond=_create_cond_fn(
            seq_len=seq_len, eos_token=eos_token, prompt_len=prompt_len
        ),
        body=loop_body,
        loop_vars=[step, caches_var, tokens_var, logits_var],
    )

    sampled_tokens = tf.transpose(tokens_var[prompt_len:], [1, 0])
    sampled_tokens_logits = tf.transpose(logits_var[prompt_len:], [1, 0, 2])
    sampled_tokens_logits = tf.reshape(
        sampled_tokens_logits, [bsz, self._max_seq_len, self._vocab_size]
    )

    # sampled_tokens_logits : [bsz, max_seq_len-prompt_len, vocab_size]
    return sampled_tokens, sampled_tokens_logits
