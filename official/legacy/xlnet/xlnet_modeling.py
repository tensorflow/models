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

"""Keras layers of XLNet model in TF 2.0."""

import copy
import warnings

import tensorflow as tf, tf_keras
from official.legacy.xlnet import data_utils
from official.nlp.modeling import networks


def gelu(x):
  return tf_keras.activations.gelu(x, approximate=True)


def _get_initializer(flags):
  """Get variable initializer."""
  if flags.init_method == "uniform":
    initializer = tf_keras.initializers.RandomUniform(
        minval=-flags.init_range, maxval=flags.init_range)
  elif flags.init_method == "normal":
    initializer = tf_keras.initializers.RandomNormal(stddev=flags.init_std)
  else:
    raise ValueError("Initializer {} not supported".format(flags.init_method))
  return initializer


def rel_shift(x, klen=-1):
  """Performs relative shift to form the relative attention score."""
  x_size = tf.shape(x)

  x = tf.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
  x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
  x = tf.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
  x = tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])

  return x


def _create_mask(qlen, mlen, dtype=tf.float32, same_length=False):
  """Creates attention mask when single-side context allowed only."""
  attn_mask = tf.ones([qlen, qlen], dtype=dtype)
  mask_u = tf.linalg.band_part(attn_mask, 0, -1)
  mask_dia = tf.linalg.band_part(attn_mask, 0, 0)
  attn_mask_pad = tf.zeros([qlen, mlen], dtype=dtype)
  ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
  if same_length:
    mask_l = tf.linalg.band_part(attn_mask, -1, 0)
    ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)

  return ret


def _cache_mem(curr_out, prev_mem, mem_len, reuse_len=None):
  """cache hidden states into memory."""

  if mem_len is None or mem_len == 0:
    return None
  else:
    if reuse_len is not None and reuse_len > 0:
      curr_out = curr_out[:reuse_len]

    if prev_mem is None:
      new_mem = curr_out[-mem_len:]
    else:
      new_mem = tf.concat([prev_mem, curr_out], 0)[-mem_len:]

  return tf_keras.backend.stop_gradient(new_mem)


def is_special_none_tensor(tensor):
  """Checks if a tensor is a special None Tensor."""
  return tensor.shape.ndims == 0 and tensor.dtype == tf.int32


@tf_keras.utils.register_keras_serializable(package="Text")
class RelativePositionEncoding(tf_keras.layers.Layer):
  """Creates a relative positional encoding.

  This layer creates a relative positional encoding as described in
  "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
  (https://arxiv.org/abs/1901.02860).

  Rather than an absolute position embedding as in Transformer, this
  formulation represents position as the relative distance between tokens using
  sinusoidal positional embeddings.

  Note: This layer is currently experimental.

  Attributes:
    hidden_size: The dimensionality of the input embeddings.
  """

  def __init__(self, hidden_size, **kwargs):
    super(RelativePositionEncoding, self).__init__(**kwargs)
    self._hidden_size = hidden_size
    self._inv_freq = 1.0 / (10000.0**(
        tf.range(0, self._hidden_size, 2.0) / self._hidden_size))

  def call(self, pos_seq, batch_size=None):
    """Implements call() for the layer.

    Args:
      pos_seq: A 1-D `Tensor`
      batch_size: The optionally provided batch size that tiles the relative
        positional encoding.

    Returns:
      The relative positional encoding of shape:
        [len(pos_seq), batch_size, hidden_size] if batch_size is provided, else
        [len(pos_seq), 1, hidden_size].
    """
    sinusoid_input = tf.einsum("i,d->id", pos_seq, self._inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_input), tf.cos(sinusoid_input)], -1)
    pos_emb = pos_emb[:, None, :]

    if batch_size is not None:
      pos_emb = tf.tile(pos_emb, [1, batch_size, 1])
    return pos_emb


class RelativeAttention(tf_keras.layers.Layer):
  """Core calculations for relative attention."""

  def __init__(self, dropout_att, scale):
    super(RelativeAttention, self).__init__()
    self.scale = scale
    self.dropout_att = dropout_att

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""

    self.attention_probs_dropout = tf_keras.layers.Dropout(
        rate=self.dropout_att)

    super(RelativeAttention, self).build(unused_input_shapes)

  def call(self, q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
           r_w_bias, r_r_bias, r_s_bias, attn_mask):
    """Implements call() for the layer."""

    # content based attention score
    ac = tf.einsum("ibnd,jbnd->ijbn", q_head + r_w_bias, k_head_h)

    # position based attention score
    bd = tf.einsum("ibnd,jbnd->ijbn", q_head + r_r_bias, k_head_r)
    bd = rel_shift(bd, klen=tf.shape(ac)[1])

    # segment-based attention score
    if seg_mat is None:
      ef = 0
    else:
      ef = tf.einsum("ibnd,snd->isbn", q_head + r_s_bias, seg_embed)
      tgt_shape = tf.shape(bd)
      ef = tf.where(
          tf.broadcast_to(tf.expand_dims(seg_mat, 3), tgt_shape),
          tf.broadcast_to(ef[:, 1:, :, :], tgt_shape),
          tf.broadcast_to(ef[:, :1, :, :], tgt_shape))

    # merges attention scores and performs masking
    attn_score = (ac + bd + ef) * self.scale
    if attn_mask is not None:
      attn_score = attn_score - 1e30 * attn_mask

    # attention probability
    attn_prob = tf.nn.softmax(attn_score, 1)
    attn_prob = self.attention_probs_dropout(attn_prob)

    # attention output
    attn_vec = tf.einsum("ijbn,jbnd->ibnd", attn_prob, v_head_h)

    return attn_vec


class PositionwiseFF(tf_keras.layers.Layer):
  """Positionwise feed-forward layer."""

  def __init__(self, d_model, d_inner, dropout, kernel_initializer,
               activation_type, **kwargs):
    super(PositionwiseFF, self).__init__(**kwargs)
    self.d_model = d_model
    self.d_inner = d_inner
    self.dropout = dropout
    self.activation_type = activation_type
    self.kernel_initializer = kernel_initializer

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    if self.activation_type == "relu":
      activation = tf.nn.relu
    elif self.activation_type == "gelu":
      activation = gelu
    else:
      raise (ValueError("Unsupported activation type {}".format(
          self.activation_type)))
    self.inner_projection_layer = (
        tf_keras.layers.Dense(
            units=self.d_inner,
            activation=activation,
            kernel_initializer=self.kernel_initializer,
            name="layer_1"))
    self.output_projection_layer = (
        tf_keras.layers.Dense(
            units=self.d_model,
            kernel_initializer=self.kernel_initializer,
            name="layer_2"))
    self.output_dropout = tf_keras.layers.Dropout(
        rate=self.dropout, name="drop_2")
    self.output_layer_norm = (
        tf_keras.layers.LayerNormalization(
            name="LayerNorm", axis=-1, epsilon=1e-12))
    super(PositionwiseFF, self).build(unused_input_shapes)

  def call(self, inp):
    """Implements call() for the layer."""

    output = self.inner_projection_layer(inp)
    output = self.output_projection_layer(output)
    output = self.output_dropout(output)
    output = self.output_layer_norm(output + inp)
    return output


class EmbeddingLookup(tf_keras.layers.Layer):
  """Looks up words embeddings for id tensor."""

  def __init__(self, n_token, d_embed, initializer, **kwargs):
    super(EmbeddingLookup, self).__init__(**kwargs)
    self.n_token = n_token
    self.d_embed = d_embed
    self.initializer = initializer

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.lookup_table = self.add_weight(
        "lookup_table",
        shape=[self.n_token, self.d_embed],
        initializer=self.initializer,
        dtype=self.dtype)

    super(EmbeddingLookup, self).build(unused_input_shapes)

  def call(self, inputs):
    return tf.nn.embedding_lookup(self.lookup_table, inputs)


class RelativeMultiheadAttention(tf_keras.layers.Layer):
  """Multi-head attention with relative embedding."""

  def __init__(self, d_model, n_head, d_head, dropout, dropout_att,
               kernel_initializer, **kwargs):
    super(RelativeMultiheadAttention, self).__init__(**kwargs)
    self.d_model = d_model
    self.n_head = n_head
    self.d_head = d_head
    self.dropout = dropout
    self.dropout_att = dropout_att
    self.initializer = kernel_initializer

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.scale = 1.0 / (self.d_head**0.5)

    self.output_layer_norm = tf_keras.layers.LayerNormalization(
        name="LayerNorm", axis=-1, epsilon=1e-12)

    self.kh_projection_layer = self.add_weight(
        "k/kernel",
        shape=[self.d_model, self.n_head, self.d_head],
        initializer=self.initializer)
    self.vh_projection_layer = self.add_weight(
        "v/kernel",
        shape=[self.d_model, self.n_head, self.d_head],
        initializer=self.initializer)
    self.kr_projection_layer = self.add_weight(
        "r/kernel",
        shape=[self.d_model, self.n_head, self.d_head],
        initializer=self.initializer)
    self.qh_projection_layer = self.add_weight(
        "q/kernel",
        shape=[self.d_model, self.n_head, self.d_head],
        initializer=self.initializer)

    self.relative_attention_layer = RelativeAttention(
        dropout_att=self.dropout_att, scale=self.scale)

    self.proj_o = self.add_weight(
        "o/kernel",
        shape=[self.d_model, self.n_head, self.d_head],
        initializer=self.initializer)

    self.attention_dropout = tf_keras.layers.Dropout(rate=self.dropout)

    super(RelativeMultiheadAttention, self).build(unused_input_shapes)

  def call(self, h, g, r, r_w_bias, r_r_bias, seg_mat, r_s_bias, seg_embed,
           attn_mask_h, attn_mask_g, mems, target_mapping):
    """Implements call() for the layer."""

    if mems is not None and mems.shape.ndims > 1:
      cat = tf.concat([mems, h], 0)
    else:
      cat = h

    # content heads
    q_head_h = tf.einsum("ibh,hnd->ibnd", h, self.qh_projection_layer)
    k_head_h = tf.einsum("ibh,hnd->ibnd", cat, self.kh_projection_layer)
    v_head_h = tf.einsum("ibh,hnd->ibnd", cat, self.vh_projection_layer)

    # positional heads
    k_head_r = tf.einsum("ibh,hnd->ibnd", r, self.kr_projection_layer)

    # core attention ops
    attn_vec_h = self.relative_attention_layer(q_head_h, k_head_h, v_head_h,
                                               k_head_r, seg_embed, seg_mat,
                                               r_w_bias, r_r_bias, r_s_bias,
                                               attn_mask_h)

    # post processing
    output_h = tf.einsum("ibnd,hnd->ibh", attn_vec_h, self.proj_o)
    output_h = self.attention_dropout(output_h)
    output_h = self.output_layer_norm(output_h + h)

    output_g = None
    if g is not None:  # enable two-stream attention
      # g-stream
      q_head_g = tf.einsum("ibh,hnd->ibnd", g, self.qh_projection_layer)
      if target_mapping is not None:
        q_head_g = tf.einsum("mbnd,mlb->lbnd", q_head_g, target_mapping)
        attn_vec_g = self.relative_attention_layer(q_head_g, k_head_h, v_head_h,
                                                   k_head_r, seg_embed, seg_mat,
                                                   r_w_bias, r_r_bias, r_s_bias,
                                                   attn_mask_g)
        attn_vec_g = tf.einsum("lbnd,mlb->mbnd", attn_vec_g, target_mapping)

      else:
        attn_vec_g = self.relative_attention_layer(q_head_g, k_head_h, v_head_h,
                                                   k_head_r, seg_embed, seg_mat,
                                                   r_w_bias, r_r_bias, r_s_bias,
                                                   attn_mask_g)

      # post processing
      output_g = tf.einsum("ibnd,hnd->ibh", attn_vec_g, self.proj_o)
      output_g = self.attention_dropout(output_g)
      output_g = self.output_layer_norm(output_g + g)

    return (output_h, output_g)


class TransformerXLModel(tf_keras.layers.Layer):
  """Defines a Transformer-XL computation graph with additional support for XLNet."""

  def __init__(self,
               n_token,
               n_layer,
               d_model,
               n_head,
               d_head,
               d_inner,
               dropout,
               dropout_att,
               attn_type,
               bi_data,
               is_training,
               initializer,
               mem_len=None,
               same_length=False,
               clamp_len=-1,
               untie_r=False,
               use_tpu=True,
               reuse_len=None,
               ff_activation="relu",
               use_cls_mask=False,
               **kwargs):
    """Initializes TransformerXLModel.

    Args:
      n_token: int, the number of tokens in vocabulary.
      n_layer: int, the number of layers.
      d_model: int, the hidden size.
      n_head: int, the number of attention heads.
      d_head: int, the dimension size of each attention head.
      d_inner: int, the hidden size in feed-forward layers.
      dropout: float, dropout rate.
      dropout_att: float, dropout rate on attention probabilities.
      attn_type: str, "uni" or "bi".
      bi_data: bool, whether to use bidirectional input pipeline. Usually set to
        True during pretraining and False during finetuning.
      is_training: bool, whether in training mode.
      initializer: A tf initializer.
      mem_len: int, the number of tokens to cache.
      same_length: bool, whether to use the same attention length for each
        token.
      clamp_len: int, clamp all relative distances larger than clamp_len. -1
        means no clamping.
      untie_r: bool, whether to untie the biases in attention.
      use_tpu: bool, whether TPUs are used.
      reuse_len: int, the number of tokens in the currect batch to be cached and
        reused in the future.
      ff_activation: str, "relu" or "gelu".
      use_cls_mask: bool, whether to introduce cls mask.
      **kwargs: Other parameters.
    """

    super(TransformerXLModel, self).__init__(**kwargs)
    warnings.warn(
        "`TransformerXLModel` is deprecated, please use `XLNetBase` instead",
        DeprecationWarning, stacklevel=2)

    self.n_token = n_token
    self.initializer = initializer
    self.attn_type = attn_type
    self.n_layer = n_layer
    self.d_model = d_model
    self.n_head = n_head
    self.d_head = d_head
    self.d_inner = d_inner
    self.ff_activation = ff_activation
    self.untie_r = untie_r
    self.use_tpu = use_tpu
    self.dropout = dropout
    self.dropout_att = dropout_att

    self.mem_len = mem_len
    self.reuse_len = reuse_len
    self.bi_data = bi_data
    self.clamp_len = clamp_len
    self.same_length = same_length
    self.use_cls_mask = use_cls_mask

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.tf_float = tf.float32

    self.embedding_lookup = EmbeddingLookup(
        n_token=self.n_token,
        d_embed=self.d_model,
        initializer=self.initializer,
        dtype=self.tf_float,
        name="word_embedding")

    self.h_dropout = tf_keras.layers.Dropout(rate=self.dropout)
    self.g_dropout = tf_keras.layers.Dropout(rate=self.dropout)

    if self.untie_r:
      self.r_w_bias = (
          self.add_weight(
              "r_w_bias",
              shape=[self.n_layer, self.n_head, self.d_head],
              dtype=self.tf_float,
              initializer=self.initializer))
      self.r_r_bias = (
          self.add_weight(
              "r_r_bias",
              shape=[self.n_layer, self.n_head, self.d_head],
              dtype=self.tf_float,
              initializer=self.initializer))
      self.r_s_bias = (
          self.add_weight(
              "r_s_bias",
              shape=[self.n_layer, self.n_head, self.d_head],
              dtype=self.tf_float,
              initializer=self.initializer))
    else:
      self.r_w_bias = (
          self.add_weight(
              "r_w_bias",
              shape=[self.n_head, self.d_head],
              dtype=self.tf_float,
              initializer=self.initializer))
      self.r_r_bias = (
          self.add_weight(
              "r_r_bias",
              shape=[self.n_head, self.d_head],
              dtype=self.tf_float,
              initializer=self.initializer))
      self.r_s_bias = (
          self.add_weight(
              "r_s_bias", [self.n_head, self.d_head],
              dtype=self.tf_float,
              initializer=self.initializer))

    self.seg_embed = self.add_weight(
        "seg_embed", [self.n_layer, 2, self.n_head, self.d_head],
        dtype=self.tf_float,
        initializer=self.initializer)

    self.mask_emb = self.add_weight(
        "mask_emb/mask_emb", shape=[1, 1, self.d_model], dtype=self.tf_float)

    self.emb_dropout = tf_keras.layers.Dropout(rate=self.dropout)
    self.fwd_position_embedding = RelativePositionEncoding(self.d_model)
    self.bwd_position_embedding = RelativePositionEncoding(self.d_model)

    self.rel_multihead_layers = []
    self.h_positionwise_ffn_layers = []
    for i in range(self.n_layer):
      self.rel_multihead_layers.append(
          RelativeMultiheadAttention(
              d_model=self.d_model,
              dropout=self.dropout,
              n_head=self.n_head,
              d_head=self.d_head,
              dropout_att=self.dropout_att,
              kernel_initializer=self.initializer,
              name="layer_%d/rel_attn" % (i)))
      self.h_positionwise_ffn_layers.append(
          PositionwiseFF(
              d_model=self.d_model,
              d_inner=self.d_inner,
              dropout=self.dropout,
              kernel_initializer=self.initializer,
              activation_type=self.ff_activation,
              name="layer_%d/ff" % (i)))

    self.output_dropout = tf_keras.layers.Dropout(rate=self.dropout)

    super(TransformerXLModel, self).build(unused_input_shapes)

  def __call__(self,
               inp_k,
               seg_id=None,
               input_mask=None,
               mems=None,
               perm_mask=None,
               target_mapping=None,
               inp_q=None,
               **kwargs):
    # Uses dict to feed inputs into call() in order to keep mems as a python
    # list.
    inputs = {
        "inp_k": inp_k,
        "seg_id": seg_id,
        "input_mask": input_mask,
        "mems": mems,
        "perm_mask": perm_mask,
        "target_mapping": target_mapping,
        "inp_q": inp_q
    }
    return super(TransformerXLModel, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    inp_k = inputs["inp_k"]
    seg_id = inputs["seg_id"]
    input_mask = inputs["input_mask"]
    mems = inputs["mems"]
    perm_mask = inputs["perm_mask"]
    target_mapping = inputs["target_mapping"]
    inp_q = inputs["inp_q"]

    new_mems = []

    bsz = tf.shape(inp_k)[1]

    qlen = inp_k.shape.as_list()[0]

    mlen = mems[0].shape.as_list()[0] if mems is not None else 0
    klen = mlen + qlen

    ##### Attention mask
    # causal attention mask
    if self.attn_type == "uni":
      attn_mask = _create_mask(qlen, mlen, self.tf_float, self.same_length)
      # pylint: enable=protected-access
      attn_mask = attn_mask[:, :, None, None]
    elif self.attn_type == "bi":
      attn_mask = None
    else:
      raise ValueError("Unsupported attention type: {}".format(self.attn_type))

    # data mask: input mask & perm mask
    if input_mask is not None and perm_mask is not None:
      data_mask = input_mask[None] + perm_mask

    elif input_mask is not None and perm_mask is None:
      data_mask = input_mask[None]
    elif input_mask is None and perm_mask is not None:
      data_mask = perm_mask
    else:
      data_mask = None

    if data_mask is not None:
      # all mems can be attended to
      mems_mask = tf.zeros([tf.shape(data_mask)[0], mlen, bsz],
                           dtype=self.tf_float)
      data_mask = tf.concat([mems_mask, data_mask], 1)
      if attn_mask is None:
        attn_mask = data_mask[:, :, :, None]
      else:
        attn_mask += data_mask[:, :, :, None]

    if attn_mask is not None:
      attn_mask = tf.cast(attn_mask > 0, dtype=self.tf_float)

    if attn_mask is not None:
      non_tgt_mask = -tf.eye(qlen, dtype=self.tf_float)
      non_tgt_mask = tf.concat(
          [tf.zeros([qlen, mlen], dtype=self.tf_float), non_tgt_mask], axis=-1)
      non_tgt_mask = tf.cast(
          (attn_mask + non_tgt_mask[:, :, None, None]) > 0, dtype=self.tf_float)
    else:
      non_tgt_mask = None

    word_emb_k = self.embedding_lookup(inp_k)

    if inp_q is not None:
      if target_mapping is not None:
        word_emb_q = tf.tile(self.mask_emb,
                             [tf.shape(target_mapping)[0], bsz, 1])
      else:
        inp_q_ext = inp_q[:, :, None]
        word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k

    output_h = self.h_dropout(word_emb_k)
    output_g = None
    if inp_q is not None:
      output_g = self.g_dropout(word_emb_q)

    ##### Segment embedding
    if seg_id is not None:

      # Convert `seg_id` to one-hot `seg_mat`

      mem_pad = tf.zeros([mlen, bsz], dtype=tf.int32)

      cat_id = tf.concat([mem_pad, seg_id], 0)

      if self.use_cls_mask:
        # `1` indicates not in the same segment [qlen x klen x bsz]
        # seg_id: [qlen x bsz] & cat_id: [klen x bsz]
        cls_mat = tf.logical_or(
            tf.equal(seg_id, tf.constant([data_utils.SEG_ID_CLS]))[:, None],
            tf.equal(cat_id, tf.constant([data_utils.SEG_ID_CLS]))[None, :])
        seg_mat = tf.equal(seg_id[:, None], cat_id[None, :])
        seg_mat = tf.logical_or(cls_mat, seg_mat)
      else:
        seg_mat = tf.logical_not(tf.equal(seg_id[:, None], cat_id[None, :]))
    else:
      seg_mat = None

    dtype = self.tf_float
    freq_seq = tf.range(0, self.d_model, 2.0)
    if dtype is not None and dtype != tf.float32:
      freq_seq = tf.cast(freq_seq, dtype=self.dtype)

    if self.attn_type == "bi":
      beg, end = klen, -qlen
    elif self.attn_type == "uni":
      beg, end = klen, -1
    else:
      raise ValueError("Unknown `attn_type` {}.".format(self.attn_type))

    if self.bi_data:
      fwd_pos_seq = tf.range(beg, end, -1.0)
      bwd_pos_seq = tf.range(-beg, -end, 1.0)

      if dtype is not None and dtype != tf.float32:
        fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
        bwd_pos_seq = tf.cast(bwd_pos_seq, dtype=dtype)

      if self.clamp_len > 0:
        fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len,
                                       self.clamp_len)
        bwd_pos_seq = tf.clip_by_value(bwd_pos_seq, -self.clamp_len,
                                       self.clamp_len)

      if bsz is not None:
        fwd_pos_emb = self.fwd_position_embedding(fwd_pos_seq, bsz // 2)
        bwd_pos_emb = self.bwd_position_embedding(bwd_pos_seq, bsz // 2)
      else:
        fwd_pos_emb = self.fwd_position_embedding(fwd_pos_seq, None)
        bwd_pos_emb = self.bwd_position_embedding(bwd_pos_seq, None)

      pos_emb = tf.concat([fwd_pos_emb, bwd_pos_emb], axis=1)
    else:
      fwd_pos_seq = tf.range(beg, end, -1.0)
      if dtype is not None and dtype != tf.float32:
        fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
      if self.clamp_len > 0:
        fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len,
                                       self.lamp_len)

      pos_emb = self.fwd_position_embedding(fwd_pos_seq, bsz)

    pos_emb = self.emb_dropout(pos_emb)

    if mems is None:
      mems = [None] * self.n_layer
    for i in range(self.n_layer):
      # cache new mems
      new_mems.append(
          _cache_mem(output_h, mems[i], self.mem_len, self.reuse_len))
      # pylint: enable=protected-access

      # segment bias
      if seg_id is None:
        r_s_bias_i = None
        seg_embed_i = None
      else:
        r_s_bias_i = self.r_s_bias if not self.untie_r else self.r_s_bias[i]
        seg_embed_i = self.seg_embed[i]

      ffn_layer = self.h_positionwise_ffn_layers[i]
      attention_layer = self.rel_multihead_layers[i]
      output_h, output_g = attention_layer(
          h=output_h,
          g=output_g,
          r=pos_emb,
          r_w_bias=self.r_w_bias if not self.untie_r else self.r_w_bias[i],
          r_r_bias=self.r_r_bias if not self.untie_r else self.r_r_bias[i],
          seg_mat=seg_mat,
          r_s_bias=r_s_bias_i,
          seg_embed=seg_embed_i,
          attn_mask_h=non_tgt_mask,
          attn_mask_g=attn_mask,
          mems=mems[i],
          target_mapping=target_mapping)
      output_h = ffn_layer(output_h)
      if output_g is not None:
        output_g = ffn_layer(output_g)

    if inp_q is not None:
      output = output_g
    else:
      output = output_h

    return output, new_mems, None


class PretrainingXLNetModel(tf_keras.Model):
  """XLNet keras model combined with pretraining LM loss layer.

  See the original paper: https://arxiv.org/pdf/1906.08237.pdf

  """

  def __init__(self, use_proj, xlnet_config, run_config, use_legacy_mask=True,
               **kwargs):
    super(PretrainingXLNetModel, self).__init__(**kwargs)
    self.run_config = run_config
    self.initializer = _get_initializer(run_config)
    self.xlnet_config = copy.deepcopy(xlnet_config)
    self._use_legacy_mask = use_legacy_mask

    self.xlnet_model = networks.XLNetBase(
        vocab_size=self.xlnet_config.n_token,
        initializer=self.initializer,
        attention_type="bi",
        num_layers=self.xlnet_config.n_layer,
        hidden_size=self.xlnet_config.d_model,
        num_attention_heads=self.xlnet_config.n_head,
        head_size=self.xlnet_config.d_head,
        inner_size=self.xlnet_config.d_inner,
        two_stream=True,
        tie_attention_biases=not self.xlnet_config.untie_r,
        inner_activation=self.xlnet_config.ff_activation,
        dropout_rate=self.run_config.dropout,
        attention_dropout_rate=self.run_config.dropout_att,
        memory_length=self.run_config.mem_len,
        reuse_length=self.run_config.reuse_len,
        bi_data=self.run_config.bi_data,
        clamp_length=self.run_config.clamp_len,
        use_cls_mask=self.run_config.use_cls_mask,
        name="xlnet_model")

    self.lmloss_layer = LMLossLayer(
        vocab_size=self.xlnet_config.n_token,
        hidden_size=self.xlnet_config.d_model,
        initializer=self.initializer,
        tie_weight=True,
        bi_data=self.run_config.bi_data,
        use_one_hot=self.run_config.use_tpu,
        use_proj=use_proj,
        name="lm_loss")

  def call(self, features):
    """Implements call() for the layer."""

    input_ids = features["input_ids"]
    masked_tokens = features["input_q"]
    seg_ids = features["seg_id"]
    if self._use_legacy_mask:
      # Legacy input mask assumes `real` values are 0 and `padding`
      # values are 1.
      perm_mask = 1 - features["perm_mask"]
    else:
      perm_mask = features["perm_mask"]
    target_mapping = features["target_mapping"]

    # target for LM loss
    target = features["target"]

    # target mask for LM loss
    tgt_mask = features["target_mask"]

    mems = features.get("mems", None)

    model_output, self.new_mems = self.xlnet_model(
        input_ids=input_ids,
        segment_ids=seg_ids,
        input_mask=None,
        state=mems,
        permutation_mask=perm_mask,
        target_mapping=target_mapping,
        masked_tokens=masked_tokens)
    lm_loss, _ = self.lmloss_layer(
        hidden=model_output,
        target=target,
        lookup_table=self.xlnet_model.get_embedding_lookup_table(),
        target_mask=tgt_mask)
    self.add_loss(lm_loss)
    return self.new_mems, model_output


class ClassificationXLNetModel(tf_keras.Model):
  """XLNet keras model combined with classification loss layer.

  See the original paper: https://arxiv.org/pdf/1906.08237.pdf

  """

  def __init__(self, xlnet_config, run_config, n_class, summary_type,
               use_legacy_mask=True, **kwargs):
    super(ClassificationXLNetModel, self).__init__(**kwargs)
    warnings.warn(
        "`ClassificationXLNetModel` is deprecated, please use `XLNetClassifier`"
        "instead.", DeprecationWarning, stacklevel=2)
    self.run_config = run_config
    self.initializer = _get_initializer(run_config)
    self.xlnet_config = copy.deepcopy(xlnet_config)
    self._use_legacy_mask = use_legacy_mask

    self.xlnet_model = networks.XLNetBase(
        vocab_size=self.xlnet_config.n_token,
        initializer=self.initializer,
        attention_type="bi",
        num_layers=self.xlnet_config.n_layer,
        hidden_size=self.xlnet_config.d_model,
        num_attention_heads=self.xlnet_config.n_head,
        head_size=self.xlnet_config.d_head,
        inner_size=self.xlnet_config.d_inner,
        two_stream=False,
        tie_attention_biases=not self.xlnet_config.untie_r,
        inner_activation=self.xlnet_config.ff_activation,
        dropout_rate=self.run_config.dropout,
        attention_dropout_rate=self.run_config.dropout_att,
        memory_length=self.run_config.mem_len,
        reuse_length=self.run_config.reuse_len,
        bi_data=self.run_config.bi_data,
        clamp_length=self.run_config.clamp_len,
        use_cls_mask=False,
        name="xlnet_model")

    self.summarization_layer = Summarization(
        hidden_size=self.xlnet_config.d_model,
        num_attention_heads=self.xlnet_config.n_head,
        head_size=self.xlnet_config.d_head,
        dropout_rate=self.run_config.dropout,
        attention_dropout_rate=self.run_config.dropout_att,
        initializer=self.initializer,
        use_proj=True,
        summary_type=summary_type,
        name="sequence_summary")

    self.cl_loss_layer = ClassificationLossLayer(
        n_class=n_class, initializer=self.initializer, name="classification")

  def call(self, features):
    """Implements call() for the layer."""
    batch_size_per_core = tf.shape(features["input_ids"])[0]

    input_ids = features["input_ids"]
    segment_ids = features["segment_ids"]
    if self._use_legacy_mask:
      # Legacy input mask assumes `real` values are 0 and `padding`
      # values are 1.
      input_mask = 1 - features["input_mask"]
    else:
      input_mask = features["input_mask"]

    label = tf.reshape(features["label_ids"], [batch_size_per_core])

    mems = features.get("mems", None)

    attention_output, new_mems = (
        self.xlnet_model(input_ids, segment_ids, input_mask, mems))

    summary = self.summarization_layer(attention_output)
    per_example_loss, logits = self.cl_loss_layer(hidden=summary, labels=label)
    self.add_loss(tf_keras.backend.mean(per_example_loss))
    return new_mems, logits


class LMLossLayer(tf_keras.layers.Layer):
  """Layer computing cross entropy loss for language modeling."""

  def __init__(self,
               vocab_size,
               hidden_size,
               initializer,
               tie_weight=False,
               bi_data=True,
               use_one_hot=False,
               use_proj=False,
               **kwargs):
    """Constructs LMLoss layer.

    Args:
      vocab_size: Number of tokens in vocabulary.
      hidden_size: The dimension of model hidden state.
      initializer: Initializer used for parameters.
      tie_weight: Whether to share weights between embedding lookup layer and
        next-token prediction layer.
      bi_data: Whether to use bidirectional input pipeline. Usually set to True
        during pretraining and False during finetuning.
      use_one_hot: bool, whether to use one hot encodings. This should be used
        when TPUs are used.
      use_proj: bool, whether to add a projection layer before LM prediction.
      **kwargs: Other parameters.
    """
    super(LMLossLayer, self).__init__(**kwargs)
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.initializer = initializer

    self.tie_weight = tie_weight
    self.bi_data = bi_data
    self.use_one_hot = use_one_hot
    self.use_proj = use_proj

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    if self.use_proj:
      self.proj_layer = tf_keras.layers.Dense(
          units=self.hidden_size,
          kernel_initializer=self.initializer,
          activation=gelu,
          name="lm_projection/dense")
      self.proj_layer_norm = tf_keras.layers.LayerNormalization(
          axis=-1, epsilon=1e-12, name="lm_projection/LayerNorm")
    if not self.tie_weight:
      self.softmax_w = self.add_weight(
          "weight",
          shape=[self.vocab_size, self.hidden_size],
          initializer=self.initializer)

    self.softmax_b = self.add_weight(
        "bias", shape=[self.vocab_size], initializer=tf.zeros_initializer())

    super(LMLossLayer, self).build(unused_input_shapes)

  def call(self, hidden, target, lookup_table, target_mask):
    """Implements call() for the layer."""
    if self.use_proj:
      hidden = self.proj_layer_norm(self.proj_layer(hidden))
    if self.tie_weight:
      logits = tf.einsum("ibd,nd->ibn", hidden, lookup_table) + self.softmax_b
    else:
      logits = tf.einsum("ibd,nd->ibn", hidden, self.softmax_w) + self.softmax_b

    if self.use_one_hot:
      one_hot_target = tf.one_hot(target, self.vocab_size, dtype=logits.dtype)
      loss = -tf.reduce_sum(tf.nn.log_softmax(logits) * one_hot_target, -1)
    else:
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=target, logits=logits)

    total_loss = tf.reduce_sum(loss * target_mask) / tf.reduce_sum(target_mask)

    return total_loss, logits


class Summarization(tf_keras.layers.Layer):
  """The layer to pool the output from XLNet model into a vector."""

  def __init__(self,
               hidden_size,
               num_attention_heads,
               head_size,
               dropout_rate,
               attention_dropout_rate,
               initializer,
               use_proj=True,
               summary_type="last",
               **kwargs):
    """Constructs Summarization layer.

    Args:
      hidden_size: int, the dimension of model hidden state.
      num_attention_heads: int, the number of attention heads.
      head_size: int, the dimension size of each attention head.
      dropout_rate: float, dropout rate.
      attention_dropout_rate: float, dropout rate on attention probabilities.
      initializer: Initializer used for parameters.
      use_proj: bool, whether to use projection layer for summarization.
      summary_type: Method used to summarize a sequence into a compact vector.
      **kwargs: Other parameters.
    """
    super(Summarization, self).__init__(**kwargs)
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.head_size = head_size
    self.initializer = initializer

    self.dropout_rate = dropout_rate
    self.attention_dropout_rate = attention_dropout_rate
    self.use_proj = use_proj
    self.summary_type = summary_type

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    if self.use_proj:
      self.proj_layer = tf_keras.layers.Dense(
          units=self.hidden_size,
          kernel_initializer=self.initializer,
          activation=tf.nn.tanh,
          name="summary")
    self.dropout_layer = tf_keras.layers.Dropout(rate=self.dropout_rate)

    super(Summarization, self).build(unused_input_shapes)

  def call(self, inputs):
    """Implements call() for the layer."""
    if self.summary_type == "last":
      summary = inputs[:, -1, :]
    elif self.summary_type == "first":
      summary = inputs[:, 0, :]
    else:
      raise ValueError("Invalid summary type provided: %s" % self.summary_type)
    if self.use_proj:
      summary = self.proj_layer(summary)
    summary = self.dropout_layer(summary)
    return summary


class ClassificationLossLayer(tf_keras.layers.Layer):
  """Layer computing cross entropy loss for classification task."""

  def __init__(self, n_class, initializer, **kwargs):
    """Constructs Summarization layer.

    Args:
      n_class: Number of tokens in vocabulary.
      initializer: Initializer used for parameters.
      **kwargs: Other parameters.
    """
    super(ClassificationLossLayer, self).__init__(**kwargs)

    self.n_class = n_class
    self.initializer = initializer

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.proj_layer = tf_keras.layers.Dense(
        units=self.n_class, kernel_initializer=self.initializer, name="logit")

    super(ClassificationLossLayer, self).build(unused_input_shapes)

  def call(self, hidden, labels):
    """Implements call() for the layer."""

    logits = self.proj_layer(hidden)
    one_hot_target = tf.one_hot(labels, self.n_class, dtype=hidden.dtype)  # pytype: disable=attribute-error
    loss = -tf.reduce_sum(tf.nn.log_softmax(logits) * one_hot_target, -1)

    return loss, logits


class QAXLNetModel(tf_keras.Model):
  """XLNet keras model combined with question answering loss layer.

  See the original paper: https://arxiv.org/pdf/1906.08237.pdf

  """

  def __init__(self, xlnet_config, run_config, start_n_top, end_n_top,
               use_legacy_mask=True, **kwargs):
    super(QAXLNetModel, self).__init__(**kwargs)
    warnings.warn(
        "`QAXLNetModel` is deprecated, please use `XLNetSpanLabeler` instead.",
        DeprecationWarning, stacklevel=2)
    self.run_config = run_config
    self.initializer = _get_initializer(run_config)
    self.xlnet_config = copy.deepcopy(xlnet_config)
    self._use_legacy_mask = use_legacy_mask

    self.xlnet_model = networks.XLNetBase(
        vocab_size=self.xlnet_config.n_token,
        initializer=self.initializer,
        attention_type="bi",
        num_layers=self.xlnet_config.n_layer,
        hidden_size=self.xlnet_config.d_model,
        num_attention_heads=self.xlnet_config.n_head,
        head_size=self.xlnet_config.d_head,
        inner_size=self.xlnet_config.d_inner,
        tie_attention_biases=not self.xlnet_config.untie_r,
        inner_activation=self.xlnet_config.ff_activation,
        dropout_rate=self.run_config.dropout,
        attention_dropout_rate=self.run_config.dropout_att,
        two_stream=False,
        memory_length=self.run_config.mem_len,
        reuse_length=self.run_config.reuse_len,
        bi_data=self.run_config.bi_data,
        clamp_length=self.run_config.clamp_len,
        use_cls_mask=False,
        name="xlnet_model")

    self.qa_loss_layer = QALossLayer(
        hidden_size=self.xlnet_config.d_model,
        start_n_top=start_n_top,
        end_n_top=end_n_top,
        initializer=self.initializer,
        dropout_rate=self.run_config.dropout,
        name="qa_loss_layer")

  def call(self, features, training=False):
    """Implements call() for the layer."""

    input_ids = features["input_ids"]
    segment_ids = features["segment_ids"]
    if self._use_legacy_mask:
      # Legacy input mask assumes `real` values are 0 and `padding`
      # values are 1.
      input_mask = 1 - features["input_mask"]
    else:
      input_mask = features["input_mask"]

    cls_index = tf.reshape(features["cls_index"], [-1])
    p_mask = features["p_mask"]

    attention_output, new_mems = (
        self.xlnet_model(input_ids, segment_ids, input_mask))

    if training:
      loss, logits = self.qa_loss_layer(
          hidden=attention_output,
          p_mask=p_mask,
          cls_index=cls_index,
          start_positions=features["start_positions"],
          end_positions=features["end_positions"],
          is_impossible=features["is_impossible"])
      self.add_loss(loss)
      return new_mems, logits
    else:
      results = self.qa_loss_layer(
          hidden=attention_output, p_mask=p_mask, cls_index=cls_index)
      return results


class QALossLayer(tf_keras.layers.Layer):
  """Layer computing position and regression loss for question answering task."""

  def __init__(self, hidden_size, start_n_top, end_n_top, initializer,
               dropout_rate, **kwargs):
    """Constructs Summarization layer.

    Args:
      hidden_size: Int, the hidden size.
      start_n_top: Beam size for span start.
      end_n_top: Beam size for span end.
      initializer: Initializer used for parameters.
      dropout_rate: float, dropout rate.
      **kwargs: Other parameters.
    """
    super(QALossLayer, self).__init__(**kwargs)
    self.hidden_size = hidden_size
    self.start_n_top = start_n_top
    self.end_n_top = end_n_top
    self.initializer = initializer
    self.dropout_rate = dropout_rate

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.start_logits_proj_layer = tf_keras.layers.Dense(
        units=1, kernel_initializer=self.initializer, name="start_logits/dense")
    self.end_logits_proj_layer0 = tf_keras.layers.Dense(
        units=self.hidden_size,
        kernel_initializer=self.initializer,
        activation=tf.nn.tanh,
        name="end_logits/dense_0")
    self.end_logits_proj_layer1 = tf_keras.layers.Dense(
        units=1, kernel_initializer=self.initializer, name="end_logits/dense_1")
    self.end_logits_layer_norm = tf_keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12, name="end_logits/LayerNorm")
    self.answer_class_proj_layer0 = tf_keras.layers.Dense(
        units=self.hidden_size,
        kernel_initializer=self.initializer,
        activation=tf.nn.tanh,
        name="answer_class/dense_0")
    self.answer_class_proj_layer1 = tf_keras.layers.Dense(
        units=1,
        kernel_initializer=self.initializer,
        use_bias=False,
        name="answer_class/dense_1")
    self.ans_feature_dropout = tf_keras.layers.Dropout(rate=self.dropout_rate)
    super(QALossLayer, self).build(unused_input_shapes)

  def __call__(self, hidden, p_mask, cls_index, **kwargs):
    return super(QALossLayer, self).__call__(
        (hidden, p_mask, cls_index, kwargs))

  def call(self, inputs, training=False):
    """Implements call() for the layer."""
    hidden, p_mask, cls_index, kwargs = inputs
    return_dict = {}
    seq_len = tf.shape(hidden)[1]

    hidden = tf.transpose(hidden, [1, 0, 2])
    start_logits = self.start_logits_proj_layer(hidden)
    start_logits = tf.transpose(tf.squeeze(start_logits, -1), [1, 0])
    start_logits_masked = start_logits * (1 - p_mask) - 1e30 * p_mask
    start_log_probs = tf.nn.log_softmax(start_logits_masked, -1)
    if training:
      start_positions = kwargs["start_positions"]
      end_positions = kwargs["end_positions"]
      is_impossible = kwargs["is_impossible"]
      start_positions = tf.reshape(start_positions, [-1])
      start_index = tf.one_hot(
          start_positions, depth=seq_len, axis=-1, dtype=tf.float32)
      start_features = tf.einsum("lbh,bl->bh", hidden, start_index)
      start_features = tf.tile(start_features[None], [seq_len, 1, 1])
      end_logits = self.end_logits_proj_layer0(
          tf.concat([hidden, start_features], axis=-1))

      end_logits = self.end_logits_layer_norm(end_logits)

      end_logits = self.end_logits_proj_layer1(end_logits)
      end_logits = tf.transpose(tf.squeeze(end_logits, -1), [1, 0])
      end_logits_masked = end_logits * (1 - p_mask) - 1e30 * p_mask
      end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)
    else:
      # during inference, compute the end logits based on beam search

      start_top_log_probs, start_top_index = tf.nn.top_k(
          start_log_probs, k=self.start_n_top)
      start_index = tf.one_hot(
          start_top_index, depth=seq_len, axis=-1, dtype=tf.float32)
      start_features = tf.einsum("lbh,bkl->bkh", hidden, start_index)
      end_input = tf.tile(hidden[:, :, None], [1, 1, self.start_n_top, 1])
      start_features = tf.tile(start_features[None], [seq_len, 1, 1, 1])
      end_input = tf.concat([end_input, start_features], axis=-1)
      end_logits = self.end_logits_proj_layer0(end_input)
      end_logits = tf.reshape(end_logits, [seq_len, -1, self.hidden_size])
      end_logits = self.end_logits_layer_norm(end_logits)

      end_logits = tf.reshape(end_logits,
                              [seq_len, -1, self.start_n_top, self.hidden_size])

      end_logits = self.end_logits_proj_layer1(end_logits)
      end_logits = tf.reshape(end_logits, [seq_len, -1, self.start_n_top])
      end_logits = tf.transpose(end_logits, [1, 2, 0])
      end_logits_masked = end_logits * (
          1 - p_mask[:, None]) - 1e30 * p_mask[:, None]
      end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)
      end_top_log_probs, end_top_index = tf.nn.top_k(
          end_log_probs, k=self.end_n_top)
      end_top_log_probs = tf.reshape(end_top_log_probs,
                                     [-1, self.start_n_top * self.end_n_top])
      end_top_index = tf.reshape(end_top_index,
                                 [-1, self.start_n_top * self.end_n_top])

    if training:
      return_dict["start_log_probs"] = start_log_probs
      return_dict["end_log_probs"] = end_log_probs
    else:
      return_dict["start_top_log_probs"] = start_top_log_probs
      return_dict["start_top_index"] = start_top_index
      return_dict["end_top_log_probs"] = end_top_log_probs
      return_dict["end_top_index"] = end_top_index
    # an additional layer to predict answerability

    # get the representation of CLS
    cls_index = tf.one_hot(cls_index, seq_len, axis=-1, dtype=tf.float32)
    cls_feature = tf.einsum("lbh,bl->bh", hidden, cls_index)

    # get the representation of START
    start_p = tf.nn.softmax(start_logits_masked, axis=-1, name="softmax_start")
    start_feature = tf.einsum("lbh,bl->bh", hidden, start_p)

    ans_feature = tf.concat([start_feature, cls_feature], -1)
    ans_feature = self.answer_class_proj_layer0(ans_feature)
    ans_feature = self.ans_feature_dropout(ans_feature)
    cls_logits = self.answer_class_proj_layer1(ans_feature)
    cls_logits = tf.squeeze(cls_logits, -1)
    return_dict["cls_logits"] = cls_logits

    if not training:
      return return_dict

    def compute_loss(log_probs, positions):
      one_hot_positions = tf.one_hot(positions, depth=seq_len, dtype=tf.float32)

      loss = -tf.reduce_sum(one_hot_positions * log_probs, axis=-1)
      loss = tf.reduce_mean(loss)
      return loss

    start_loss = compute_loss(start_log_probs, start_positions)
    end_loss = compute_loss(end_log_probs, end_positions)

    total_loss = (start_loss + end_loss) * 0.5

    is_impossible = tf.reshape(is_impossible, [-1])
    regression_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=is_impossible, logits=cls_logits)
    regression_loss = tf.reduce_mean(regression_loss)

    total_loss += regression_loss * 0.5
    return total_loss, cls_logits
