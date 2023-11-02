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

"""Perceiver modeling utils."""

import functools
import tensorflow as tf, tf_keras


def make_cross_attention_mask(query_mask, kv_mask):
  """Compute the outer product between `query_mask` and `kv_mask`."""
  # Porting `mask = jax.vmap(jnp.outer)(query_mask, kv_mask)`
  return tf.einsum("ab,ac->abc", query_mask, kv_mask)


def build_cross_attention_block_args(
    input_shape,
    widening_factor=1,
    dropout_prob=0.0,
    dropout_attn_prob=0.0,
    num_heads=8,
    att_init_scale=1.0,
    dense_init_scale=1.0,
    shape_for_attn="kv",
    use_query_residual=True,
    norm_epsilon=1e-5,
    qk_last_dim=None,
    v_last_dim=None):
  """Builds cross attention block arguments for `TransformerEncoderBlock`.

  Build cross attention block arguments for `TransformerEncoderBlock` used
  in Perceiver.

  The last dimension of the output of the attention block or `output_last_dim`
  of `TransformerEncoderBlocks` is set to the first `input_shape`'s last
  dimension.

  `diff_q_kv_att_layer_norm` is set to `True`.

  `inner_dropout` is set to 0.

  `norm_first` is set to `True`.

  `inner_activation` is set to gelu.

  `kernel_initializer` and `attention_initializer` are both
  `tf_keras.initializers.VarianceScaling`.

  Args:
    input_shape:
      Check `input_shape` doc in `_build_transformer_encoder_block_args`.
    widening_factor:
      Check `widening_factor` doc in `_build_transformer_encoder_block_args`.
    dropout_prob:
      Check `dropout_prob` doc in `_build_transformer_encoder_block_args`.
    dropout_attn_prob:
      Check `dropout_attn_prob` doc in `_build_transformer_encoder_block_args`.
    num_heads:
      Check `num_heads` doc in `_build_transformer_encoder_block_args`.
    att_init_scale:
      Check `att_init_scale` doc in `_build_transformer_encoder_block_args`.
    dense_init_scale:
      Check `dense_init_scale` doc in `_build_transformer_encoder_block_args`.
    shape_for_attn:
      Valid values are `q` or `kv`. This value is used to determine the last
      dimension of the attention score output attention last dimension.
      `qk_last_dim` has higher precedence over `shape_for_attn`.
    use_query_residual:
      Toggle to execute residual connection after attention.
    norm_epsilon:
      Check `norm_epsilon` doc in `_build_transformer_encoder_block_args`.
    qk_last_dim:
      When set, determines the last dimension of the attention score output.
      When it's `None`, it uses the first `input_shape`'s last dimension as the
      last dimension of the attention score output. `qk_last_dim` has higher
      precedence over `shape_for_attn`.
    v_last_dim:
      Check `v_last_dim` doc in `_build_transformer_encoder_block_args`.

  Returns:
    A `dict` mapping `TransformerEncoderBlock` arguments.

  References:
    [Perceiver: General Perception with Iterative
    Attention](https://arxiv.org/abs/2103.03206)
    (https://github.com/deepmind/deepmind-research/blob/master/perceiver/perceiver.py)
    (https://github.com/tensorflow/models/blob/871c4e0a393ef4385534bee55354a5df8aa1ccf4/official/nlp/modeling/layers/transformer_encoder_block.py)
  """
  inputs_q_shape = input_shape[0]
  inputs_kv_shape = input_shape[1]

  output_last_dim = inputs_q_shape[-1]

  if shape_for_attn == "q":
    f_qk_last_dim = inputs_q_shape[-1]
  elif shape_for_attn == "kv":
    f_qk_last_dim = inputs_kv_shape[-1]
  else:
    raise ValueError(f"Unknown value {shape_for_attn} for "
                     "shape_for_attention.")

  f_v_last_dim = None
  if qk_last_dim is not None:
    f_qk_last_dim = qk_last_dim
  if v_last_dim is not None:
    f_v_last_dim = v_last_dim

  return _build_transformer_encoder_block_args(
      input_shape=input_shape,
      widening_factor=widening_factor,
      dropout_prob=dropout_prob,
      dropout_attn_prob=dropout_attn_prob,
      num_heads=num_heads,
      att_init_scale=att_init_scale,
      dense_init_scale=dense_init_scale,
      use_query_residual=use_query_residual,
      norm_epsilon=norm_epsilon,
      qk_last_dim=f_qk_last_dim,
      v_last_dim=f_v_last_dim,
      diff_q_kv_att_layer_norm=True,
      output_last_dim=output_last_dim)


def build_self_attention_block_args(
    input_shape,
    widening_factor=4,
    dropout_prob=0.0,
    dropout_attn_prob=0.0,
    num_heads=8,
    att_init_scale=1.0,
    dense_init_scale=1.0,
    norm_epsilon=1e-5,
    qk_last_dim=None,
    v_last_dim=None):
  """Builds self attention block arguments for `TransformerEncoderBlock`.

  Light wrapper around `_build_transformer_encoder_block_args` with some
  assumptions around self attention block. Builds the arguments for
  `TransformerEncoderBlock` used in Perceiver.

  The last dimension of the output of the attention block or `output_last_dim`
  of `TransformerEncoderBlocks` is set using the logic described in the
  doc associated with `output_last_dim` in
  `_build_transformer_encoder_block_args`.

  `diff_q_kv_att_layer_norm` is set to `False`.

  `use_query_residual` is set to `True`.

  `inner_dropout` is set to 0.

  `norm_first` is set to `True`.

  `inner_activation` is set to gelu.

  `kernel_initializer` and `attention_initializer` are both
  `tf_keras.initializers.VarianceScaling`.

  Args:
    input_shape:
      Check `input_shape` doc in `_build_transformer_encoder_block_args`.
    widening_factor:
      Check `widening_factor` doc in `_build_transformer_encoder_block_args`.
    dropout_prob:
      Check `dropout_prob` doc in `_build_transformer_encoder_block_args`.
    dropout_attn_prob:
      Check `dropout_attn_prob` doc in `_build_transformer_encoder_block_args`.
    num_heads:
      Check `num_heads` doc in `_build_transformer_encoder_block_args`.
    att_init_scale:
      Check `att_init_scale` doc in `_build_transformer_encoder_block_args`.
    dense_init_scale:
      Check `dense_init_scale` doc in `_build_transformer_encoder_block_args`.
    norm_epsilon:
      Check `norm_epsilon` doc in `_build_transformer_encoder_block_args`.
    qk_last_dim:
      Check `qk_last_dim` doc in `_build_transformer_encoder_block_args`.
    v_last_dim:
      Check `v_last_dim` doc in `_build_transformer_encoder_block_args`.

  Returns:
    A `dict` mapping `TransformerEncoderBlock` arguments.

  References:
    [Perceiver: General Perception with Iterative
    Attention](https://arxiv.org/abs/2103.03206)
    (https://github.com/deepmind/deepmind-research/blob/master/perceiver/perceiver.py)
    (https://github.com/tensorflow/models/blob/871c4e0a393ef4385534bee55354a5df8aa1ccf4/official/nlp/modeling/layers/transformer_encoder_block.py)
  """

  return _build_transformer_encoder_block_args(
      input_shape=input_shape,
      widening_factor=widening_factor,
      dropout_prob=dropout_prob,
      dropout_attn_prob=dropout_attn_prob,
      num_heads=num_heads,
      att_init_scale=att_init_scale,
      dense_init_scale=dense_init_scale,
      use_query_residual=True,
      norm_epsilon=norm_epsilon,
      qk_last_dim=qk_last_dim,
      v_last_dim=v_last_dim,
      diff_q_kv_att_layer_norm=False,
      output_last_dim=None)


def _build_transformer_encoder_block_args(
    input_shape,
    widening_factor,
    dropout_prob,
    dropout_attn_prob,
    num_heads,
    att_init_scale,
    dense_init_scale,
    use_query_residual,
    norm_epsilon,
    qk_last_dim,
    v_last_dim,
    diff_q_kv_att_layer_norm,
    output_last_dim):
  """Build arguments for `TransformerEncoderBlock`.

  `inner_dropout` is set to 0.

  `norm_first` is set to `True`.

  `inner_activation` is set to gelu.

  `kernel_initializer` and `attention_initializer` are both
  `tf_keras.initializers.VarianceScaling`.

  Args:
    input_shape:
      input shape(s). Usually passed through `build` method in
      `tf_keras.layers.Layer`.
    widening_factor:
      Multiplier used to widen on the inner layer of the MLP step within a
      transformer attention block.
    dropout_prob:
      Dropout probability for the post-attention and output dropout.
    dropout_attn_prob:
      Dropout probability for within the attention layer.
    num_heads:
      Number of attention heads.
    att_init_scale:
      Scale for the `tf_keras.initializers.VarianceScaling` used in attention
      kernel.
    dense_init_scale:
      Scale for the `tf_keras.initializers.VarianceScaling` used in MLP kernel.
    use_query_residual:
      Toggle to execute residual connection after attention.
    norm_epsilon:
      Epsilon value to initialize normalization layers.
    qk_last_dim:
      When set, determines the last dimension of the attention score output.
      When it's `None`, it uses the first `input_shape`'s last dimension as the
      last dimension of the attention score output.
    v_last_dim:
      When set, determines the value's last dimension in the multi-head
      attention.
      When it's `None`, it uses the `qk_last_dim` for `inner_dim` and
      `value_dim`.
      If `qk_last_dim` is `None`, the first input_shape's last dimension is used
      as the last dimension of the attention score output.
      If `output_last_dim` is `None`, `v_last_dim` is used to set the
      `TransformerEncoderBlock`'s output's last dimension.
    diff_q_kv_att_layer_norm:
      If `True`, create a separate attention layer norm layer for query and
      key-value if `norm_first` is `True`. Invalid to set to `True` if
      `norm_first` is `False`.
    output_last_dim:
      When set, the value determines the last dimension of the output of the
      attention block or `output_last_dim`.
      When it's `None`, it uses, in order of decreasing precedence,
      `v_last_dim`, `qk_last_dim`, and finally first `input_shape`'s last
      dimension. To clarify, if `v_last_dim` or `qk_last_dim` is `None`, the
      next order of precedence is used. The value is used to determine the last
      dimension of the output of the attention block or `output_last_dim`.

  Returns:
    A `dict` mapping `TransformerEncoderBlock` arguments.

  References:
    [Perceiver: General Perception with Iterative
    Attention](https://arxiv.org/abs/2103.03206)
    (https://github.com/deepmind/deepmind-research/blob/master/perceiver/perceiver.py)
    (https://github.com/tensorflow/models/blob/871c4e0a393ef4385534bee55354a5df8aa1ccf4/official/nlp/modeling/layers/transformer_encoder_block.py)
  """

  inputs_q_shape = input_shape[0]
  # Q and K must have the same number of last dim.
  # Default to preserving Q's input's shape.
  if qk_last_dim is None:
    qk_last_dim = inputs_q_shape[-1]
  # V's number of last dim determines the shape of the output of QKV-attention.
  # Default to the same number of last dim used in the key-query operation.
  if v_last_dim is None:
    v_last_dim = qk_last_dim
  # Project the output of QKV attention to a desired number of last dim.
  # Default to the same number as the output of the QKV attention operation.
  if output_last_dim is None:
    output_last_dim = v_last_dim

  assert qk_last_dim % num_heads == 0
  assert v_last_dim % num_heads == 0
  qk_last_dim_per_head = qk_last_dim // num_heads
  v_last_dim_per_head = v_last_dim // num_heads

  return {
      "num_attention_heads":
          num_heads,
      "inner_dim":
          output_last_dim * widening_factor,
      "inner_activation":
          functools.partial(tf_keras.activations.gelu, approximate=True),
      "kernel_initializer":
          tf_keras.initializers.VarianceScaling(scale=dense_init_scale),
      "attention_initializer":
          tf_keras.initializers.VarianceScaling(scale=att_init_scale),
      "norm_first":
          True,
      "norm_epsilon":
          norm_epsilon,
      "output_dropout":
          dropout_prob,
      "attention_dropout":
          dropout_attn_prob,
      "inner_dropout":
          0.0,
      "use_query_residual":
          use_query_residual,
      "value_dim":
          v_last_dim_per_head,
      "key_dim":
          qk_last_dim_per_head,
      "output_last_dim":
          output_last_dim,
      "diff_q_kv_att_layer_norm":
          diff_q_kv_att_layer_norm,
  }
