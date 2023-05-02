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

"""Perceiver encode processor."""

import tensorflow as tf

from official.nlp.modeling import layers
from official.projects.perceiver.modeling.layers import utils


class Encoder(tf.keras.layers.Layer):
  """Perceiver Encoder and Processor(s) layer.

  This layer implements the Perceiver Encoder and Processor stack from
  "Perceiver: General Perception with Iterative Attention".
  (https://arxiv.org/abs/2103.03206)
  It uses SelfAttention and CrossAttention modules.
  It allows the user to choose the initial latent positional encodings.

  References:
    [Perceiver: General Perception with Iterative
    Attention](https://arxiv.org/abs/2103.03206)
    (https://github.com/deepmind/deepmind-research/blob/master/perceiver/perceiver.py)
    (https://github.com/tensorflow/models/blob/871c4e0a393ef4385534bee55354a5df8aa1ccf4/official/nlp/modeling/layers/transformer_encoder_block.py)
  """

  def __init__(self,
               self_attention_num_heads=8,
               self_attention_widening_factor=1,
               cross_attention_num_heads=8,
               cross_attention_widening_factor=1,
               num_self_attends_per_block=6,
               num_blocks=8,
               qk_last_dim=None,
               v_last_dim=None,
               dropout_prob=0.0,
               dropout_attn_prob=0.0,
               att_init_scale=1.0,
               dense_init_scale=1.0,
               norm_epsilon=1e-5,
               name="encode_processor",
               **kwargs):
    """Init.

    Args:
      self_attention_num_heads:
        Number of attention heads in the self-attention transformer block.
      self_attention_widening_factor:
        Multiplier used to widen on the inner layer of the MLP step within the
        self-attention transformer block.
      cross_attention_num_heads:
        Number of attention heads in the cross-attention transformer block.
      cross_attention_widening_factor:
        Multiplier used to widen on the inner layer of the MLP step within the
        cross-attention transformer block.
      num_self_attends_per_block:
        Number of different self-attention encoders initialized per latent
        perceiver block.
      num_blocks:
        Number of latent perceiver blocks.
      qk_last_dim:
        When set, determines the last dimension of the attention score output.
        Check `qk_last_dim` doc in `utils.build_cross_attention_block_args` for
        more details.
      v_last_dim:
        It can impact the last dimension size of value projection in mult-head
        attention output and `TransformerEncoderBlock`'s output.
        For more details, check `v_last_dim` doc in
        `utils._build_transformer_encoder_block_args`.
      dropout_prob:
        Dropout probability for the post-attention and output dropout.
      dropout_attn_prob:
        Dropout probability for within the attention layer.
      att_init_scale:
        Scale for the `tf.keras.initializers.VarianceScaling` used in attention
        kernel.
      dense_init_scale:
        Scale for the `tf.keras.initializers.VarianceScaling` used in MLP
        kernel.
      norm_epsilon:
        Epsilon value to initialize normalization layers.
      name:
        Sets the `tf.keras.layers.Layer` name.
      **kwargs:
        Any keyword arguments to pass through to `tf.keras.layers.Layer`.
    """
    super().__init__(name=name, **kwargs)

    self._input_is_1d = True

    self._num_self_attends_per_block = num_self_attends_per_block
    self._dropout_prob = dropout_prob
    self._qk_last_dim = qk_last_dim
    self._v_last_dim = v_last_dim
    self._norm_epsilon = norm_epsilon
    self._dropout_attn_prob = dropout_attn_prob
    self._att_init_scale = att_init_scale
    self._dense_init_scale = dense_init_scale
    self._num_blocks = num_blocks

    self._self_attention_widening_factor = self_attention_widening_factor
    self._self_attention_num_heads = self_attention_num_heads

    self._cross_attention_widening_factor = cross_attention_widening_factor
    self._cross_attention_num_heads = cross_attention_num_heads
    self._cross_attention_shape_for_attn = "kv"
    self._cross_attention_use_query_residual = True

  def build(self, input_shape):
    embeddings_shape = input_shape[0]
    z_shape = input_shape[1]
    self._self_attention_encoder_blocks = []
    for i in range(self._num_self_attends_per_block):
      self._self_attention_encoder_blocks.append(layers.TransformerEncoderBlock(
          name=f"self_attention_encoder_{i}",
          **utils.build_self_attention_block_args(
              (z_shape,),
              widening_factor=self._self_attention_widening_factor,
              dropout_prob=self._dropout_prob,
              dropout_attn_prob=self._dropout_attn_prob,
              num_heads=self._self_attention_num_heads,
              att_init_scale=self._att_init_scale,
              dense_init_scale=self._dense_init_scale,
              qk_last_dim=self._qk_last_dim,
              v_last_dim=self._v_last_dim,
              norm_epsilon=self._norm_epsilon)))

    self._cross_attention_encoder_block = layers.TransformerEncoderBlock(
        name="cross_attention_encoder",
        **utils.build_cross_attention_block_args(
            (z_shape, embeddings_shape),
            widening_factor=self._cross_attention_widening_factor,
            dropout_prob=self._dropout_prob,
            dropout_attn_prob=self._dropout_attn_prob,
            num_heads=self._cross_attention_num_heads,
            att_init_scale=self._att_init_scale,
            dense_init_scale=self._dense_init_scale,
            shape_for_attn=self._cross_attention_shape_for_attn,
            use_query_residual=self._cross_attention_use_query_residual,
            norm_epsilon=self._norm_epsilon,
            qk_last_dim=self._qk_last_dim,
            v_last_dim=self._v_last_dim))

  def call(self, inputs, input_mask=None, training=None):
    embeddings = inputs[0]
    z = inputs[1]
    if input_mask is None:
      input_mask = tf.ones(tf.shape(embeddings)[:2], dtype=tf.int32)
    attention_mask = utils.make_cross_attention_mask(
        query_mask=tf.ones(tf.shape(z)[:2], dtype=tf.int32),
        kv_mask=input_mask)
    z = self._cross_attention_encoder_block(
        (z, embeddings, attention_mask),
        training=training)
    for _ in range(self._num_blocks):
      for self_attention_block in self._self_attention_encoder_blocks:
        z = self_attention_block(z, training=training)
    return z
