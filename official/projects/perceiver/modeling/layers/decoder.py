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

"""Perceiver basic decoder."""

import collections

import tensorflow as tf

from official.nlp.modeling import layers
from official.projects.perceiver.modeling.layers import utils


class Decoder(tf.keras.layers.Layer):
  """Perceiver Decoder layer.

  Uses cross attention decoder layer.
  This layer implements a Perceiver Decoder from
  "Perceiver: General Perception with Iterative Attention".
  (https://arxiv.org/abs/2103.03206)

  References:
    [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
    [Perceiver: General Perception with Iterative
    Attention](https://arxiv.org/abs/2103.03206)
    (https://github.com/deepmind/deepmind-research/blob/master/perceiver/perceiver.py)
    (https://github.com/tensorflow/models/blob/871c4e0a393ef4385534bee55354a5df8aa1ccf4/official/nlp/modeling/layers/transformer_encoder_block.py)
  """

  def __init__(self,
               output_last_dim,
               qk_last_dim=None,
               v_last_dim=None,
               use_query_residual=False,
               output_w_init=None,
               num_heads=1,
               name="decoder",
               **kwargs):
    """Init.

    Args:
      output_last_dim:
        Last dim size for output.
      qk_last_dim:
        When set, determines the last dimension of the attention score output.
        Check `qk_last_dim` doc in `utils.build_cross_attention_block_args`.
      v_last_dim:
        When set, determines the value's last dimension in the multi-head
        attention.
        Check `v_last_dim` doc in `utils._build_transformer_encoder_block_args`.
      use_query_residual:
        Toggle to execute residual connection after attention.
      output_w_init:
        Ouptut layer kernel initializer.
      num_heads:
        Number of attention heads for the `TransformerEncoderBlock`.
      name:
        Sets the `tf.keras.layers.Layer` name.
      **kwargs:
        Any keyword arguments to pass through to `tf.keras.layers.Layer`.
    """
    super().__init__(name=name, **kwargs)

    self._output_last_dim = output_last_dim
    self._output_w_init = output_w_init
    self._use_query_residual = use_query_residual
    self._qk_last_dim = qk_last_dim
    self._v_last_dim = v_last_dim
    self._final_project = False  # Make variable if needed
    self._num_heads = num_heads

    # Omitted `concat_preprocessed_input` for MLM use-case.

  def build(self, input_shape):
    """Build layers using `input_shape`.

    Args:
      input_shape:
        Input shape(s) of the layer call.
    """
    decoder_query_shape = input_shape[0]
    z_shape = input_shape[1]
    self._decoding_cross_attn = layers.TransformerEncoderBlock(
        **utils.build_cross_attention_block_args(
            (decoder_query_shape, z_shape),
            widening_factor=1,
            dropout_prob=0.0,
            num_heads=self._num_heads,
            shape_for_attn="kv",
            qk_last_dim=self._qk_last_dim,
            v_last_dim=self._v_last_dim,
            use_query_residual=self._use_query_residual))

  def call(self, inputs, training=None, query_mask=None):
    """Return decoded output of latent vector via the query.

    Args:
      inputs:
        Expect inputs to be a tuple of perceiver's decoder query tensor and
        latent tensor (z). For the cross attention block, `z` is the key-value
        tensor and decoder query is the query tensor.
        Latent tensor comes from the self-attention processing blocks and
        decoder query comes from users to query for the desired output.
      training:
        Flag to indicate training status.
      query_mask:
        mask used to create the attention mask for the query tensor in the
        cross attention block.

    Returns:
      `tf.Tensor` decoded output of latent vector via the query.
    """
    if not isinstance(inputs, collections.abc.Sequence):
      raise ValueError("`inputs` must be a sequence.")
    if len(inputs) != 2:
      raise ValueError("`inputs` must have two elements.")

    query, z = inputs
    # Cross-attention decoding.
    # key, value: B x N x K; query: B x M x K
    # Attention maps -> B x N x M
    # Output -> B x M x K
    # Construct cross attention and linear layer lazily, in case we don't need
    # them.
    if query_mask is None:
      attention_mask = None
    else:
      attention_mask = utils.make_cross_attention_mask(
          query_mask=query_mask,
          kv_mask=tf.ones(tf.shape(z)[:2], dtype=tf.int32))

    output = self._decoding_cross_attn(
        (query, z, attention_mask),
        training=training)

    return output
