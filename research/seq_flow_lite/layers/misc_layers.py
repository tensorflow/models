# Copyright 2020 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================
"""Layers for embedding."""
import math
import tensorflow as tf

from layers import base_layers # import seq_flow_lite module
from layers import conv_layers # import seq_flow_lite module
from layers import dense_layers # import seq_flow_lite module
from layers import embedding_layers # import seq_flow_lite module
from layers import quantization_layers # import seq_flow_lite module


class AttentionPooling(base_layers.BaseLayer):
  """A basic attention pooling layer."""

  def __init__(self, scalar=True, normalize=True, **kwargs):
    self.scalar = scalar
    # Attention logits should not have activation post linear layer so it can
    # be positive or negative. This would enable the attention distribution to
    # be anything that the network likes. Using relu activation makes the
    # attention distribution biased towards uniform distribution.
    # This gets better results for attention pooling. Though some outputs are
    # emphasized for making classification decision, all other outputs have
    # a non zero probability of influencing the class. This seems to result
    # in better backprop.
    self.attention = dense_layers.BaseQDenseVarLen(
        units=1, rank=3, normalize=normalize, **kwargs)
    self.qactivation = quantization_layers.ActivationQuantization(**kwargs)
    super(AttentionPooling, self).__init__(**kwargs)

  def build(self, input_shapes):
    self.feature_size = input_shapes[-1]

  def call(self, inputs, mask, inverse_normalizer):
    self._assert_rank_and_type(inputs, 3)
    self._assert_rank_and_type(mask, 3)
    batch_size = self.get_batch_dimension(inputs)
    attn_logits = self.attention(inputs, mask, inverse_normalizer)
    if self.parameters.mode not in [base_layers.PREDICT, base_layers.TFLITE]:
      invalid_mask = (1 - mask) * self.parameters.invalid_logit
      attn_logits = attn_logits * mask + invalid_mask
    attn_logits = tf.reshape(attn_logits, [batch_size, -1])
    attention = tf.nn.softmax(attn_logits, axis=-1)
    attention = self.qrange_sigmoid(attention, tf_only=True)
    if self.parameters.mode in [base_layers.PREDICT, base_layers.TFLITE]:
      inputs = tf.reshape(inputs, [-1, self.feature_size])
    else:
      attention = tf.expand_dims(attention, axis=1)
    pre_logits = self.qactivation(tf.matmul(attention, inputs))
    return tf.reshape(pre_logits, [batch_size, self.feature_size])


class TreeInductionLayer(base_layers.BaseLayer):
  """A basic tree induction layer."""

  def __init__(self, **kwargs):
    self.qactivation = quantization_layers.ActivationQuantization(**kwargs)
    super(TreeInductionLayer, self).__init__(**kwargs)

  def call(self, keys, queries, sequence_length):
    key_dim = keys.get_shape().as_list()[-1]
    query_dim = queries.get_shape().as_list()[-1]
    assert key_dim == query_dim, "Last dimension of keys/queries should match."

    if self.parameters.mode not in [base_layers.PREDICT, base_layers.TFLITE]:
      sequence_mask = tf.sequence_mask(
          sequence_length, maxlen=tf.shape(keys)[1], dtype=tf.float32)
      sequence_mask = tf.expand_dims(sequence_mask, axis=2)
      attn_mask = tf.matmul(sequence_mask, sequence_mask, transpose_b=True)

      attn_logits = self.qactivation(tf.matmul(keys, queries, transpose_b=True))
      invalid_attn_mask = (1 - attn_mask) * self.parameters.invalid_logit
      return attn_logits * attn_mask + invalid_attn_mask
    else:
      assert self.get_batch_dimension(keys) == 1
      assert self.get_batch_dimension(queries) == 1
      keys = tf.reshape(keys, [-1, key_dim])
      queries = tf.reshape(queries, [-1, key_dim])

      result = self.qactivation(tf.matmul(keys, queries, transpose_b=True))
      # TODO(b/171063452): Bug needs to be fixed to handle this correctly.
      # seq_dim = tf.shape(result)[1]
      # result = tf.reshape(result, [1, seq_dim, seq_dim])
      return result


class GBSTLayerV2(base_layers.BaseLayer):
  """Tokenization layer."""

  def __init__(self,
               feature_size,
               max_seq_len,
               downsample_rate=2,
               max_subword_block_width=4,
               conv_kernel_size=5,
               block_mixing_mode=None,
               add_block_pos_embed=False,
               **kwargs):
    super(GBSTLayerV2, self).__init__(**kwargs)
    self.feature_size = feature_size
    self.max_seq_len = max_seq_len
    self.downsample_rate = downsample_rate
    self.subword_blocks_width = [1, 2, 3, 4]
    self.max_subword_block_width = len(self.subword_blocks_width)
    self.block_mixing_mode = block_mixing_mode

    self.add_block_pos_embed = add_block_pos_embed
    if self.add_block_pos_embed:
      self.block_pos_embedding = embedding_layers.EmbeddingLayer(
          shape=[self.max_subword_block_width, self.feature_size], **kwargs)
    self.conv_kernel_size = conv_kernel_size
    self.conv_layer = conv_layers.EncoderQConvolution(
        filters=feature_size,
        ksize=conv_kernel_size,
        rank=3,
        padding="VALID",
        activation=None,
        **kwargs)
    padding = [conv_kernel_size - 1, 0]
    self.zero_pad = tf.keras.layers.ZeroPadding1D(padding=padding)
    self.block_attn = dense_layers.BaseQDense(
        units=1,
        rank=3,
        activation=None,
        normalize=False,
        quantize_output=False,
        **kwargs)
    self.scores_concat = quantization_layers.ConcatQuantization(
        axis=3, **kwargs)
    self.attn_concat = quantization_layers.ConcatQuantization(axis=0, **kwargs)
    self.qact = quantization_layers.ActivationQuantization(**kwargs)
    self.qact_dot = quantization_layers.ActivationQuantization(**kwargs)
    self.qoutput = quantization_layers.ActivationQuantization(**kwargs)

  def call(self, inputs, seq_length):
    """Performs downsampling on the character-scale input representation.

    Based in principle on https://arxiv.org/pdf/2106.12672.pdf.

    Args:
      inputs: float Tensor of shape [batch_size, seq_length, embedding_size].
      seq_length: sequence length of shape [batch_size].

    Returns:
      <float>[batch_size, seq_length / downsample_rate, embedding_size].
        Downsampled sequences.
    """
    self._assert_rank_and_type(inputs, 3)
    bsz = self.get_batch_dimension(inputs)
    max_seq_len = self.max_seq_len

    if self.parameters.mode in [base_layers.PREDICT, base_layers.TFLITE]:
      num_steps = tf.shape(inputs)[1]

    inputs = self.zero_pad(inputs)
    inputs = self.conv_layer(inputs)

    all_block_scores = []
    all_sequences = []
    for subword_len in self.subword_blocks_width:
      if self.add_block_pos_embed:
        block_pos_indices = tf.range(subword_len, dtype=tf.int32)
        block_pos_indices = tf.reshape(block_pos_indices, [1, -1])
        block_pos_embeds = self.block_pos_embedding(block_pos_indices)
        tile_len = math.ceil(max_seq_len / float(subword_len))
        retiled_block_pos_embeds = tf.repeat(block_pos_embeds, tile_len, axis=1)
        inputs += retiled_block_pos_embeds
      # For this block size, form candidate block embeddings and scores.
      # candidates shape: [batch, seq_len/subword_len, dim]
      # block_scores shape: [batch, seq_len/subword_len, 1]
      candidates = tf.nn.avg_pool(
          inputs, [subword_len], strides=[subword_len], padding="SAME")
      candidates = self.conv_layer.quantize_using_output_range(candidates)

      block_scores = self.block_attn(candidates)
      # Upsample it back to the original sequence length.
      retiled_seq = tf.repeat(candidates, subword_len, axis=1)
      retiled_block_scores = tf.repeat(block_scores, subword_len, axis=1)

      # Make sure everything is the right length and add new dimension to concat
      # candidate blocks on.
      if self.parameters.mode in [base_layers.PREDICT, base_layers.TFLITE]:
        retiled_block_scores = retiled_block_scores[:, :num_steps, :]
        retiled_seq = retiled_seq[:, :num_steps, :]
      else:
        retiled_block_scores = retiled_block_scores[:, :max_seq_len, :]
        retiled_seq = retiled_seq[:, :max_seq_len, :]
      retiled_seq = tf.expand_dims(retiled_seq, axis=-1)
      retiled_block_scores = tf.expand_dims(retiled_block_scores, axis=-1)
      all_sequences.append(retiled_seq)
      all_block_scores.append(retiled_block_scores)

    block_net = self.scores_concat(all_block_scores)
    if self.block_mixing_mode == "score_attention":
      if self.parameters.mode in [base_layers.PREDICT, base_layers.TFLITE]:
        block_attn_steps = []
        self.attn_concat(None)
        for i in range(num_steps):
          block_i = tf.reshape(block_net[:, i:i + 1, :, :], [1, -1])
          block_attn_steps.append(tf.matmul(block_i, block_i, transpose_b=True))
        block_attn = self.attn_concat(block_attn_steps)
        block_attn = tf.reshape(block_attn, [bsz, -1, 1, 1])
      else:
        block_attn = self.attn_concat(
            [tf.matmul(block_net, block_net, transpose_b=True)])

      block_attn = tf.nn.softmax(block_attn, axis=1)
      block_attn = self.qrange_sigmoid(block_attn, tf_only=True)
      block_net_scaled = self.qact(block_attn * block_net)
    else:
      block_net_scaled = block_net

    candidate_embeds = self.conv_layer.quantize_using_output_range(
        tf.concat(all_sequences, axis=3))
    dot_product = self.qact_dot(block_net_scaled * candidate_embeds)
    output = self.qoutput(tf.reduce_mean(dot_product, axis=-1, keepdims=True))
    output = tf.reshape(output, [bsz, -1, self.feature_size])

    # Removing pad entries for inference mode.
    if self.parameters.mode in [base_layers.PREDICT, base_layers.TFLITE]:
      output = output[:, :num_steps, :]
    # Downsample by mean pooling.
    if self.downsample_rate > 1:
      output = tf.nn.avg_pool(
          output, (self.downsample_rate,),
          strides=(self.downsample_rate,),
          padding="VALID")
    return output
