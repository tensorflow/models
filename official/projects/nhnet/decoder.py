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

"""Transformer decoder that mimics a BERT encoder, to load BERT checkpoints."""

import tensorflow as tf

from official.legacy.transformer import model_utils as transformer_utils
from official.modeling import tf_utils
from official.nlp.modeling import layers


class TransformerDecoder(tf.keras.layers.Layer):
  """Transformer decoder stack."""

  def __init__(self,
               num_hidden_layers=12,
               hidden_size=768,
               num_attention_heads=12,
               intermediate_size=3072,
               intermediate_activation="gelu",
               hidden_dropout_prob=0.0,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               attend_to_last_layer=True,
               multi_channel_cross_attention=False,
               **kwargs):
    super(TransformerDecoder, self).__init__(**kwargs)
    self.num_hidden_layers = num_hidden_layers
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.intermediate_activation = tf_utils.get_activation(
        intermediate_activation)
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.attend_to_last_layer = attend_to_last_layer
    self.multi_channel_cross_attention = multi_channel_cross_attention

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.layers = []
    for i in range(self.num_hidden_layers):
      self.layers.append(
          layers.TransformerDecoderBlock(
              num_attention_heads=self.num_attention_heads,
              intermediate_size=self.intermediate_size,
              intermediate_activation=self.intermediate_activation,
              dropout_rate=self.hidden_dropout_prob,
              attention_dropout_rate=self.attention_probs_dropout_prob,
              kernel_initializer=tf.keras.initializers.TruncatedNormal(
                  stddev=self.initializer_range),
              multi_channel_cross_attention=self.multi_channel_cross_attention,
              name=("layer_%d" % i)))
    super(TransformerDecoder, self).build(unused_input_shapes)

  def call(self, inputs, cache=None, decode_loop_step=None):
    """Return the output of the decoder layer stacks.

    Args:
      inputs: A dictionary of inputs. `decoder_inputs` is a tf.int32 tensor for
        input ids. `encoder_outputs` is a list of tensors with shape
        [batch_size, input_length, hidden_size]. `self_attention_mask` is the
        bias for decoder self-attention layer. [1, 1, target_length,
        target_length]. `attention_mask` is the bias for encoder-decoder
        attention layer, [batch_size, 1, 1, input_length].
      cache: A dictionary of cache tensors, including key & value attentions.
      decode_loop_step: an integer to indicate the step inside a decoding loop.

    Returns:
      Output of decoder layer stack.
      float32 tensor with shape [batch_size, target_length, hidden_size]
    """
    decoder_inputs = inputs["decoder_inputs"]
    encoder_outputs = inputs["encoder_outputs"]
    self_attention_mask = inputs["self_attention_mask"]
    attention_mask = inputs["attention_mask"]
    decoder_shape = tf_utils.get_shape_list(decoder_inputs, expected_rank=3)
    batch_size = decoder_shape[0]
    decoder_length = decoder_shape[1]

    def _to_bert_self_attention_mask(matrix):
      """[1, 1, target_len, target_len] -> [bs, target_len, target_len]."""
      matrix = tf.squeeze(matrix, axis=[1])
      matrix = tf.tile(matrix, [batch_size, 1, 1])
      return matrix

    def _to_bert_encdec_attention_mask(matrix):
      """[bs, 1, 1, input_len] -> [bs, target_len, input_len]."""
      if self.multi_channel_cross_attention:
        matrix = tf.expand_dims(matrix, axis=2)
        matrix = tf.tile(matrix, [1, 1, decoder_length, 1])
      else:
        matrix = tf.squeeze(matrix, axis=[1])
        matrix = tf.tile(matrix, [1, decoder_length, 1])
      return matrix

    attention_mask = _to_bert_encdec_attention_mask(attention_mask)
    self_attention_mask = _to_bert_self_attention_mask(self_attention_mask)

    output_tensor = decoder_inputs
    for layer_idx in range(self.num_hidden_layers):
      if self.attend_to_last_layer:
        memory = encoder_outputs[-1]
      else:
        memory = encoder_outputs[layer_idx]
      if self.multi_channel_cross_attention:
        transformer_inputs = [
            output_tensor, memory, attention_mask, self_attention_mask,
            inputs["doc_attention_probs"]
        ]
      else:
        transformer_inputs = [
            output_tensor, memory, attention_mask, self_attention_mask
        ]
      # Gets the cache for decoding.
      if cache is None:
        output_tensor, _ = self.layers[layer_idx](transformer_inputs)
      else:
        cache_layer_idx = str(layer_idx)
        output_tensor, cache[cache_layer_idx] = self.layers[layer_idx](
            transformer_inputs,
            cache=cache[cache_layer_idx],
            decode_loop_step=decode_loop_step)
    return output_tensor, cache


def get_attention_bias(input_tensor,
                       bias_type,
                       padding_value=0,
                       max_length=None):
  """A helper function to get various attention bias tensors."""
  if bias_type not in ("single_cross", "multi_cross", "decoder_self"):
    raise ValueError("Invalid attention bias type: %s" % bias_type)
  if bias_type == "single_cross":
    length = tf_utils.get_shape_list(input_tensor, expected_rank=2)[1]
    bias = transformer_utils.get_padding_bias(
        input_tensor, padding_value=padding_value)
  elif bias_type == "multi_cross":
    length = tf_utils.get_shape_list(input_tensor, expected_rank=3)[2]
    padding = transformer_utils.get_padding(
        input_tensor, padding_value=padding_value)
    bias = padding * -1e9
  else:
    if max_length is not None:
      length = max_length
    else:
      length = tf_utils.get_shape_list(input_tensor, expected_rank=2)[1]
    bias = transformer_utils.get_decoder_self_attention_bias(length)

  return tf.where(bias < 0, tf.zeros_like(bias), tf.ones_like(bias))


class AttentionBias(tf.keras.layers.Layer):

  def __init__(self, bias_type, **kwargs):
    super(AttentionBias, self).__init__(**kwargs)
    self.bias_type = bias_type

  def call(self, inputs):
    return get_attention_bias(inputs, self.bias_type)


class EmbeddingPostprocessor(tf.keras.layers.Layer):
  """Performs various post-processing on a word embedding tensor."""

  def __init__(self,
               use_type_embeddings=False,
               token_type_vocab_size=None,
               use_position_embeddings=True,
               max_position_embeddings=512,
               dropout_prob=0.0,
               initializer_range=0.02,
               initializer=None,
               **kwargs):
    super(EmbeddingPostprocessor, self).__init__(**kwargs)
    self.use_type_embeddings = use_type_embeddings
    self.token_type_vocab_size = token_type_vocab_size
    self.use_position_embeddings = use_position_embeddings
    self.max_position_embeddings = max_position_embeddings
    self.dropout_prob = dropout_prob
    self.initializer_range = initializer_range

    if not initializer:
      self.initializer = tf.keras.initializers.TruncatedNormal(
          stddev=initializer_range)
    else:
      self.initializer = initializer

    if self.use_type_embeddings and not self.token_type_vocab_size:
      raise ValueError("If `use_type_embeddings` is True, then "
                       "`token_type_vocab_size` must be specified.")

  def build(self, input_shapes):
    """Implements build() for the layer."""
    (word_embeddings_shape, _) = input_shapes
    width = word_embeddings_shape.as_list()[-1]
    self.type_embeddings = None
    if self.use_type_embeddings:
      self.type_embeddings = self.add_weight(
          "type_embeddings",
          shape=[self.token_type_vocab_size, width],
          initializer=tf.keras.initializers.TruncatedNormal(
              stddev=self.initializer_range),
          dtype=self.dtype)

    self.position_embeddings = None
    if self.use_position_embeddings:
      self.position_embeddings = self.add_weight(
          "position_embeddings",
          shape=[self.max_position_embeddings, width],
          initializer=tf.keras.initializers.TruncatedNormal(
              stddev=self.initializer_range),
          dtype=self.dtype)

    self.output_layer_norm = tf.keras.layers.LayerNormalization(
        name="layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)
    self.output_dropout = tf.keras.layers.Dropout(
        rate=self.dropout_prob, dtype=tf.float32)
    super(EmbeddingPostprocessor, self).build(input_shapes)

  def __call__(self, word_embeddings, token_type_ids=None, **kwargs):
    inputs = tf_utils.pack_inputs([word_embeddings, token_type_ids])
    return super(EmbeddingPostprocessor, self).__call__(inputs, **kwargs)  # pytype: disable=attribute-error  # typed-keras

  def call(self, inputs):
    """Implements call() for the layer."""
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    word_embeddings = unpacked_inputs[0]
    token_type_ids = unpacked_inputs[1]
    input_shape = tf_utils.get_shape_list(word_embeddings, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = word_embeddings
    if self.use_type_embeddings:
      flat_token_type_ids = tf.reshape(token_type_ids, [-1])
      token_type_embeddings = tf.gather(self.type_embeddings,
                                        flat_token_type_ids)
      token_type_embeddings = tf.reshape(token_type_embeddings,
                                         [batch_size, seq_length, width])
      output += token_type_embeddings

    if self.use_position_embeddings:
      position_embeddings = tf.expand_dims(
          tf.slice(self.position_embeddings, [0, 0], [seq_length, width]),
          axis=0)

      output += position_embeddings

    output = self.output_layer_norm(output)
    output = self.output_dropout(output)

    return output


class Decoder(tf.keras.layers.Layer):
  """The decoder network which can reuse encoder embeddings for target."""

  def __init__(self, config, embedding_lookup=None, **kwargs):
    super(Decoder, self).__init__(**kwargs)
    self.config = config
    # Shares vocabulary embedding.
    self.embedding_lookup = None
    if embedding_lookup:
      self.embedding_lookup = embedding_lookup

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    if self.embedding_lookup is None:
      self.embedding_lookup = layers.OnDeviceEmbedding(
          vocab_size=self.config.vocab_size,
          embedding_width=self.config.hidden_size,
          initializer=tf.keras.initializers.TruncatedNormal(
              stddev=self.config.initializer_range),
          name="target_embeddings")
    self.embedding_postprocessor = EmbeddingPostprocessor(
        use_type_embeddings=False,
        use_position_embeddings=True,
        max_position_embeddings=self.config.max_position_embeddings,
        dropout_prob=self.config.hidden_dropout_prob,
        initializer=tf.keras.initializers.VarianceScaling(
            scale=self.config.initializer_gain,
            mode="fan_avg",
            distribution="uniform"),
        name="embedding_postprocessor")
    # Decoder can use a different intermediate size.
    self.multi_channel_cross_attention = self.config.get(
        "multi_channel_cross_attention", False)
    self.decoder = TransformerDecoder(
        num_hidden_layers=self.config.num_decoder_layers,
        hidden_size=self.config.hidden_size,
        num_attention_heads=self.config.num_decoder_attn_heads,
        intermediate_size=self.config.decoder_intermediate_size,
        intermediate_activation=self.config.hidden_act,
        hidden_dropout_prob=self.config.hidden_dropout_prob,
        attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
        initializer_range=self.config.initializer_range,
        multi_channel_cross_attention=self.multi_channel_cross_attention,
        name="decoder")
    super(Decoder, self).build(unused_input_shapes)

  def _decoding_step_time_signal(self, target_embeds, decode_loop_step):
    """Applies time signal (positional embeddings) for decoded embeddings."""
    # TODO(hongkuny): migrate to keras bert and design a module to handle this.
    output = target_embeds
    if self.embedding_postprocessor.use_position_embeddings:
      position_embeddings = tf.gather(
          self.embedding_postprocessor.position_embeddings, [decode_loop_step])
      # Broadcasts to all sequences inside a batch.
      output += position_embeddings

    output = self.embedding_postprocessor.output_layer_norm(output)
    output = self.embedding_postprocessor.output_dropout(output)
    return output

  def call(self,
           inputs,
           cache=None,
           decode_loop_step=None,
           padded_decode=False):
    """Implements call() for the layer.

    Args:
      inputs: a list of input tensors.
      cache: A dictionary of cache tensors, including key & value attentions.
        Due to the limit of keras, we uses the side effect to update cache and
        states of tensors will be mutated.
      decode_loop_step: an integer to indicate the step inside a decoding loop.
      padded_decode: a boolean indicates if the pass is for padded decoding.

    Returns:
      Decoder output tensors.
    """
    attention_bias = inputs["attention_bias"]
    target_ids = inputs["target_ids"]
    all_encoder_outputs = inputs["all_encoder_outputs"]
    self_attention_bias = inputs["self_attention_bias"]
    if not isinstance(all_encoder_outputs, list):
      all_encoder_outputs = [all_encoder_outputs]

    target_embeds = self.embedding_lookup(target_ids)
    if decode_loop_step is None:
      target_embeds = self.embedding_postprocessor(target_embeds)
    else:
      target_embeds = self._decoding_step_time_signal(target_embeds,
                                                      decode_loop_step)
    decoder_inputs = dict(
        decoder_inputs=target_embeds,
        encoder_outputs=all_encoder_outputs,
        self_attention_mask=self_attention_bias,
        attention_mask=attention_bias)
    if self.multi_channel_cross_attention:
      decoder_inputs["doc_attention_probs"] = inputs["doc_attention_probs"]
    decode_outputs, cache = self.decoder(
        decoder_inputs, cache, decode_loop_step if padded_decode else None)
    return decode_outputs
