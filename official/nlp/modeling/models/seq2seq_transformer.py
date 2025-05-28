# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Implement Seq2Seq Transformer model by TF official NLP library.

Model paper: https://arxiv.org/pdf/1706.03762.pdf
"""
import inspect
import math

import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.nlp.modeling import layers
from official.nlp.modeling.ops import beam_search

EOS_ID = 1


class Seq2SeqTransformer(tf_keras.Model):
  """Transformer model with Keras.

  Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf

  The Transformer model consists of an encoder and decoder. The input is an int
  sequence (or a batch of sequences). The encoder produces a continuous
  representation, and the decoder uses the encoder output to generate
  probabilities for the output sequence.
  """

  def __init__(self,
               vocab_size=33708,
               embedding_width=512,
               dropout_rate=0.0,
               padded_decode=False,
               decode_max_length=None,
               extra_decode_length=0,
               beam_size=4,
               alpha=0.6,
               encoder_layer=None,
               decoder_layer=None,
               eos_id=EOS_ID,
               **kwargs):
    """Initialize layers to build Transformer model.

    Args:
      vocab_size: Size of vocabulary.
      embedding_width: Size of hidden layer for embedding.
      dropout_rate: Dropout probability.
      padded_decode: Whether to max_sequence_length padding is used. If set
        False, max_sequence_length padding is not used.
      decode_max_length: maximum number of steps to decode a sequence.
      extra_decode_length: Beam search will run extra steps to decode.
      beam_size: Number of beams for beam search
      alpha: The strength of length normalization for beam search.
      encoder_layer: An initialized encoder layer.
      decoder_layer: An initialized decoder layer.
      eos_id: Id of end of sentence token.
      **kwargs: other keyword arguments.
    """
    super().__init__(**kwargs)
    self._vocab_size = vocab_size
    self._embedding_width = embedding_width
    self._dropout_rate = dropout_rate
    self._padded_decode = padded_decode
    self._decode_max_length = decode_max_length
    self._extra_decode_length = extra_decode_length
    self._beam_size = beam_size
    self._alpha = alpha
    self._eos_id = eos_id
    self.embedding_lookup = layers.OnDeviceEmbedding(
        vocab_size=self._vocab_size,
        embedding_width=self._embedding_width,
        initializer=tf.random_normal_initializer(
            mean=0., stddev=self._embedding_width**-0.5),
        scale_factor=self._embedding_width**0.5)
    self.encoder_layer = encoder_layer
    self.decoder_layer = decoder_layer
    self.position_embedding = layers.RelativePositionEmbedding(
        hidden_size=self._embedding_width)
    self.encoder_dropout = tf_keras.layers.Dropout(rate=self._dropout_rate)
    self.decoder_dropout = tf_keras.layers.Dropout(rate=self._dropout_rate)

  def get_config(self):
    config = {
        "vocab_size": self._vocab_size,
        "hidden_size": self._embedding_width,
        "dropout_rate": self._dropout_rate,
        "padded_decode": self._padded_decode,
        "decode_max_length": self._decode_max_length,
        "eos_id": self._eos_id,
        "extra_decode_length": self._extra_decode_length,
        "beam_size": self._beam_size,
        "alpha": self._alpha,
        "encoder_layer": self.encoder_layer,
        "decoder_layer": self.decoder_layer,
    }
    base_config = super(Seq2SeqTransformer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _embedding_linear(self, embedding_matrix, x):
    """Uses embeddings as linear transformation weights."""
    embedding_matrix = tf.cast(embedding_matrix, dtype=self.compute_dtype)
    x = tf.cast(x, dtype=self.compute_dtype)
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]
    hidden_size = tf.shape(x)[2]
    vocab_size = tf.shape(embedding_matrix)[0]

    x = tf.reshape(x, [-1, hidden_size])
    logits = tf.matmul(x, embedding_matrix, transpose_b=True)

    return tf.reshape(logits, [batch_size, length, vocab_size])

  def _parse_inputs(self, inputs):
    """Parses the `call` inputs and returns an uniformed output."""
    sources = inputs.get("inputs", None)
    input_mask = inputs.get("input_masks", None)
    embedded = inputs.get("embedded_inputs", None)

    if sources is None and embedded is not None:
      embedded_inputs = embedded
      boolean_mask = input_mask
      input_shape = tf_utils.get_shape_list(embedded, expected_rank=3)
      source_dtype = embedded.dtype
    elif sources is not None:
      embedded_inputs = self.embedding_lookup(sources)
      boolean_mask = tf.not_equal(sources, 0)
      input_shape = tf_utils.get_shape_list(sources, expected_rank=2)
      source_dtype = sources.dtype
    else:
      raise KeyError(
          "The call method expects either `inputs` or `embedded_inputs` and "
          "`input_masks` as input features.")

    return embedded_inputs, boolean_mask, input_shape, source_dtype

  def call(self, inputs):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    """Calculate target logits or inferred target sequences.

    Args:
      inputs: a dictionary of tensors.
        Feature `inputs` (optional): int tensor with shape
          `[batch_size, input_length]`.
        Feature `embedded_inputs` (optional): float tensor with shape
          `[batch_size, input_length, embedding_width]`.
        Feature `targets` (optional): None or int tensor with shape
          `[batch_size, target_length]`.
        Feature `input_masks` (optional): When providing the `embedded_inputs`,
          the dictionary must provide a boolean mask marking the filled time
          steps. The shape of the tensor is `[batch_size, input_length]`.
        Either `inputs` or `embedded_inputs` and `input_masks` must be present
        in the input dictionary. In the second case the projection of the
        integer tokens to the transformer embedding space is skipped and
        `input_masks` is expected to be present.

    Returns:
      If targets is defined, then return logits for each word in the target
      sequence, which is a float tensor with shape
      `(batch_size, target_length, vocab_size)`. If target is `None`, then
      generate output sequence one token at a time and
      returns a dictionary {
          outputs: `(batch_size, decoded_length)`
          scores: `(batch_size, 1)`}
      Even when `float16` is used, the output tensor(s) are always `float32`.

    Raises:
      NotImplementedError: If try to use padded decode method on CPU/GPUs.
    """
    # Prepare inputs to the layer stack by adding positional encodings and
    # applying dropout.
    targets = inputs.get("targets", None)
    (embedded_inputs, boolean_mask,
     input_shape, source_dtype) = self._parse_inputs(inputs)
    embedding_mask = tf.cast(boolean_mask, embedded_inputs.dtype)
    embedded_inputs *= tf.expand_dims(embedding_mask, -1)
    # Attention_mask generation.
    attention_mask = tf.cast(
        tf.reshape(boolean_mask, [input_shape[0], 1, input_shape[1]]),
        dtype=source_dtype)
    broadcast_ones = tf.ones(
        shape=[input_shape[0], input_shape[1], 1], dtype=source_dtype)
    attention_mask = broadcast_ones * attention_mask

    pos_encoding = self.position_embedding(embedded_inputs)
    pos_encoding = tf.cast(pos_encoding, embedded_inputs.dtype)
    encoder_inputs = embedded_inputs + pos_encoding

    encoder_inputs = self.encoder_dropout(encoder_inputs)

    encoder_outputs = self.encoder_layer(
        encoder_inputs, attention_mask=attention_mask)

    if targets is None:
      if self._padded_decode:
        max_decode_length = self._decode_max_length
      else:
        max_decode_length = self._decode_max_length or (
            tf.shape(encoder_outputs)[1] + self._extra_decode_length)
      symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

      batch_size = tf.shape(encoder_outputs)[0]
      # Create initial set of IDs that will be passed to symbols_to_logits_fn.
      initial_ids = tf.zeros([batch_size], dtype=tf.int32)

      # Create cache storing decoder attention values for each layer.
      init_decode_length = (max_decode_length if self._padded_decode else 0)
      num_heads = self.decoder_layer.num_attention_heads
      dim_per_head = self._embedding_width // num_heads

      # Cache dtype needs to match beam_search dtype.
      # pylint: disable=g-complex-comprehension
      cache = {
          str(layer): {
              "key":
                  tf.zeros(
                      [batch_size, init_decode_length, num_heads, dim_per_head],
                      dtype=self.compute_dtype),
              "value":
                  tf.zeros(
                      [batch_size, init_decode_length, num_heads, dim_per_head],
                      dtype=self.compute_dtype)
          } for layer in range(self.decoder_layer.num_layers)
      }
      # pylint: enable=g-complex-comprehension

      # Add encoder output and attention bias to the cache.
      encoder_outputs = tf.cast(encoder_outputs, dtype=self.compute_dtype)
      attention_mask = tf.cast(
          tf.reshape(boolean_mask, [input_shape[0], 1, input_shape[1]]),
          dtype=self.compute_dtype)
      cache["encoder_outputs"] = encoder_outputs
      cache["encoder_decoder_attention_mask"] = attention_mask

      # Use beam search to find the top beam_size sequences and scores.
      decoded_ids, scores = beam_search.sequence_beam_search(
          symbols_to_logits_fn=symbols_to_logits_fn,
          initial_ids=initial_ids,
          initial_cache=cache,
          vocab_size=self._vocab_size,
          beam_size=self._beam_size,
          alpha=self._alpha,
          max_decode_length=max_decode_length,
          eos_id=self._eos_id,
          padded_decode=self._padded_decode,
          dtype=self.compute_dtype)

      # Get the top sequence for each batch element
      top_decoded_ids = decoded_ids[:, 0, 1:]
      top_scores = scores[:, 0]

      return {"outputs": top_decoded_ids, "scores": top_scores}

    # Shift targets to the right, and remove the last element
    targets = tf.pad(targets, [[0, 0], [1, 0]])[:, :-1]
    decoder_inputs = self.embedding_lookup(targets)
    length = tf.shape(decoder_inputs)[1]
    pos_encoding = self.position_embedding(decoder_inputs)
    pos_encoding = tf.cast(pos_encoding, embedded_inputs.dtype)
    decoder_inputs += pos_encoding

    decoder_inputs = self.decoder_dropout(decoder_inputs)

    decoder_shape = tf_utils.get_shape_list(decoder_inputs, expected_rank=3)
    batch_size = decoder_shape[0]
    decoder_length = decoder_shape[1]

    self_attention_mask = tf.linalg.band_part(tf.ones([length, length]), -1, 0)
    self_attention_mask = tf.reshape(self_attention_mask, [1, length, length])
    self_attention_mask = tf.tile(self_attention_mask, [batch_size, 1, 1])

    attention_mask = tf.cast(
        tf.expand_dims(boolean_mask, axis=1), dtype=source_dtype)
    attention_mask = tf.tile(attention_mask, [1, decoder_length, 1])

    outputs = self.decoder_layer(
        decoder_inputs,
        encoder_outputs,
        self_attention_mask=self_attention_mask,
        cross_attention_mask=attention_mask)
    logits = self._embedding_linear(self.embedding_lookup.embeddings, outputs)
    # Model outputs should be float32 to avoid numeric issues.
    # https://www.tensorflow.org/guide/mixed_precision#building_the_model
    logits = tf.cast(logits, tf.float32)
    return logits

  def _get_symbols_to_logits_fn(self, max_decode_length):
    """Returns a decoding function that calculates logits of the next tokens."""
    timing_signal = self.position_embedding(
        inputs=None, length=max_decode_length + 1)
    timing_signal = tf.cast(timing_signal, dtype=self.compute_dtype)
    decoder_self_attention_mask = tf.linalg.band_part(
        tf.ones([max_decode_length, max_decode_length],
                dtype=self.compute_dtype), -1, 0)
    decoder_self_attention_mask = tf.reshape(
        decoder_self_attention_mask, [1, max_decode_length, max_decode_length])

    def symbols_to_logits_fn(ids, i, cache):
      """Generate logits for next potential IDs.

      Args:
        ids: Current decoded sequences. int tensor with shape `(batch_size *
          beam_size, i + 1)`.
        i: Loop index.
        cache: Dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.

      Returns:
        Tuple of
          (logits with shape `(batch_size * beam_size, vocab_size)`,
           updated cache values)
      """
      # Set decoder input to the last generated IDs
      decoder_input = ids[:, -1:]

      # Preprocess decoder input by getting embeddings and adding timing signal.
      decoder_input = self.embedding_lookup(decoder_input)
      decoder_input += timing_signal[i]
      if self._padded_decode:
        # indexing does not work on TPU.
        bias_shape = decoder_self_attention_mask.shape.as_list()
        self_attention_mask = tf.slice(decoder_self_attention_mask, [0, i, 0],
                                       [bias_shape[0], 1, bias_shape[2]])
      else:
        self_attention_mask = decoder_self_attention_mask[:, i:i + 1, :i + 1]
      decoder_shape = tf_utils.get_shape_list(decoder_input, expected_rank=3)
      batch_size = decoder_shape[0]
      decoder_length = decoder_shape[1]

      self_attention_mask = tf.tile(self_attention_mask, [batch_size, 1, 1])
      attention_mask = cache.get("encoder_decoder_attention_mask")
      attention_mask = tf.tile(attention_mask, [1, decoder_length, 1])

      decoder_outputs = self.decoder_layer(
          decoder_input,
          cache.get("encoder_outputs"),
          self_attention_mask=self_attention_mask,
          cross_attention_mask=attention_mask,
          cache=cache,
          decode_loop_step=i if self._padded_decode else None)

      decoder_outputs = tf.cast(decoder_outputs, dtype=self.compute_dtype)
      logits = self._embedding_linear(self.embedding_lookup.embeddings,
                                      decoder_outputs)
      logits = tf.squeeze(logits, axis=[1])
      return logits, cache

    return symbols_to_logits_fn


class TransformerEncoder(tf_keras.layers.Layer):
  """Transformer encoder.

  Transformer encoder is made up of N identical layers. Each layer is composed
  of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
  """

  def __init__(self,
               num_layers=6,
               num_attention_heads=8,
               intermediate_size=2048,
               activation="relu",
               dropout_rate=0.0,
               attention_dropout_rate=0.0,
               use_bias=False,
               norm_first=True,
               norm_epsilon=1e-6,
               intermediate_dropout=0.0,
               **kwargs):
    """Initialize a Transformer encoder.

    Args:
      num_layers: Number of layers.
      num_attention_heads: Number of attention heads.
      intermediate_size: Size of the intermediate (Feedforward) layer.
      activation: Activation for the intermediate layer.
      dropout_rate: Dropout probability.
      attention_dropout_rate: Dropout probability for attention layers.
      use_bias: Whether to enable use_bias in attention layer. If set False,
        use_bias in attention layer is disabled.
      norm_first: Whether to normalize inputs to attention and intermediate
        dense layers. If set False, output of attention and intermediate dense
        layers is normalized.
      norm_epsilon: Epsilon value to initialize normalization layers.
      intermediate_dropout: Dropout probability for intermediate_dropout_layer.
      **kwargs: key word arguemnts passed to tf_keras.layers.Layer.
    """

    super(TransformerEncoder, self).__init__(**kwargs)
    self.num_layers = num_layers
    self.num_attention_heads = num_attention_heads
    self._intermediate_size = intermediate_size
    self._activation = activation
    self._dropout_rate = dropout_rate
    self._attention_dropout_rate = attention_dropout_rate
    self._use_bias = use_bias
    self._norm_first = norm_first
    self._norm_epsilon = norm_epsilon
    self._intermediate_dropout = intermediate_dropout

  def build(self, input_shape):
    """Implements build() for the layer."""
    self.encoder_layers = []
    for i in range(self.num_layers):
      self.encoder_layers.append(
          layers.TransformerEncoderBlock(
              num_attention_heads=self.num_attention_heads,
              inner_dim=self._intermediate_size,
              inner_activation=self._activation,
              output_dropout=self._dropout_rate,
              attention_dropout=self._attention_dropout_rate,
              use_bias=self._use_bias,
              norm_first=self._norm_first,
              norm_epsilon=self._norm_epsilon,
              inner_dropout=self._intermediate_dropout,
              attention_initializer=attention_initializer(input_shape[2]),
              name=("layer_%d" % i)))
    self.output_normalization = tf_keras.layers.LayerNormalization(
        epsilon=self._norm_epsilon, dtype="float32")
    super(TransformerEncoder, self).build(input_shape)

  def get_config(self):
    config = {
        "num_layers": self.num_layers,
        "num_attention_heads": self.num_attention_heads,
        "intermediate_size": self._intermediate_size,
        "activation": self._activation,
        "dropout_rate": self._dropout_rate,
        "attention_dropout_rate": self._attention_dropout_rate,
        "use_bias": self._use_bias,
        "norm_first": self._norm_first,
        "norm_epsilon": self._norm_epsilon,
        "intermediate_dropout": self._intermediate_dropout
    }
    base_config = super(TransformerEncoder, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, encoder_inputs, attention_mask=None):
    """Return the output of the encoder.

    Args:
      encoder_inputs: A tensor with shape `(batch_size, input_length,
        hidden_size)`.
      attention_mask: A mask for the encoder self-attention layer with shape
        `(batch_size, input_length, input_length)`.

    Returns:
      Output of encoder which is a `float32` tensor with shape
        `(batch_size, input_length, hidden_size)`.
    """
    for layer_idx in range(self.num_layers):
      encoder_inputs = self.encoder_layers[layer_idx](
          [encoder_inputs, attention_mask])

    output_tensor = encoder_inputs
    output_tensor = self.output_normalization(output_tensor)

    return output_tensor


class TransformerDecoder(tf_keras.layers.Layer):
  """Transformer decoder.

  Like the encoder, the decoder is made up of N identical layers.
  Each layer is composed of the sublayers:
    1. Self-attention layer
    2. Multi-headed attention layer combining encoder outputs with results from
       the previous self-attention layer.
    3. Feedforward network (2 fully-connected layers)
  """

  def __init__(self,
               num_layers=6,
               num_attention_heads=8,
               intermediate_size=2048,
               activation="relu",
               dropout_rate=0.0,
               attention_dropout_rate=0.0,
               use_bias=False,
               norm_first=True,
               norm_epsilon=1e-6,
               intermediate_dropout=0.0,
               self_attention_cls=None,
               cross_attention_cls=None,
               **kwargs):
    """Initialize a Transformer decoder.

    Args:
      num_layers: Number of layers.
      num_attention_heads: Number of attention heads.
      intermediate_size: Size of the intermediate (Feedforward) layer.
      activation: Activation for the intermediate layer.
      dropout_rate: Dropout probability.
      attention_dropout_rate: Dropout probability for attention layers.
      use_bias: Whether to enable use_bias in attention layer. If set `False`,
        use_bias in attention layer is disabled.
      norm_first: Whether to normalize inputs to attention and intermediate
        dense layers. If set `False`, output of attention and intermediate dense
        layers is normalized.
      norm_epsilon: Epsilon value to initialize normalization layers.
      intermediate_dropout: Dropout probability for intermediate_dropout_layer.
      self_attention_cls: An optional class to use for self attention
        or a function that provides the class per layer.
      cross_attention_cls: An optional class to use for cross attention
        or a function that provides the class per layer.
      **kwargs: key word arguemnts passed to tf_keras.layers.Layer.
    """
    super(TransformerDecoder, self).__init__(**kwargs)
    self.num_layers = num_layers
    self.num_attention_heads = num_attention_heads
    self._intermediate_size = intermediate_size
    self._activation = activation
    self._dropout_rate = dropout_rate
    self._attention_dropout_rate = attention_dropout_rate
    self._use_bias = use_bias
    self._norm_first = norm_first
    self._norm_epsilon = norm_epsilon
    self._intermediate_dropout = intermediate_dropout
    self._self_attention_cls = self_attention_cls
    self._cross_attention_cls = cross_attention_cls

  def build(self, input_shape):
    """Implements build() for the layer."""

    def _select_attention_cls(attention_cls, index):
      cls = None
      if attention_cls is not None:
        cls = (
            attention_cls(index)
            if inspect.isfunction(attention_cls)
            else attention_cls
        )
      return cls

    self.decoder_layers = []
    for i in range(self.num_layers):
      self_attention_cls = _select_attention_cls(self._self_attention_cls, i)
      cross_attention_cls = _select_attention_cls(self._cross_attention_cls, i)
      self.decoder_layers.append(
          layers.TransformerDecoderBlock(
              num_attention_heads=self.num_attention_heads,
              intermediate_size=self._intermediate_size,
              intermediate_activation=self._activation,
              dropout_rate=self._dropout_rate,
              attention_dropout_rate=self._attention_dropout_rate,
              use_bias=self._use_bias,
              norm_first=self._norm_first,
              norm_epsilon=self._norm_epsilon,
              intermediate_dropout=self._intermediate_dropout,
              attention_initializer=attention_initializer(input_shape[2]),
              name=("layer_%d" % i),
              self_attention_cls=self_attention_cls,
              cross_attention_cls=cross_attention_cls))
    self.output_normalization = tf_keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(TransformerDecoder, self).build(input_shape)

  def get_config(self):
    config = {
        "num_layers": self.num_layers,
        "num_attention_heads": self.num_attention_heads,
        "intermediate_size": self._intermediate_size,
        "activation": self._activation,
        "dropout_rate": self._dropout_rate,
        "attention_dropout_rate": self._attention_dropout_rate,
        "use_bias": self._use_bias,
        "norm_first": self._norm_first,
        "norm_epsilon": self._norm_epsilon,
        "intermediate_dropout": self._intermediate_dropout,
        "self_attention_cls": self._self_attention_cls,
        "cross_attention_cls": self._cross_attention_cls,
    }
    base_config = super(TransformerDecoder, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self,
           target,
           memory,
           self_attention_mask=None,
           cross_attention_mask=None,
           cache=None,
           decode_loop_step=None,
           return_all_decoder_outputs=False):
    """Return the output of the decoder layer stacks.

    Args:
      target: A tensor with shape `(batch_size, target_length, hidden_size)`.
      memory: A tensor with shape `(batch_size, input_length, hidden_size)`.
      self_attention_mask: A tensor with shape `(batch_size, target_len,
        target_length)`, the mask for decoder self-attention layer.
      cross_attention_mask: A tensor with shape `(batch_size, target_length,
        input_length)` which is the mask for encoder-decoder attention layer.
      cache: (Used for fast decoding) A nested dictionary storing previous
        decoder self-attention values. The items are:
        {layer_n: {"k": A tensor with shape `(batch_size, i, key_channels)`,
                   "v": A tensor with shape `(batch_size, i, value_channels)`},
                     ...}
      decode_loop_step: An integer, the step number of the decoding loop. Used
        only for autoregressive inference on TPU.
      return_all_decoder_outputs: Return all decoder layer outputs.
        Note that the outputs are layer normed.
        This is useful when introducing per layer auxiliary loss.

    Returns:
      Output of decoder.
      float32 tensor with shape `(batch_size, target_length, hidden_size`).
    """

    output_tensor = target
    decoder_outputs = []
    for layer_idx in range(self.num_layers):
      transformer_inputs = [
          output_tensor, memory, cross_attention_mask, self_attention_mask
      ]
      # Gets the cache for decoding.
      if cache is None:
        output_tensor, _ = self.decoder_layers[layer_idx](transformer_inputs)
      else:
        cache_layer_idx = str(layer_idx)
        output_tensor, cache[cache_layer_idx] = self.decoder_layers[layer_idx](
            transformer_inputs,
            cache=cache[cache_layer_idx],
            decode_loop_step=decode_loop_step)
      if return_all_decoder_outputs:
        decoder_outputs.append(self.output_normalization(output_tensor))

    if return_all_decoder_outputs:
      return decoder_outputs
    else:
      return self.output_normalization(output_tensor)


def attention_initializer(hidden_size):
  """Initializer for attention layers in Seq2SeqTransformer."""
  hidden_size = int(hidden_size)
  limit = math.sqrt(6.0 / (hidden_size + hidden_size))
  return tf_keras.initializers.RandomUniform(minval=-limit, maxval=limit)
