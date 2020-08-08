# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Defines the Transformer model in TF 2.0.

Model paper: https://arxiv.org/pdf/1706.03762.pdf
Transformer model code source: https://github.com/tensorflow/tensor2tensor
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from official.modeling import tf_utils
from official.modeling.activations import attention_initializer
from official.nlp.modeling import layers
from official.nlp.modeling.layers import position_embedding
from official.nlp.modeling.layers import transformer
from official.nlp.modeling.ops import beam_search
from official.nlp.transformer import metrics
from official.nlp.transformer import model_utils
from official.nlp.transformer.utils.tokenizer import EOS_ID


# Disable the not-callable lint error, since it claims many objects are not
# callable when they actually are.
# pylint: disable=not-callable

@tf.keras.utils.register_keras_serializable(package="Text")
class Seq2SeqTransformer(tf.keras.Model):
  """Transformer model with Keras.

  Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf

  The Transformer model consists of an encoder and decoder. The input is an int
  sequence (or a batch of sequences). The encoder produces a continuous
  representation, and the decoder uses the encoder output to generate
  probabilities for the output sequence.
  """

  def __init__(self, params, name=None):
    """Initialize layers to build Transformer model.

    Args:
      params: hyperparameter object defining layer sizes, dropout values, etc.
      name: name of the model.
    """
    super(Seq2SeqTransformer, self).__init__(name=name)
    self.params = params
    self.embedding_lookup = layers.OnDeviceEmbedding(
        vocab_size=params["vocab_size"],
        embedding_width=params["hidden_size"],
        initializer=tf.random_normal_initializer(
            mean=0., stddev=params["hidden_size"]**-0.5),
        use_scale=True)
    self.encoder_layer = TransformerEncoder(
        num_layers=self.params["num_hidden_layers"],
        num_attention_heads=self.params["num_heads"],
        intermediate_size=self.params["filter_size"],
        activation="relu",
        dropout_rate=self.params["relu_dropout"],
        attention_dropout_rate=self.params["attention_dropout"],
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-6,
        intermediate_dropout=self.params["relu_dropout"])
    self.decoder_layer = TransformerDecoder(
        num_layers=self.params["num_hidden_layers"],
        num_attention_heads=self.params["num_heads"],
        intermediate_size=self.params["filter_size"],
        activation="relu",
        dropout_rate=self.params["relu_dropout"],
        attention_dropout_rate=self.params["attention_dropout"],
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-6,
        intermediate_dropout=self.params["relu_dropout"])
    self.position_embedding = position_embedding.RelativePositionEmbedding(
        hidden_size=self.params["hidden_size"])
    self.encoder_dropout = tf.keras.layers.Dropout(
        rate=self.params["layer_postprocess_dropout"])
    self.decoder_dropout = tf.keras.layers.Dropout(
        rate=self.params["layer_postprocess_dropout"])

  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self, inputs, training):
    """Calculate target logits or inferred target sequences.

    Args:
      inputs: input tensor list of size 1 or 2.
        First item, inputs: int tensor with shape [batch_size, input_length].
        Second item (optional), targets: None or int tensor with shape
          [batch_size, target_length].
      training: boolean, whether in training mode or not.

    Returns:
      If targets is defined, then return logits for each word in the target
      sequence. float tensor with shape [batch_size, target_length, vocab_size]
      If target is none, then generate output sequence one token at a time.
        returns a dictionary {
          outputs: [batch_size, decoded length]
          scores: [batch_size, float]}
      Even when float16 is used, the output tensor(s) are always float32.

    Raises:
      NotImplementedError: If try to use padded decode method on CPU/GPUs.
    """
    if len(inputs) == 2:
      inputs, targets = inputs[0], inputs[1]
    else:
      # Decoding path.
      inputs, targets = inputs[0], None
      if self.params["padded_decode"]:
        if not self.params["num_replicas"]:
          raise NotImplementedError(
              "Padded decoding on CPU/GPUs is not supported.")
        decode_batch_size = int(self.params["decode_batch_size"] /
                                self.params["num_replicas"])
        inputs.set_shape([
            decode_batch_size, self.params["decode_max_length"]
        ])

    with tf.name_scope("Transformer"):
      attention_bias = model_utils.get_padding_bias(inputs)
      attention_bias = tf.cast(attention_bias, self.params["dtype"])
      with tf.name_scope("encode"):
        # Prepare inputs to the layer stack by adding positional encodings and
        # applying dropout.
        embedded_inputs = self.embedding_lookup(inputs)
        embedding_mask = tf.cast(tf.not_equal(inputs, 0),
                                 self.embedding_lookup.embeddings.dtype)
        embedded_inputs *= tf.expand_dims(embedding_mask, -1)
        embedded_inputs = tf.cast(embedded_inputs, self.params["dtype"])

        # Attention_mask generation.
        input_shape = tf_utils.get_shape_list(inputs, expected_rank=2)
        attention_mask = tf.cast(
            tf.reshape(tf.not_equal(inputs, 0),
                       [input_shape[0], 1, input_shape[1]]),
            dtype=inputs.dtype)
        broadcast_ones = tf.ones(
            shape=[input_shape[0], input_shape[1], 1], dtype=inputs.dtype)
        attention_mask = broadcast_ones * attention_mask

        with tf.name_scope("add_pos_encoding"):
          pos_encoding = self.position_embedding(inputs=embedded_inputs)
          pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
          encoder_inputs = embedded_inputs + pos_encoding

        # if training:
        #   encoder_inputs = tf.nn.dropout(
        #       encoder_inputs, rate=self.params["layer_postprocess_dropout"])

        encoder_inputs = self.encoder_dropout(encoder_inputs)

        encoder_outputs = self.encoder_layer(encoder_inputs,
                                             attention_mask=attention_mask)

      if targets is None:
        # return self.predict(encoder_outputs, attention_bias, training)
        encoder_decoder_attention_bias = attention_bias
        encoder_outputs = tf.cast(encoder_outputs, self.params["dtype"])
        if self.params["padded_decode"]:
          batch_size = encoder_outputs.shape.as_list()[0]
          input_length = encoder_outputs.shape.as_list()[1]
        else:
          batch_size = tf.shape(encoder_outputs)[0]
          input_length = tf.shape(encoder_outputs)[1]
        max_decode_length = input_length + self.params["extra_decode_length"]
        encoder_decoder_attention_bias = tf.cast(encoder_decoder_attention_bias,
                                                 self.params["dtype"])

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(
            max_decode_length, training)

        # Create initial set of IDs that will be passed to symbols_to_logits_fn.
        initial_ids = tf.zeros([batch_size], dtype=tf.int32)

        # Create cache storing decoder attention values for each layer.
        # pylint: disable=g-complex-comprehension
        init_decode_length = (
            max_decode_length if self.params["padded_decode"] else 0)
        num_heads = self.params["num_heads"]
        dim_per_head = self.params["hidden_size"] // num_heads

        cache = {
            str(layer): {
                "key":
                    tf.zeros([
                        batch_size, init_decode_length, num_heads, dim_per_head
                    ],
                             dtype=self.params["dtype"]),
                "value":
                    tf.zeros([
                        batch_size, init_decode_length, num_heads, dim_per_head
                    ],
                             dtype=self.params["dtype"])
            } for layer in range(self.params["num_hidden_layers"])
        }

        # pylint: enable=g-complex-comprehension

        # Add encoder output and attention bias to the cache.
        cache["encoder_outputs"] = encoder_outputs
        cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

        # Use beam search to find the top beam_size sequences and scores.
        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.params["vocab_size"],
            beam_size=self.params["beam_size"],
            alpha=self.params["alpha"],
            max_decode_length=max_decode_length,
            eos_id=EOS_ID,
            padded_decode=self.params["padded_decode"],
            dtype=self.params["dtype"])

        # Get the top sequence for each batch element
        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        return {"outputs": top_decoded_ids, "scores": top_scores}

      else:
        with tf.name_scope("decode"):
          decoder_inputs = self.embedding_lookup(targets)
          embedding_mask = tf.cast(tf.not_equal(targets, 0),
                                   self.embedding_lookup.embeddings.dtype)
          decoder_inputs *= tf.expand_dims(embedding_mask, -1)
          decoder_inputs = tf.cast(decoder_inputs, self.params["dtype"])
          with tf.name_scope("shift_targets"):
            # Shift targets to the right, and remove the last element
            decoder_inputs = tf.pad(decoder_inputs,
                                    [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
          with tf.name_scope("add_pos_encoding"):
            length = tf.shape(decoder_inputs)[1]
            pos_encoding = self.position_embedding(decoder_inputs)
            pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
            decoder_inputs += pos_encoding

          # if training:
          #   decoder_inputs = tf.nn.dropout(
          #       decoder_inputs, rate=self.params["layer_postprocess_dropout"])

          decoder_inputs = self.decoder_dropout(decoder_inputs)

          decoder_shape = tf_utils.get_shape_list(decoder_inputs,
                                                  expected_rank=3)
          batch_size = decoder_shape[0]
          decoder_length = decoder_shape[1]

          self_attention_mask = tf.linalg.band_part(
              tf.ones([length, length], dtype=tf.float32), -1, 0)
          self_attention_mask = tf.reshape(self_attention_mask,
                                           [1, length, length])
          self_attention_mask = tf.tile(self_attention_mask, [batch_size, 1, 1])

          attention_mask = tf.cast(
              tf.expand_dims(tf.not_equal(inputs, 0), axis=1),
              dtype=inputs.dtype)
          attention_mask = tf.tile(attention_mask, [1, decoder_length, 1])

          outputs = self.decoder_layer(
              decoder_inputs,
              encoder_outputs,
              memory_mask=self_attention_mask,
              target_mask=attention_mask)
          logits = embedding_linear(self.embedding_lookup.embeddings, outputs)
          logits = tf.cast(logits, tf.float32)

        return logits


  def _get_symbols_to_logits_fn(self, max_decode_length, training):
    """Returns a decoding function that calculates logits of the next tokens."""
    timing_signal = self.position_embedding(
        inputs=None, length=max_decode_length + 1)
    timing_signal = tf.cast(timing_signal, self.params["dtype"])
    decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
        max_decode_length, dtype=self.params["dtype"])

    def symbols_to_logits_fn(ids, i, cache):
      """Generate logits for next potential IDs.

      Args:
        ids: Current decoded sequences. int tensor with shape [batch_size *
          beam_size, i + 1].
        i: Loop index.
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.

      Returns:
        Tuple of
          (logits with shape [batch_size * beam_size, vocab_size],
           updated cache values)
      """
      # Set decoder input to the last generated IDs
      decoder_input = ids[:, -1:]

      # Preprocess decoder input by getting embeddings and adding timing signal.
      # decoder_input = self.embedding_softmax_layer(decoder_input)
      source_decoder_input = decoder_input
      decoder_input = self.embedding_lookup(decoder_input)
      embedding_mask = tf.cast(tf.not_equal(source_decoder_input, 0),
                               self.embedding_lookup.embeddings.dtype)
      decoder_input *= tf.expand_dims(embedding_mask, -1)

      if self.params["padded_decode"]:
        timing_signal_shape = timing_signal.shape.as_list()
        decoder_input += tf.slice(timing_signal, [i, 0],
                                  [1, timing_signal_shape[1]])

        bias_shape = decoder_self_attention_bias.shape.as_list()
        self_attention_bias = tf.slice(
            decoder_self_attention_bias, [0, 0, i, 0],
            [bias_shape[0], bias_shape[1], 1, bias_shape[3]])
      else:
        decoder_input += timing_signal[i:i + 1]

        self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      decoder_shape = tf_utils.get_shape_list(decoder_input, expected_rank=3)
      batch_size = decoder_shape[0]
      decoder_length = decoder_shape[1]

      attention_bias = cache.get("encoder_decoder_attention_bias")
      attention_bias = tf.where(attention_bias < 0,
                                tf.zeros_like(attention_bias),
                                tf.ones_like(attention_bias))
      attention_bias = tf.squeeze(attention_bias, axis=[1])
      attention_mask = tf.tile(attention_bias, [1, decoder_length, 1])

      self_attention_bias = tf.where(self_attention_bias < 0,
                                     tf.zeros_like(self_attention_bias),
                                     tf.ones_like(self_attention_bias))
      self_attention_bias = tf.squeeze(self_attention_bias, axis=[1])
      self_attention_mask = tf.tile(self_attention_bias, [batch_size, 1, 1])


      decoder_outputs = self.decoder_layer(
          decoder_input,
          cache.get("encoder_outputs"),
          memory_mask=self_attention_mask,
          target_mask=attention_mask,
          cache=cache,
          decode_loop_step=i if self.params["padded_decode"] else None)

      logits = embedding_linear(self.embedding_lookup.embeddings,
                                decoder_outputs)
      logits = tf.squeeze(logits, axis=[1])
      return logits, cache

    return symbols_to_logits_fn

  def predict(self, encoder_outputs, encoder_decoder_attention_bias, training):
    """Return predicted sequence."""
    encoder_outputs = tf.cast(encoder_outputs, self.params["dtype"])
    if self.params["padded_decode"]:
      batch_size = encoder_outputs.shape.as_list()[0]
      input_length = encoder_outputs.shape.as_list()[1]
    else:
      batch_size = tf.shape(encoder_outputs)[0]
      input_length = tf.shape(encoder_outputs)[1]
    max_decode_length = input_length + self.params["extra_decode_length"]
    encoder_decoder_attention_bias = tf.cast(encoder_decoder_attention_bias,
                                             self.params["dtype"])

    symbols_to_logits_fn = self._get_symbols_to_logits_fn(
        max_decode_length, training)

    # Create initial set of IDs that will be passed into symbols_to_logits_fn.
    initial_ids = tf.zeros([batch_size], dtype=tf.int32)

    # Create cache storing decoder attention values for each layer.
    # pylint: disable=g-complex-comprehension
    init_decode_length = (
        max_decode_length if self.params["padded_decode"] else 0)
    num_heads = self.params["num_heads"]
    dim_per_head = self.params["hidden_size"] // num_heads
    cache = {
        str(layer): {
            "key":
                tf.zeros([
                    batch_size, init_decode_length, num_heads, dim_per_head
                ],
                         dtype=self.params["dtype"]),
            "value":
                tf.zeros([
                    batch_size, init_decode_length, num_heads, dim_per_head
                ],
                         dtype=self.params["dtype"])
        } for layer in range(self.params["num_hidden_layers"])
    }
    # pylint: enable=g-complex-comprehension

    # Add encoder output and attention bias to the cache.
    cache["encoder_outputs"] = encoder_outputs
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    # Use beam search to find the top beam_size sequences and scores.
    decoded_ids, scores = beam_search.sequence_beam_search(
        symbols_to_logits_fn=symbols_to_logits_fn,
        initial_ids=initial_ids,
        initial_cache=cache,
        vocab_size=self.params["vocab_size"],
        beam_size=self.params["beam_size"],
        alpha=self.params["alpha"],
        max_decode_length=max_decode_length,
        eos_id=EOS_ID,
        padded_decode=self.params["padded_decode"],
        dtype=self.params["dtype"])

    # Get the top sequence for each batch element
    top_decoded_ids = decoded_ids[:, 0, 1:]
    top_scores = scores[:, 0]

    return {"outputs": top_decoded_ids, "scores": top_scores}

class TransformerEncoder(tf.keras.layers.Layer):
  """Transformer decoder stack.
  Like the encoder stack, the decoder stack is made up of N identical layers.
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
               intermediate_dropout=0.0):
    super(TransformerEncoder, self).__init__()
    self._num_layers = num_layers
    self._num_attention_heads = num_attention_heads
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
    for i in range(self._num_layers):
      self.encoder_layers.append(
          transformer.Transformer(
              num_attention_heads=self._num_attention_heads,
              intermediate_size=self._intermediate_size,
              intermediate_activation=self._activation,
              dropout_rate=self._dropout_rate,
              attention_dropout_rate=self._attention_dropout_rate,
              use_bias=self._use_bias,
              norm_first=self._norm_first,
              norm_epsilon=self._norm_epsilon,
              intermediate_dropout=self._intermediate_dropout,
              attention_initializer=attention_initializer.attention_initializer(
                  input_shape[2]),
              name=("layer_%d" % i)))
    self.output_normalization = tf.keras.layers.LayerNormalization(
        epsilon=self._norm_epsilon, dtype="float32")
    super(TransformerEncoder, self).build(input_shape)

  def get_config(self):
    return {
        "num_layers":
            self._num_layers,
         "num_attention_heads":
            self._num_attention_heads,
         "intermediate_size":
            self._intermediate_size,
         "activation":
            self._activation,
         "dropout_rate":
            self._dropout_rate,
         "attention_dropout_rate":
            self._attention_dropout_rate,
         "use_bias":
            self._use_bias,
         "norm_first":
            self._norm_first,
         "norm_epsilon":
            self._norm_epsilon,
         "intermediate_dropout":
            self._intermediate_dropout
    }

  def call(self,
           encoder_inputs,
           attention_mask=None):
    """Return the output of the decoder layer stacks.
    Args:
      decoder_inputs: A tensor with shape
        [batch_size, target_length, hidden_size].
      encoder_outputs: A tensor with shape
        [batch_size, input_length, hidden_size]
      decoder_self_attention_bias: A tensor with shape
        [1, 1, target_len, target_length], the bias for decoder self-attention
        layer.
      attention_bias: A tensor with shape [batch_size, 1, 1, input_length],
        the bias for encoder-decoder attention layer.
      training: A bool, whether in training mode or not.
      cache: (Used for fast decoding) A nested dictionary storing previous
        decoder self-attention values. The items are:
          {layer_n: {"k": A tensor with shape [batch_size, i, key_channels],
                     "v": A tensor with shape [batch_size, i, value_channels]},
                       ...}
      decode_loop_step: An integer, the step number of the decoding loop. Used
        only for autoregressive inference on TPU.
    Returns:
      Output of decoder layer stack.
      float32 tensor with shape [batch_size, target_length, hidden_size]
    """
    for layer_idx in range(self._num_layers):
      encoder_inputs = self.encoder_layers[layer_idx](
          [encoder_inputs, attention_mask])

    output_tensor = encoder_inputs
    output_tensor = self.output_normalization(output_tensor)

    return output_tensor

class TransformerDecoder(tf.keras.layers.Layer):
  """Transformer decoder stack.
  Like the encoder stack, the decoder stack is made up of N identical layers.
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
               intermediate_dropout=0.0):
    super(TransformerDecoder, self).__init__()
    self._num_layers = num_layers
    self._num_attention_heads = num_attention_heads
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
    self.decoder_layers = []
    for i in range(self._num_layers):
      self.decoder_layers.append(
          transformer.TransformerDecoderLayer(
              num_attention_heads=self._num_attention_heads,
              intermediate_size=self._intermediate_size,
              intermediate_activation=self._activation,
              dropout_rate=self._dropout_rate,
              attention_dropout_rate=self._attention_dropout_rate,
              use_bias=self._use_bias,
              norm_first=self._norm_first,
              norm_epsilon=self._norm_epsilon,
              intermediate_dropout=self._intermediate_dropout,
              attention_initializer=attention_initializer.attention_initializer(
                  input_shape[2]),
              name=("layer_%d" % i)))
    self.output_normalization = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(TransformerDecoder, self).build(input_shape)

  def get_config(self):
    return {
        "num_layers":
            self._num_layers,
         "num_attention_heads":
            self._num_attention_heads,
         "intermediate_size":
            self._intermediate_size,
         "activation":
            self._activation,
         "dropout_rate":
            self._dropout_rate,
         "attention_dropout_rate":
            self._attention_dropout_rate,
         "use_bias":
            self._use_bias,
         "norm_first":
            self._norm_first,
         "norm_epsilon":
            self._norm_epsilon,
         "intermediate_dropout":
            self._intermediate_dropout
    }

  def call(self,
           target,
           memory,
           memory_mask=None,
           target_mask=None,
           cache=None,
           decode_loop_step=None):
    """Return the output of the decoder layer stacks.
    Args:
      decoder_inputs: A tensor with shape
        [batch_size, target_length, hidden_size].
      encoder_outputs: A tensor with shape
        [batch_size, input_length, hidden_size]
      decoder_self_attention_bias: A tensor with shape
        [1, 1, target_len, target_length], the bias for decoder self-attention
        layer.
      attention_bias: A tensor with shape [batch_size, 1, 1, input_length],
        the bias for encoder-decoder attention layer.
      training: A bool, whether in training mode or not.
      cache: (Used for fast decoding) A nested dictionary storing previous
        decoder self-attention values. The items are:
          {layer_n: {"k": A tensor with shape [batch_size, i, key_channels],
                     "v": A tensor with shape [batch_size, i, value_channels]},
                       ...}
      decode_loop_step: An integer, the step number of the decoding loop. Used
        only for autoregressive inference on TPU.
    Returns:
      Output of decoder layer stack.
      float32 tensor with shape [batch_size, target_length, hidden_size]
    """

    output_tensor = target
    for layer_idx in range(self._num_layers):
      transformer_inputs = [
          output_tensor, memory, target_mask, memory_mask
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
    return self.output_normalization(output_tensor)


def embedding_linear(embedding_matrix, x):
  """Uses embeddings as linear transformation weights."""
  with tf.name_scope("presoftmax_linear"):
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]
    hidden_size = tf.shape(x)[2]
    vocab_size = tf.shape(embedding_matrix)[0]

    x = tf.reshape(x, [-1, hidden_size])
    logits = tf.matmul(x, embedding_matrix, transpose_b=True)

    return tf.reshape(logits, [batch_size, length, vocab_size])
