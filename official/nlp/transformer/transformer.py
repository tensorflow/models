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
from official.nlp.modeling.layers import position_embedding
from official.nlp.modeling.ops import beam_search
from official.nlp.transformer import attention_layer
from official.nlp.transformer import embedding_layer
from official.nlp.transformer import ffn_layer
from official.nlp.transformer import metrics
from official.nlp.transformer import model_utils
from official.nlp.transformer.utils.tokenizer import EOS_ID

# Disable the not-callable lint error, since it claims many objects are not
# callable when they actually are.
# pylint: disable=not-callable


def create_model(params, is_train):
  """Creates transformer model."""
  with tf.name_scope("model"):
    if is_train:
      inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
      targets = tf.keras.layers.Input((None,), dtype="int64", name="targets")
      internal_model = Transformer(params, name="transformer_v2")
      logits = internal_model([inputs, targets], training=is_train)
      vocab_size = params["vocab_size"]
      label_smoothing = params["label_smoothing"]
      if params["enable_metrics_in_training"]:
        logits = metrics.MetricLayer(vocab_size)([logits, targets])
      logits = tf.keras.layers.Lambda(
          lambda x: x, name="logits", dtype=tf.float32)(
              logits)
      model = tf.keras.Model([inputs, targets], logits)
      loss = metrics.transformer_loss(logits, targets, label_smoothing,
                                      vocab_size)
      model.add_loss(loss)
      return model

    else:
      inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
      internal_model = Transformer(params, name="transformer_v2")
      ret = internal_model([inputs], training=is_train)
      outputs, scores = ret["outputs"], ret["scores"]
      return tf.keras.Model(inputs, [outputs, scores])


class Transformer(tf.keras.Model):
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
    super(Transformer, self).__init__(name=name)
    self.params = params
    self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
        params["vocab_size"], params["hidden_size"])
    self.encoder_stack = EncoderStack(params)
    self.decoder_stack = DecoderStack(params)
    self.position_embedding = position_embedding.RelativePositionEmbedding(
        hidden_size=self.params["hidden_size"])

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
    inputs = inputs if isinstance(inputs, list) else [inputs]
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
        inputs.set_shape([decode_batch_size, self.params["decode_max_length"]])

    # Variance scaling is used here because it seems to work in many problems.
    # Other reasonable initializers may also work just as well.
    with tf.name_scope("Transformer"):
      # Calculate attention bias for encoder self-attention and decoder
      # multi-headed attention layers.
      attention_bias = model_utils.get_padding_bias(inputs)

      # Run the inputs through the encoder layer to map the symbol
      # representations to continuous representations.
      encoder_outputs = self.encode(inputs, attention_bias, training)
      # Generate output sequence if targets is None, or return logits if target
      # sequence is known.
      if targets is None:
        return self.predict(encoder_outputs, attention_bias, training)
      else:
        logits = self.decode(targets, encoder_outputs, attention_bias, training)
        return logits

  def encode(self, inputs, attention_bias, training):
    """Generate continuous representation for inputs.

    Args:
      inputs: int tensor with shape [batch_size, input_length].
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length].
      training: boolean, whether in training mode or not.

    Returns:
      float tensor with shape [batch_size, input_length, hidden_size]
    """
    with tf.name_scope("encode"):
      # Prepare inputs to the layer stack by adding positional encodings and
      # applying dropout.
      embedded_inputs = self.embedding_softmax_layer(inputs)
      embedded_inputs = tf.cast(embedded_inputs, self.params["dtype"])
      inputs_padding = model_utils.get_padding(inputs)
      attention_bias = tf.cast(attention_bias, self.params["dtype"])

      with tf.name_scope("add_pos_encoding"):
        pos_encoding = self.position_embedding(inputs=embedded_inputs)
        pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
        encoder_inputs = embedded_inputs + pos_encoding

      if training:
        encoder_inputs = tf.nn.dropout(
            encoder_inputs, rate=self.params["layer_postprocess_dropout"])

      return self.encoder_stack(
          encoder_inputs, attention_bias, inputs_padding, training=training)

  def decode(self, targets, encoder_outputs, attention_bias, training):
    """Generate logits for each value in the target sequence.

    Args:
      targets: target values for the output sequence. int tensor with shape
        [batch_size, target_length]
      encoder_outputs: continuous representation of input sequence. float tensor
        with shape [batch_size, input_length, hidden_size]
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length]
      training: boolean, whether in training mode or not.

    Returns:
      float32 tensor with shape [batch_size, target_length, vocab_size]
    """
    with tf.name_scope("decode"):
      # Prepare inputs to decoder layers by shifting targets, adding positional
      # encoding and applying dropout.
      decoder_inputs = self.embedding_softmax_layer(targets)
      decoder_inputs = tf.cast(decoder_inputs, self.params["dtype"])
      attention_bias = tf.cast(attention_bias, self.params["dtype"])
      with tf.name_scope("shift_targets"):
        # Shift targets to the right, and remove the last element
        decoder_inputs = tf.pad(decoder_inputs,
                                [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
      with tf.name_scope("add_pos_encoding"):
        length = tf.shape(decoder_inputs)[1]
        pos_encoding = self.position_embedding(decoder_inputs)
        pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
        decoder_inputs += pos_encoding
      if training:
        decoder_inputs = tf.nn.dropout(
            decoder_inputs, rate=self.params["layer_postprocess_dropout"])

      # Run values
      decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
          length, dtype=self.params["dtype"])
      outputs = self.decoder_stack(
          decoder_inputs,
          encoder_outputs,
          decoder_self_attention_bias,
          attention_bias,
          training=training)
      logits = self.embedding_softmax_layer(outputs, mode="linear")
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
      decoder_input = self.embedding_softmax_layer(decoder_input)

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

      decoder_outputs = self.decoder_stack(
          decoder_input,
          cache.get("encoder_outputs"),
          self_attention_bias,
          cache.get("encoder_decoder_attention_bias"),
          training=training,
          cache=cache,
          decode_loop_step=i if self.params["padded_decode"] else None)
      logits = self.embedding_softmax_layer(decoder_outputs, mode="linear")
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
        "layer_%d" % layer: {
            "k":
                tf.zeros(
                    [batch_size, init_decode_length, num_heads, dim_per_head],
                    dtype=self.params["dtype"]),
            "v":
                tf.zeros(
                    [batch_size, init_decode_length, num_heads, dim_per_head],
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


class PrePostProcessingWrapper(tf.keras.layers.Layer):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, params):
    super(PrePostProcessingWrapper, self).__init__()
    self.layer = layer
    self.params = params
    self.postprocess_dropout = params["layer_postprocess_dropout"]

  def build(self, input_shape):
    # Create normalization layer
    self.layer_norm = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(PrePostProcessingWrapper, self).build(input_shape)

  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self, x, *args, **kwargs):
    """Calls wrapped layer with same parameters."""
    # Preprocessing: apply layer normalization
    training = kwargs["training"]

    y = self.layer_norm(x)

    # Get layer output
    y = self.layer(y, *args, **kwargs)

    # Postprocessing: apply dropout and residual connection
    if training:
      y = tf.nn.dropout(y, rate=self.postprocess_dropout)
    return x + y


class EncoderStack(tf.keras.layers.Layer):
  """Transformer encoder stack.

  The encoder stack is made up of N identical layers. Each layer is composed
  of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
  """

  def __init__(self, params):
    super(EncoderStack, self).__init__()
    self.params = params
    self.layers = []

  def build(self, input_shape):
    """Builds the encoder stack."""
    params = self.params
    for _ in range(params["num_hidden_layers"]):
      # Create sublayers for each layer.
      self_attention_layer = attention_layer.SelfAttention(
          params["hidden_size"], params["num_heads"],
          params["attention_dropout"])
      feed_forward_network = ffn_layer.FeedForwardNetwork(
          params["hidden_size"], params["filter_size"], params["relu_dropout"])

      self.layers.append([
          PrePostProcessingWrapper(self_attention_layer, params),
          PrePostProcessingWrapper(feed_forward_network, params)
      ])

    # Create final layer normalization layer.
    self.output_normalization = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(EncoderStack, self).build(input_shape)

  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self, encoder_inputs, attention_bias, inputs_padding, training):
    """Return the output of the encoder layer stacks.

    Args:
      encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
      attention_bias: bias for the encoder self-attention layer. [batch_size, 1,
        1, input_length]
      inputs_padding: tensor with shape [batch_size, input_length], inputs with
        zero paddings.
      training: boolean, whether in training mode or not.

    Returns:
      Output of encoder layer stack.
      float32 tensor with shape [batch_size, input_length, hidden_size]
    """
    for n, layer in enumerate(self.layers):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      with tf.name_scope("layer_%d" % n):
        with tf.name_scope("self_attention"):
          encoder_inputs = self_attention_layer(
              encoder_inputs, attention_bias, training=training)
        with tf.name_scope("ffn"):
          encoder_inputs = feed_forward_network(
              encoder_inputs, training=training)

    return self.output_normalization(encoder_inputs)


class DecoderStack(tf.keras.layers.Layer):
  """Transformer decoder stack.

  Like the encoder stack, the decoder stack is made up of N identical layers.
  Each layer is composed of the sublayers:
    1. Self-attention layer
    2. Multi-headed attention layer combining encoder outputs with results from
       the previous self-attention layer.
    3. Feedforward network (2 fully-connected layers)
  """

  def __init__(self, params):
    super(DecoderStack, self).__init__()
    self.params = params
    self.layers = []

  def build(self, input_shape):
    """Builds the decoder stack."""
    params = self.params
    for _ in range(params["num_hidden_layers"]):
      self_attention_layer = attention_layer.SelfAttention(
          params["hidden_size"], params["num_heads"],
          params["attention_dropout"])
      enc_dec_attention_layer = attention_layer.Attention(
          params["hidden_size"], params["num_heads"],
          params["attention_dropout"])
      feed_forward_network = ffn_layer.FeedForwardNetwork(
          params["hidden_size"], params["filter_size"], params["relu_dropout"])

      self.layers.append([
          PrePostProcessingWrapper(self_attention_layer, params),
          PrePostProcessingWrapper(enc_dec_attention_layer, params),
          PrePostProcessingWrapper(feed_forward_network, params)
      ])
    self.output_normalization = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(DecoderStack, self).build(input_shape)

  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self,
           decoder_inputs,
           encoder_outputs,
           decoder_self_attention_bias,
           attention_bias,
           training,
           cache=None,
           decode_loop_step=None):
    """Return the output of the decoder layer stacks.

    Args:
      decoder_inputs: A tensor with shape [batch_size, target_length,
        hidden_size].
      encoder_outputs: A tensor with shape [batch_size, input_length,
        hidden_size]
      decoder_self_attention_bias: A tensor with shape [1, 1, target_len,
        target_length], the bias for decoder self-attention layer.
      attention_bias: A tensor with shape [batch_size, 1, 1, input_length], the
        bias for encoder-decoder attention layer.
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
    for n, layer in enumerate(self.layers):
      self_attention_layer = layer[0]
      enc_dec_attention_layer = layer[1]
      feed_forward_network = layer[2]

      # Run inputs through the sublayers.
      layer_name = "layer_%d" % n
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.name_scope(layer_name):
        with tf.name_scope("self_attention"):
          decoder_inputs = self_attention_layer(
              decoder_inputs,
              decoder_self_attention_bias,
              training=training,
              cache=layer_cache,
              decode_loop_step=decode_loop_step)
        with tf.name_scope("encdec_attention"):
          decoder_inputs = enc_dec_attention_layer(
              decoder_inputs,
              encoder_outputs,
              attention_bias,
              training=training)
        with tf.name_scope("ffn"):
          decoder_inputs = feed_forward_network(
              decoder_inputs, training=training)

    return self.output_normalization(decoder_inputs)
