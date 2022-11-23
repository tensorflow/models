# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""tf.keras Models for NHNet."""
from typing import Optional, Text

from absl import logging
import gin
import tensorflow as tf

from official.modeling import tf_utils
from official.modeling.hyperparams import params_dict
from official.nlp.modeling import networks
from official.nlp.modeling.layers import multi_channel_attention
from official.nlp.modeling.ops import beam_search
from official.projects.nhnet import configs
from official.projects.nhnet import decoder
from official.projects.nhnet import utils


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


def _add_sos_to_seq(seq, start_token_id):
  """Add a start sequence token while keeping seq length."""
  batch_size = tf.shape(seq)[0]
  seq_len = tf.shape(seq)[1]
  sos_ids = tf.ones([batch_size], tf.int32) * start_token_id
  targets = tf.concat([tf.expand_dims(sos_ids, axis=1), seq], axis=1)
  targets = targets[:, :-1]
  tf.assert_equal(tf.shape(targets), (batch_size, seq_len))
  return targets


def remove_sos_from_seq(seq, pad_token_id):
  """Remove the start sequence token while keeping seq length."""
  batch_size, seq_len = tf_utils.get_shape_list(seq, expected_rank=2)
  # remove <s>
  targets = seq[:, 1:]
  # pad
  pad_ids = tf.ones([batch_size], tf.int32) * pad_token_id
  targets = tf.concat([targets, tf.expand_dims(pad_ids, axis=1)], axis=1)
  tf.assert_equal(tf.shape(targets), (batch_size, seq_len))
  return targets


class Bert2Bert(tf.keras.Model):
  """Bert2Bert encoder decoder model for training."""

  def __init__(self, params, bert_layer, decoder_layer, name=None):
    super(Bert2Bert, self).__init__(name=name)
    self.params = params
    if not bert_layer.built:
      raise ValueError("bert_layer should be built.")
    if not decoder_layer.built:
      raise ValueError("decoder_layer should be built.")
    self.bert_layer = bert_layer
    self.decoder_layer = decoder_layer

  def get_config(self):
    return {"params": self.params.as_dict()}

  def get_decode_logits(self,
                        decoder_inputs,
                        ids,
                        decoder_self_attention_bias,
                        step,
                        cache=None):
    if cache:
      if self.params.get("padded_decode", False):
        bias_shape = decoder_self_attention_bias.shape.as_list()
        self_attention_bias = tf.slice(
            decoder_self_attention_bias, [0, 0, step, 0],
            [bias_shape[0], bias_shape[1], 1, bias_shape[3]])
      else:
        self_attention_bias = decoder_self_attention_bias[:, :, step:step +
                                                          1, :step + 1]
      # Sets decoder input to the last generated IDs.
      decoder_input = ids[:, -1:]
    else:
      self_attention_bias = decoder_self_attention_bias[:, :, :step + 1, :step +
                                                        1]
      decoder_input = ids
    decoder_inputs["target_ids"] = decoder_input
    decoder_inputs["self_attention_bias"] = self_attention_bias
    if cache:
      decoder_outputs = self.decoder_layer(
          decoder_inputs,
          cache,
          decode_loop_step=step,
          padded_decode=self.params.get("padded_decode", False))
    else:
      decoder_outputs = self.decoder_layer(decoder_inputs)
    logits = embedding_linear(self.decoder_layer.embedding_lookup.embeddings,
                              decoder_outputs[:, -1:, :])
    logits = tf.squeeze(logits, axis=[1])
    return logits

  def _get_symbols_to_logits_fn(self, max_decode_length):
    """Returns a decoding function that calculates logits of the next tokens."""
    # Max decode length should be smaller than the positional embedding max
    # sequence length.
    decoder_self_attention_bias = decoder.get_attention_bias(
        input_tensor=None,
        bias_type="decoder_self",
        max_length=max_decode_length)

    def _symbols_to_logits_fn(ids, i, cache):
      """Generate logits for next candidate IDs.

      Args:
        ids: Current decoded sequences. int tensor with shape [batch_size *
          beam_size, i + 1]
        i: Loop index
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.

      Returns:
        Tuple of
          (logits with shape [batch_size * beam_size, vocab_size],
           updated cache values)
      """
      decoder_inputs = dict(
          all_encoder_outputs=cache["all_encoder_outputs"],
          attention_bias=cache["attention_bias"])
      logits = self.get_decode_logits(
          decoder_inputs,
          ids,
          decoder_self_attention_bias,
          step=i,
          cache=cache if self.params.use_cache else None)
      return logits, cache

    return _symbols_to_logits_fn

  def train_decode(self, decode_outputs):
    logits = embedding_linear(self.decoder_layer.embedding_lookup.embeddings,
                              decode_outputs)
    decode_output_ids = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    output_log_probs = tf.nn.log_softmax(logits, axis=-1)
    return logits, decode_output_ids, output_log_probs

  def predict_decode(self, start_token_ids, cache):
    symbols_to_logits_fn = self._get_symbols_to_logits_fn(self.params.len_title)
    # Use beam search to find the top beam_size sequences and scores.
    decoded_ids, scores = beam_search.sequence_beam_search(
        symbols_to_logits_fn=symbols_to_logits_fn,
        initial_ids=start_token_ids,
        initial_cache=cache,
        vocab_size=self.params.vocab_size,
        beam_size=self.params.beam_size,
        alpha=self.params.alpha,
        max_decode_length=self.params.len_title,
        padded_decode=self.params.get("padded_decode", False),
        eos_id=self.params.end_token_id)
    return decoded_ids, scores

  def _get_logits_for_decode_ids(self, decoder_inputs, top_decoded_ids):
    """Returns the log probabilities for ids."""
    target_ids = _add_sos_to_seq(top_decoded_ids, self.params.start_token_id)
    decoder_inputs["self_attention_bias"] = decoder.get_attention_bias(
        target_ids, bias_type="decoder_self")
    decoder_inputs["target_ids"] = target_ids
    decoder_outputs = self.decoder_layer(decoder_inputs)
    logits = embedding_linear(self.decoder_layer.embedding_lookup.embeddings,
                              decoder_outputs)
    return logits

  def _init_cache(self, batch_size):
    num_heads = self.params.num_decoder_attn_heads
    dim_per_head = self.params.hidden_size // num_heads
    init_decode_length = (
        self.params.len_title if self.params.get("padded_decode", False) else 0)
    cache = {}
    for layer in range(self.params.num_decoder_layers):
      cache[str(layer)] = {
          "key":
              tf.zeros(
                  [batch_size, init_decode_length, num_heads, dim_per_head],
                  dtype=tf.float32),
          "value":
              tf.zeros(
                  [batch_size, init_decode_length, num_heads, dim_per_head],
                  dtype=tf.float32)
      }
    return cache

  def call(self, inputs, mode="train"):
    """Implements call().

    Args:
      inputs: a dictionary of tensors.
      mode: string, an enum for mode, train/eval.

    Returns:
      logits, decode_output_ids, output_log_probs for training. top_decoded_ids
      for eval.
    """
    input_ids = inputs["input_ids"]
    input_mask = inputs["input_mask"]
    segment_ids = inputs["segment_ids"]
    all_encoder_outputs, _ = self.bert_layer(
        [input_ids, input_mask, segment_ids])

    if mode not in ("train", "eval", "predict"):
      raise ValueError("Invalid call mode: %s" % mode)
    encoder_decoder_attention_bias = decoder.get_attention_bias(
        input_ids,
        bias_type="single_cross",
        padding_value=self.params.pad_token_id)
    if mode == "train":
      self_attention_bias = decoder.get_attention_bias(
          inputs["target_ids"], bias_type="decoder_self")
      decoder_inputs = dict(
          attention_bias=encoder_decoder_attention_bias,
          all_encoder_outputs=all_encoder_outputs,
          target_ids=inputs["target_ids"],
          self_attention_bias=self_attention_bias)
      decoder_outputs = self.decoder_layer(decoder_inputs)
      return self.train_decode(decoder_outputs)

    batch_size = tf.shape(input_ids)[0]
    start_token_ids = tf.ones([batch_size],
                              tf.int32) * self.params.start_token_id
    # Add encoder output and attention bias to the cache.
    if self.params.use_cache:
      cache = self._init_cache(batch_size)
    else:
      cache = {}
    cache["all_encoder_outputs"] = all_encoder_outputs
    cache["attention_bias"] = encoder_decoder_attention_bias
    decoded_ids, scores = self.predict_decode(start_token_ids, cache)
    if mode == "predict":
      return decoded_ids[:, :self.params.beam_size,
                         1:], scores[:, :self.params.beam_size]

    decoder_inputs = dict(
        attention_bias=encoder_decoder_attention_bias,
        all_encoder_outputs=all_encoder_outputs)
    top_decoded_ids = decoded_ids[:, 0, 1:]
    return self._get_logits_for_decode_ids(decoder_inputs, top_decoded_ids)


class NHNet(Bert2Bert):
  """NHNet model which performs multi-doc decoding."""

  def __init__(self, params, bert_layer, decoder_layer, name=None):
    super(NHNet, self).__init__(params, bert_layer, decoder_layer, name=name)
    self.doc_attention = multi_channel_attention.VotingAttention(
        num_heads=params.num_decoder_attn_heads,
        head_size=params.hidden_size // params.num_decoder_attn_heads)

  def _expand_doc_attention_probs(self, doc_attention_probs, target_length):
    """Expands doc attention probs to fit the decoding sequence length."""
    doc_attention_probs = tf.expand_dims(
        doc_attention_probs, axis=[1])  # [B, 1, A]
    doc_attention_probs = tf.expand_dims(
        doc_attention_probs, axis=[2])  # [B, 1, 1, A]
    return tf.tile(doc_attention_probs,
                   [1, self.params.num_decoder_attn_heads, target_length, 1])

  def _get_symbols_to_logits_fn(self, max_decode_length):
    """Returns a decoding function that calculates logits of the next tokens."""
    # Max decode length should be smaller than the positional embedding max
    # sequence length.
    decoder_self_attention_bias = decoder.get_attention_bias(
        input_tensor=None,
        bias_type="decoder_self",
        max_length=max_decode_length)

    def _symbols_to_logits_fn(ids, i, cache):
      """Generate logits for next candidate IDs."""
      if self.params.use_cache:
        target_length = 1
      else:
        target_length = i + 1
      decoder_inputs = dict(
          doc_attention_probs=self._expand_doc_attention_probs(
              cache["doc_attention_probs"], target_length),
          all_encoder_outputs=cache["all_encoder_outputs"],
          attention_bias=cache["attention_bias"])
      logits = self.get_decode_logits(
          decoder_inputs,
          ids,
          decoder_self_attention_bias,
          step=i,
          cache=cache if self.params.use_cache else None)
      return logits, cache

    return _symbols_to_logits_fn

  def call(self, inputs, mode="training"):  # pytype: disable=signature-mismatch  # overriding-default-value-checks
    input_shape = tf_utils.get_shape_list(inputs["input_ids"], expected_rank=3)
    batch_size, num_docs, len_passage = (input_shape[0], input_shape[1],
                                         input_shape[2])
    input_ids = tf.reshape(inputs["input_ids"], [-1, len_passage])
    input_mask = tf.reshape(inputs["input_mask"], [-1, len_passage])
    segment_ids = tf.reshape(inputs["segment_ids"], [-1, len_passage])
    all_encoder_outputs, _ = self.bert_layer(
        [input_ids, input_mask, segment_ids])
    encoder_outputs = tf.reshape(
        all_encoder_outputs[-1],
        [batch_size, num_docs, len_passage, self.params.hidden_size])
    doc_attention_mask = tf.reshape(
        tf.cast(
            tf.math.count_nonzero(input_mask, axis=1, dtype=tf.int32) > 2,
            tf.int32), [batch_size, num_docs])

    doc_attention_probs = self.doc_attention(encoder_outputs,
                                             doc_attention_mask)
    encoder_decoder_attention_bias = decoder.get_attention_bias(
        inputs["input_ids"],
        bias_type="multi_cross",
        padding_value=self.params.pad_token_id)

    if mode == "train":
      target_length = tf_utils.get_shape_list(
          inputs["target_ids"], expected_rank=2)[1]
      doc_attention_probs = self._expand_doc_attention_probs(
          doc_attention_probs, target_length)
      self_attention_bias = decoder.get_attention_bias(
          inputs["target_ids"], bias_type="decoder_self")
      decoder_inputs = dict(
          attention_bias=encoder_decoder_attention_bias,
          self_attention_bias=self_attention_bias,
          target_ids=inputs["target_ids"],
          all_encoder_outputs=encoder_outputs,
          doc_attention_probs=doc_attention_probs)
      decoder_outputs = self.decoder_layer(decoder_inputs)
      return self.train_decode(decoder_outputs)

    # Adds encoder output and attention bias to the cache.
    if self.params.use_cache:
      cache = self._init_cache(batch_size)
    else:
      cache = {}
    cache["all_encoder_outputs"] = [encoder_outputs]
    cache["attention_bias"] = encoder_decoder_attention_bias
    cache["doc_attention_probs"] = doc_attention_probs

    start_token_ids = tf.ones([batch_size],
                              tf.int32) * self.params.start_token_id
    decoded_ids, scores = self.predict_decode(start_token_ids, cache)
    if mode == "predict":
      return decoded_ids[:, :self.params.beam_size,
                         1:], scores[:, :self.params.beam_size]

    top_decoded_ids = decoded_ids[:, 0, 1:]
    target_length = tf_utils.get_shape_list(top_decoded_ids)[-1]
    decoder_inputs = dict(
        attention_bias=encoder_decoder_attention_bias,
        all_encoder_outputs=[encoder_outputs],
        doc_attention_probs=self._expand_doc_attention_probs(
            doc_attention_probs, target_length))
    return self._get_logits_for_decode_ids(decoder_inputs, top_decoded_ids)


def get_bert2bert_layers(params: configs.BERT2BERTConfig):
  """Creates a Bert2Bert stem model and returns Bert encoder/decoder.

  We use funtional-style to create stem model because we need to make all layers
  built to restore variables in a customized way. The layers are called with
  placeholder inputs to make them fully built.

  Args:
    params: ParamsDict.

  Returns:
    two keras Layers, bert_model_layer and decoder_layer
  """
  input_ids = tf.keras.layers.Input(
      shape=(None,), name="input_ids", dtype=tf.int32)
  input_mask = tf.keras.layers.Input(
      shape=(None,), name="input_mask", dtype=tf.int32)
  segment_ids = tf.keras.layers.Input(
      shape=(None,), name="segment_ids", dtype=tf.int32)
  target_ids = tf.keras.layers.Input(
      shape=(None,), name="target_ids", dtype=tf.int32)
  bert_config = utils.get_bert_config_from_params(params)
  bert_model_layer = networks.BertEncoder(
      vocab_size=bert_config.vocab_size,
      hidden_size=bert_config.hidden_size,
      num_layers=bert_config.num_hidden_layers,
      num_attention_heads=bert_config.num_attention_heads,
      intermediate_size=bert_config.intermediate_size,
      activation=tf_utils.get_activation(bert_config.hidden_act),
      dropout_rate=bert_config.hidden_dropout_prob,
      attention_dropout_rate=bert_config.attention_probs_dropout_prob,
      max_sequence_length=bert_config.max_position_embeddings,
      type_vocab_size=bert_config.type_vocab_size,
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=bert_config.initializer_range),
      return_all_encoder_outputs=True,
      name="bert_encoder")
  all_encoder_outputs, _ = bert_model_layer(
      [input_ids, input_mask, segment_ids])
  # pylint: disable=protected-access
  decoder_layer = decoder.Decoder(params, bert_model_layer._embedding_layer)
  # pylint: enable=protected-access
  cross_attention_bias = decoder.AttentionBias(bias_type="single_cross")(
      input_ids)
  self_attention_bias = decoder.AttentionBias(bias_type="decoder_self")(
      target_ids)
  decoder_inputs = dict(
      attention_bias=cross_attention_bias,
      self_attention_bias=self_attention_bias,
      target_ids=target_ids,
      all_encoder_outputs=all_encoder_outputs)
  _ = decoder_layer(decoder_inputs)

  return bert_model_layer, decoder_layer


def get_nhnet_layers(params: configs.NHNetConfig):
  """Creates a Mult-doc encoder/decoder.

  Args:
    params: ParamsDict.

  Returns:
    two keras Layers, bert_model_layer and decoder_layer
  """
  input_ids = tf.keras.layers.Input(
      shape=(None,), name="input_ids", dtype=tf.int32)
  input_mask = tf.keras.layers.Input(
      shape=(None,), name="input_mask", dtype=tf.int32)
  segment_ids = tf.keras.layers.Input(
      shape=(None,), name="segment_ids", dtype=tf.int32)
  bert_config = utils.get_bert_config_from_params(params)
  bert_model_layer = networks.BertEncoder(
      vocab_size=bert_config.vocab_size,
      hidden_size=bert_config.hidden_size,
      num_layers=bert_config.num_hidden_layers,
      num_attention_heads=bert_config.num_attention_heads,
      intermediate_size=bert_config.intermediate_size,
      activation=tf_utils.get_activation(bert_config.hidden_act),
      dropout_rate=bert_config.hidden_dropout_prob,
      attention_dropout_rate=bert_config.attention_probs_dropout_prob,
      max_sequence_length=bert_config.max_position_embeddings,
      type_vocab_size=bert_config.type_vocab_size,
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=bert_config.initializer_range),
      return_all_encoder_outputs=True,
      name="bert_encoder")
  bert_model_layer([input_ids, input_mask, segment_ids])

  input_ids = tf.keras.layers.Input(
      shape=(None, None), name="input_ids", dtype=tf.int32)
  all_encoder_outputs = tf.keras.layers.Input((None, None, params.hidden_size),
                                              dtype=tf.float32)
  target_ids = tf.keras.layers.Input(
      shape=(None,), name="target_ids", dtype=tf.int32)
  doc_attention_probs = tf.keras.layers.Input(
      (params.num_decoder_attn_heads, None, None), dtype=tf.float32)
  # pylint: disable=protected-access
  decoder_layer = decoder.Decoder(params, bert_model_layer._embedding_layer)
  # pylint: enable=protected-access
  cross_attention_bias = decoder.AttentionBias(bias_type="multi_cross")(
      input_ids)
  self_attention_bias = decoder.AttentionBias(bias_type="decoder_self")(
      target_ids)
  decoder_inputs = dict(
      attention_bias=cross_attention_bias,
      self_attention_bias=self_attention_bias,
      target_ids=target_ids,
      all_encoder_outputs=all_encoder_outputs,
      doc_attention_probs=doc_attention_probs)
  _ = decoder_layer(decoder_inputs)

  return bert_model_layer, decoder_layer


def create_transformer_model(params,
                             init_checkpoint: Optional[Text] = None
                            ) -> tf.keras.Model:
  """A helper to create Transformer model."""
  bert_layer, decoder_layer = get_bert2bert_layers(params=params)
  model = Bert2Bert(
      params=params,
      bert_layer=bert_layer,
      decoder_layer=decoder_layer,
      name="transformer")

  if init_checkpoint:
    logging.info(
        "Checkpoint file %s found and restoring from "
        "initial checkpoint.", init_checkpoint)
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(init_checkpoint).expect_partial()

  return model


def create_bert2bert_model(
    params: configs.BERT2BERTConfig,
    cls=Bert2Bert,
    init_checkpoint: Optional[Text] = None) -> tf.keras.Model:
  """A helper to create Bert2Bert model."""
  bert_layer, decoder_layer = get_bert2bert_layers(params=params)
  if init_checkpoint:
    utils.initialize_bert2bert_from_pretrained_bert(bert_layer, decoder_layer,
                                                    init_checkpoint)
  return cls(
      params=params,
      bert_layer=bert_layer,
      decoder_layer=decoder_layer,
      name="bert2bert")


def create_nhnet_model(
    params: configs.NHNetConfig,
    cls=NHNet,
    init_checkpoint: Optional[Text] = None) -> tf.keras.Model:
  """A helper to create NHNet model."""
  bert_layer, decoder_layer = get_nhnet_layers(params=params)
  model = cls(
      params=params,
      bert_layer=bert_layer,
      decoder_layer=decoder_layer,
      name="nhnet")
  if init_checkpoint:
    logging.info(
        "Checkpoint file %s found and restoring from "
        "initial checkpoint.", init_checkpoint)
    if params.init_from_bert2bert:
      ckpt = tf.train.Checkpoint(model=model)
      ckpt.restore(init_checkpoint).assert_existing_objects_matched()
    else:
      utils.initialize_bert2bert_from_pretrained_bert(bert_layer, decoder_layer,
                                                      init_checkpoint)
  return model


@gin.configurable
def get_model_params(model: Optional[Text] = "bert2bert",
                     config_class=None) -> params_dict.ParamsDict:
  """Helper function to convert config file to ParamsDict."""
  if model == "bert2bert":
    return configs.BERT2BERTConfig()
  elif model == "nhnet":
    return configs.NHNetConfig()
  elif config_class:
    return config_class()
  else:
    raise KeyError("The model type is not defined: %s" % model)


@gin.configurable
def create_model(model_type: Text,
                 params,
                 init_checkpoint: Optional[Text] = None):
  """A factory function to create different types of models."""
  if model_type == "bert2bert":
    return create_bert2bert_model(params, init_checkpoint=init_checkpoint)
  elif model_type == "nhnet":
    return create_nhnet_model(params, init_checkpoint=init_checkpoint)
  elif "transformer" in model_type:
    return create_transformer_model(params, init_checkpoint=init_checkpoint)
  else:
    raise KeyError("The model type is not defined: %s" % model_type)
