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
"""Implementation of Transformer decoder model."""

import math

from absl import logging
from tensor2tensor.utils import beam_search
import tensorflow as tf

from layers import base_layers # import seq_flow_lite module
from layers import dense_layers # import seq_flow_lite module
from layers import embedding_layers # import seq_flow_lite module
from layers import normalization_layers # import seq_flow_lite module
from layers import quantization_layers # import seq_flow_lite module
from layers import transformer_layers # import seq_flow_lite module


class TransformerUniformAttnDecoder(base_layers.BaseLayer):
  """Transformer Uniform Attention Decoder."""

  def __init__(self,
               model_dimension,
               max_time_step,
               num_heads,
               intermediate_size,
               activation_dropout_rate=0.0,
               attention_dropout_rate=0.0,
               beam_size=1,
               cached_kv=False,
               **kwargs):
    self.model_dimension = model_dimension
    self.decoder_uniform_attn = transformer_layers.DecoderUniformAttention(
        model_dimension,
        max_time_step,
        attention_dropout_rate=attention_dropout_rate,
        beam_size=beam_size,
        **kwargs)
    self.multihead_cross_attn = transformer_layers.DecoderMultiheadAttention(
        model_dimension,
        num_heads,
        cached_kv=cached_kv,
        attention_dropout_rate=attention_dropout_rate,
        **kwargs)
    self.prx = dense_layers.BaseQDense(
        model_dimension, activation=None, normalize=False, bias=False, **kwargs)
    self.upprx = dense_layers.BaseQDense(
        intermediate_size, normalize=False, **kwargs)
    self.downprx = dense_layers.BaseQDense(
        model_dimension, activation=None, normalize=False, **kwargs)
    self.activation_dropout_rate = activation_dropout_rate
    self.ln1 = normalization_layers.LayerNormalization(**kwargs)
    self.ln2 = normalization_layers.LayerNormalization(**kwargs)
    self.q0 = quantization_layers.ActivationQuantization(**kwargs)
    self.q1 = quantization_layers.ActivationQuantization(**kwargs)
    self.q2 = quantization_layers.ActivationQuantization(**kwargs)
    super(TransformerUniformAttnDecoder, self).__init__(**kwargs)

  def call(self,
           dec_inputs,
           dec_mask,
           dec_inverse_normalizer,
           enc_output,
           enc_mask,
           enc_inverse_normalizer,
           cross_attn_mask=None,
           step=None,
           selected_beams=None,
           cache=None):
    batch_size = self.get_batch_dimension(dec_inputs)
    self._assert_rank_and_type(dec_inputs, 3)
    self._assert_rank_and_type(dec_mask, 3)
    assert dec_inputs.get_shape().as_list()[-1] == self.model_dimension

    self_attn_output = self.decoder_uniform_attn(
        dec_inputs,
        dec_mask,
        dec_inverse_normalizer,
        step=step,
        beam_indices=selected_beams,
        cache=cache)
    cross_attn_output = self.multihead_cross_attn(dec_inputs, dec_mask,
                                                  dec_inverse_normalizer,
                                                  enc_output, enc_mask,
                                                  enc_inverse_normalizer,
                                                  cross_attn_mask)
    layer_out = self.q0(cross_attn_output + self_attn_output)
    layer_out = tf.reshape(layer_out, [-1, self.model_dimension])
    layer_out = self.prx(layer_out)
    if self.parameters.mode == base_layers.TRAIN:
      layer_out = tf.nn.dropout(layer_out, rate=self.activation_dropout_rate)

    dec_inputs = tf.reshape(dec_inputs, [-1, self.model_dimension])
    dec_inputs_updated = self.q1(self.ln1(dec_inputs + layer_out))

    # Feed forward network.
    layer_out = self.upprx(dec_inputs_updated)
    layer_out = self.downprx(layer_out)
    if self.parameters.mode == base_layers.TRAIN:
      layer_out = tf.nn.dropout(layer_out, rate=self.activation_dropout_rate)

    outputs = self.q2(self.ln2(dec_inputs_updated + layer_out))
    return tf.reshape(outputs, [batch_size, -1, self.model_dimension])


class TransformerUniformAttnDecoderStack(base_layers.BaseLayer):
  """TransformerUniformAttnDecoderStack Decoder."""

  def __init__(self,
               num_layers,
               max_time_step,
               vocabulary_size,
               embedding_size,
               model_dimension,
               num_heads,
               intermediate_size,
               beam_size=1,
               activation_dropout_rate=0.1,
               attention_dropout_rate=0.0,
               cached_kv=False,
               **kwargs):
    super(TransformerUniformAttnDecoderStack, self).__init__(**kwargs)
    self.max_time_step = max_time_step
    self.vocabulary_size = vocabulary_size
    self.embedding_size = embedding_size
    self.activation_dropout_rate = activation_dropout_rate
    self.layers = []
    for _ in range(num_layers):
      self.layers.append(
          TransformerUniformAttnDecoder(
              model_dimension=model_dimension,
              max_time_step=max_time_step,
              num_heads=num_heads,
              intermediate_size=intermediate_size,
              beam_size=beam_size,
              cached_kv=cached_kv,
              activation_dropout_rate=activation_dropout_rate,
              attention_dropout_rate=attention_dropout_rate,
              **kwargs))

  def call(self,
           dec_inputs,
           dec_mask,
           enc_output,
           enc_mask,
           step=None,
           selected_beams=None,
           cache=None):
    self._assert_rank_and_type(dec_mask, 2)
    self._assert_rank_and_type(enc_mask, 2)
    dec_mask_rank3 = tf.expand_dims(dec_mask, axis=2)
    dec_inverse_normalizer = tf.math.reciprocal(tf.reduce_sum(dec_mask_rank3))
    enc_mask_rank3 = tf.expand_dims(enc_mask, 1)
    enc_inverse_normalizer = tf.math.reciprocal(tf.reduce_sum(enc_mask_rank3))
    cross_attn_mask = enc_mask_rank3
    layer_in = dec_inputs
    if self.parameters.mode == base_layers.TRAIN:
      layer_in = tf.nn.dropout(layer_in, rate=self.activation_dropout_rate)

    enc_output_feature_dim = enc_output.get_shape().as_list()[2]
    enc_output = tf.reshape(enc_output, [-1, enc_output_feature_dim])
    for i, layer in enumerate(self.layers):
      layer_cache = cache["layer_%d" % i] if cache is not None else None
      layer_in = layer(
          layer_in,
          dec_mask_rank3,
          dec_inverse_normalizer,
          enc_output,
          enc_mask,
          enc_inverse_normalizer,
          cross_attn_mask,
          step=step,
          selected_beams=selected_beams,
          cache=layer_cache)
    return layer_in


class Model(tf.keras.layers.Layer):
  """Quantized transformer decoder."""

  def __init__(self, config, mode):

    super(Model, self).__init__()

    def _get_params(varname, default_value=None):
      value = config[varname] if varname in config else default_value
      default = "" if varname in config else " (default)"
      logging.info("%s = %s%s", varname, value, default)
      setattr(self, varname, value)

    _get_params("intermediate_size")
    _get_params("max_dec_time_step")
    _get_params("max_enc_time_step")
    _get_params("embedding_size")
    _get_params("vocabulary_size")
    _get_params("num_layers")
    _get_params("labels")
    _get_params("regularizer_scale")
    _get_params("num_heads")
    _get_params("model_dimension")
    _get_params("beam_size", 1)
    _get_params("quantize", True)
    _get_params("cached_kv", False)
    _get_params("attention_dropout_rate", 0.0)
    _get_params("activation_dropout_rate", 0.0)
    # If set, a separate dense layer is used to generate the logits instead of
    # re-using the input embedding table.
    _get_params("use_output_layer", False)
    self.parameters = base_layers.Parameters(mode, self.quantize,
                                             self.regularizer_scale)
    # Activation/Normalization enabled on input bottleneck as there is no
    # temporal information.
    self.input_bottleneck = dense_layers.BaseQDenseVarLen(
        self.model_dimension, rank=3, parameters=self.parameters)
    self.output_bottleneck = dense_layers.BaseQDense(
        self.embedding_size,
        normalize=False,
        activation=None,
        bias=False,
        parameters=self.parameters)

    self.embedding = embedding_layers.EmbeddingFullyConnected(
        shape=[self.vocabulary_size, self.embedding_size],
        initializer=tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3)),
        parameters=self.parameters)
    if self.use_output_layer:
      self.output_layer = dense_layers.BaseQDense(
          self.vocabulary_size,
          activation=None,
          normalize=False,
          bias=False,
          parameters=self.parameters)
    self.positional_embedding = embedding_layers.EmbeddingLayer(
        shape=[self.max_dec_time_step, self.model_dimension],
        initializer=tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3)),
        parameters=self.parameters)
    self.ln = normalization_layers.LayerNormalization(
        parameters=self.parameters)
    self.qact = quantization_layers.ActivationQuantization(
        parameters=self.parameters)
    # Scales the weights for computing logits.
    self.logits_fc_weights_scale_factor = None
    self.logits_fc_bias = self.add_weight(
        "logits_fc_bias",
        shape=[self.vocabulary_size],
        initializer=tf.constant_initializer(0),
        dtype="float32")
    # Optional bias which can be used to mask logits output.
    self.output_bias = None
    self.transformer_uniform_attn_decoder = TransformerUniformAttnDecoderStack(
        parameters=self.parameters,
        num_layers=self.num_layers,
        intermediate_size=self.intermediate_size,
        embedding_size=self.embedding_size,
        max_time_step=self.max_dec_time_step,
        num_heads=self.num_heads,
        model_dimension=self.model_dimension,
        vocabulary_size=self.vocabulary_size,
        beam_size=self.beam_size,
        cached_kv=self.cached_kv,
        attention_dropout_rate=self.attention_dropout_rate,
        activation_dropout_rate=self.activation_dropout_rate)
    # Beam search output.
    self.finished_seq = None
    self.finished_scores = None

  def call(self,
           decode_ids,
           decode_ids_mask,
           enc_output,
           enc_mask,
           start_ids=None,
           eos_id=None,
           pad_id=None,
           input_id=None,
           time_step=None,
           selected_beams=None):

    if self.parameters.mode == base_layers.TRAIN:
      inputs = self.training_inputs(decode_ids, decode_ids_mask)
      layer_out = self.transformer_uniform_attn_decoder(inputs, decode_ids_mask,
                                                        enc_output, enc_mask)
      logits, predicted_ids = self.model_outputs(layer_out)
    elif self.parameters.mode in [base_layers.EVAL, base_layers.PREDICT]:
      logits, predicted_ids = self.decode_beam_search(start_ids, eos_id, pad_id,
                                                      enc_output, enc_mask)
    elif self.parameters.mode == base_layers.TFLITE:
      input_values = self.embedding(input_id)
      # time_step starts from 1.
      pos_values = self.positional_embedding(time_step - 1)
      pos_values = tf.reshape(pos_values, [-1, 1, self.embedding_size])
      input_mask = tf.ones(tf.shape(input_values)[:-1], dtype=tf.float32)
      inputs = self.qact(self.ln(input_values + pos_values))
      layer_out = self.transformer_uniform_attn_decoder(
          inputs,
          input_mask,
          enc_output,
          enc_mask,
          step=time_step,
          selected_beams=selected_beams)
      logits, predicted_ids = self.model_outputs(layer_out)
    else:
      assert "Invalid mode."
    return logits, predicted_ids

  def training_inputs(self, input_ids, input_mask):
    input_values = self.embedding(input_ids)
    if self.embedding_size != self.model_dimension:
      input_values = self.input_bottleneck(input_values, input_mask)
    pos_indices = tf.cumsum(input_mask, axis=1, exclusive=True)
    pos_indices = tf.cast(pos_indices, dtype=tf.int32)
    pos_values = self.positional_embedding(pos_indices)
    inputs = self.qact(self.ln(input_values + pos_values))
    return inputs

  def model_outputs(self, layer_in):
    bsz = layer_in.get_shape().as_list()[0] or tf.shape(layer_in)[0]
    layer_out = tf.reshape(layer_in, [-1, self.model_dimension])

    if self.use_output_layer:
      logits = self.output_layer(layer_out)
    else:
      if self.model_dimension != self.embedding_size:
        layer_out = self.output_bottleneck(layer_out)
      logits = self.embedding.fully_connected(
          layer_out,
          bias=self.logits_fc_bias,
          weights_scale_factor=self.logits_fc_weights_scale_factor)

    logits = tf.reshape(logits, [bsz, -1, self.vocabulary_size])
    # Optional bias to mask out logits before applying argmax.
    if self.output_bias is not None:
      logits += self.output_bias
    predicted_ids = tf.argmax(logits, axis=2, output_type=tf.int64)
    return logits, predicted_ids

  def decode_beam_search(self,
                         start_ids,
                         eos_id,
                         pad_id,
                         enc_output,
                         enc_mask,
                         scope="model"):
    batch_size = tf.shape(start_ids)[0]
    cache = {  # pylint: disable=g-complex-comprehension
        "layer_%d" % layer: {
            "uniform_avg": tf.zeros([batch_size, 1, self.model_dimension]),
        } for layer in range(self.num_layers)
    }
    cache["logits"] = tf.zeros([batch_size, 0, self.vocabulary_size])
    pos_indices = tf.range(self.max_dec_time_step, dtype=tf.int32)
    pos_indices = tf.reshape(pos_indices, [1, -1])
    pos_values = self.positional_embedding(pos_indices)

    def beam_search_tile(output, tile_pattern, final_shape):
      x = tf.tile(output, tile_pattern)
      x = tf.reshape(x, final_shape)
      return x

    enc_output_feature_dim = enc_output.get_shape().as_list()[2]
    enc_output = beam_search_tile(
        enc_output, [1, self.beam_size, 1],
        [batch_size * self.beam_size, -1, enc_output_feature_dim])
    enc_mask = beam_search_tile(enc_mask, [1, self.beam_size],
                                [batch_size * self.beam_size, -1])

    def symbols_to_logits_fn(ids, step, cache):
      """Looks up ids to logits."""
      logging.info("Running symbols to logits. ids=%s, step=%s, cache=%s", ids,
                   step, cache)
      curr_id = ids[:, -1:]
      with tf.name_scope(scope):
        curr_embed = self.embedding(curr_id)
        input_mask = tf.ones(tf.shape(curr_embed)[:-1], dtype=tf.float32)
        if self.embedding_size != self.model_dimension:
          curr_embed = self.input_bottleneck(curr_embed, input_mask)
        inputs = self.qact(
            self.ln(curr_embed + pos_values[:, step:step + 1, :]))
        layer_out = self.transformer_uniform_attn_decoder(
            inputs,
            input_mask,
            enc_output,
            enc_mask,
            step=step + 1,
            cache=cache)
        next_logits, _ = self.model_outputs(layer_out)
        cache["logits"] = tf.concat([cache["logits"], next_logits], axis=1)
        return next_logits, cache

    self.finished_seq, self.finished_scores, states = beam_search.beam_search(
        symbols_to_logits_fn,
        initial_ids=start_ids,
        beam_size=self.beam_size,
        decode_length=self.max_dec_time_step,
        vocab_size=self.vocabulary_size,
        alpha=0.6,
        eos_id=eos_id,
        states=cache)
    beam_ids = self.finished_seq[:, 0, 1:]
    beam_ids = tf.pad(
        beam_ids, [[0, 0], [0, self.max_dec_time_step - tf.shape(beam_ids)[1]]],
        constant_values=pad_id)
    logits = states["logits"][:, 0, :, :]
    logits = tf.pad(
        logits,
        [[0, 0], [0, self.max_dec_time_step - tf.shape(logits)[1]], [0, 0]],
        constant_values=self.parameters.invalid_logit)
    return logits, beam_ids


class ModelEvalWithGTLogitsAndPredictions(Model):
  """Model with EVAL mode logits and predictions based on ground truth inputs at each step."""

  def call(self,
           decode_ids,
           decode_ids_mask,
           enc_output,
           enc_mask,
           start_ids=None,
           eos_id=None,
           pad_id=None,
           input_id=None,
           time_step=None,
           selected_beams=None):
    if self.parameters.mode in [base_layers.TRAIN, base_layers.EVAL]:
      inputs = self.training_inputs(decode_ids, decode_ids_mask)
      layer_out = self.transformer_uniform_attn_decoder(inputs, decode_ids_mask,
                                                        enc_output, enc_mask)
      logits, predicted_ids = self.model_outputs(layer_out)
    elif self.parameters.mode == base_layers.PREDICT:
      logits, predicted_ids = self.decode_beam_search(
          start_ids,
          eos_id,
          pad_id,
          enc_output,
          enc_mask,
          scope="model_eval_with_gt_logits_and_predictions")
    elif self.parameters.mode == base_layers.TFLITE:
      input_values = self.embedding(input_id)
      # time_step starts from 1.
      pos_values = self.positional_embedding(time_step - 1)
      pos_values = tf.reshape(pos_values, [-1, 1, self.embedding_size])
      input_mask = tf.ones(tf.shape(input_values)[:-1], dtype=tf.float32)
      inputs = self.qact(self.ln(input_values + pos_values))
      layer_out = self.transformer_uniform_attn_decoder(
          inputs,
          input_mask,
          enc_output,
          enc_mask,
          step=time_step,
          selected_beams=selected_beams)
      logits, predicted_ids = self.model_outputs(layer_out)
    else:
      assert "Invalid mode."
    return logits, predicted_ids


class ModelEvalWithGTLogits(Model):
  """Model with EVAL mode logits computed based on ground truth input at each step."""

  def call(self,
           decode_ids,
           decode_ids_mask,
           enc_output,
           enc_mask,
           start_ids=None,
           eos_id=None,
           pad_id=None,
           input_id=None,
           time_step=None,
           selected_beams=None):
    logits = None
    if self.parameters.mode in [base_layers.TRAIN, base_layers.EVAL]:
      inputs = self.training_inputs(decode_ids, decode_ids_mask)
      layer_out = self.transformer_uniform_attn_decoder(inputs, decode_ids_mask,
                                                        enc_output, enc_mask)
      logits, predicted_ids = self.model_outputs(layer_out)

    if self.parameters.mode in [base_layers.EVAL, base_layers.PREDICT]:
      # EVAL mode predictions are based on beam search path.
      _, predicted_ids = self.decode_beam_search(
          start_ids,
          eos_id,
          pad_id,
          enc_output,
          enc_mask,
          scope="model_eval_with_gt_logits")
    if self.parameters.mode == base_layers.TFLITE:
      input_values = self.embedding(input_id)
      # time_step starts from 1.
      pos_values = self.positional_embedding(time_step - 1)
      pos_values = tf.reshape(pos_values, [-1, 1, self.embedding_size])
      input_mask = tf.ones(tf.shape(input_values)[:-1], dtype=tf.float32)
      inputs = self.qact(self.ln(input_values + pos_values))
      layer_out = self.transformer_uniform_attn_decoder(
          inputs,
          input_mask,
          enc_output,
          enc_mask,
          step=time_step,
          selected_beams=selected_beams)
      logits, predicted_ids = self.model_outputs(layer_out)

    return logits, predicted_ids
