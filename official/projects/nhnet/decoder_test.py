# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for projects.nhnet.decoder."""

import numpy as np
import tensorflow as tf
from official.nlp.modeling import layers
from official.projects.nhnet import configs
from official.projects.nhnet import decoder
from official.projects.nhnet import utils


class DecoderTest(tf.test.TestCase):

  def setUp(self):
    super(DecoderTest, self).setUp()
    self._config = utils.get_test_params()

  def test_transformer_decoder(self):
    decoder_block = decoder.TransformerDecoder(
        num_hidden_layers=self._config.num_hidden_layers,
        hidden_size=self._config.hidden_size,
        num_attention_heads=self._config.num_attention_heads,
        intermediate_size=self._config.intermediate_size,
        intermediate_activation=self._config.hidden_act,
        hidden_dropout_prob=self._config.hidden_dropout_prob,
        attention_probs_dropout_prob=self._config.attention_probs_dropout_prob,
        initializer_range=self._config.initializer_range)
    decoder_block.build(None)
    self.assertEqual(len(decoder_block.layers), self._config.num_hidden_layers)

  def test_bert_decoder(self):
    seq_length = 10
    encoder_input_ids = tf.keras.layers.Input(
        shape=(seq_length,), name="encoder_input_ids", dtype=tf.int32)
    target_ids = tf.keras.layers.Input(
        shape=(seq_length,), name="target_ids", dtype=tf.int32)
    encoder_outputs = tf.keras.layers.Input(
        shape=(seq_length, self._config.hidden_size),
        name="all_encoder_outputs",
        dtype=tf.float32)
    embedding_lookup = layers.OnDeviceEmbedding(
        vocab_size=self._config.vocab_size,
        embedding_width=self._config.hidden_size,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=self._config.initializer_range),
        name="word_embeddings")
    cross_attention_bias = decoder.AttentionBias(bias_type="single_cross")(
        encoder_input_ids)
    self_attention_bias = decoder.AttentionBias(bias_type="decoder_self")(
        target_ids)
    inputs = dict(
        attention_bias=cross_attention_bias,
        self_attention_bias=self_attention_bias,
        target_ids=target_ids,
        all_encoder_outputs=encoder_outputs)
    decoder_layer = decoder.Decoder(self._config, embedding_lookup)
    outputs = decoder_layer(inputs)
    model_inputs = dict(
        encoder_input_ids=encoder_input_ids,
        target_ids=target_ids,
        all_encoder_outputs=encoder_outputs)
    model = tf.keras.Model(inputs=model_inputs, outputs=outputs, name="test")
    self.assertLen(decoder_layer.trainable_weights, 30)
    # Forward path.
    fake_inputs = {
        "encoder_input_ids": np.zeros((2, 10), dtype=np.int32),
        "target_ids": np.zeros((2, 10), dtype=np.int32),
        "all_encoder_outputs": np.zeros((2, 10, 16), dtype=np.float32),
    }
    output_tensor = model(fake_inputs)
    self.assertEqual(output_tensor.shape, (2, 10, 16))

  def test_multi_doc_decoder(self):
    self._config = utils.get_test_params(cls=configs.NHNetConfig)
    seq_length = 10
    num_docs = 5
    encoder_input_ids = tf.keras.layers.Input(
        shape=(num_docs, seq_length), name="encoder_input_ids", dtype=tf.int32)
    target_ids = tf.keras.layers.Input(
        shape=(seq_length,), name="target_ids", dtype=tf.int32)
    encoder_outputs = tf.keras.layers.Input(
        shape=(num_docs, seq_length, self._config.hidden_size),
        name="all_encoder_outputs",
        dtype=tf.float32)
    embedding_lookup = layers.OnDeviceEmbedding(
        vocab_size=self._config.vocab_size,
        embedding_width=self._config.hidden_size,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=self._config.initializer_range),
        name="word_embeddings")
    doc_attention_probs = tf.keras.layers.Input(
        shape=(self._config.num_decoder_attn_heads, seq_length, num_docs),
        name="doc_attention_probs",
        dtype=tf.float32)
    cross_attention_bias = decoder.AttentionBias(bias_type="multi_cross")(
        encoder_input_ids)
    self_attention_bias = decoder.AttentionBias(bias_type="decoder_self")(
        target_ids)

    inputs = dict(
        attention_bias=cross_attention_bias,
        self_attention_bias=self_attention_bias,
        target_ids=target_ids,
        all_encoder_outputs=encoder_outputs,
        doc_attention_probs=doc_attention_probs)

    decoder_layer = decoder.Decoder(self._config, embedding_lookup)
    outputs = decoder_layer(inputs)
    model_inputs = dict(
        encoder_input_ids=encoder_input_ids,
        target_ids=target_ids,
        all_encoder_outputs=encoder_outputs,
        doc_attention_probs=doc_attention_probs)
    model = tf.keras.Model(inputs=model_inputs, outputs=outputs, name="test")
    self.assertLen(decoder_layer.trainable_weights, 30)
    # Forward path.
    fake_inputs = {
        "encoder_input_ids":
            np.zeros((2, num_docs, seq_length), dtype=np.int32),
        "target_ids":
            np.zeros((2, seq_length), dtype=np.int32),
        "all_encoder_outputs":
            np.zeros((2, num_docs, seq_length, 16), dtype=np.float32),
        "doc_attention_probs":
            np.zeros(
                (2, self._config.num_decoder_attn_heads, seq_length, num_docs),
                dtype=np.float32)
    }
    output_tensor = model(fake_inputs)
    self.assertEqual(output_tensor.shape, (2, seq_length, 16))


if __name__ == "__main__":
  tf.test.main()
