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

"""Tests for pack_optimization."""

import tensorflow as tf, tf_keras
from official.nlp.modeling.layers import pack_optimization


class PackOptimizationTest(tf.test.TestCase):

  def test_bert_embedding_packing(self):
    batch_size, seq_len, embed_dim = 2, 4, 8
    pack_sequences = 2
    token_and_position_embed = tf.ones((batch_size, seq_len, embed_dim),
                                       dtype=tf.float32)
    input_mask = tf.ones((batch_size, seq_len), dtype=tf.int32)

    layer = pack_optimization.PackBertEmbeddings(pack_sequences=pack_sequences)
    outputs = layer(token_and_position_embed, input_mask)
    self.assertEqual(outputs["packed_embeddings"].shape, (1, 8, embed_dim))
    self.assertEqual(outputs["combined_attention_mask"].shape, (1, 8, 8))

  def test_strided_transformer_encoder_block(self):
    inputs = tf.zeros((2, 4, 8), dtype=tf.float32)
    attention_mask = tf.ones((2, 4, 4), dtype=tf.float32)
    transformer = pack_optimization.StridedTransformerEncoderBlock(
        num_attention_heads=2, inner_dim=4, inner_activation="relu")
    outputs = transformer([inputs, attention_mask],
                          stride=tf.constant(2, dtype=tf.int32))
    self.assertEqual(outputs.shape, (2, 2, 8))

  def test_strided_rezero_transformer(self):
    inputs = tf.zeros((2, 4, 8), dtype=tf.float32)
    attention_mask = tf.ones((2, 4, 4), dtype=tf.float32)
    transformer = pack_optimization.StridedReZeroTransformer(
        num_attention_heads=2, inner_dim=4, inner_activation="relu")
    outputs = transformer([inputs, attention_mask],
                          stride=tf.constant(2, dtype=tf.int32))
    self.assertEqual(outputs.shape, (2, 2, 8))

  def test_strided_scaffold(self):
    inputs = tf.zeros((2, 4, 8), dtype=tf.float32)
    attention_mask = tf.ones((2, 4, 4), dtype=tf.float32)
    test_layer = pack_optimization.StridedTransformerScaffold(
        num_attention_heads=2,
        inner_dim=128,
        inner_activation="relu")
    outputs = test_layer([inputs, attention_mask],
                         stride=tf.constant(2, dtype=tf.int32))
    self.assertEqual(outputs.shape, (2, 2, 8))


if __name__ == "__main__":
  tf.test.main()
