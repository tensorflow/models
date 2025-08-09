# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for context_rcnn_lib."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from absl.testing import parameterized
import tensorflow.compat.v1 as tf

from object_detection.meta_architectures import detr_lib
from object_detection.utils import test_case
from object_detection.utils import tf_version
from official.nlp.modeling.layers import position_embedding

_NEGATIVE_PADDING_VALUE = -100000


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class DETRLibTest(parameterized.TestCase, test_case.TestCase,
                  tf.test.TestCase):
  """Tests for the functions in detr_lib."""

  @parameterized.named_parameters(
      ("small_error", 7, 15),
      ("small", 25, 25),
      ("medium_error", 46, 49),
      ("large", 600, 2500)
  )
  def test_2d_encoding(self, hidden_size, length):
    if hidden_size % 4 != 0:
      self.assertRaises(ValueError, detr_lib.TwoDimensionalPositionEmbedding,
                        hidden_size)
    else:
      encoding = detr_lib.TwoDimensionalPositionEmbedding(hidden_size)
      self.assertAllEqual(encoding(tf.ones([4, length, hidden_size])).shape,
                          [length, hidden_size])

  def test_wrapper(self):
    layer = lambda x, training: x
    wrapper = detr_lib.PrePostProcessingWrapper(layer, 0.0)
    result = wrapper(tf.ones([2]), tf.ones([2]))
    self.assertAllEqual(result, tf.zeros([2]))

  @parameterized.named_parameters(
      ("max_heads", 8, 8, 3, 7),
      ("two_heads", 2, 8, 3, 7),
      ("large", 8, 128, 300, 70)
  )
  def test_multihead_attention(self, num_heads, hidden_size,
                               query_length, kv_length):
    attention_layer = detr_lib.Attention(hidden_size, num_heads, 0.0)
    batch_size = 5
    query = tf.ones([batch_size, query_length, hidden_size])
    key = tf.ones([batch_size, kv_length, hidden_size])
    value = tf.ones([batch_size, kv_length, hidden_size])
    result = attention_layer(query, key, value, training=False)
    self.assertAllEqual(result.shape, [batch_size, query_length, hidden_size])

  @parameterized.named_parameters(
      ("max_heads", 8, 8, 3),
      ("two_heads", 2, 8, 3),
      ("large", 8, 128, 300)
  )
  def test_self_attention(self, num_heads, hidden_size,
                          qkv_length):
    attention_layer = detr_lib.SelfAttention(hidden_size, num_heads, 0.0)
    batch_size = 5
    query = tf.ones([batch_size, qkv_length, hidden_size])
    value = tf.ones([batch_size, qkv_length, hidden_size])
    result = attention_layer(query, value, training=False)
    self.assertAllEqual(result.shape, [batch_size, qkv_length, hidden_size])

  @parameterized.named_parameters(
      ("ffn_small", 20, 10),
      ("ffn_large", 2000, 10000)
  )
  def test_ffn(self, hidden_size, filter_size):
    ffn = detr_lib.FeedForwardNetwork(hidden_size, filter_size, 0.0)
    batch_size = 5
    input_data = tf.ones([batch_size, hidden_size])
    result = ffn(input_data, training=False)
    self.assertAllEqual(result.shape, [batch_size, hidden_size])

  @parameterized.named_parameters(
      ("encoder_small", 20),
      ("encoder_large", 2000)
  )
  def test_encoder(self, hidden_size):
    batch_size = 6
    input_data = tf.ones([batch_size, 25, hidden_size], dtype=tf.float32)
    encoder_stack = detr_lib.EncoderStack(hidden_size=hidden_size,
                                          num_heads=5)
    result = encoder_stack(input_data, False,
                           encoding=detr_lib.TwoDimensionalPositionEmbedding(
                               hidden_size)(input_data))
    self.assertAllEqual(result.shape, [batch_size, 25, hidden_size])

  @parameterized.named_parameters(
      ("decoder_small", 20, 45),
      ("decoder_large", 2000, 203)
  )
  def test_decoder(self, hidden_size, num_queries):
    batch_size = 6
    input_queries = tf.ones([batch_size, num_queries, hidden_size],
                            dtype=tf.float32)
    encoder_outputs = tf.ones([batch_size, 25, hidden_size])
    decoder_stack = detr_lib.DecoderStack(hidden_size=hidden_size,
                                          num_heads=2)
    result = decoder_stack(input_queries, encoder_outputs, False,
                           encoding=detr_lib.TwoDimensionalPositionEmbedding(
                               hidden_size)(encoder_outputs))
    self.assertAllEqual(result.shape, [batch_size, num_queries, hidden_size])

  @parameterized.named_parameters(
      ("transformer_small", 20, 45),
      ("transformer_large", 2000, 203)
  )
  def test_transformer(self, hidden_size, num_queries):
    transformer = detr_lib.Transformer(hidden_size=hidden_size,
                                       num_heads=4)
    batch_size = 5
    input_data = tf.ones([batch_size, 25, hidden_size])
    input_queries = tf.ones([batch_size, num_queries, hidden_size],
                            dtype=tf.float32)
    result = transformer([input_data, input_queries], training=False)
    self.assertAllEqual(result.shape, [batch_size, num_queries, hidden_size])

if __name__ == '__main__':
  tf.test.main()
