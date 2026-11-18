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

if __name__ == '__main__':
  tf.test.main()
