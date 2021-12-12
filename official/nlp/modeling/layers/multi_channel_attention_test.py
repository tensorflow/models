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

"""Tests for projects.nhnet.multi_channel_attention."""

import numpy as np
import tensorflow as tf

from official.nlp.modeling.layers import multi_channel_attention


class MultiChannelAttentionTest(tf.test.TestCase):

  def test_doc_attention(self):
    num_heads = 2
    doc_attention = multi_channel_attention.VotingAttention(
        num_heads, head_size=8)
    num_docs = 3
    inputs = np.zeros((2, num_docs, 10, 16), dtype=np.float32)
    doc_mask = np.zeros((2, num_docs), dtype=np.float32)
    outputs = doc_attention(inputs, doc_mask)
    self.assertEqual(outputs.shape, (2, num_docs))

  def test_multi_channel_attention(self):
    num_heads = 2
    num_docs = 5
    attention_layer = multi_channel_attention.MultiChannelAttention(
        num_heads, key_dim=2)

    from_data = 10 * np.random.random_sample((3, 4, 8))
    to_data = 10 * np.random.random_sample((3, num_docs, 2, 8))
    mask_data = np.random.randint(2, size=(3, num_docs, 4, 2))
    doc_probs = np.random.randint(
        2, size=(3, num_heads, 4, num_docs)).astype(float)
    outputs = attention_layer(
        query=from_data,
        value=to_data,
        context_attention_weights=doc_probs,
        attention_mask=mask_data)
    self.assertEqual(outputs.shape, (3, 4, 8))


if __name__ == "__main__":
  tf.test.main()
