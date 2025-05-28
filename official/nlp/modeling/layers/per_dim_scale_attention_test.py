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

"""Tests for PerDimScaleAttention."""

import tensorflow as tf, tf_keras

from official.nlp.modeling.layers import per_dim_scale_attention as attention


class PerDimScaleAttentionTest(tf.test.TestCase):

  def test_attention(self):
    num_heads = 12
    key_dim = 64
    seq_length = 1024
    batch_size = 2
    test_layer = attention.PerDimScaleAttention(
        num_heads=num_heads, key_dim=key_dim)
    query = tf.random.normal(
        shape=(batch_size, seq_length, key_dim * num_heads))
    value = query
    output = test_layer(query=query, value=value)
    self.assertEqual(output.shape,
                     [batch_size, seq_length, key_dim * num_heads])

  def test_config(self):
    num_heads = 12
    key_dim = 64
    test_layer = attention.PerDimScaleAttention(
        num_heads=num_heads, key_dim=key_dim)
    print(test_layer.get_config())
    new_layer = attention.PerDimScaleAttention.from_config(
        test_layer.get_config())

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(test_layer.get_config(), new_layer.get_config())


if __name__ == '__main__':
  tf.test.main()
