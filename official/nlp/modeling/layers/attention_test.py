# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for the attention layer."""

import numpy as np
import tensorflow as tf

from official.nlp.modeling.layers import attention


def _create_cache(batch_size, init_decode_length, num_heads, head_size):
  return {
      "key":
          tf.zeros([batch_size, init_decode_length, num_heads, head_size],
                   dtype=tf.float32),
      "value":
          tf.zeros([batch_size, init_decode_length, num_heads, head_size],
                   dtype=tf.float32)
  }


class CachedAttentionTest(tf.test.TestCase):

  def test_masked_attention(self):
    """Test with a mask tensor."""
    num_heads, head_size = 2, 2
    # Create a 3-dimensional input (the first dimension is implicit).
    from_seq_length = 4
    batch_size = 3
    # GPU/CPU case.
    init_decode_length = 0
    # Directly tests the keras layer.
    cache = _create_cache(batch_size, init_decode_length, num_heads, head_size)
    layer = attention.CachedAttention(num_heads=num_heads, key_dim=head_size)

    # Generate data for the input (non-mask) tensors.
    from_data = tf.zeros((batch_size, from_seq_length, 8), dtype=np.float32)
    # Invoke the data with a random set of mask data. This should mask at least
    # one element.
    mask_data = np.random.randint(
        2, size=(batch_size, from_seq_length, from_seq_length))
    masked_output_data, cache = layer(
        query=from_data, value=from_data, attention_mask=mask_data, cache=cache)
    self.assertEqual(masked_output_data.shape, (3, 4, 8))
    self.assertEqual(cache["value"].shape, (3, 4, 2, 2))

    # Tests inputs without cache.
    masked_output_data, cache = layer(
        query=from_data, value=from_data, attention_mask=mask_data)
    self.assertEqual(masked_output_data.shape, (3, 4, 8))
    self.assertIsNone(cache)

  def test_padded_decode(self):
    """Test with a mask tensor."""
    num_heads, head_size = 2, 2
    from_seq_length = 4
    # TPU decoding should pre-allocate the entire sequence.
    batch_size = 3
    init_decode_length = from_seq_length

    # Directly tests the keras layer.
    cache = _create_cache(batch_size, init_decode_length, num_heads, head_size)
    layer = attention.CachedAttention(num_heads=num_heads, key_dim=head_size)

    # Generate data for the input (non-mask) tensors.
    from_data = tf.zeros((batch_size, from_seq_length, 8), dtype=np.float32)
    decode_loop_step = 2
    mask_data = np.random.randint(
        2, size=(batch_size, from_seq_length, from_seq_length), dtype=np.int32)
    # Testing the invocation directly as Keras cannot consume inputs correctly.
    masked_output_data, cache = layer(
        query=from_data,
        value=from_data,
        attention_mask=mask_data,
        cache=cache,
        decode_loop_step=decode_loop_step)
    self.assertEqual(masked_output_data.shape, (3, 4, 8))
    self.assertEqual(cache["value"].shape, (3, 4, 2, 2))


if __name__ == "__main__":
  tf.test.main()
