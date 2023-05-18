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

"""Tests for Keras-based transformer block layer."""

import tensorflow as tf

from official.nlp.modeling.layers import transformer


def _create_cache(batch_size, init_decode_length, num_heads, head_size):
  return {
      'key':
          tf.zeros([batch_size, init_decode_length, num_heads, head_size],
                   dtype=tf.float32),
      'value':
          tf.zeros([batch_size, init_decode_length, num_heads, head_size],
                   dtype=tf.float32)
  }


class TransformerDecoderBlockTest(tf.test.TestCase):

  def test_decoder_block_with_cache(self):
    num_attention_heads = 2
    hidden_size = 16
    decoder_block = transformer.TransformerDecoderBlock(
        num_attention_heads=num_attention_heads,
        intermediate_size=32,
        intermediate_activation='relu',
        dropout_rate=0.1,
        attention_dropout_rate=0.1)
    # Forward path.
    dummy_tensor = tf.zeros([2, 4, 16], dtype=tf.float32)
    dummy_mask = tf.zeros([2, 4, 4], dtype=tf.float32)
    inputs = [dummy_tensor, dummy_tensor, dummy_mask, dummy_mask]
    cache = _create_cache(2, 0, num_attention_heads,
                          hidden_size // num_attention_heads)
    output, cache = decoder_block(inputs, cache)
    self.assertEqual(output.shape, (2, 4, hidden_size))
    self.assertEqual(cache['value'].shape, (2, 4, 2, 8))

  def test_use_bias_norm_first(self):
    num_attention_heads = 2
    hidden_size = 16
    decoder_block = transformer.TransformerDecoderBlock(
        num_attention_heads=num_attention_heads,
        intermediate_size=32,
        intermediate_activation='relu',
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-6,
        intermediate_dropout=0.1,
        attention_initializer=tf.keras.initializers.RandomUniform(
            minval=0., maxval=1.))
    # Forward path.
    dummy_tensor = tf.zeros([2, 4, 16], dtype=tf.float32)
    dummy_mask = tf.zeros([2, 4, 4], dtype=tf.float32)
    inputs = [dummy_tensor, dummy_tensor, dummy_mask, dummy_mask]
    output, _ = decoder_block(inputs)
    self.assertEqual(output.shape, (2, 4, hidden_size))

  def test_get_config(self):
    num_attention_heads = 2
    decoder_block = transformer.TransformerDecoderBlock(
        num_attention_heads=num_attention_heads,
        intermediate_size=32,
        intermediate_activation='relu',
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-6,
        intermediate_dropout=0.1,
        attention_initializer=tf.keras.initializers.RandomUniform(
            minval=0., maxval=1.))
    decoder_block_config = decoder_block.get_config()
    new_decoder_block = transformer.TransformerDecoderBlock.from_config(
        decoder_block_config)
    self.assertEqual(decoder_block_config, new_decoder_block.get_config())


if __name__ == '__main__':
  tf.test.main()
