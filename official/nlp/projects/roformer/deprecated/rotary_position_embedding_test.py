
"""Tests for Keras-based rotary positional embedding layer."""

import numpy as np
import tensorflow as tf

from official.nlp.projects.roformer import rotary_position_embedding


class RotaryPositionEmbeddingTest(tf.test.TestCase):

  def test_rotary_tensor_input(self):
    seq_len = 512
    batch_size = 2
    num_heads = 4
    head_size = 64
    test_layer = rotary_position_embedding.RotaryPositionEmbedding(
        hidden_size=head_size)

    q = np.random.randn(batch_size, seq_len, num_heads, head_size)
    k = np.random.randn(batch_size, seq_len, num_heads, head_size)
    v = np.random.randn(batch_size, seq_len, num_heads, head_size)
    new_q, new_k, new_v = test_layer([q,k,v])

    self.assertEqual(new_q.shape, (batch_size, seq_len, num_heads, head_size))
    self.assertEqual(new_k.shape, (batch_size, seq_len, num_heads, head_size))
    self.assertEqual(new_v.shape, (batch_size, seq_len, num_heads, head_size))