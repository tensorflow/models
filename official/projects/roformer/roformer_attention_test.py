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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import combinations
from official.projects.roformer import roformer_attention


def _create_mock_attention_data(num_heads,
                                key_dim,
                                value_dim,
                                q_seq_length,
                                kv_seq_length,
                                batch_size,
                                include_mask=False):
  """Creates mock testing data.

  Args:
    num_heads: `int`, Number of attention heads.
    key_dim: `int`, Size of query head.
    value_dim: `int`, Size of key, value dim.
    q_seq_length: query sequence length.
    kv_seq_length: key/value sequence length.
    batch_size: `int`, the batch size.
    include_mask: optional `bool`, whether or not to include mask data.

  Returns:
    A dictionary with `str` as keys and `Tensor` as values.
  """
  query_shape = (batch_size, q_seq_length, key_dim)
  value_shape = (batch_size, kv_seq_length, value_dim)

  data = dict(
      query=tf.random.normal(shape=query_shape),
      value=tf.random.normal(shape=value_shape),
      key=tf.random.normal(shape=value_shape))

  total_seq_length = kv_seq_length

  if include_mask:
    mask_shape = (batch_size, num_heads, q_seq_length, total_seq_length)
    mask_data = np.random.randint(2, size=mask_shape).astype("float32")
    mask_data = dict(attention_mask=mask_data)
    data.update(mask_data)

  return data


class RoformerAttentionTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(RoformerAttentionTest, self).setUp()
    np.random.seed(0)
    tf.random.set_seed(0)

  @combinations.generate(
      combinations.combine(length=[8, 50], key_dim=[64, 128]))
  def test_trig_vector(self, length, key_dim):
    sin_emb, cos_emb = roformer_attention._build_trig_vector(length, key_dim)
    length = tf.shape(sin_emb)[1]
    key_dim = tf.shape(sin_emb)[3]
    for m in range(0, length):
      half_d = key_dim // 2
      std_emb = tf.range(half_d, dtype=tf.float32)
      std_emb = tf.pow(10000.0, -std_emb / float(half_d))
      std_emb = m * std_emb
      std_sin_emb = tf.sin(std_emb)
      std_cos_emb = tf.cos(std_emb)
      tf.assert_equal(sin_emb[:, m, :, 0::2], std_sin_emb)
      tf.assert_equal(sin_emb[:, m, :, 1::2], std_sin_emb)
      tf.assert_equal(cos_emb[:, m, :, 0::2], std_cos_emb)
      tf.assert_equal(cos_emb[:, m, :, 1::2], std_cos_emb)

  @combinations.generate(
      combinations.combine(value_dim=[32, 64], mask=[True, False]))
  def test_attention_scores(self, value_dim, mask):
    """Tests combinations of attention score calculations."""
    batch_size, num_heads, key_dim, seq_length = 2, 12, 64, 8
    test_layer = roformer_attention.RoformerAttention(
        q_max_sequence_length=seq_length,
        kv_max_sequence_length=seq_length,
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim)
    data = _create_mock_attention_data(
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        q_seq_length=seq_length,
        kv_seq_length=seq_length,
        batch_size=batch_size,
        include_mask=mask)
    output = test_layer(**data)
    self.assertEqual(output.shape, [batch_size, seq_length, key_dim])


if __name__ == "__main__":
  tf.test.main()
