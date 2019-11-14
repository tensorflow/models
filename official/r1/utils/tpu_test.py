# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Test TPU optimized matmul embedding."""

import numpy as np
import tensorflow as tf

from official.r1.utils import tpu as tpu_utils


TEST_CASES = [
    dict(embedding_dim=256, vocab_size=1000, sequence_length=64,
         batch_size=32, seed=54131),
    dict(embedding_dim=8, vocab_size=15, sequence_length=12,
         batch_size=256, seed=536413),
    dict(embedding_dim=2048, vocab_size=512, sequence_length=50,
         batch_size=8, seed=35124)
]


class TPUBaseTester(tf.test.TestCase):
  def construct_embedding_and_values(self, embedding_dim, vocab_size,
                                     sequence_length, batch_size, seed):
    np.random.seed(seed)

    embeddings = np.random.random(size=(vocab_size, embedding_dim))
    embedding_table = tf.convert_to_tensor(value=embeddings, dtype=tf.float32)

    tokens = np.random.randint(low=1, high=vocab_size-1,
                               size=(batch_size, sequence_length))
    for i in range(batch_size):
      tokens[i, np.random.randint(low=0, high=sequence_length-1):] = 0
    values = tf.convert_to_tensor(value=tokens, dtype=tf.int32)
    mask = tf.cast(tf.not_equal(values, 0), dtype=tf.float32)
    return embedding_table, values, mask

  def _test_embedding(self, embedding_dim, vocab_size,
                      sequence_length, batch_size, seed):
    """Test that matmul embedding matches embedding lookup (gather)."""

    with self.test_session():
      embedding_table, values, mask = self.construct_embedding_and_values(
          embedding_dim=embedding_dim,
          vocab_size=vocab_size,
          sequence_length=sequence_length,
          batch_size=batch_size,
          seed=seed
      )

      embedding = (tf.nn.embedding_lookup(params=embedding_table, ids=values) *
                   tf.expand_dims(mask, -1))

      matmul_embedding = tpu_utils.embedding_matmul(
          embedding_table=embedding_table, values=values, mask=mask)

      self.assertAllClose(embedding, matmul_embedding)

  def _test_masking(self, embedding_dim, vocab_size,
                    sequence_length, batch_size, seed):
    """Test that matmul embedding properly zeros masked positions."""
    with self.test_session():
      embedding_table, values, mask = self.construct_embedding_and_values(
          embedding_dim=embedding_dim,
          vocab_size=vocab_size,
          sequence_length=sequence_length,
          batch_size=batch_size,
          seed=seed
      )

      matmul_embedding = tpu_utils.embedding_matmul(
          embedding_table=embedding_table, values=values, mask=mask)

      self.assertAllClose(matmul_embedding,
                          matmul_embedding * tf.expand_dims(mask, -1))

  def test_embedding_0(self):
    self._test_embedding(**TEST_CASES[0])

  def test_embedding_1(self):
    self._test_embedding(**TEST_CASES[1])

  def test_embedding_2(self):
    self._test_embedding(**TEST_CASES[2])

  def test_masking_0(self):
    self._test_masking(**TEST_CASES[0])

  def test_masking_1(self):
    self._test_masking(**TEST_CASES[1])

  def test_masking_2(self):
    self._test_masking(**TEST_CASES[2])


if __name__ == "__main__":
  tf.test.main()
