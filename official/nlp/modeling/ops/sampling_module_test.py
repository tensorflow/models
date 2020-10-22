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
"""Tests for Sampling Strategies."""

from absl.testing import parameterized
import tensorflow as tf

from official.nlp.modeling.ops import sampling_module


def length_norm(length, dtype):
  """Return length normalization factor."""
  return tf.pow(((5. + tf.cast(length, dtype)) / 6.), 0.0)


class SamplingModuleTest(tf.test.TestCase, parameterized.TestCase):

  cache = {'layer_%d' % layer: {'k': tf.zeros([2, 2, 2, 2], dtype=tf.float32),
                                'v': tf.zeros([2, 2, 2, 2], dtype=tf.float32)
                               } for layer in range(2)}
  probabilities = tf.constant([[[0.3, 0.4, 0.3], [0.3, 0.3, 0.4],
                                [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]],
                               [[0.2, 0.4, 0.4], [0.2, 0.7, 0.1],
                                [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]])

  def _get_test_symbols_to_logits_fn(self):
    """Calculates logits of the next tokens."""
    def symbols_to_logits_fn(ids, i, cache):
      del ids
      logits = tf.cast(tf.math.log(self.probabilities[:, i, :]), tf.float32)
      return logits, cache
    return symbols_to_logits_fn

  @parameterized.named_parameters([
      ('padded_decode_true', True),
      ('padded_decode_false', False),
  ])
  def test_greedy(self, padded_decode):
    greedy_obj = sampling_module.SamplingModule(
        length_normalization_fn=None,
        dtype=tf.float32,
        symbols_to_logits_fn=self._get_test_symbols_to_logits_fn(),
        vocab_size=3,
        max_decode_length=4,
        eos_id=10,
        padded_decode=padded_decode)
    ids, _ = greedy_obj.generate(
        initial_ids=tf.constant([9, 1]), initial_cache=self.cache)
    self.assertAllEqual([[9, 1, 2, 2, 2], [1, 1, 1, 2, 2]], ids)

  @parameterized.named_parameters([
      ('padded_decode_true', True),
      ('padded_decode_false', False),
  ])
  def test_topk(self, padded_decode):
    top_k_obj = sampling_module.SamplingModule(
        length_normalization_fn=length_norm,
        dtype=tf.float32,
        symbols_to_logits_fn=self._get_test_symbols_to_logits_fn(),
        vocab_size=3,
        max_decode_length=4,
        eos_id=10,
        sample_temperature=tf.constant(0.1),
        top_k=tf.constant(3),
        padded_decode=padded_decode)
    ids, _ = top_k_obj.generate(
        initial_ids=tf.constant([9, 1]), initial_cache=self.cache)
    self.assertAllEqual([2, 5], ids.shape)

if __name__ == '__main__':
  tf.test.main()
