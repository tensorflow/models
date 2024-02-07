# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for official.nlp.data.create_pretraining_data."""
import random

import tensorflow as tf, tf_keras

from official.nlp.data import create_pretraining_data as cpd

_VOCAB_WORDS = ["vocab_1", "vocab_2"]


class CreatePretrainingDataTest(tf.test.TestCase):

  def assertTokens(self, input_tokens, output_tokens, masked_positions,
                   masked_labels):
    # Ensure the masked positions are unique.
    self.assertCountEqual(masked_positions, set(masked_positions))

    # Ensure we can reconstruct the input from the output.
    reconstructed_tokens = output_tokens
    for pos, label in zip(masked_positions, masked_labels):
      reconstructed_tokens[pos] = label
    self.assertEqual(input_tokens, reconstructed_tokens)

    # Ensure each label is valid.
    for pos, label in zip(masked_positions, masked_labels):
      output_token = output_tokens[pos]
      if (output_token == "[MASK]" or output_token in _VOCAB_WORDS or
          output_token == input_tokens[pos]):
        continue
      self.fail("invalid mask value: {}".format(output_token))

  def test_tokens_to_grams(self):
    tests = [
        (["That", "cone"], [(0, 1), (1, 2)]),
        (["That", "cone", "##s"], [(0, 1), (1, 3)]),
        (["Swit", "##zer", "##land"], [(0, 3)]),
        (["[CLS]", "Up", "##dog"], [(1, 3)]),
        (["[CLS]", "Up", "##dog", "[SEP]", "Down"], [(1, 3), (4, 5)]),
    ]
    for inp, expected in tests:
      output = cpd._tokens_to_grams(inp)
      self.assertEqual(expected, output)

  def test_window(self):
    input_list = [1, 2, 3, 4]
    window_outputs = [
        (1, [[1], [2], [3], [4]]),
        (2, [[1, 2], [2, 3], [3, 4]]),
        (3, [[1, 2, 3], [2, 3, 4]]),
        (4, [[1, 2, 3, 4]]),
        (5, []),
    ]
    for window, expected in window_outputs:
      output = cpd._window(input_list, window)
      self.assertEqual(expected, list(output))

  def test_create_masked_lm_predictions(self):
    tokens = ["[CLS]", "a", "##a", "b", "##b", "c", "##c", "[SEP]"]
    rng = random.Random(123)
    for _ in range(0, 5):
      output_tokens, masked_positions, masked_labels = (
          cpd.create_masked_lm_predictions(
              tokens=tokens,
              masked_lm_prob=1.0,
              max_predictions_per_seq=3,
              vocab_words=_VOCAB_WORDS,
              rng=rng,
              do_whole_word_mask=False,
              max_ngram_size=None))
      self.assertLen(masked_positions, 3)
      self.assertLen(masked_labels, 3)
      self.assertTokens(tokens, output_tokens, masked_positions, masked_labels)

  def test_create_masked_lm_predictions_whole_word(self):
    tokens = ["[CLS]", "a", "##a", "b", "##b", "c", "##c", "[SEP]"]
    rng = random.Random(345)
    for _ in range(0, 5):
      output_tokens, masked_positions, masked_labels = (
          cpd.create_masked_lm_predictions(
              tokens=tokens,
              masked_lm_prob=1.0,
              max_predictions_per_seq=3,
              vocab_words=_VOCAB_WORDS,
              rng=rng,
              do_whole_word_mask=True,
              max_ngram_size=None))
      # since we can't get exactly three tokens without breaking a word we
      # only take two.
      self.assertLen(masked_positions, 2)
      self.assertLen(masked_labels, 2)
      self.assertTokens(tokens, output_tokens, masked_positions, masked_labels)
      # ensure that we took an entire word.
      self.assertIn(masked_labels, [["a", "##a"], ["b", "##b"], ["c", "##c"]])

  def test_create_masked_lm_predictions_ngram(self):
    tokens = ["[CLS]"] + ["tok{}".format(i) for i in range(0, 512)] + ["[SEP]"]
    rng = random.Random(345)
    for _ in range(0, 5):
      output_tokens, masked_positions, masked_labels = (
          cpd.create_masked_lm_predictions(
              tokens=tokens,
              masked_lm_prob=1.0,
              max_predictions_per_seq=76,
              vocab_words=_VOCAB_WORDS,
              rng=rng,
              do_whole_word_mask=True,
              max_ngram_size=3))
      self.assertLen(masked_positions, 76)
      self.assertLen(masked_labels, 76)
      self.assertTokens(tokens, output_tokens, masked_positions, masked_labels)


if __name__ == "__main__":
  tf.test.main()
