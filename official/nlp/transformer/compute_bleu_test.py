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
"""Test functions in compute_blue.py."""

import tempfile

import tensorflow as tf

from official.nlp.transformer import compute_bleu


class ComputeBleuTest(tf.test.TestCase):

  def _create_temp_file(self, text):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with tf.io.gfile.GFile(temp_file.name, "w") as w:
      w.write(text)
    return temp_file.name

  def test_bleu_same(self):
    ref = self._create_temp_file("test 1 two 3\nmore tests!")
    hyp = self._create_temp_file("test 1 two 3\nmore tests!")

    uncased_score = compute_bleu.bleu_wrapper(ref, hyp, False)
    cased_score = compute_bleu.bleu_wrapper(ref, hyp, True)
    self.assertEqual(100, uncased_score)
    self.assertEqual(100, cased_score)

  def test_bleu_same_different_case(self):
    ref = self._create_temp_file("Test 1 two 3\nmore tests!")
    hyp = self._create_temp_file("test 1 two 3\nMore tests!")
    uncased_score = compute_bleu.bleu_wrapper(ref, hyp, False)
    cased_score = compute_bleu.bleu_wrapper(ref, hyp, True)
    self.assertEqual(100, uncased_score)
    self.assertLess(cased_score, 100)

  def test_bleu_different(self):
    ref = self._create_temp_file("Testing\nmore tests!")
    hyp = self._create_temp_file("Dog\nCat")
    uncased_score = compute_bleu.bleu_wrapper(ref, hyp, False)
    cased_score = compute_bleu.bleu_wrapper(ref, hyp, True)
    self.assertLess(uncased_score, 100)
    self.assertLess(cased_score, 100)

  def test_bleu_tokenize(self):
    s = "Test0, 1 two, 3"
    tokenized = compute_bleu.bleu_tokenize(s)
    self.assertEqual(["Test0", ",", "1", "two", ",", "3"], tokenized)


if __name__ == "__main__":
  tf.test.main()
