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

"""Test Subtokenizer and string helper methods."""

import collections
import tempfile

import tensorflow as tf, tf_keras

from official.legacy.transformer.utils import tokenizer


class SubtokenizerTest(tf.test.TestCase):

  def _init_subtokenizer(self, vocab_list):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with tf.io.gfile.GFile(temp_file.name, "w") as w:
      for subtoken in vocab_list:
        w.write("'%s'" % subtoken)
        w.write("\n")
    return tokenizer.Subtokenizer(temp_file.name, reserved_tokens=[])

  def test_encode(self):
    vocab_list = ["123_", "test", "ing_"]
    subtokenizer = self._init_subtokenizer(vocab_list)
    s = "testing 123"
    encoded_list = subtokenizer.encode(s)
    self.assertEqual([1, 2, 0], encoded_list)

  def test_decode(self):
    vocab_list = ["123_", "test", "ing_"]
    subtokenizer = self._init_subtokenizer(vocab_list)
    encoded_list = [1, 2, 0]  # testing 123
    decoded_str = subtokenizer.decode(encoded_list)
    self.assertEqual("testing 123", decoded_str)

  def test_subtoken_ids_to_tokens(self):
    vocab_list = ["123_", "test", "ing_"]
    subtokenizer = self._init_subtokenizer(vocab_list)
    encoded_list = [1, 2, 0]  # testing 123
    token_list = subtokenizer._subtoken_ids_to_tokens(encoded_list)
    self.assertEqual([u"testing", u"123"], token_list)


class StringHelperTest(tf.test.TestCase):

  def test_split_string_to_tokens(self):
    text = "test? testing 123."

    tokens = tokenizer._split_string_to_tokens(text,
                                               tokenizer._ALPHANUMERIC_CHAR_SET)
    self.assertEqual(["test", "? ", "testing", "123", "."], tokens)

  def test_join_tokens_to_string(self):
    tokens = ["test", "? ", "testing", "123", "."]

    s = tokenizer._join_tokens_to_string(tokens,
                                         tokenizer._ALPHANUMERIC_CHAR_SET)
    self.assertEqual("test? testing 123.", s)

  def test_escape_token(self):
    token = u"abc_\\4"
    alphabet = set("abc_\\u;")

    escaped_token = tokenizer._escape_token(token, alphabet)
    self.assertEqual("abc\\u\\\\\\52;_", escaped_token)

  def test_unescape_token(self):
    escaped_token = u"Underline: \\u, Backslash: \\\\, Unicode: \\52;"

    unescaped_token = tokenizer._unescape_token(escaped_token)
    self.assertEqual("Underline: _, Backslash: \\, Unicode: 4", unescaped_token)

  def test_list_to_index_dict(self):
    lst = ["test", "strings"]

    d = tokenizer._list_to_index_dict(lst)
    self.assertDictEqual({"test": 0, "strings": 1}, d)

  def test_split_token_to_subtokens(self):
    token = "abc"
    subtoken_dict = {"a": 0, "b": 1, "c": 2, "ab": 3}
    max_subtoken_length = 2

    subtokens = tokenizer._split_token_to_subtokens(token, subtoken_dict,
                                                    max_subtoken_length)
    self.assertEqual(["ab", "c"], subtokens)

  def test_generate_alphabet_dict(self):
    s = ["testing", "123"]
    reserved_tokens = ["???"]

    alphabet = tokenizer._generate_alphabet_dict(s, reserved_tokens)
    self.assertIn("?", alphabet)
    self.assertIn("t", alphabet)
    self.assertIn("e", alphabet)
    self.assertIn("s", alphabet)
    self.assertIn("i", alphabet)
    self.assertIn("n", alphabet)
    self.assertIn("g", alphabet)
    self.assertIn("1", alphabet)
    self.assertIn("2", alphabet)
    self.assertIn("3", alphabet)

  def test_count_and_gen_subtokens(self):
    token_counts = {"abc": 5}
    alphabet = set("abc_")
    subtoken_dict = {"a": 0, "b": 1, "c": 2, "_": 3}
    max_subtoken_length = 2

    subtoken_counts = tokenizer._count_and_gen_subtokens(
        token_counts, alphabet, subtoken_dict, max_subtoken_length)

    self.assertIsInstance(subtoken_counts, collections.defaultdict)
    self.assertDictEqual(
        {
            "a": 5,
            "b": 5,
            "c": 5,
            "_": 5,
            "ab": 5,
            "bc": 5,
            "c_": 5,
            "abc": 5,
            "bc_": 5,
            "abc_": 5
        }, subtoken_counts)

  def test_filter_and_bucket_subtokens(self):
    subtoken_counts = collections.defaultdict(int, {
        "a": 2,
        "b": 4,
        "c": 1,
        "ab": 6,
        "ac": 3,
        "abbc": 5
    })
    min_count = 3

    subtoken_buckets = tokenizer._filter_and_bucket_subtokens(
        subtoken_counts, min_count)

    self.assertEqual(len(subtoken_buckets[0]), 0)
    self.assertEqual(set("b"), subtoken_buckets[1])
    self.assertEqual(set(["ab", "ac"]), subtoken_buckets[2])
    self.assertEqual(len(subtoken_buckets[3]), 0)
    self.assertEqual(set(["abbc"]), subtoken_buckets[4])

  def test_gen_new_subtoken_list(self):
    subtoken_counts = collections.defaultdict(int, {
        "translate": 10,
        "t": 40,
        "tr": 16,
        "tra": 12
    })
    min_count = 5
    alphabet = set("translate")
    reserved_tokens = ["reserved", "tokens"]

    subtoken_list, max_token_length = tokenizer._gen_new_subtoken_list(
        subtoken_counts, min_count, alphabet, reserved_tokens)

    # Check that "tra" isn"t in the list (its count should be decremented to 2,
    # so it should not be added to the canddiate list).
    self.assertNotIn("tra", subtoken_list)

    self.assertIn("tr", subtoken_list)
    self.assertIn("t", subtoken_list)

    self.assertEqual(len("translate"), max_token_length)

  def test_generate_subtokens(self):
    token_counts = {"ab": 1, "bc": 3, "abc": 5}
    alphabet = set("abc_")
    min_count = 100
    num_iterations = 1
    reserved_tokens = ["reserved", "tokens"]

    vocab_list = tokenizer._generate_subtokens(token_counts, alphabet,
                                               min_count, num_iterations,
                                               reserved_tokens)

    # Check that reserved tokens are at the front of the list
    self.assertEqual(vocab_list[:2], reserved_tokens)

    # Check that each character in alphabet is in the vocab list
    for c in alphabet:
      self.assertIn(c, vocab_list)


if __name__ == "__main__":
  tf.test.main()
