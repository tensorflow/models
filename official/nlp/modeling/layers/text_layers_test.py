# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Tests bert.text_layers."""

import os
import tempfile

import numpy as np
import tensorflow as tf

from sentencepiece import SentencePieceTrainer
from official.nlp.modeling.layers import text_layers


class RoundRobinTruncatorTest(tf.test.TestCase):

  def _test_input(self, start, lengths):
    return tf.ragged.constant([[start + 10 * j + i
                                for i in range(length)]
                               for j, length in enumerate(lengths)],
                              dtype=tf.int32)

  def test_single_segment(self):
    # Single segment.
    single_input = self._test_input(11, [4, 5, 6])
    expected_single_output = tf.ragged.constant(
        [[11, 12, 13, 14],
         [21, 22, 23, 24, 25],
         [31, 32, 33, 34, 35],  # Truncated.
        ])

    self.assertAllEqual(
        expected_single_output,
        text_layers.round_robin_truncate_inputs(single_input, limit=5))
    # Test wrapping in a singleton list.
    actual_single_list_output = text_layers.round_robin_truncate_inputs(
        [single_input], limit=5)
    self.assertIsInstance(actual_single_list_output, list)
    self.assertAllEqual(expected_single_output, actual_single_list_output[0])

  def test_two_segments(self):
    input_a = self._test_input(111, [1, 2, 2, 3, 4, 5])
    input_b = self._test_input(211, [1, 3, 4, 2, 2, 5])
    expected_a = tf.ragged.constant(
        [[111],
         [121, 122],
         [131, 132],
         [141, 142, 143],
         [151, 152, 153],  # Truncated.
         [161, 162, 163],  # Truncated.
        ])
    expected_b = tf.ragged.constant(
        [[211],
         [221, 222, 223],
         [231, 232, 233],  # Truncated.
         [241, 242],
         [251, 252],
         [261, 262],  # Truncated.
        ])
    actual_a, actual_b = text_layers.round_robin_truncate_inputs(
        [input_a, input_b], limit=5)
    self.assertAllEqual(expected_a, actual_a)
    self.assertAllEqual(expected_b, actual_b)

  def test_three_segments(self):
    input_a = self._test_input(111, [1, 2, 2, 3, 4, 5, 1])
    input_b = self._test_input(211, [1, 3, 4, 2, 2, 5, 8])
    input_c = self._test_input(311, [1, 3, 4, 2, 2, 5, 10])
    seg_limit = 8
    expected_a = tf.ragged.constant([
        [111],
        [121, 122],
        [131, 132],
        [141, 142, 143],
        [151, 152, 153, 154],
        [161, 162, 163],  # Truncated
        [171]
    ])
    expected_b = tf.ragged.constant([
        [211],
        [221, 222, 223],
        [231, 232, 233],  # Truncated
        [241, 242],
        [251, 252],
        [261, 262, 263],  # Truncated
        [271, 272, 273, 274]  # Truncated
    ])
    expected_c = tf.ragged.constant([
        [311],
        [321, 322, 323],
        [331, 332, 333],  # Truncated
        [341, 342],
        [351, 352],
        [361, 362],  # Truncated
        [371, 372, 373]  # Truncated
    ])
    actual_a, actual_b, actual_c = text_layers.round_robin_truncate_inputs(
        [input_a, input_b, input_c], limit=seg_limit)
    self.assertAllEqual(expected_a, actual_a)
    self.assertAllEqual(expected_b, actual_b)
    self.assertAllEqual(expected_c, actual_c)
    input_cap = tf.math.reduce_sum(
        tf.stack([rt.row_lengths() for rt in [input_a, input_b, input_c]]),
        axis=0)
    per_example_usage = tf.math.reduce_sum(
        tf.stack([rt.row_lengths() for rt in [actual_a, actual_b, actual_c]]),
        axis=0)
    self.assertTrue(all(per_example_usage <= tf.minimum(seg_limit, input_cap)))


# This test covers the in-process behavior of a BertTokenizer layer.
# For saving, restoring, and the restored behavior (incl. shape inference),
# see nlp/tools/export_tfhub_lib_test.py.
class BertTokenizerTest(tf.test.TestCase):

  def _make_vocab_file(self, vocab, filename="vocab.txt"):
    path = os.path.join(
        tempfile.mkdtemp(dir=self.get_temp_dir()),  # New subdir each time.
        filename)
    with tf.io.gfile.GFile(path, "w") as f:
      f.write("\n".join(vocab + [""]))
    return path

  def test_uncased(self):
    vocab_file = self._make_vocab_file(
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "d", "##ef", "abc", "xy"])
    bert_tokenize = text_layers.BertTokenizer(
        vocab_file=vocab_file, lower_case=True)
    inputs = tf.constant(["abc def", "ABC DEF d"])
    token_ids = bert_tokenize(inputs)
    self.assertAllEqual(token_ids, tf.ragged.constant([[[6], [4, 5]],
                                                       [[6], [4, 5], [4]]]))
    bert_tokenize.tokenize_with_offsets = True
    token_ids_2, start_offsets, limit_offsets = bert_tokenize(inputs)
    self.assertAllEqual(token_ids, token_ids_2)
    self.assertAllEqual(start_offsets, tf.ragged.constant([[[0], [4, 5]],
                                                           [[0], [4, 5], [8]]]))
    self.assertAllEqual(limit_offsets, tf.ragged.constant([[[3], [5, 7]],
                                                           [[3], [5, 7], [9]]]))
    self.assertEqual(bert_tokenize.vocab_size.numpy(), 8)

  # Repeat the above and test that case matters with lower_case=False.
  def test_cased(self):
    vocab_file = self._make_vocab_file(
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "d", "##ef", "abc", "ABC"])
    bert_tokenize = text_layers.BertTokenizer(
        vocab_file=vocab_file, lower_case=False, tokenize_with_offsets=True)
    inputs = tf.constant(["abc def", "ABC DEF"])
    token_ids, start_offsets, limit_offsets = bert_tokenize(inputs)
    self.assertAllEqual(token_ids, tf.ragged.constant([[[6], [4, 5]],
                                                       [[7], [1]]]))
    self.assertAllEqual(start_offsets, tf.ragged.constant([[[0], [4, 5]],
                                                           [[0], [4]]]))
    self.assertAllEqual(limit_offsets, tf.ragged.constant([[[3], [5, 7]],
                                                           [[3], [7]]]))

  def test_special_tokens_complete(self):
    vocab_file = self._make_vocab_file(
        ["foo", "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "xy"])
    bert_tokenize = text_layers.BertTokenizer(
        vocab_file=vocab_file, lower_case=True)
    self.assertDictEqual(bert_tokenize.get_special_tokens_dict(),
                         dict(padding_id=1,
                              start_of_sequence_id=3,
                              end_of_segment_id=4,
                              mask_id=5,
                              vocab_size=7))

  def test_special_tokens_partial(self):
    vocab_file = self._make_vocab_file(
        ["[PAD]", "[CLS]", "[SEP]"])
    bert_tokenize = text_layers.BertTokenizer(
        vocab_file=vocab_file, lower_case=True)
    self.assertDictEqual(bert_tokenize.get_special_tokens_dict(),
                         dict(padding_id=0,
                              start_of_sequence_id=1,
                              end_of_segment_id=2,
                              vocab_size=3))  # No mask_id,

  def test_special_tokens_in_estimator(self):
    """Tests getting special tokens without an Eager init context."""
    vocab_file = self._make_vocab_file(
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "d", "##ef", "abc", "xy"])

    def input_fn():
      with tf.init_scope():
        self.assertFalse(tf.executing_eagerly())
      # Build a preprocessing Model.
      sentences = tf.keras.layers.Input(shape=[], dtype=tf.string)
      bert_tokenizer = text_layers.BertTokenizer(
          vocab_file=vocab_file, lower_case=True)
      special_tokens_dict = bert_tokenizer.get_special_tokens_dict()
      for k, v in special_tokens_dict.items():
        self.assertIsInstance(v, int, "Unexpected type for {}".format(k))
      tokens = bert_tokenizer(sentences)
      packed_inputs = text_layers.BertPackInputs(
          4, special_tokens_dict=special_tokens_dict)(tokens)
      preprocessing = tf.keras.Model(sentences, packed_inputs)
      # Map the dataset.
      ds = tf.data.Dataset.from_tensors(
          (tf.constant(["abc", "DEF"]), tf.constant([0, 1])))
      ds = ds.map(lambda features, labels: (preprocessing(features), labels))
      return ds

    def model_fn(features, labels, mode):
      del labels  # Unused.
      return tf.estimator.EstimatorSpec(mode=mode,
                                        predictions=features["input_word_ids"])

    estimator = tf.estimator.Estimator(model_fn=model_fn)
    outputs = list(estimator.predict(input_fn))
    self.assertAllEqual(outputs, np.array([[2, 6, 3, 0],
                                           [2, 4, 5, 3]]))


# This test covers the in-process behavior of a SentencepieceTokenizer layer.
class SentencepieceTokenizerTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    # Make a sentencepiece model.
    tmp_dir = self.get_temp_dir()
    tempfile.mkdtemp(dir=tmp_dir)
    vocab = ["a", "b", "c", "d", "e", "abc", "def", "ABC", "DEF"]
    model_prefix = os.path.join(tmp_dir, "spm_model")
    input_text_file_path = os.path.join(tmp_dir, "train_input.txt")
    with tf.io.gfile.GFile(input_text_file_path, "w") as f:
      f.write(" ".join(vocab + ["\n"]))
    # Add 7 more tokens: <pad>, <unk>, [CLS], [SEP], [MASK], <s>, </s>.
    full_vocab_size = len(vocab) + 7
    flags = dict(
        model_prefix=model_prefix,
        model_type="word",
        input=input_text_file_path,
        pad_id=0, unk_id=1, control_symbols="[CLS],[SEP],[MASK]",
        vocab_size=full_vocab_size,
        bos_id=full_vocab_size-2, eos_id=full_vocab_size-1)
    SentencePieceTrainer.Train(
        " ".join(["--{}={}".format(k, v) for k, v in flags.items()]))
    self._spm_path = model_prefix + ".model"

  def test_uncased(self):
    sentencepiece_tokenizer = text_layers.SentencepieceTokenizer(
        model_file_path=self._spm_path, lower_case=True, nbest_size=0)

    inputs = tf.constant(["abc def", "ABC DEF d"])
    token_ids = sentencepiece_tokenizer(inputs)
    self.assertAllEqual(
        token_ids,
        tf.ragged.constant([[8, 12], [8, 12, 11]]))
    sentencepiece_tokenizer.tokenize_with_offsets = True
    token_ids_2, start_offsets, limit_offsets = sentencepiece_tokenizer(inputs)
    self.assertAllEqual(token_ids, token_ids_2)
    self.assertAllEqual(
        start_offsets, tf.ragged.constant([[0, 3], [0, 3, 7]]))
    self.assertAllEqual(
        limit_offsets, tf.ragged.constant([[3, 7], [3, 7, 9]]))
    self.assertEqual(sentencepiece_tokenizer.vocab_size.numpy(), 16)

  # Repeat the above and test that case matters with lower_case=False.
  def test_cased(self):
    sentencepiece_tokenizer = text_layers.SentencepieceTokenizer(
        model_file_path=self._spm_path,
        lower_case=False,
        nbest_size=0,
        tokenize_with_offsets=False)

    inputs = tf.constant(["abc def", "ABC DEF d"])
    token_ids = sentencepiece_tokenizer(inputs)
    self.assertAllEqual(
        token_ids,
        tf.ragged.constant([[8, 12], [5, 6, 11]]))
    sentencepiece_tokenizer.tokenize_with_offsets = True
    token_ids_2, start_offsets, limit_offsets = sentencepiece_tokenizer(inputs)
    self.assertAllEqual(token_ids, token_ids_2)
    self.assertAllEqual(
        start_offsets,
        tf.ragged.constant([[0, 3], [0, 3, 7]]))
    self.assertAllEqual(
        limit_offsets,
        tf.ragged.constant([[3, 7], [3, 7, 9]]))

  def test_special_tokens(self):
    sentencepiece_tokenizer = text_layers.SentencepieceTokenizer(
        model_file_path=self._spm_path, lower_case=True, nbest_size=0)
    self.assertDictEqual(sentencepiece_tokenizer.get_special_tokens_dict(),
                         dict(padding_id=0,
                              start_of_sequence_id=2,
                              end_of_segment_id=3,
                              mask_id=4,
                              vocab_size=16))

  def test_special_tokens_in_estimator(self):
    """Tests getting special tokens without an Eager init context."""

    def input_fn():
      with tf.init_scope():
        self.assertFalse(tf.executing_eagerly())
      # Build a preprocessing Model.
      sentences = tf.keras.layers.Input(shape=[], dtype=tf.string)
      sentencepiece_tokenizer = text_layers.SentencepieceTokenizer(
          model_file_path=self._spm_path, lower_case=True, nbest_size=0)
      special_tokens_dict = sentencepiece_tokenizer.get_special_tokens_dict()
      for k, v in special_tokens_dict.items():
        self.assertIsInstance(v, int, "Unexpected type for {}".format(k))
      tokens = sentencepiece_tokenizer(sentences)
      packed_inputs = text_layers.BertPackInputs(
          4, special_tokens_dict=special_tokens_dict)(tokens)
      preprocessing = tf.keras.Model(sentences, packed_inputs)
      # Map the dataset.
      ds = tf.data.Dataset.from_tensors(
          (tf.constant(["abc", "DEF"]), tf.constant([0, 1])))
      ds = ds.map(lambda features, labels: (preprocessing(features), labels))
      return ds

    def model_fn(features, labels, mode):
      del labels  # Unused.
      return tf.estimator.EstimatorSpec(mode=mode,
                                        predictions=features["input_word_ids"])

    estimator = tf.estimator.Estimator(model_fn=model_fn)
    outputs = list(estimator.predict(input_fn))
    self.assertAllEqual(outputs, np.array([[2, 8, 3, 0],
                                           [2, 12, 3, 0]]))

  def test_strip_diacritics(self):
    sentencepiece_tokenizer = text_layers.SentencepieceTokenizer(
        model_file_path=self._spm_path,
        lower_case=True,
        nbest_size=0,
        strip_diacritics=True)
    inputs = tf.constant(["a b c d e", "ă ḅ č ḓ é"])
    token_ids = sentencepiece_tokenizer(inputs)
    self.assertAllEqual(
        token_ids,
        tf.ragged.constant([[7, 9, 10, 11, 13], [7, 9, 10, 11, 13]]))

  def test_fail_on_tokenize_with_offsets_and_strip_diacritics(self):
    # Raise an error in init().
    with self.assertRaises(ValueError):
      text_layers.SentencepieceTokenizer(
          model_file_path=self._spm_path,
          tokenize_with_offsets=True,
          lower_case=True,
          nbest_size=0,
          strip_diacritics=True)

    sentencepiece_tokenizer = text_layers.SentencepieceTokenizer(
        model_file_path=self._spm_path,
        lower_case=True,
        nbest_size=0,
        strip_diacritics=True)
    sentencepiece_tokenizer.tokenize_with_offsets = True

    # Raise an error in call():
    inputs = tf.constant(["abc def", "ABC DEF d", "Äffin"])
    with self.assertRaises(ValueError):
      sentencepiece_tokenizer(inputs)

  def test_serialize_deserialize(self):
    self.skipTest("b/170480226")
    sentencepiece_tokenizer = text_layers.SentencepieceTokenizer(
        model_file_path=self._spm_path,
        lower_case=False,
        nbest_size=0,
        tokenize_with_offsets=False,
        name="sentencepiece_tokenizer_layer")
    config = sentencepiece_tokenizer.get_config()
    new_tokenizer = text_layers.SentencepieceTokenizer.from_config(config)
    self.assertEqual(config, new_tokenizer.get_config())
    inputs = tf.constant(["abc def", "ABC DEF d"])
    token_ids = sentencepiece_tokenizer(inputs)
    token_ids_2 = new_tokenizer(inputs)
    self.assertAllEqual(token_ids, token_ids_2)

  # TODO(b/170480226): Remove once tf_hub_export_lib_test.py covers saving.
  def test_saving(self):
    sentencepiece_tokenizer = text_layers.SentencepieceTokenizer(
        model_file_path=self._spm_path, lower_case=True, nbest_size=0)
    inputs = tf.keras.layers.Input([], dtype=tf.string)
    outputs = sentencepiece_tokenizer(inputs)
    model = tf.keras.Model(inputs, outputs)
    export_path = tempfile.mkdtemp(dir=self.get_temp_dir())
    model.save(export_path, signatures={})


class BertPackInputsTest(tf.test.TestCase):

  def test_round_robin_correct_outputs(self):
    bpi = text_layers.BertPackInputs(
        10,
        start_of_sequence_id=1001,
        end_of_segment_id=1002,
        padding_id=999,
        truncator="round_robin")
    # Single input, rank 2.
    bert_inputs = bpi(
        tf.ragged.constant([[11, 12, 13],
                            [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]))
    self.assertAllEqual(
        bert_inputs["input_word_ids"],
        tf.constant([[1001, 11, 12, 13, 1002, 999, 999, 999, 999, 999],
                     [1001, 21, 22, 23, 24, 25, 26, 27, 28, 1002]]))
    self.assertAllEqual(
        bert_inputs["input_mask"],
        tf.constant([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))
    self.assertAllEqual(
        bert_inputs["input_type_ids"],
        tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    # Two inputs, rank 3. Truncation does not respect word boundaries.
    bert_inputs = bpi([
        tf.ragged.constant([[[111], [112, 113]],
                            [[121, 122, 123], [124, 125, 126], [127, 128]]]),
        tf.ragged.constant([[[211, 212], [213]],
                            [[221, 222], [223, 224, 225], [226, 227, 228]]])
    ])
    self.assertAllEqual(
        bert_inputs["input_word_ids"],
        tf.constant([[1001, 111, 112, 113, 1002, 211, 212, 213, 1002, 999],
                     [1001, 121, 122, 123, 124, 1002, 221, 222, 223, 1002]]))
    self.assertAllEqual(
        bert_inputs["input_mask"],
        tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))
    self.assertAllEqual(
        bert_inputs["input_type_ids"],
        tf.constant([[0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]]))

    # Three inputs. rank 3.
    bert_inputs = bpi([
        tf.ragged.constant([[[111], [112, 113]],
                            [[121, 122, 123], [124, 125, 126], [127, 128]]]),
        tf.ragged.constant([[[211, 212], [213]],
                            [[221, 222], [223, 224, 225], [226, 227, 228]]]),
        tf.ragged.constant([[[311, 312], [313]],
                            [[321, 322], [323, 324, 325], [326, 327, 328]]])
    ])
    self.assertAllEqual(
        bert_inputs["input_word_ids"],
        tf.constant([[1001, 111, 112, 1002, 211, 212, 1002, 311, 312, 1002],
                     [1001, 121, 122, 1002, 221, 222, 1002, 321, 322, 1002]]))

  def test_waterfall_correct_outputs(self):
    bpi = text_layers.BertPackInputs(
        10,
        start_of_sequence_id=1001,
        end_of_segment_id=1002,
        padding_id=999,
        truncator="waterfall")
    # Single input, rank 2.
    bert_inputs = bpi(
        tf.ragged.constant([[11, 12, 13],
                            [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]))
    self.assertAllEqual(
        bert_inputs["input_word_ids"],
        tf.constant([[1001, 11, 12, 13, 1002, 999, 999, 999, 999, 999],
                     [1001, 21, 22, 23, 24, 25, 26, 27, 28, 1002]]))
    self.assertAllEqual(
        bert_inputs["input_mask"],
        tf.constant([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))
    self.assertAllEqual(
        bert_inputs["input_type_ids"],
        tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    # Two inputs, rank 3. Truncation does not respect word boundaries.
    bert_inputs = bpi([
        tf.ragged.constant([[[111], [112, 113]],
                            [[121, 122, 123], [124, 125, 126], [127, 128]]]),
        tf.ragged.constant([[[211, 212], [213]],
                            [[221, 222], [223, 224, 225], [226, 227, 228]]])
    ])
    self.assertAllEqual(
        bert_inputs["input_word_ids"],
        tf.constant([[1001, 111, 112, 113, 1002, 211, 212, 213, 1002, 999],
                     [1001, 121, 122, 123, 124, 125, 126, 127, 1002, 1002]]))
    self.assertAllEqual(
        bert_inputs["input_mask"],
        tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))
    self.assertAllEqual(
        bert_inputs["input_type_ids"],
        tf.constant([[0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]))

    # Three inputs, rank 3. Truncation does not respect word boundaries.
    bert_inputs = bpi([
        tf.ragged.constant([[[111], [112, 113]],
                            [[121, 122, 123], [124, 125, 126], [127, 128]]]),
        tf.ragged.constant([[[211], [212]],
                            [[221, 222], [223, 224, 225], [226, 227, 228]]]),
        tf.ragged.constant([[[311, 312], [313]],
                            [[321, 322], [323, 324, 325], [326, 327]]])
    ])
    self.assertAllEqual(
        bert_inputs["input_word_ids"],
        tf.constant([[1001, 111, 112, 113, 1002, 211, 212, 1002, 311, 1002],
                     [1001, 121, 122, 123, 124, 125, 126, 1002, 1002, 1002]]))
    self.assertAllEqual(
        bert_inputs["input_mask"],
        tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))
    self.assertAllEqual(
        bert_inputs["input_type_ids"],
        tf.constant([[0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 2]]))

  def test_special_tokens_dict(self):
    special_tokens_dict = dict(start_of_sequence_id=1001,
                               end_of_segment_id=1002,
                               padding_id=999,
                               extraneous_key=666)
    bpi = text_layers.BertPackInputs(10,
                                     special_tokens_dict=special_tokens_dict)
    bert_inputs = bpi(
        tf.ragged.constant([[11, 12, 13],
                            [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]))
    self.assertAllEqual(
        bert_inputs["input_word_ids"],
        tf.constant([[1001, 11, 12, 13, 1002, 999, 999, 999, 999, 999],
                     [1001, 21, 22, 23, 24, 25, 26, 27, 28, 1002]]))


if __name__ == "__main__":
  tf.test.main()
