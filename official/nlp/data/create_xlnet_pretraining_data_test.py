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

"""Tests for official.nlp.data.create_xlnet_pretraining_data."""
import os
import tempfile
from typing import List

from absl import logging
from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from official.nlp.data import create_xlnet_pretraining_data as cpd

_VOCAB_WORDS = ["vocab_1", "vocab_2"]


# pylint: disable=invalid-name
def _create_files(
    temp_dir: str, file_contents: List[List[str]]) -> List[str]:
  """Writes arbitrary documents into files."""
  root_dir = tempfile.mkdtemp(dir=temp_dir)
  files = []

  for i, file_content in enumerate(file_contents):
    destination = os.path.join(root_dir, "%d.txt" % i)
    with open(destination, "wb") as f:
      for line in file_content:
        f.write(line.encode("utf-8"))
    files.append(destination)
  return files


def _get_mock_tokenizer():
  """Creates a mock tokenizer."""

  class MockSpieceModel:
    """Mock Spiece model for testing."""

    def __init__(self):
      self._special_piece_to_id = {
          "<unk>": 0,
      }
      for piece in set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~')):
        self._special_piece_to_id[piece] = 1

    def EncodeAsPieces(self, inputs: str) -> List[str]:
      return inputs

    def SampleEncodeAsPieces(self,
                             inputs: str,
                             nbest_size: int,
                             theta: float) -> List[str]:
      del nbest_size, theta
      return inputs

    def PieceToId(self, piece: str) -> int:
      return ord(piece[0])

    def IdToPiece(self, id_: int) -> str:
      return chr(id_) * 3

  class Tokenizer:
    """Mock Tokenizer for testing."""

    def __init__(self):
      self.sp_model = MockSpieceModel()

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
      return [self.sp_model.IdToPiece(id_) for id_ in ids]

  return Tokenizer()


class PreprocessDataTest(tf.test.TestCase):

  def test_remove_extraneous_space(self):
    line = "   abc   "
    output = cpd._preprocess_line(line)
    self.assertEqual(output, "abc")

  def test_symbol_replacements(self):
    self.assertEqual(cpd._preprocess_line("``abc``"), "\"abc\"")
    self.assertEqual(cpd._preprocess_line("''abc''"), "\"abc\"")

  def test_accent_replacements(self):
    self.assertEqual(cpd._preprocess_line("åbc"), "abc")

  def test_lower_case(self):
    self.assertEqual(cpd._preprocess_line("ABC", do_lower_case=True), "abc")

  def test_end_to_end(self):
    self.assertEqual(
        cpd._preprocess_line("HelLo ``wórLd``", do_lower_case=True),
        "hello \"world\"")


class PreprocessAndTokenizeFilesTest(tf.test.TestCase):

  def test_basic_end_to_end(self):
    documents = [
        [
            "This is sentence 1.\n",
            "This is sentence 2.\n",
            "Sentence 3 is what this is.\n",
        ],
        [
            "This is the second document.\n",
            "This is the second line of the second document.\n"
        ],
    ]
    input_files = _create_files(temp_dir=self.get_temp_dir(),
                                file_contents=documents)
    all_data = cpd.preprocess_and_tokenize_input_files(
        input_files=input_files,
        tokenizer=_get_mock_tokenizer(),
        log_example_freq=1)

    self.assertEqual(len(all_data), len(documents))
    for token_ids, sentence_ids in all_data:
      self.assertEqual(len(token_ids), len(sentence_ids))

  def test_basic_correctness(self):
    documents = [["a\n", "b\n", "c\n"]]
    input_files = _create_files(temp_dir=self.get_temp_dir(),
                                file_contents=documents)
    all_data = cpd.preprocess_and_tokenize_input_files(
        input_files=input_files,
        tokenizer=_get_mock_tokenizer(),
        log_example_freq=1)

    token_ids, sentence_ids = all_data[0]

    self.assertAllClose(token_ids, [97, 98, 99])
    self.assertAllClose(sentence_ids, [True, False, True])

  def test_correctness_with_spaces_and_accents(self):
    documents = [[
        "       å   \n",
        "b          \n",
        "   c      \n",
    ]]
    input_files = _create_files(temp_dir=self.get_temp_dir(),
                                file_contents=documents)
    all_data = cpd.preprocess_and_tokenize_input_files(
        input_files=input_files,
        tokenizer=_get_mock_tokenizer(),
        log_example_freq=1)

    token_ids, sentence_ids = all_data[0]

    self.assertAllClose(token_ids, [97, 98, 99])
    self.assertAllClose(sentence_ids, [True, False, True])


class BatchReshapeTests(tf.test.TestCase):

  def test_basic_functionality(self):
    per_host_batch_size = 3
    mock_shape = (20,)

    # Should truncate and reshape.
    expected_result_shape = (3, 6)

    tokens = np.zeros(mock_shape)
    sentence_ids = np.zeros(mock_shape)

    reshaped_data = cpd._reshape_to_batch_dimensions(
        tokens=tokens,
        sentence_ids=sentence_ids,
        per_host_batch_size=per_host_batch_size)
    for values in reshaped_data:
      self.assertEqual(len(values.flatten()) % per_host_batch_size, 0)
      self.assertAllClose(values.shape, expected_result_shape)


class CreateSegmentsTest(tf.test.TestCase):

  def test_basic_functionality(self):
    data_length = 10
    tokens = np.arange(data_length)
    sentence_ids = np.concatenate([np.zeros(data_length // 2),
                                   np.ones(data_length // 2)])
    begin_index = 0
    total_length = 8
    a_data, b_data, label = cpd._create_a_and_b_segments(
        tokens=tokens,
        sentence_ids=sentence_ids,
        begin_index=begin_index,
        total_length=total_length,
        no_cut_probability=0.)
    self.assertAllClose(a_data, [0, 1, 2, 3])
    self.assertAllClose(b_data, [5, 6, 7, 8])
    self.assertEqual(label, 1)

  def test_no_cut(self):
    data_length = 10
    tokens = np.arange(data_length)
    sentence_ids = np.zeros(data_length)

    begin_index = 0
    total_length = 8
    a_data, b_data, label = cpd._create_a_and_b_segments(
        tokens=tokens,
        sentence_ids=sentence_ids,
        begin_index=begin_index,
        total_length=total_length,
        no_cut_probability=0.)
    self.assertGreater(len(a_data), 0)
    self.assertGreater(len(b_data), 0)
    self.assertEqual(label, 0)

  def test_no_cut_with_probability(self):
    data_length = 10
    tokens = np.arange(data_length)
    sentence_ids = np.concatenate([np.zeros(data_length // 2),
                                   np.ones(data_length // 2)])
    begin_index = 0
    total_length = 8
    a_data, b_data, label = cpd._create_a_and_b_segments(
        tokens=tokens,
        sentence_ids=sentence_ids,
        begin_index=begin_index,
        total_length=total_length,
        no_cut_probability=1.)
    self.assertGreater(len(a_data), 0)
    self.assertGreater(len(b_data), 0)
    self.assertEqual(label, 0)


class CreateInstancesTest(tf.test.TestCase):
  """Tests conversions of Token/Sentence IDs to training instances."""

  def test_basic(self):
    data_length = 12
    tokens = np.arange(data_length)
    sentence_ids = np.zeros(data_length)
    seq_length = 8
    instances = cpd._convert_tokens_to_instances(
        tokens=tokens,
        sentence_ids=sentence_ids,
        per_host_batch_size=2,
        seq_length=seq_length,
        reuse_length=4,
        tokenizer=_get_mock_tokenizer(),
        bi_data=False,
        num_cores_per_host=1,
        logging_frequency=1)
    for instance in instances:
      self.assertEqual(len(instance.data), seq_length)
      self.assertEqual(len(instance.segment_ids), seq_length)
      self.assertIsInstance(instance.label, int)
      self.assertIsInstance(instance.boundary_indices, list)


class TFRecordPathTests(tf.test.TestCase):

  def test_basic(self):
    base_kwargs = dict(
        per_host_batch_size=1,
        num_cores_per_host=1,
        seq_length=2,
        reuse_length=1)

    config1 = dict(
        prefix="test",
        suffix="",
        bi_data=True,
        use_eod_token=False,
        do_lower_case=True)
    config1.update(base_kwargs)
    expectation1 = "test_seqlen-2_reuse-1_bs-1_cores-1_uncased_bi.tfrecord"
    self.assertEqual(cpd.get_tfrecord_name(**config1), expectation1)

    config2 = dict(
        prefix="",
        suffix="test",
        bi_data=False,
        use_eod_token=False,
        do_lower_case=False)
    config2.update(base_kwargs)
    expectation2 = "seqlen-2_reuse-1_bs-1_cores-1_cased_uni_test.tfrecord"
    self.assertEqual(cpd.get_tfrecord_name(**config2), expectation2)

    config3 = dict(
        prefix="",
        suffix="",
        use_eod_token=True,
        bi_data=False,
        do_lower_case=True)
    config3.update(base_kwargs)
    expectation3 = "seqlen-2_reuse-1_bs-1_cores-1_uncased_eod_uni.tfrecord"
    self.assertEqual(cpd.get_tfrecord_name(**config3), expectation3)


class TestCreateTFRecords(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ("bi_data_only", True, False, False),
      ("eod_token_only", False, True, True),
      ("lower_case_only", False, False, True),
      ("all_enabled", True, True, True),
      )
  def test_end_to_end(self,
                      bi_data: bool,
                      use_eod_token: bool,
                      do_lower_case: bool):
    tokenizer = _get_mock_tokenizer()

    num_documents = 5
    sentences_per_document = 10
    document_length = 50

    documents = [
        ["a " * document_length for _ in range(sentences_per_document)]
        for _ in range(num_documents)]

    save_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    files = _create_files(temp_dir=self.get_temp_dir(), file_contents=documents)

    cpd.create_tfrecords(
        tokenizer=tokenizer,
        input_file_or_files=",".join(files),
        use_eod_token=use_eod_token,
        do_lower_case=do_lower_case,
        per_host_batch_size=8,
        seq_length=8,
        reuse_length=4,
        bi_data=bi_data,
        num_cores_per_host=2,
        save_dir=save_dir)

    self.assertTrue(any(filter(lambda x: x.endswith(".json"),
                               os.listdir(save_dir))))
    self.assertTrue(any(filter(lambda x: x.endswith(".tfrecord"),
                               os.listdir(save_dir))))


if __name__ == "__main__":
  np.random.seed(0)
  logging.set_verbosity(logging.INFO)
  tf.test.main()
