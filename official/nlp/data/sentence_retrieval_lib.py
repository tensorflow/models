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
"""BERT library to process data for cross lingual sentence retrieval task."""

import os

from absl import logging
from official.nlp.bert import tokenization
from official.nlp.data import classifier_data_lib


class BuccProcessor(classifier_data_lib.DataProcessor):
  """Procssor for Xtreme BUCC data set."""
  supported_languages = ["de", "fr", "ru", "zh"]

  def __init__(self, process_text_fn=tokenization.convert_to_unicode):
    super(BuccProcessor, self).__init__(process_text_fn)
    self.languages = BuccProcessor.supported_languages

  def get_dev_examples(self, data_dir, file_pattern):
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, file_pattern.format("dev"))),
        "sample")

  def get_test_examples(self, data_dir, file_pattern):
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, file_pattern.format("test"))),
        "test")

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "BUCC"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      example_id = int(line[0].split("-")[1])
      text_a = self.process_text_fn(line[1])
      examples.append(
          classifier_data_lib.InputExample(
              guid=guid, text_a=text_a, example_id=example_id))
    return examples


class TatoebaProcessor(classifier_data_lib.DataProcessor):
  """Procssor for Xtreme Tatoeba data set."""
  supported_languages = [
      "af", "ar", "bg", "bn", "de", "el", "es", "et", "eu", "fa", "fi", "fr",
      "he", "hi", "hu", "id", "it", "ja", "jv", "ka", "kk", "ko", "ml", "mr",
      "nl", "pt", "ru", "sw", "ta", "te", "th", "tl", "tr", "ur", "vi", "zh"
  ]

  def __init__(self, process_text_fn=tokenization.convert_to_unicode):
    super(TatoebaProcessor, self).__init__(process_text_fn)
    self.languages = TatoebaProcessor.supported_languages

  def get_test_examples(self, data_dir, file_path):
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, file_path)), "test")

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "TATOEBA"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = self.process_text_fn(line[0])
      examples.append(
          classifier_data_lib.InputExample(
              guid=guid, text_a=text_a, example_id=i))
    return examples


def generate_sentence_retrevial_tf_record(processor,
                                          data_dir,
                                          tokenizer,
                                          eval_data_output_path=None,
                                          test_data_output_path=None,
                                          max_seq_length=128):
  """Generates the tf records for retrieval tasks.

  Args:
    processor: Input processor object to be used for generating data. Subclass
      of `DataProcessor`.
      data_dir: Directory that contains train/eval data to process. Data files
        should be in from.
      tokenizer: The tokenizer to be applied on the data.
      eval_data_output_path: Output to which processed tf record for evaluation
        will be saved.
      test_data_output_path: Output to which processed tf record for testing
        will be saved. Must be a pattern template with {} if processor has
        language specific test data.
      max_seq_length: Maximum sequence length of the to be generated
        training/eval data.

  Returns:
      A dictionary containing input meta data.
  """
  assert eval_data_output_path or test_data_output_path

  if processor.get_processor_name() == "BUCC":
    path_pattern = "{}-en.{{}}.{}"

  if processor.get_processor_name() == "TATOEBA":
    path_pattern = "{}-en.{}"

  meta_data = {
      "processor_type": processor.get_processor_name(),
      "max_seq_length": max_seq_length,
      "number_eval_data": {},
      "number_test_data": {},
  }
  logging.info("Start to process %s task data", processor.get_processor_name())

  for lang_a in processor.languages:
    for lang_b in [lang_a, "en"]:
      if eval_data_output_path:
        eval_input_data_examples = processor.get_dev_examples(
            data_dir, os.path.join(path_pattern.format(lang_a, lang_b)))

        num_eval_data = len(eval_input_data_examples)
        logging.info("Processing %d dev examples of %s-en.%s", num_eval_data,
                     lang_a, lang_b)
        output_file = os.path.join(
            eval_data_output_path,
            "{}-en-{}.{}.tfrecords".format(lang_a, lang_b, "dev"))
        classifier_data_lib.file_based_convert_examples_to_features(
            eval_input_data_examples, None, max_seq_length, tokenizer,
            output_file, None)
        meta_data["number_eval_data"][f"{lang_a}-en.{lang_b}"] = num_eval_data

      if test_data_output_path:
        test_input_data_examples = processor.get_test_examples(
            data_dir, os.path.join(path_pattern.format(lang_a, lang_b)))

        num_test_data = len(test_input_data_examples)
        logging.info("Processing %d test examples of %s-en.%s", num_test_data,
                     lang_a, lang_b)
        output_file = os.path.join(
            test_data_output_path,
            "{}-en-{}.{}.tfrecords".format(lang_a, lang_b, "test"))
        classifier_data_lib.file_based_convert_examples_to_features(
            test_input_data_examples, None, max_seq_length, tokenizer,
            output_file, None)
        meta_data["number_test_data"][f"{lang_a}-en.{lang_b}"] = num_test_data

  return meta_data
