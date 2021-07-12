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

"""Library to process data for tagging task such as NER/POS."""
import collections
import os

from absl import logging
import tensorflow as tf

from official.nlp.bert import tokenization
from official.nlp.data import classifier_data_lib

# A negative label id for the padding label, which will not contribute
# to loss/metrics in training.
_PADDING_LABEL_ID = -1

# The special unknown token, used to substitute a word which has too many
# subwords after tokenization.
_UNK_TOKEN = "[UNK]"


class InputExample(object):
  """A single training/test example for token classification."""

  def __init__(self,
               sentence_id,
               sub_sentence_id=0,
               words=None,
               label_ids=None):
    """Constructs an InputExample."""
    self.sentence_id = sentence_id
    self.sub_sentence_id = sub_sentence_id
    self.words = words if words else []
    self.label_ids = label_ids if label_ids else []

  def add_word_and_label_id(self, word, label_id):
    """Adds word and label_id pair in the example."""
    self.words.append(word)
    self.label_ids.append(label_id)


def _read_one_file(file_name, label_list):
  """Reads one file and returns a list of `InputExample` instances."""
  lines = tf.io.gfile.GFile(file_name, "r").readlines()
  examples = []
  label_id_map = {label: i for i, label in enumerate(label_list)}
  sentence_id = 0
  example = InputExample(sentence_id=0)
  for line in lines:
    line = line.strip("\n")
    if line:
      # The format is: <token>\t<label> for train/dev set and <token> for test.
      items = line.split("\t")
      assert len(items) == 2 or len(items) == 1
      token = items[0].strip()

      # Assign a dummy label_id for test set
      label_id = label_id_map[items[1].strip()] if len(items) == 2 else 0
      example.add_word_and_label_id(token, label_id)
    else:
      # Empty line indicates a new sentence.
      if example.words:
        examples.append(example)
        sentence_id += 1
        example = InputExample(sentence_id=sentence_id)

  if example.words:
    examples.append(example)
  return examples


class PanxProcessor(classifier_data_lib.DataProcessor):
  """Processor for the Panx data set."""
  supported_languages = [
      "ar", "he", "vi", "id", "jv", "ms", "tl", "eu", "ml", "ta", "te", "af",
      "nl", "en", "de", "el", "bn", "hi", "mr", "ur", "fa", "fr", "it", "pt",
      "es", "bg", "ru", "ja", "ka", "ko", "th", "sw", "yo", "my", "zh", "kk",
      "tr", "et", "fi", "hu"
  ]

  def __init__(self,
               process_text_fn=tokenization.convert_to_unicode,
               only_use_en_train=True,
               only_use_en_dev=True):
    """See base class.

    Args:
      process_text_fn: See base class.
      only_use_en_train: If True, only use english training data. Otherwise, use
        training data from all languages.
      only_use_en_dev: If True, only use english dev data. Otherwise, use dev
        data from all languages.
    """
    super(PanxProcessor, self).__init__(process_text_fn)
    self.only_use_en_train = only_use_en_train
    self.only_use_en_dev = only_use_en_dev

  def get_train_examples(self, data_dir):
    examples = _read_one_file(
        os.path.join(data_dir, "train-en.tsv"), self.get_labels())
    if not self.only_use_en_train:
      for language in self.supported_languages:
        if language == "en":
          continue
        examples.extend(
            _read_one_file(
                os.path.join(data_dir, f"train-{language}.tsv"),
                self.get_labels()))
    return examples

  def get_dev_examples(self, data_dir):
    examples = _read_one_file(
        os.path.join(data_dir, "dev-en.tsv"), self.get_labels())
    if not self.only_use_en_dev:
      for language in self.supported_languages:
        if language == "en":
          continue
        examples.extend(
            _read_one_file(
                os.path.join(data_dir, f"dev-{language}.tsv"),
                self.get_labels()))
    return examples

  def get_test_examples(self, data_dir):
    examples_dict = {}
    for language in self.supported_languages:
      examples_dict[language] = _read_one_file(
          os.path.join(data_dir, "test-%s.tsv" % language), self.get_labels())
    return examples_dict

  def get_labels(self):
    return ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]

  @staticmethod
  def get_processor_name():
    return "panx"


class UdposProcessor(classifier_data_lib.DataProcessor):
  """Processor for the Udpos data set."""
  supported_languages = [
      "af", "ar", "bg", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fr",
      "he", "hi", "hu", "id", "it", "ja", "kk", "ko", "mr", "nl", "pt", "ru",
      "ta", "te", "th", "tl", "tr", "ur", "vi", "yo", "zh"
  ]

  def __init__(self,
               process_text_fn=tokenization.convert_to_unicode,
               only_use_en_train=True,
               only_use_en_dev=True):
    """See base class.

    Args:
      process_text_fn: See base class.
      only_use_en_train: If True, only use english training data. Otherwise, use
        training data from all languages.
      only_use_en_dev: If True, only use english dev data. Otherwise, use dev
        data from all languages.
    """
    super(UdposProcessor, self).__init__(process_text_fn)
    self.only_use_en_train = only_use_en_train
    self.only_use_en_dev = only_use_en_dev

  def get_train_examples(self, data_dir):
    if self.only_use_en_train:
      examples = _read_one_file(
          os.path.join(data_dir, "train-en.tsv"), self.get_labels())
    else:
      examples = []
      # Uses glob because some languages are missing in train.
      for filepath in tf.io.gfile.glob(os.path.join(data_dir, "train-*.tsv")):
        examples.extend(
            _read_one_file(
                filepath,
                self.get_labels()))
    return examples

  def get_dev_examples(self, data_dir):
    if self.only_use_en_dev:
      examples = _read_one_file(
          os.path.join(data_dir, "dev-en.tsv"), self.get_labels())
    else:
      examples = []
      for filepath in tf.io.gfile.glob(os.path.join(data_dir, "dev-*.tsv")):
        examples.extend(
            _read_one_file(
                filepath,
                self.get_labels()))
    return examples

  def get_test_examples(self, data_dir):
    examples_dict = {}
    for language in self.supported_languages:
      examples_dict[language] = _read_one_file(
          os.path.join(data_dir, "test-%s.tsv" % language), self.get_labels())
    return examples_dict

  def get_labels(self):
    return [
        "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
        "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"
    ]

  @staticmethod
  def get_processor_name():
    return "udpos"


def _tokenize_example(example, max_length, tokenizer, text_preprocessing=None):
  """Tokenizes words and breaks long example into short ones."""
  # Needs additional [CLS] and [SEP] tokens.
  max_length = max_length - 2
  new_examples = []
  new_example = InputExample(sentence_id=example.sentence_id, sub_sentence_id=0)
  if any([x < 0 for x in example.label_ids]):
    raise ValueError("Unexpected negative label_id: %s" % example.label_ids)

  for i, word in enumerate(example.words):
    if text_preprocessing:
      word = text_preprocessing(word)
    subwords = tokenizer.tokenize(word)
    if (not subwords or len(subwords) > max_length) and word:
      subwords = [_UNK_TOKEN]

    if len(subwords) + len(new_example.words) > max_length:
      # Start a new example.
      new_examples.append(new_example)
      last_sub_sentence_id = new_example.sub_sentence_id
      new_example = InputExample(
          sentence_id=example.sentence_id,
          sub_sentence_id=last_sub_sentence_id + 1)

    for j, subword in enumerate(subwords):
      # Use the real label for the first subword, and pad label for
      # the remainings.
      subword_label = example.label_ids[i] if j == 0 else _PADDING_LABEL_ID
      new_example.add_word_and_label_id(subword, subword_label)

  if new_example.words:
    new_examples.append(new_example)

  return new_examples


def _convert_single_example(example, max_seq_length, tokenizer):
  """Converts an `InputExample` instance to a `tf.train.Example` instance."""
  tokens = ["[CLS]"]
  tokens.extend(example.words)
  tokens.append("[SEP]")
  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  label_ids = [_PADDING_LABEL_ID]
  label_ids.extend(example.label_ids)
  label_ids.append(_PADDING_LABEL_ID)

  segment_ids = [0] * len(input_ids)
  input_mask = [1] * len(input_ids)

  # Pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
    label_ids.append(_PADDING_LABEL_ID)

  def create_int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

  features = collections.OrderedDict()
  features["input_ids"] = create_int_feature(input_ids)
  features["input_mask"] = create_int_feature(input_mask)
  features["segment_ids"] = create_int_feature(segment_ids)
  features["label_ids"] = create_int_feature(label_ids)
  features["sentence_id"] = create_int_feature([example.sentence_id])
  features["sub_sentence_id"] = create_int_feature([example.sub_sentence_id])

  tf_example = tf.train.Example(features=tf.train.Features(feature=features))
  return tf_example


def write_example_to_file(examples,
                          tokenizer,
                          max_seq_length,
                          output_file,
                          text_preprocessing=None):
  """Writes `InputExample`s into a tfrecord file with `tf.train.Example` protos.

  Note that the words inside each example will be tokenized and be applied by
  `text_preprocessing` if available. Also, if the length of sentence (plus
  special [CLS] and [SEP] tokens) exceeds `max_seq_length`, the long sentence
  will be broken into multiple short examples. For example:

  Example (text_preprocessing=lowercase, max_seq_length=5)
    words:        ["What", "a", "great", "weekend"]
    labels:       [     7,   5,       9,        10]
    sentence_id:  0
    preprocessed: ["what", "a", "great", "weekend"]
    tokenized:    ["what", "a", "great", "week", "##end"]

  will result in two tf.example protos:

    tokens:      ["[CLS]", "what", "a", "great", "[SEP]"]
    label_ids:   [-1,       7,     5,     9,     -1]
    input_mask:  [ 1,       1,     1,     1,      1]
    segment_ids: [ 0,       0,     0,     0,      0]
    input_ids:   [ tokenizer.convert_tokens_to_ids(tokens) ]
    sentence_id: 0

    tokens:      ["[CLS]", "week", "##end", "[SEP]", "[PAD]"]
    label_ids:   [-1,       10,     -1,    -1,       -1]
    input_mask:  [ 1,       1,       1,     0,        0]
    segment_ids: [ 0,       0,       0,     0,        0]
    input_ids:   [ tokenizer.convert_tokens_to_ids(tokens) ]
    sentence_id: 0

    Note the use of -1 in `label_ids` to indicate that a token should not be
    considered for classification (e.g., trailing ## wordpieces or special
    token). Token classification models should accordingly ignore these when
    calculating loss, metrics, etc...

  Args:
    examples: A list of `InputExample` instances.
    tokenizer: The tokenizer to be applied on the data.
    max_seq_length: Maximum length of generated sequences.
    output_file: The name of the output tfrecord file.
    text_preprocessing: optional preprocessing run on each word prior to
      tokenization.

  Returns:
    The total number of tf.train.Example proto written to file.
  """
  tf.io.gfile.makedirs(os.path.dirname(output_file))
  writer = tf.io.TFRecordWriter(output_file)
  num_tokenized_examples = 0
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logging.info("Writing example %d of %d to %s", ex_index, len(examples),
                   output_file)

    tokenized_examples = _tokenize_example(example, max_seq_length, tokenizer,
                                           text_preprocessing)
    num_tokenized_examples += len(tokenized_examples)
    for per_tokenized_example in tokenized_examples:
      tf_example = _convert_single_example(per_tokenized_example,
                                           max_seq_length, tokenizer)
      writer.write(tf_example.SerializeToString())

  writer.close()
  return num_tokenized_examples


def token_classification_meta_data(train_data_size,
                                   max_seq_length,
                                   num_labels,
                                   eval_data_size=None,
                                   test_data_size=None,
                                   label_list=None,
                                   processor_type=None):
  """Creates metadata for tagging (token classification) datasets."""
  meta_data = {
      "train_data_size": train_data_size,
      "max_seq_length": max_seq_length,
      "num_labels": num_labels,
      "task_type": "tagging",
      "label_type": "int",
      "label_shape": [max_seq_length],
  }
  if eval_data_size:
    meta_data["eval_data_size"] = eval_data_size
  if test_data_size:
    meta_data["test_data_size"] = test_data_size
  if label_list:
    meta_data["label_list"] = label_list
  if processor_type:
    meta_data["processor_type"] = processor_type

  return meta_data


def generate_tf_record_from_data_file(processor, data_dir, tokenizer,
                                      max_seq_length, train_data_output_path,
                                      eval_data_output_path,
                                      test_data_output_path,
                                      text_preprocessing):
  """Generates tfrecord files from the raw data."""
  common_kwargs = dict(
      tokenizer=tokenizer,
      max_seq_length=max_seq_length,
      text_preprocessing=text_preprocessing)
  train_examples = processor.get_train_examples(data_dir)
  train_data_size = write_example_to_file(
      train_examples, output_file=train_data_output_path, **common_kwargs)

  eval_examples = processor.get_dev_examples(data_dir)
  eval_data_size = write_example_to_file(
      eval_examples, output_file=eval_data_output_path, **common_kwargs)

  test_input_data_examples = processor.get_test_examples(data_dir)
  test_data_size = {}
  for language, examples in test_input_data_examples.items():
    test_data_size[language] = write_example_to_file(
        examples,
        output_file=test_data_output_path.format(language),
        **common_kwargs)

  labels = processor.get_labels()
  meta_data = token_classification_meta_data(
      train_data_size,
      max_seq_length,
      len(labels),
      eval_data_size,
      test_data_size,
      label_list=labels,
      processor_type=processor.get_processor_name())
  return meta_data
