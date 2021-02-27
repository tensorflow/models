# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""BERT library to process data for classification task."""

import collections
import csv
import importlib
import json
import os

from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from official.nlp.bert import tokenization


class InputExample(object):
  """A single training/test example for simple seq regression/classification."""

  def __init__(self,
               guid,
               text_a,
               text_b=None,
               label=None,
               weight=None,
               example_id=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string for classification, float for regression. The
        label of the example. This should be specified for train and dev
        examples, but not for test examples.
      weight: (Optional) float. The weight of the example to be used during
        training.
      example_id: (Optional) int. The int identification number of example in
        the corpus.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    self.weight = weight
    self.example_id = example_id


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True,
               weight=None,
               example_id=None):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example
    self.weight = weight
    self.example_id = example_id


class DataProcessor(object):
  """Base class for converters for seq regression/classification datasets."""

  def __init__(self, process_text_fn=tokenization.convert_to_unicode):
    self.process_text_fn = process_text_fn
    self.is_regression = False
    self.label_type = None

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @staticmethod
  def get_processor_name():
    """Gets the string identifier of the processor."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.io.gfile.GFile(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  @classmethod
  def _read_jsonl(cls, input_file):
    """Reads a json line file."""
    with tf.io.gfile.GFile(input_file, "r") as f:
      lines = []
      for json_str in f:
        lines.append(json.loads(json_str))
    return lines


class AxProcessor(DataProcessor):
  """Processor for the AX dataset (GLUE diagnostics dataset)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "AX"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training/dev/test sets."""
    text_a_index = 1 if set_type == "test" else 8
    text_b_index = 2 if set_type == "test" else 9
    examples = []
    for i, line in enumerate(lines):
      # Skip header.
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, self.process_text_fn(line[0]))
      text_a = self.process_text_fn(line[text_a_index])
      text_b = self.process_text_fn(line[text_b_index])
      if set_type == "test":
        label = "contradiction"
      else:
        label = self.process_text_fn(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "COLA"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training/dev/test sets."""
    examples = []
    for i, line in enumerate(lines):
      # Only the test set has a header.
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = self.process_text_fn(line[1])
        label = "0"
      else:
        text_a = self.process_text_fn(line[3])
        label = self.process_text_fn(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


class ImdbProcessor(DataProcessor):
  """Processor for the IMDb dataset."""

  def get_labels(self):
    return ["neg", "pos"]

  def get_train_examples(self, data_dir):
    return self._create_examples(os.path.join(data_dir, "train"))

  def get_dev_examples(self, data_dir):
    return self._create_examples(os.path.join(data_dir, "test"))

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "IMDB"

  def _create_examples(self, data_dir):
    """Creates examples."""
    examples = []
    for label in ["neg", "pos"]:
      cur_dir = os.path.join(data_dir, label)
      for filename in tf.io.gfile.listdir(cur_dir):
        if not filename.endswith("txt"):
          continue

        if len(examples) % 1000 == 0:
          logging.info("Loading dev example %d", len(examples))

        path = os.path.join(cur_dir, filename)
        with tf.io.gfile.GFile(path, "r") as f:
          text = f.read().strip().replace("<br />", " ")
        examples.append(
            InputExample(
                guid="unused_id", text_a=text, text_b=None, label=label))
    return examples


class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def __init__(self,
               mnli_type="matched",
               process_text_fn=tokenization.convert_to_unicode):
    super(MnliProcessor, self).__init__(process_text_fn)
    if mnli_type not in ("matched", "mismatched"):
      raise ValueError("Invalid `mnli_type`: %s" % mnli_type)
    self.mnli_type = mnli_type

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    if self.mnli_type == "matched":
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
          "dev_matched")
    else:
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
          "dev_mismatched")

  def get_test_examples(self, data_dir):
    """See base class."""
    if self.mnli_type == "matched":
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")
    else:
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "MNLI"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training/dev/test sets."""
    examples = []
    for i, line in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, self.process_text_fn(line[0]))
      text_a = self.process_text_fn(line[8])
      text_b = self.process_text_fn(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = self.process_text_fn(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "MRPC"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training/dev/test sets."""
    examples = []
    for i, line in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = self.process_text_fn(line[3])
      text_b = self.process_text_fn(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = self.process_text_fn(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class PawsxProcessor(DataProcessor):
  """Processor for the PAWS-X data set."""
  supported_languages = ["de", "en", "es", "fr", "ja", "ko", "zh"]

  def __init__(self,
               language="en",
               process_text_fn=tokenization.convert_to_unicode):
    super(PawsxProcessor, self).__init__(process_text_fn)
    if language == "all":
      self.languages = PawsxProcessor.supported_languages
    elif language not in PawsxProcessor.supported_languages:
      raise ValueError("language %s is not supported for PAWS-X task." %
                       language)
    else:
      self.languages = [language]

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = []
    for language in self.languages:
      if language == "en":
        train_tsv = "train.tsv"
      else:
        train_tsv = "translated_train.tsv"
      # Skips the header.
      lines.extend(
          self._read_tsv(os.path.join(data_dir, language, train_tsv))[1:])

    examples = []
    for i, line in enumerate(lines):
      guid = "train-%d" % i
      text_a = self.process_text_fn(line[1])
      text_b = self.process_text_fn(line[2])
      label = self.process_text_fn(line[3])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = []
    for lang in PawsxProcessor.supported_languages:
      lines.extend(
          self._read_tsv(os.path.join(data_dir, lang, "dev_2k.tsv"))[1:])

    examples = []
    for i, line in enumerate(lines):
      guid = "dev-%d" % i
      text_a = self.process_text_fn(line[1])
      text_b = self.process_text_fn(line[2])
      label = self.process_text_fn(line[3])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_test_examples(self, data_dir):
    """See base class."""
    examples_by_lang = {k: [] for k in self.supported_languages}
    for lang in self.supported_languages:
      lines = self._read_tsv(os.path.join(data_dir, lang, "test_2k.tsv"))[1:]
      for i, line in enumerate(lines):
        guid = "test-%d" % i
        text_a = self.process_text_fn(line[1])
        text_b = self.process_text_fn(line[2])
        label = self.process_text_fn(line[3])
        examples_by_lang[lang].append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples_by_lang

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "XTREME-PAWS-X"


class QnliProcessor(DataProcessor):
  """Processor for the QNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["entailment", "not_entailment"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "QNLI"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training/dev/test sets."""
    examples = []
    for i, line in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, 1)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        text_b = tokenization.convert_to_unicode(line[2])
        label = "entailment"
      else:
        text_a = tokenization.convert_to_unicode(line[1])
        text_b = tokenization.convert_to_unicode(line[2])
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class QqpProcessor(DataProcessor):
  """Processor for the QQP data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "QQP"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training/dev/test sets."""
    examples = []
    for i, line in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, line[0])
      if set_type == "test":
        text_a = line[1]
        text_b = line[2]
        label = "0"
      else:
        # There appear to be some garbage lines in the train dataset.
        try:
          text_a = line[3]
          text_b = line[4]
          label = line[5]
        except IndexError:
          continue
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class RteProcessor(DataProcessor):
  """Processor for the RTE data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    # All datasets are converted to 2-class split, where for 3-class datasets we
    # collapse neutral and contradiction into not_entailment.
    return ["entailment", "not_entailment"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "RTE"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training/dev/test sets."""
    examples = []
    for i, line in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[1])
      text_b = tokenization.convert_to_unicode(line[2])
      if set_type == "test":
        label = "entailment"
      else:
        label = tokenization.convert_to_unicode(line[3])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class SstProcessor(DataProcessor):
  """Processor for the SST-2 data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "SST-2"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training/dev/test sets."""
    examples = []
    for i, line in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[0])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


class StsBProcessor(DataProcessor):
  """Processor for the STS-B data set (GLUE version)."""

  def __init__(self, process_text_fn=tokenization.convert_to_unicode):
    super(StsBProcessor, self).__init__(process_text_fn=process_text_fn)
    self.is_regression = True
    self.label_type = float
    self._labels = None

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return self._labels

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "STS-B"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training/dev/test sets."""
    examples = []
    for i, line in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[7])
      text_b = tokenization.convert_to_unicode(line[8])
      if set_type == "test":
        label = 0.0
      else:
        label = self.label_type(tokenization.convert_to_unicode(line[9]))
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class TfdsProcessor(DataProcessor):
  """Processor for generic text classification and regression TFDS data set.

  The TFDS parameters are expected to be provided in the tfds_params string, in
  a comma-separated list of parameter assignments.
  Examples:
    tfds_params="dataset=scicite,text_key=string"
    tfds_params="dataset=imdb_reviews,test_split=,dev_split=test"
    tfds_params="dataset=glue/cola,text_key=sentence"
    tfds_params="dataset=glue/sst2,text_key=sentence"
    tfds_params="dataset=glue/qnli,text_key=question,text_b_key=sentence"
    tfds_params="dataset=glue/mrpc,text_key=sentence1,text_b_key=sentence2"
    tfds_params="dataset=glue/stsb,text_key=sentence1,text_b_key=sentence2,"
                "is_regression=true,label_type=float"
    tfds_params="dataset=snli,text_key=premise,text_b_key=hypothesis,"
                "skip_label=-1"
  Possible parameters (please refer to the documentation of Tensorflow Datasets
  (TFDS) for the meaning of individual parameters):
    dataset: Required dataset name (potentially with subset and version number).
    data_dir: Optional TFDS source root directory.
    module_import: Optional Dataset module to import.
    train_split: Name of the train split (defaults to `train`).
    dev_split: Name of the dev split (defaults to `validation`).
    test_split: Name of the test split (defaults to `test`).
    text_key: Key of the text_a feature (defaults to `text`).
    text_b_key: Key of the second text feature if available.
    label_key: Key of the label feature (defaults to `label`).
    test_text_key: Key of the text feature to use in test set.
    test_text_b_key: Key of the second text feature to use in test set.
    test_label: String to be used as the label for all test examples.
    label_type: Type of the label key (defaults to `int`).
    weight_key: Key of the float sample weight (is not used if not provided).
    is_regression: Whether the task is a regression problem (defaults to False).
    skip_label: Skip examples with given label (defaults to None).
  """

  def __init__(self,
               tfds_params,
               process_text_fn=tokenization.convert_to_unicode):
    super(TfdsProcessor, self).__init__(process_text_fn)
    self._process_tfds_params_str(tfds_params)
    if self.module_import:
      importlib.import_module(self.module_import)

    self.dataset, info = tfds.load(
        self.dataset_name, data_dir=self.data_dir, with_info=True)
    if self.is_regression:
      self._labels = None
    else:
      self._labels = list(range(info.features[self.label_key].num_classes))

  def _process_tfds_params_str(self, params_str):
    """Extracts TFDS parameters from a comma-separated assignements string."""
    dtype_map = {"int": int, "float": float}
    cast_str_to_bool = lambda s: s.lower() not in ["false", "0"]

    tuples = [x.split("=") for x in params_str.split(",")]
    d = {k.strip(): v.strip() for k, v in tuples}
    self.dataset_name = d["dataset"]  # Required.
    self.data_dir = d.get("data_dir", None)
    self.module_import = d.get("module_import", None)
    self.train_split = d.get("train_split", "train")
    self.dev_split = d.get("dev_split", "validation")
    self.test_split = d.get("test_split", "test")
    self.text_key = d.get("text_key", "text")
    self.text_b_key = d.get("text_b_key", None)
    self.label_key = d.get("label_key", "label")
    self.test_text_key = d.get("test_text_key", self.text_key)
    self.test_text_b_key = d.get("test_text_b_key", self.text_b_key)
    self.test_label = d.get("test_label", "test_example")
    self.label_type = dtype_map[d.get("label_type", "int")]
    self.is_regression = cast_str_to_bool(d.get("is_regression", "False"))
    self.weight_key = d.get("weight_key", None)
    self.skip_label = d.get("skip_label", None)
    if self.skip_label is not None:
      self.skip_label = self.label_type(self.skip_label)

  def get_train_examples(self, data_dir):
    assert data_dir is None
    return self._create_examples(self.train_split, "train")

  def get_dev_examples(self, data_dir):
    assert data_dir is None
    return self._create_examples(self.dev_split, "dev")

  def get_test_examples(self, data_dir):
    assert data_dir is None
    return self._create_examples(self.test_split, "test")

  def get_labels(self):
    return self._labels

  def get_processor_name(self):
    return "TFDS_" + self.dataset_name

  def _create_examples(self, split_name, set_type):
    """Creates examples for the training/dev/test sets."""
    if split_name not in self.dataset:
      raise ValueError("Split {} not available.".format(split_name))
    dataset = self.dataset[split_name].as_numpy_iterator()
    examples = []
    text_b, weight = None, None
    for i, example in enumerate(dataset):
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = self.process_text_fn(example[self.test_text_key])
        if self.test_text_b_key:
          text_b = self.process_text_fn(example[self.test_text_b_key])
        label = self.test_label
      else:
        text_a = self.process_text_fn(example[self.text_key])
        if self.text_b_key:
          text_b = self.process_text_fn(example[self.text_b_key])
        label = self.label_type(example[self.label_key])
        if self.skip_label is not None and label == self.skip_label:
          continue
      if self.weight_key:
        weight = float(example[self.weight_key])
      examples.append(
          InputExample(
              guid=guid,
              text_a=text_a,
              text_b=text_b,
              label=label,
              weight=weight))
    return examples


class WnliProcessor(DataProcessor):
  """Processor for the WNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "WNLI"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training/dev/test sets."""
    examples = []
    for i, line in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[1])
      text_b = tokenization.convert_to_unicode(line[2])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[3])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""
  supported_languages = [
      "ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr",
      "ur", "vi", "zh"
  ]

  def __init__(self,
               language="en",
               process_text_fn=tokenization.convert_to_unicode):
    super(XnliProcessor, self).__init__(process_text_fn)
    if language == "all":
      self.languages = XnliProcessor.supported_languages
    elif language not in XnliProcessor.supported_languages:
      raise ValueError("language %s is not supported for XNLI task." % language)
    else:
      self.languages = [language]

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = []
    for language in self.languages:
      # Skips the header.
      lines.extend(
          self._read_tsv(
              os.path.join(data_dir, "multinli",
                           "multinli.train.%s.tsv" % language))[1:])

    examples = []
    for i, line in enumerate(lines):
      guid = "train-%d" % i
      text_a = self.process_text_fn(line[0])
      text_b = self.process_text_fn(line[1])
      label = self.process_text_fn(line[2])
      if label == self.process_text_fn("contradictory"):
        label = self.process_text_fn("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
    examples = []
    for i, line in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % i
      text_a = self.process_text_fn(line[6])
      text_b = self.process_text_fn(line[7])
      label = self.process_text_fn(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_test_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.test.tsv"))
    examples_by_lang = {k: [] for k in XnliProcessor.supported_languages}
    for i, line in enumerate(lines):
      if i == 0:
        continue
      guid = "test-%d" % i
      language = self.process_text_fn(line[0])
      text_a = self.process_text_fn(line[6])
      text_b = self.process_text_fn(line[7])
      label = self.process_text_fn(line[1])
      examples_by_lang[language].append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples_by_lang

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "XNLI"


class XtremePawsxProcessor(DataProcessor):
  """Processor for the XTREME PAWS-X data set."""
  supported_languages = ["de", "en", "es", "fr", "ja", "ko", "zh"]

  def __init__(self,
               process_text_fn=tokenization.convert_to_unicode,
               translated_data_dir=None,
               only_use_en_dev=True):
    """See base class.

    Args:
      process_text_fn: See base class.
      translated_data_dir: If specified, will also include translated data in
        the training and testing data.
      only_use_en_dev: If True, only use english dev data. Otherwise, use dev
        data from all languages.
    """
    super(XtremePawsxProcessor, self).__init__(process_text_fn)
    self.translated_data_dir = translated_data_dir
    self.only_use_en_dev = only_use_en_dev

  def get_train_examples(self, data_dir):
    """See base class."""
    examples = []
    if self.translated_data_dir is None:
      lines = self._read_tsv(os.path.join(data_dir, "train-en.tsv"))
      for i, line in enumerate(lines):
        guid = "train-%d" % i
        text_a = self.process_text_fn(line[0])
        text_b = self.process_text_fn(line[1])
        label = self.process_text_fn(line[2])
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    else:
      for lang in self.supported_languages:
        lines = self._read_tsv(
            os.path.join(self.translated_data_dir, "translate-train",
                         f"en-{lang}-translated.tsv"))
        for i, line in enumerate(lines):
          guid = f"train-{lang}-{i}"
          text_a = self.process_text_fn(line[2])
          text_b = self.process_text_fn(line[3])
          label = self.process_text_fn(line[4])
          examples.append(
              InputExample(
                  guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    examples = []
    if self.only_use_en_dev:
      lines = self._read_tsv(os.path.join(data_dir, "dev-en.tsv"))
      for i, line in enumerate(lines):
        guid = "dev-%d" % i
        text_a = self.process_text_fn(line[0])
        text_b = self.process_text_fn(line[1])
        label = self.process_text_fn(line[2])
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    else:
      for lang in self.supported_languages:
        lines = self._read_tsv(os.path.join(data_dir, f"dev-{lang}.tsv"))
        for i, line in enumerate(lines):
          guid = f"dev-{lang}-{i}"
          text_a = self.process_text_fn(line[0])
          text_b = self.process_text_fn(line[1])
          label = self.process_text_fn(line[2])
          examples.append(
              InputExample(
                  guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_test_examples(self, data_dir):
    """See base class."""
    examples_by_lang = {}
    for lang in self.supported_languages:
      examples_by_lang[lang] = []
      lines = self._read_tsv(os.path.join(data_dir, f"test-{lang}.tsv"))
      for i, line in enumerate(lines):
        guid = f"test-{lang}-{i}"
        text_a = self.process_text_fn(line[0])
        text_b = self.process_text_fn(line[1])
        label = "0"
        examples_by_lang[lang].append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    if self.translated_data_dir is not None:
      for lang in self.supported_languages:
        if lang == "en":
          continue
        examples_by_lang[f"{lang}-en"] = []
        lines = self._read_tsv(
            os.path.join(self.translated_data_dir, "translate-test",
                         f"test-{lang}-en-translated.tsv"))
        for i, line in enumerate(lines):
          guid = f"test-{lang}-en-{i}"
          text_a = self.process_text_fn(line[2])
          text_b = self.process_text_fn(line[3])
          label = "0"
          examples_by_lang[f"{lang}-en"].append(
              InputExample(
                  guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples_by_lang

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "XTREME-PAWS-X"


class XtremeXnliProcessor(DataProcessor):
  """Processor for the XTREME XNLI data set."""
  supported_languages = [
      "ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr",
      "ur", "vi", "zh"
  ]

  def __init__(self,
               process_text_fn=tokenization.convert_to_unicode,
               translated_data_dir=None,
               only_use_en_dev=True):
    """See base class.

    Args:
      process_text_fn: See base class.
      translated_data_dir: If specified, will also include translated data in
        the training data.
      only_use_en_dev: If True, only use english dev data. Otherwise, use dev
        data from all languages.
    """
    super(XtremeXnliProcessor, self).__init__(process_text_fn)
    self.translated_data_dir = translated_data_dir
    self.only_use_en_dev = only_use_en_dev

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "train-en.tsv"))

    examples = []
    if self.translated_data_dir is None:
      for i, line in enumerate(lines):
        guid = "train-%d" % i
        text_a = self.process_text_fn(line[0])
        text_b = self.process_text_fn(line[1])
        label = self.process_text_fn(line[2])
        if label == self.process_text_fn("contradictory"):
          label = self.process_text_fn("contradiction")
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    else:
      for lang in self.supported_languages:
        lines = self._read_tsv(
            os.path.join(self.translated_data_dir, "translate-train",
                         f"en-{lang}-translated.tsv"))
        for i, line in enumerate(lines):
          guid = f"train-{lang}-{i}"
          text_a = self.process_text_fn(line[2])
          text_b = self.process_text_fn(line[3])
          label = self.process_text_fn(line[4])
          if label == self.process_text_fn("contradictory"):
            label = self.process_text_fn("contradiction")
          examples.append(
              InputExample(
                  guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    examples = []
    if self.only_use_en_dev:
      lines = self._read_tsv(os.path.join(data_dir, "dev-en.tsv"))
      for i, line in enumerate(lines):
        guid = "dev-%d" % i
        text_a = self.process_text_fn(line[0])
        text_b = self.process_text_fn(line[1])
        label = self.process_text_fn(line[2])
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    else:
      for lang in self.supported_languages:
        lines = self._read_tsv(os.path.join(data_dir, f"dev-{lang}.tsv"))
        for i, line in enumerate(lines):
          guid = f"dev-{lang}-{i}"
          text_a = self.process_text_fn(line[0])
          text_b = self.process_text_fn(line[1])
          label = self.process_text_fn(line[2])
          if label == self.process_text_fn("contradictory"):
            label = self.process_text_fn("contradiction")
          examples.append(
              InputExample(
                  guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_test_examples(self, data_dir):
    """See base class."""
    examples_by_lang = {}
    for lang in self.supported_languages:
      examples_by_lang[lang] = []
      lines = self._read_tsv(os.path.join(data_dir, f"test-{lang}.tsv"))
      for i, line in enumerate(lines):
        guid = f"test-{lang}-{i}"
        text_a = self.process_text_fn(line[0])
        text_b = self.process_text_fn(line[1])
        label = "contradiction"
        examples_by_lang[lang].append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    if self.translated_data_dir is not None:
      for lang in self.supported_languages:
        if lang == "en":
          continue
        examples_by_lang[f"{lang}-en"] = []
        lines = self._read_tsv(
            os.path.join(self.translated_data_dir, "translate-test",
                         f"test-{lang}-en-translated.tsv"))
        for i, line in enumerate(lines):
          guid = f"test-{lang}-en-{i}"
          text_a = self.process_text_fn(line[2])
          text_b = self.process_text_fn(line[3])
          label = "contradiction"
          examples_by_lang[f"{lang}-en"].append(
              InputExample(
                  guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples_by_lang

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "XTREME-XNLI"


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  label_map = {}
  if label_list:
    for (i, label) in enumerate(label_list):
      label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  seg_id_a = 0
  seg_id_b = 1
  seg_id_cls = 0
  seg_id_pad = 0

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(seg_id_cls)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(seg_id_a)
  tokens.append("[SEP]")
  segment_ids.append(seg_id_a)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(seg_id_b)
    tokens.append("[SEP]")
    segment_ids.append(seg_id_b)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(seg_id_pad)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label] if label_map else example.label
  if ex_index < 5:
    logging.info("*** Example ***")
    logging.info("guid: %s", (example.guid))
    logging.info("tokens: %s",
                 " ".join([tokenization.printable_text(x) for x in tokens]))
    logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
    logging.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
    logging.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
    logging.info("label: %s (id = %s)", example.label, str(label_id))
    logging.info("weight: %s", example.weight)
    logging.info("example_id: %s", example.example_id)

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True,
      weight=example.weight,
      example_id=example.example_id)

  return feature


class AXgProcessor(DataProcessor):
  """Processor for the AXg dataset (SuperGLUE diagnostics dataset)."""

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_jsonl(os.path.join(data_dir, "AX-g.jsonl")), "test")

  def get_labels(self):
    """See base class."""
    return ["entailment", "not_entailment"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "AXg"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training/dev/test sets."""
    examples = []
    for line in lines:
      guid = "%s-%s" % (set_type, self.process_text_fn(str(line["idx"])))
      text_a = self.process_text_fn(line["premise"])
      text_b = self.process_text_fn(line["hypothesis"])
      label = self.process_text_fn(line["label"])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class SuperGLUERTEProcessor(DataProcessor):
  """Processor for the RTE dataset (SuperGLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

  def get_labels(self):
    """See base class."""
    # All datasets are converted to 2-class split, where for 3-class datasets we
    # collapse neutral and contradiction into not_entailment.
    return ["entailment", "not_entailment"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "RTESuperGLUE"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training/dev/test sets."""
    examples = []
    for i, line in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = self.process_text_fn(line["premise"])
      text_b = self.process_text_fn(line["hypothesis"])
      if set_type == "test":
        label = "entailment"
      else:
        label = self.process_text_fn(line["label"])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def file_based_convert_examples_to_features(examples,
                                            label_list,
                                            max_seq_length,
                                            tokenizer,
                                            output_file,
                                            label_type=None):
  """Convert a set of `InputExample`s to a TFRecord file."""

  tf.io.gfile.makedirs(os.path.dirname(output_file))
  writer = tf.io.TFRecordWriter(output_file)

  for ex_index, example in enumerate(examples):
    if ex_index % 10000 == 0:
      logging.info("Writing example %d of %d", ex_index, len(examples))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    if label_type is not None and label_type == float:
      features["label_ids"] = create_float_feature([feature.label_id])
    elif feature.label_id is not None:
      features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])
    if feature.weight is not None:
      features["weight"] = create_float_feature([feature.weight])
    if feature.example_id is not None:
      features["example_id"] = create_int_feature([feature.example_id])
    else:
      features["example_id"] = create_int_feature([ex_index])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def generate_tf_record_from_data_file(processor,
                                      data_dir,
                                      tokenizer,
                                      train_data_output_path=None,
                                      eval_data_output_path=None,
                                      test_data_output_path=None,
                                      max_seq_length=128):
  """Generates and saves training data into a tf record file.

  Args:
      processor: Input processor object to be used for generating data. Subclass
        of `DataProcessor`.
      data_dir: Directory that contains train/eval/test data to process.
      tokenizer: The tokenizer to be applied on the data.
      train_data_output_path: Output to which processed tf record for training
        will be saved.
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
  assert train_data_output_path or eval_data_output_path

  label_list = processor.get_labels()
  label_type = getattr(processor, "label_type", None)
  is_regression = getattr(processor, "is_regression", False)
  has_sample_weights = getattr(processor, "weight_key", False)

  num_training_data = 0
  if train_data_output_path:
    train_input_data_examples = processor.get_train_examples(data_dir)
    file_based_convert_examples_to_features(train_input_data_examples,
                                            label_list, max_seq_length,
                                            tokenizer, train_data_output_path,
                                            label_type)
    num_training_data = len(train_input_data_examples)

  if eval_data_output_path:
    eval_input_data_examples = processor.get_dev_examples(data_dir)
    file_based_convert_examples_to_features(eval_input_data_examples,
                                            label_list, max_seq_length,
                                            tokenizer, eval_data_output_path,
                                            label_type)

  meta_data = {
      "processor_type": processor.get_processor_name(),
      "train_data_size": num_training_data,
      "max_seq_length": max_seq_length,
  }

  if test_data_output_path:
    test_input_data_examples = processor.get_test_examples(data_dir)
    if isinstance(test_input_data_examples, dict):
      for language, examples in test_input_data_examples.items():
        file_based_convert_examples_to_features(
            examples, label_list, max_seq_length, tokenizer,
            test_data_output_path.format(language), label_type)
        meta_data["test_{}_data_size".format(language)] = len(examples)
    else:
      file_based_convert_examples_to_features(test_input_data_examples,
                                              label_list, max_seq_length,
                                              tokenizer, test_data_output_path,
                                              label_type)
      meta_data["test_data_size"] = len(test_input_data_examples)

  if is_regression:
    meta_data["task_type"] = "bert_regression"
    meta_data["label_type"] = {int: "int", float: "float"}[label_type]
  else:
    meta_data["task_type"] = "bert_classification"
    meta_data["num_labels"] = len(processor.get_labels())
  if has_sample_weights:
    meta_data["has_sample_weights"] = True

  if eval_data_output_path:
    meta_data["eval_data_size"] = len(eval_input_data_examples)

  return meta_data
