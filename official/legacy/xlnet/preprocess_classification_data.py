# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Script to pre-process classification data into tfrecords."""

import collections
import csv
import os

# Import libraries
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

import sentencepiece as spm
from official.legacy.xlnet import classifier_utils
from official.legacy.xlnet import preprocess_utils


flags.DEFINE_bool(
    "overwrite_data",
    default=False,
    help="If False, will use cached data if available.")
flags.DEFINE_string("output_dir", default="", help="Output dir for TF records.")
flags.DEFINE_string(
    "spiece_model_file", default="", help="Sentence Piece model path.")
flags.DEFINE_string("data_dir", default="", help="Directory for input data.")

# task specific
flags.DEFINE_string("eval_split", default="dev", help="could be dev or test")
flags.DEFINE_string("task_name", default=None, help="Task name")
flags.DEFINE_integer(
    "eval_batch_size", default=64, help="batch size for evaluation")
flags.DEFINE_integer("max_seq_length", default=128, help="Max sequence length")
flags.DEFINE_integer(
    "num_passes",
    default=1,
    help="Num passes for processing training data. "
    "This is use to batch data without loss for TPUs.")
flags.DEFINE_bool("uncased", default=False, help="Use uncased.")
flags.DEFINE_bool(
    "is_regression", default=False, help="Whether it's a regression task.")
flags.DEFINE_bool(
    "use_bert_format",
    default=False,
    help="Whether to use BERT format to arrange input data.")

FLAGS = flags.FLAGS


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

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

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.io.gfile.GFile(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        # pylint: disable=g-explicit-length-test
        if len(line) == 0:
          continue
        lines.append(line)
      return lines


class GLUEProcessor(DataProcessor):
  """GLUEProcessor."""

  def __init__(self):
    self.train_file = "train.tsv"
    self.dev_file = "dev.tsv"
    self.test_file = "test.tsv"
    self.label_column = None
    self.text_a_column = None
    self.text_b_column = None
    self.contains_header = True
    self.test_text_a_column = None
    self.test_text_b_column = None
    self.test_contains_header = True

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, self.train_file)), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, self.dev_file)), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    if self.test_text_a_column is None:
      self.test_text_a_column = self.text_a_column
    if self.test_text_b_column is None:
      self.test_text_b_column = self.text_b_column

    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, self.test_file)), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0 and self.contains_header and set_type != "test":
        continue
      if i == 0 and self.test_contains_header and set_type == "test":
        continue
      guid = "%s-%s" % (set_type, i)

      a_column = (
          self.text_a_column if set_type != "test" else self.test_text_a_column)
      b_column = (
          self.text_b_column if set_type != "test" else self.test_text_b_column)

      # there are some incomplete lines in QNLI
      if len(line) <= a_column:
        logging.warning("Incomplete line, ignored.")
        continue
      text_a = line[a_column]

      if b_column is not None:
        if len(line) <= b_column:
          logging.warning("Incomplete line, ignored.")
          continue
        text_b = line[b_column]
      else:
        text_b = None

      if set_type == "test":
        label = self.get_labels()[0]
      else:
        if len(line) <= self.label_column:
          logging.warning("Incomplete line, ignored.")
          continue
        label = line[self.label_column]
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class Yelp5Processor(DataProcessor):
  """Yelp5Processor."""

  def get_train_examples(self, data_dir):
    return self._create_examples(os.path.join(data_dir, "train.csv"))

  def get_dev_examples(self, data_dir):
    return self._create_examples(os.path.join(data_dir, "test.csv"))

  def get_labels(self):
    """See base class."""
    return ["1", "2", "3", "4", "5"]

  def _create_examples(self, input_file):
    """Creates examples for the training and dev sets."""
    examples = []
    with tf.io.gfile.GFile(input_file) as f:
      reader = csv.reader(f)
      for i, line in enumerate(reader):

        label = line[0]
        text_a = line[1].replace('""', '"').replace('\\"', '"')
        examples.append(
            InputExample(guid=str(i), text_a=text_a, text_b=None, label=label))
    return examples


class ImdbProcessor(DataProcessor):
  """ImdbProcessor."""

  def get_labels(self):
    return ["neg", "pos"]

  def get_train_examples(self, data_dir):
    return self._create_examples(os.path.join(data_dir, "train"))

  def get_dev_examples(self, data_dir):
    return self._create_examples(os.path.join(data_dir, "test"))

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
        with tf.io.gfile.GFile(path) as f:
          text = f.read().strip().replace("<br />", " ")
        examples.append(
            InputExample(
                guid="unused_id", text_a=text, text_b=None, label=label))
    return examples


class MnliMatchedProcessor(GLUEProcessor):
  """MnliMatchedProcessor."""

  def __init__(self):
    super(MnliMatchedProcessor, self).__init__()
    self.dev_file = "dev_matched.tsv"
    self.test_file = "test_matched.tsv"
    self.label_column = -1
    self.text_a_column = 8
    self.text_b_column = 9

  def get_labels(self):
    return ["contradiction", "entailment", "neutral"]


class MnliMismatchedProcessor(MnliMatchedProcessor):

  def __init__(self):
    super(MnliMismatchedProcessor, self).__init__()
    self.dev_file = "dev_mismatched.tsv"
    self.test_file = "test_mismatched.tsv"


class StsbProcessor(GLUEProcessor):
  """StsbProcessor."""

  def __init__(self):
    super(StsbProcessor, self).__init__()
    self.label_column = 9
    self.text_a_column = 7
    self.text_b_column = 8

  def get_labels(self):
    return [0.0]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0 and self.contains_header and set_type != "test":
        continue
      if i == 0 and self.test_contains_header and set_type == "test":
        continue
      guid = "%s-%s" % (set_type, i)

      a_column = (
          self.text_a_column if set_type != "test" else self.test_text_a_column)
      b_column = (
          self.text_b_column if set_type != "test" else self.test_text_b_column)

      # there are some incomplete lines in QNLI
      if len(line) <= a_column:
        logging.warning("Incomplete line, ignored.")
        continue
      text_a = line[a_column]

      if b_column is not None:
        if len(line) <= b_column:
          logging.warning("Incomplete line, ignored.")
          continue
        text_b = line[b_column]
      else:
        text_b = None

      if set_type == "test":
        label = self.get_labels()[0]
      else:
        if len(line) <= self.label_column:
          logging.warning("Incomplete line, ignored.")
          continue
        label = float(line[self.label_column])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    return examples


def file_based_convert_examples_to_features(examples,
                                            label_list,
                                            max_seq_length,
                                            tokenize_fn,
                                            output_file,
                                            num_passes=1):
  """Convert a set of `InputExample`s to a TFRecord file."""

  # do not create duplicated records
  if tf.io.gfile.exists(output_file) and not FLAGS.overwrite_data:
    logging.info("Do not overwrite tfrecord %s exists.", output_file)
    return

  logging.info("Create new tfrecord %s.", output_file)

  writer = tf.io.TFRecordWriter(output_file)

  examples *= num_passes

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logging.info("Writing example %d of %d", ex_index, len(examples))

    feature = classifier_utils.convert_single_example(ex_index, example,
                                                      label_list,
                                                      max_seq_length,
                                                      tokenize_fn,
                                                      FLAGS.use_bert_format)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_float_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    if label_list is not None:
      features["label_ids"] = create_int_feature([feature.label_id])
    else:
      features["label_ids"] = create_float_feature([float(feature.label_id)])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def main(_):
  logging.set_verbosity(logging.INFO)
  processors = {
      "mnli_matched": MnliMatchedProcessor,
      "mnli_mismatched": MnliMismatchedProcessor,
      "sts-b": StsbProcessor,
      "imdb": ImdbProcessor,
      "yelp5": Yelp5Processor
  }

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()
  label_list = processor.get_labels() if not FLAGS.is_regression else None

  sp = spm.SentencePieceProcessor()
  sp.Load(FLAGS.spiece_model_file)

  def tokenize_fn(text):
    text = preprocess_utils.preprocess_text(text, lower=FLAGS.uncased)
    return preprocess_utils.encode_ids(sp, text)

  spm_basename = os.path.basename(FLAGS.spiece_model_file)

  train_file_base = "{}.len-{}.train.tf_record".format(spm_basename,
                                                       FLAGS.max_seq_length)
  train_file = os.path.join(FLAGS.output_dir, train_file_base)
  logging.info("Use tfrecord file %s", train_file)

  train_examples = processor.get_train_examples(FLAGS.data_dir)
  np.random.shuffle(train_examples)
  logging.info("Num of train samples: %d", len(train_examples))

  file_based_convert_examples_to_features(train_examples, label_list,
                                          FLAGS.max_seq_length, tokenize_fn,
                                          train_file, FLAGS.num_passes)
  if FLAGS.eval_split == "dev":
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
  else:
    eval_examples = processor.get_test_examples(FLAGS.data_dir)

  logging.info("Num of eval samples: %d", len(eval_examples))

  # TPU requires a fixed batch size for all batches, therefore the number
  # of examples must be a multiple of the batch size, or else examples
  # will get dropped. So we pad with fake examples which are ignored
  # later on. These do NOT count towards the metric (all tf.metrics
  # support a per-instance weight, and these get a weight of 0.0).
  #
  # Modified in XL: We also adopt the same mechanism for GPUs.
  while len(eval_examples) % FLAGS.eval_batch_size != 0:
    eval_examples.append(classifier_utils.PaddingInputExample())

  eval_file_base = "{}.len-{}.{}.eval.tf_record".format(spm_basename,
                                                        FLAGS.max_seq_length,
                                                        FLAGS.eval_split)
  eval_file = os.path.join(FLAGS.output_dir, eval_file_base)

  file_based_convert_examples_to_features(eval_examples, label_list,
                                          FLAGS.max_seq_length, tokenize_fn,
                                          eval_file)


if __name__ == "__main__":
  app.run(main)
