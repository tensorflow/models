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
"""BERT finetuning task dataset generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os

from absl import app
from absl import flags
import tensorflow as tf
from official.nlp.bert import tokenization
from official.nlp.data import classifier_data_lib
# word-piece tokenizer based squad_lib
from official.nlp.data import squad_lib as squad_lib_wp
# sentence-piece tokenizer based squad_lib
from official.nlp.data import squad_lib_sp

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "fine_tuning_task_type", "classification",
    ["classification", "regression", "squad"],
    "The name of the BERT fine tuning task for which data "
    "will be generated..")

# BERT classification specific flags.
flags.DEFINE_string(
    "input_data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_enum("classification_task_name", "MNLI",
                  ["COLA", "MNLI", "MRPC", "QNLI", "QQP", "SST-2", "XNLI",
                   "PAWS-X"],
                  "The name of the task to train BERT classifier.")

# XNLI task specific flag.
flags.DEFINE_string(
    "xnli_language", "en",
    "Language of training data for XNIL task. If the value is 'all', the data "
    "of all languages will be used for training.")

# PAWS-X task specific flag.
flags.DEFINE_string(
    "pawsx_language", "en",
    "Language of trainig data for PAWS-X task. If the value is 'all', the data "
    "of all languages will be used for training.")

# BERT Squad task specific flags.
flags.DEFINE_string(
    "squad_data_file", None,
    "The input data file in for generating training data for BERT squad task.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")

# Shared flags across BERT fine-tuning tasks.
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "train_data_output_path", None,
    "The path in which generated training input data will be written as tf"
    " records.")

flags.DEFINE_string(
    "eval_data_output_path", None,
    "The path in which generated evaluation input data will be written as tf"
    " records.")

flags.DEFINE_string(
    "test_data_output_path", None,
    "The path in which generated test input data will be written as tf"
    " records. If None, do not generate test data. Must be a pattern template"
    " as test_{}.tfrecords if processor has language specific test data.")

flags.DEFINE_string("meta_data_file_path", None,
                    "The path in which input meta data will be written.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string("sp_model_file", "",
                    "The path to the model used by sentence piece tokenizer.")

flags.DEFINE_enum(
    "tokenizer_impl", "word_piece", ["word_piece", "sentence_piece"],
    "Specifies the tokenizer implementation, i.e., whehter to use word_piece "
    "or sentence_piece tokenizer. Canonical BERT uses word_piece tokenizer, "
    "while ALBERT uses sentence_piece tokenizer.")

flags.DEFINE_string("tfds_params", "",
                    "Comma-separated list of TFDS parameter assigments for "
                    "generic classfication data import (for more details "
                    "see the TfdsProcessor class documentation).")


def generate_classifier_dataset():
  """Generates classifier dataset and returns input meta data."""
  assert (FLAGS.input_data_dir and FLAGS.classification_task_name
          or FLAGS.tfds_params)

  if FLAGS.tokenizer_impl == "word_piece":
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    processor_text_fn = tokenization.convert_to_unicode
  else:
    assert FLAGS.tokenizer_impl == "sentence_piece"
    tokenizer = tokenization.FullSentencePieceTokenizer(FLAGS.sp_model_file)
    processor_text_fn = functools.partial(
        tokenization.preprocess_text, lower=FLAGS.do_lower_case)

  if FLAGS.tfds_params:
    processor = classifier_data_lib.TfdsProcessor(
        tfds_params=FLAGS.tfds_params,
        process_text_fn=processor_text_fn)
    return classifier_data_lib.generate_tf_record_from_data_file(
        processor,
        None,
        tokenizer,
        train_data_output_path=FLAGS.train_data_output_path,
        eval_data_output_path=FLAGS.eval_data_output_path,
        test_data_output_path=FLAGS.test_data_output_path,
        max_seq_length=FLAGS.max_seq_length)
  else:
    processors = {
        "cola":
            classifier_data_lib.ColaProcessor,
        "mnli":
            classifier_data_lib.MnliProcessor,
        "mrpc":
            classifier_data_lib.MrpcProcessor,
        "qnli":
            classifier_data_lib.QnliProcessor,
        "qqp": classifier_data_lib.QqpProcessor,
        "sst-2":
            classifier_data_lib.SstProcessor,
        "xnli":
            functools.partial(classifier_data_lib.XnliProcessor,
                              language=FLAGS.xnli_language),
        "paws-x":
            functools.partial(classifier_data_lib.PawsxProcessor,
                              language=FLAGS.pawsx_language)
    }
    task_name = FLAGS.classification_task_name.lower()
    if task_name not in processors:
      raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name](process_text_fn=processor_text_fn)
    return classifier_data_lib.generate_tf_record_from_data_file(
        processor,
        FLAGS.input_data_dir,
        tokenizer,
        train_data_output_path=FLAGS.train_data_output_path,
        eval_data_output_path=FLAGS.eval_data_output_path,
        test_data_output_path=FLAGS.test_data_output_path,
        max_seq_length=FLAGS.max_seq_length)


def generate_regression_dataset():
  """Generates regression dataset and returns input meta data."""
  if FLAGS.tokenizer_impl == "word_piece":
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    processor_text_fn = tokenization.convert_to_unicode
  else:
    assert FLAGS.tokenizer_impl == "sentence_piece"
    tokenizer = tokenization.FullSentencePieceTokenizer(FLAGS.sp_model_file)
    processor_text_fn = functools.partial(
        tokenization.preprocess_text, lower=FLAGS.do_lower_case)

  if FLAGS.tfds_params:
    processor = classifier_data_lib.TfdsProcessor(
        tfds_params=FLAGS.tfds_params,
        process_text_fn=processor_text_fn)
    return classifier_data_lib.generate_tf_record_from_data_file(
        processor,
        None,
        tokenizer,
        train_data_output_path=FLAGS.train_data_output_path,
        eval_data_output_path=FLAGS.eval_data_output_path,
        test_data_output_path=FLAGS.test_data_output_path,
        max_seq_length=FLAGS.max_seq_length)
  else:
    raise ValueError("No data processor found for the given regression task.")


def generate_squad_dataset():
  """Generates squad training dataset and returns input meta data."""
  assert FLAGS.squad_data_file
  if FLAGS.tokenizer_impl == "word_piece":
    return squad_lib_wp.generate_tf_record_from_json_file(
        FLAGS.squad_data_file, FLAGS.vocab_file, FLAGS.train_data_output_path,
        FLAGS.max_seq_length, FLAGS.do_lower_case, FLAGS.max_query_length,
        FLAGS.doc_stride, FLAGS.version_2_with_negative)
  else:
    assert FLAGS.tokenizer_impl == "sentence_piece"
    return squad_lib_sp.generate_tf_record_from_json_file(
        FLAGS.squad_data_file, FLAGS.sp_model_file,
        FLAGS.train_data_output_path, FLAGS.max_seq_length, FLAGS.do_lower_case,
        FLAGS.max_query_length, FLAGS.doc_stride, FLAGS.version_2_with_negative)


def main(_):
  if FLAGS.tokenizer_impl == "word_piece":
    if not FLAGS.vocab_file:
      raise ValueError(
          "FLAG vocab_file for word-piece tokenizer is not specified.")
  else:
    assert FLAGS.tokenizer_impl == "sentence_piece"
    if not FLAGS.sp_model_file:
      raise ValueError(
          "FLAG sp_model_file for sentence-piece tokenizer is not specified.")

  if FLAGS.fine_tuning_task_type == "classification":
    input_meta_data = generate_classifier_dataset()
  elif FLAGS.fine_tuning_task_type == "regression":
    input_meta_data = generate_regression_dataset()
  else:
    input_meta_data = generate_squad_dataset()

  tf.io.gfile.makedirs(os.path.dirname(FLAGS.meta_data_file_path))
  with tf.io.gfile.GFile(FLAGS.meta_data_file_path, "w") as writer:
    writer.write(json.dumps(input_meta_data, indent=4) + "\n")


if __name__ == "__main__":
  flags.mark_flag_as_required("train_data_output_path")
  flags.mark_flag_as_required("meta_data_file_path")
  app.run(main)
