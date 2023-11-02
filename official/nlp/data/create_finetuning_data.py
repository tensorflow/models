# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""BERT finetuning task dataset generator."""

import functools
import json
import os

# Import libraries
from absl import app
from absl import flags
import tensorflow as tf, tf_keras
from official.nlp.data import classifier_data_lib
from official.nlp.data import sentence_retrieval_lib
# word-piece tokenizer based squad_lib
from official.nlp.data import squad_lib as squad_lib_wp
# sentence-piece tokenizer based squad_lib
from official.nlp.data import squad_lib_sp
from official.nlp.data import tagging_data_lib
from official.nlp.tools import tokenization

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "fine_tuning_task_type", "classification",
    ["classification", "regression", "squad", "retrieval", "tagging"],
    "The name of the BERT fine tuning task for which data "
    "will be generated.")

# BERT classification specific flags.
flags.DEFINE_string(
    "input_data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_enum(
    "classification_task_name", "MNLI", [
        "AX", "COLA", "IMDB", "MNLI", "MRPC", "PAWS-X", "QNLI", "QQP", "RTE",
        "SST-2", "STS-B", "WNLI", "XNLI", "XTREME-XNLI", "XTREME-PAWS-X",
        "AX-g", "SUPERGLUE-RTE", "CB", "BoolQ", "WIC"
    ], "The name of the task to train BERT classifier. The "
    "difference between XTREME-XNLI and XNLI is: 1. the format "
    "of input tsv files; 2. the dev set for XTREME is english "
    "only and for XNLI is all languages combined. Same for "
    "PAWS-X.")

# MNLI task-specific flag.
flags.DEFINE_enum("mnli_type", "matched", ["matched", "mismatched"],
                  "The type of MNLI dataset.")

# XNLI task-specific flag.
flags.DEFINE_string(
    "xnli_language", "en",
    "Language of training data for XNLI task. If the value is 'all', the data "
    "of all languages will be used for training.")

# PAWS-X task-specific flag.
flags.DEFINE_string(
    "pawsx_language", "en",
    "Language of training data for PAWS-X task. If the value is 'all', the data "
    "of all languages will be used for training.")

# XTREME classification specific flags. Only used in XtremePawsx and XtremeXnli.
flags.DEFINE_string(
    "translated_input_data_dir", None,
    "The translated input data dir. Should contain the .tsv files (or other "
    "data files) for the task.")

# Retrieval task-specific flags.
flags.DEFINE_enum("retrieval_task_name", "bucc", ["bucc", "tatoeba"],
                  "The name of sentence retrieval task for scoring")

# Tagging task-specific flags.
flags.DEFINE_enum("tagging_task_name", "panx", ["panx", "udpos"],
                  "The name of BERT tagging (token classification) task.")

flags.DEFINE_bool("tagging_only_use_en_train", True,
                  "Whether only use english training data in tagging.")

# BERT Squad task-specific flags.
flags.DEFINE_string(
    "squad_data_file", None,
    "The input data file in for generating training data for BERT squad task.")

flags.DEFINE_string(
    "translated_squad_data_folder", None,
    "The translated data folder for generating training data for BERT squad "
    "task.")

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

flags.DEFINE_bool(
    "xlnet_format", False,
    "If true, then data will be preprocessed in a paragraph, query, class order"
    " instead of the BERT-style class, paragraph, query order.")

# XTREME specific flags.
flags.DEFINE_bool("only_use_en_dev", True, "Whether only use english dev data.")

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
    "tokenization", "WordPiece", ["WordPiece", "SentencePiece"],
    "Specifies the tokenizer implementation, i.e., whether to use WordPiece "
    "or SentencePiece tokenizer. Canonical BERT uses WordPiece tokenizer, "
    "while ALBERT uses SentencePiece tokenizer.")

flags.DEFINE_string(
    "tfds_params", "", "Comma-separated list of TFDS parameter assigments for "
    "generic classfication data import (for more details "
    "see the TfdsProcessor class documentation).")


def generate_classifier_dataset():
  """Generates classifier dataset and returns input meta data."""
  if FLAGS.classification_task_name in [
      "COLA",
      "WNLI",
      "SST-2",
      "MRPC",
      "QQP",
      "STS-B",
      "MNLI",
      "QNLI",
      "RTE",
      "AX",
      "SUPERGLUE-RTE",
      "CB",
      "BoolQ",
      "WIC",
  ]:
    assert not FLAGS.input_data_dir or FLAGS.tfds_params
  else:
    assert (FLAGS.input_data_dir and FLAGS.classification_task_name or
            FLAGS.tfds_params)

  if FLAGS.tokenization == "WordPiece":
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    processor_text_fn = tokenization.convert_to_unicode
  else:
    assert FLAGS.tokenization == "SentencePiece"
    tokenizer = tokenization.FullSentencePieceTokenizer(FLAGS.sp_model_file)
    processor_text_fn = functools.partial(
        tokenization.preprocess_text, lower=FLAGS.do_lower_case)

  if FLAGS.tfds_params:
    processor = classifier_data_lib.TfdsProcessor(
        tfds_params=FLAGS.tfds_params, process_text_fn=processor_text_fn)
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
        "ax":
            classifier_data_lib.AxProcessor,
        "cola":
            classifier_data_lib.ColaProcessor,
        "imdb":
            classifier_data_lib.ImdbProcessor,
        "mnli":
            functools.partial(
                classifier_data_lib.MnliProcessor, mnli_type=FLAGS.mnli_type),
        "mrpc":
            classifier_data_lib.MrpcProcessor,
        "qnli":
            classifier_data_lib.QnliProcessor,
        "qqp":
            classifier_data_lib.QqpProcessor,
        "rte":
            classifier_data_lib.RteProcessor,
        "sst-2":
            classifier_data_lib.SstProcessor,
        "sts-b":
            classifier_data_lib.StsBProcessor,
        "xnli":
            functools.partial(
                classifier_data_lib.XnliProcessor,
                language=FLAGS.xnli_language),
        "paws-x":
            functools.partial(
                classifier_data_lib.PawsxProcessor,
                language=FLAGS.pawsx_language),
        "wnli":
            classifier_data_lib.WnliProcessor,
        "xtreme-xnli":
            functools.partial(
                classifier_data_lib.XtremeXnliProcessor,
                translated_data_dir=FLAGS.translated_input_data_dir,
                only_use_en_dev=FLAGS.only_use_en_dev),
        "xtreme-paws-x":
            functools.partial(
                classifier_data_lib.XtremePawsxProcessor,
                translated_data_dir=FLAGS.translated_input_data_dir,
                only_use_en_dev=FLAGS.only_use_en_dev),
        "ax-g":
            classifier_data_lib.AXgProcessor,
        "superglue-rte":
            classifier_data_lib.SuperGLUERTEProcessor,
        "cb":
            classifier_data_lib.CBProcessor,
        "boolq":
            classifier_data_lib.BoolQProcessor,
        "wic":
            classifier_data_lib.WnliProcessor,
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
  if FLAGS.tokenization == "WordPiece":
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    processor_text_fn = tokenization.convert_to_unicode
  else:
    assert FLAGS.tokenization == "SentencePiece"
    tokenizer = tokenization.FullSentencePieceTokenizer(FLAGS.sp_model_file)
    processor_text_fn = functools.partial(
        tokenization.preprocess_text, lower=FLAGS.do_lower_case)

  if FLAGS.tfds_params:
    processor = classifier_data_lib.TfdsProcessor(
        tfds_params=FLAGS.tfds_params, process_text_fn=processor_text_fn)
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
  if FLAGS.tokenization == "WordPiece":
    return squad_lib_wp.generate_tf_record_from_json_file(
        input_file_path=FLAGS.squad_data_file,
        vocab_file_path=FLAGS.vocab_file,
        output_path=FLAGS.train_data_output_path,
        translated_input_folder=FLAGS.translated_squad_data_folder,
        max_seq_length=FLAGS.max_seq_length,
        do_lower_case=FLAGS.do_lower_case,
        max_query_length=FLAGS.max_query_length,
        doc_stride=FLAGS.doc_stride,
        version_2_with_negative=FLAGS.version_2_with_negative,
        xlnet_format=FLAGS.xlnet_format)
  else:
    assert FLAGS.tokenization == "SentencePiece"
    return squad_lib_sp.generate_tf_record_from_json_file(
        input_file_path=FLAGS.squad_data_file,
        sp_model_file=FLAGS.sp_model_file,
        output_path=FLAGS.train_data_output_path,
        translated_input_folder=FLAGS.translated_squad_data_folder,
        max_seq_length=FLAGS.max_seq_length,
        do_lower_case=FLAGS.do_lower_case,
        max_query_length=FLAGS.max_query_length,
        doc_stride=FLAGS.doc_stride,
        xlnet_format=FLAGS.xlnet_format,
        version_2_with_negative=FLAGS.version_2_with_negative)


def generate_retrieval_dataset():
  """Generate retrieval test and dev dataset and returns input meta data."""
  assert (FLAGS.input_data_dir and FLAGS.retrieval_task_name)
  if FLAGS.tokenization == "WordPiece":
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    processor_text_fn = tokenization.convert_to_unicode
  else:
    assert FLAGS.tokenization == "SentencePiece"
    tokenizer = tokenization.FullSentencePieceTokenizer(FLAGS.sp_model_file)
    processor_text_fn = functools.partial(
        tokenization.preprocess_text, lower=FLAGS.do_lower_case)

  processors = {
      "bucc": sentence_retrieval_lib.BuccProcessor,
      "tatoeba": sentence_retrieval_lib.TatoebaProcessor,
  }

  task_name = FLAGS.retrieval_task_name.lower()
  if task_name not in processors:
    raise ValueError("Task not found: %s" % task_name)

  processor = processors[task_name](process_text_fn=processor_text_fn)

  return sentence_retrieval_lib.generate_sentence_retrevial_tf_record(
      processor, FLAGS.input_data_dir, tokenizer, FLAGS.eval_data_output_path,
      FLAGS.test_data_output_path, FLAGS.max_seq_length)


def generate_tagging_dataset():
  """Generates tagging dataset."""
  processors = {
      "panx":
          functools.partial(
              tagging_data_lib.PanxProcessor,
              only_use_en_train=FLAGS.tagging_only_use_en_train,
              only_use_en_dev=FLAGS.only_use_en_dev),
      "udpos":
          functools.partial(
              tagging_data_lib.UdposProcessor,
              only_use_en_train=FLAGS.tagging_only_use_en_train,
              only_use_en_dev=FLAGS.only_use_en_dev),
  }
  task_name = FLAGS.tagging_task_name.lower()
  if task_name not in processors:
    raise ValueError("Task not found: %s" % task_name)

  if FLAGS.tokenization == "WordPiece":
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    processor_text_fn = tokenization.convert_to_unicode
  elif FLAGS.tokenization == "SentencePiece":
    tokenizer = tokenization.FullSentencePieceTokenizer(FLAGS.sp_model_file)
    processor_text_fn = functools.partial(
        tokenization.preprocess_text, lower=FLAGS.do_lower_case)
  else:
    raise ValueError("Unsupported tokenization: %s" % FLAGS.tokenization)

  processor = processors[task_name]()
  return tagging_data_lib.generate_tf_record_from_data_file(
      processor, FLAGS.input_data_dir, tokenizer, FLAGS.max_seq_length,
      FLAGS.train_data_output_path, FLAGS.eval_data_output_path,
      FLAGS.test_data_output_path, processor_text_fn)


def main(_):
  if FLAGS.tokenization == "WordPiece":
    if not FLAGS.vocab_file:
      raise ValueError(
          "FLAG vocab_file for word-piece tokenizer is not specified.")
  else:
    assert FLAGS.tokenization == "SentencePiece"
    if not FLAGS.sp_model_file:
      raise ValueError(
          "FLAG sp_model_file for sentence-piece tokenizer is not specified.")

  if FLAGS.fine_tuning_task_type != "retrieval":
    flags.mark_flag_as_required("train_data_output_path")

  if FLAGS.fine_tuning_task_type == "classification":
    input_meta_data = generate_classifier_dataset()
  elif FLAGS.fine_tuning_task_type == "regression":
    input_meta_data = generate_regression_dataset()
  elif FLAGS.fine_tuning_task_type == "retrieval":
    input_meta_data = generate_retrieval_dataset()
  elif FLAGS.fine_tuning_task_type == "squad":
    input_meta_data = generate_squad_dataset()
  else:
    assert FLAGS.fine_tuning_task_type == "tagging"
    input_meta_data = generate_tagging_dataset()

  tf.io.gfile.makedirs(os.path.dirname(FLAGS.meta_data_file_path))
  with tf.io.gfile.GFile(FLAGS.meta_data_file_path, "w") as writer:
    writer.write(json.dumps(input_meta_data, indent=4) + "\n")


if __name__ == "__main__":
  flags.mark_flag_as_required("meta_data_file_path")
  app.run(main)
