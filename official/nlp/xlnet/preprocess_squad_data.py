# coding=utf-8
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
"""Script to pre-process SQUAD data into tfrecords."""

import os
import random

# Import libraries
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import sentencepiece as spm
from official.nlp.xlnet import squad_utils

flags.DEFINE_integer(
    "num_proc", default=1, help="Number of preprocessing processes.")
flags.DEFINE_integer("proc_id", default=0, help="Process id for preprocessing.")

# I/O paths
flags.DEFINE_string("output_dir", default="", help="Output dir for TF records.")
flags.DEFINE_string(
    "spiece_model_file", default="", help="Sentence Piece model path.")
flags.DEFINE_string("train_file", default="", help="Path of train file.")
flags.DEFINE_string("predict_file", default="", help="Path of prediction file.")

# Data preprocessing config
flags.DEFINE_integer("max_seq_length", default=512, help="Max sequence length")
flags.DEFINE_integer("max_query_length", default=64, help="Max query length")
flags.DEFINE_integer("doc_stride", default=128, help="Doc stride")
flags.DEFINE_bool("uncased", default=False, help="Use uncased data.")
flags.DEFINE_bool(
    "create_train_data", default=True, help="Whether to create training data.")
flags.DEFINE_bool(
    "create_eval_data", default=False, help="Whether to create eval data.")

FLAGS = flags.FLAGS


def preprocess():
  """Preprocesses SQUAD data."""
  sp_model = spm.SentencePieceProcessor()
  sp_model.Load(FLAGS.spiece_model_file)
  spm_basename = os.path.basename(FLAGS.spiece_model_file)
  if FLAGS.create_train_data:
    train_rec_file = os.path.join(
        FLAGS.output_dir,
        "{}.{}.slen-{}.qlen-{}.train.tf_record".format(spm_basename,
                                                       FLAGS.proc_id,
                                                       FLAGS.max_seq_length,
                                                       FLAGS.max_query_length))

    logging.info("Read examples from %s", FLAGS.train_file)
    train_examples = squad_utils.read_squad_examples(
        FLAGS.train_file, is_training=True)
    train_examples = train_examples[FLAGS.proc_id::FLAGS.num_proc]

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in the `input_fn`.
    random.shuffle(train_examples)
    write_to_logging = "Write to " + train_rec_file
    logging.info(write_to_logging)
    train_writer = squad_utils.FeatureWriter(
        filename=train_rec_file, is_training=True)
    squad_utils.convert_examples_to_features(
        examples=train_examples,
        sp_model=sp_model,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=True,
        output_fn=train_writer.process_feature,
        uncased=FLAGS.uncased)
    train_writer.close()
  if FLAGS.create_eval_data:
    eval_examples = squad_utils.read_squad_examples(
        FLAGS.predict_file, is_training=False)
    squad_utils.create_eval_data(spm_basename, sp_model, eval_examples,
                                 FLAGS.max_seq_length, FLAGS.max_query_length,
                                 FLAGS.doc_stride, FLAGS.uncased,
                                 FLAGS.output_dir)


def main(_):
  logging.set_verbosity(logging.INFO)

  if not tf.io.gfile.exists(FLAGS.output_dir):
    tf.io.gfile.mkdir(FLAGS.output_dir)

  preprocess()


if __name__ == "__main__":
  app.run(main)
