# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Create TFRecord files of SequenceExample protos from dataset.

Constructs 3 datasets:
  1. Labeled data for the LSTM classification model, optionally with label gain.
     "*_classification.tfrecords" (for both unidirectional and bidirectional
     models).
  2. Data for the unsupervised LM-LSTM model that predicts the next token.
     "*_lm.tfrecords" (generates forward and reverse data).
  3. Data for the unsupervised SA-LSTM model that uses Seq2Seq.
     "*_sa.tfrecords".
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import string

# Dependency imports

import tensorflow as tf

from adversarial_text.data import data_utils
from adversarial_text.data import document_generators

data = data_utils
flags = tf.app.flags
FLAGS = flags.FLAGS

# Flags for input data are in document_generators.py
flags.DEFINE_string('vocab_file', '', 'Path to the vocabulary file. Defaults '
                    'to FLAGS.output_dir/vocab.txt.')
flags.DEFINE_string('output_dir', '', 'Path to save tfrecords.')

# Config
flags.DEFINE_boolean('label_gain', False,
                     'Enable linear label gain. If True, sentiment label will '
                     'be included at each timestep with linear weight '
                     'increase.')


def build_shuffling_tf_record_writer(fname):
  return data.ShufflingTFRecordWriter(os.path.join(FLAGS.output_dir, fname))


def build_tf_record_writer(fname):
  return tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_dir, fname))


def build_input_sequence(doc, vocab_ids):
  """Builds input sequence from file.

  Splits lines on whitespace. Treats punctuation as whitespace. For word-level
  sequences, only keeps terms that are in the vocab.

  Terms are added as token in the SequenceExample.  The EOS_TOKEN is also
  appended. Label and weight features are set to 0.

  Args:
    doc: Document (defined in `document_generators`) from which to build the
      sequence.
    vocab_ids: dict<term, id>.

  Returns:
    SequenceExampleWrapper.
  """
  seq = data.SequenceWrapper()
  for token in document_generators.tokens(doc):
    if token in vocab_ids:
      seq.add_timestep().set_token(vocab_ids[token])

  # Add EOS token to end
  seq.add_timestep().set_token(vocab_ids[data.EOS_TOKEN])

  return seq


def make_vocab_ids(vocab_filename):
  if FLAGS.output_char:
    ret = dict([(char, i) for i, char in enumerate(string.printable)])
    ret[data.EOS_TOKEN] = len(string.printable)
    return ret
  else:
    with open(vocab_filename) as vocab_f:
      return dict([(line.strip(), i) for i, line in enumerate(vocab_f)])


def generate_training_data(vocab_ids, writer_lm_all, writer_seq_ae_all):
  """Generates training data."""

  # Construct training data writers
  writer_lm = build_shuffling_tf_record_writer(data.TRAIN_LM)
  writer_seq_ae = build_shuffling_tf_record_writer(data.TRAIN_SA)
  writer_class = build_shuffling_tf_record_writer(data.TRAIN_CLASS)
  writer_valid_class = build_tf_record_writer(data.VALID_CLASS)
  writer_rev_lm = build_shuffling_tf_record_writer(data.TRAIN_REV_LM)
  writer_bd_class = build_shuffling_tf_record_writer(data.TRAIN_BD_CLASS)
  writer_bd_valid_class = build_shuffling_tf_record_writer(data.VALID_BD_CLASS)

  for doc in document_generators.documents(
      dataset='train', include_unlabeled=True, include_validation=True):
    input_seq = build_input_sequence(doc, vocab_ids)
    if len(input_seq) < 2:
      continue
    rev_seq = data.build_reverse_sequence(input_seq)
    lm_seq = data.build_lm_sequence(input_seq)
    rev_lm_seq = data.build_lm_sequence(rev_seq)
    seq_ae_seq = data.build_seq_ae_sequence(input_seq)
    if doc.label is not None:
      # Used for sentiment classification.
      label_seq = data.build_labeled_sequence(
          input_seq,
          doc.label,
          label_gain=(FLAGS.label_gain and not doc.is_validation))
      bd_label_seq = data.build_labeled_sequence(
          data.build_bidirectional_seq(input_seq, rev_seq),
          doc.label,
          label_gain=(FLAGS.label_gain and not doc.is_validation))
      class_writer = writer_valid_class if doc.is_validation else writer_class
      bd_class_writer = (writer_bd_valid_class
                         if doc.is_validation else writer_bd_class)
      class_writer.write(label_seq.seq.SerializeToString())
      bd_class_writer.write(bd_label_seq.seq.SerializeToString())

    # Write
    lm_seq_ser = lm_seq.seq.SerializeToString()
    seq_ae_seq_ser = seq_ae_seq.seq.SerializeToString()
    writer_lm_all.write(lm_seq_ser)
    writer_seq_ae_all.write(seq_ae_seq_ser)
    if not doc.is_validation:
      writer_lm.write(lm_seq_ser)
      writer_rev_lm.write(rev_lm_seq.seq.SerializeToString())
      writer_seq_ae.write(seq_ae_seq_ser)

  # Close writers
  writer_lm.close()
  writer_seq_ae.close()
  writer_class.close()
  writer_valid_class.close()
  writer_rev_lm.close()
  writer_bd_class.close()
  writer_bd_valid_class.close()


def generate_test_data(vocab_ids, writer_lm_all, writer_seq_ae_all):
  """Generates test data."""
  # Construct test data writers
  writer_lm = build_shuffling_tf_record_writer(data.TEST_LM)
  writer_rev_lm = build_shuffling_tf_record_writer(data.TEST_REV_LM)
  writer_seq_ae = build_shuffling_tf_record_writer(data.TEST_SA)
  writer_class = build_tf_record_writer(data.TEST_CLASS)
  writer_bd_class = build_shuffling_tf_record_writer(data.TEST_BD_CLASS)

  for doc in document_generators.documents(
      dataset='test', include_unlabeled=False, include_validation=True):
    input_seq = build_input_sequence(doc, vocab_ids)
    if len(input_seq) < 2:
      continue
    rev_seq = data.build_reverse_sequence(input_seq)
    lm_seq = data.build_lm_sequence(input_seq)
    rev_lm_seq = data.build_lm_sequence(rev_seq)
    seq_ae_seq = data.build_seq_ae_sequence(input_seq)
    label_seq = data.build_labeled_sequence(input_seq, doc.label)
    bd_label_seq = data.build_labeled_sequence(
        data.build_bidirectional_seq(input_seq, rev_seq), doc.label)

    # Write
    writer_class.write(label_seq.seq.SerializeToString())
    writer_bd_class.write(bd_label_seq.seq.SerializeToString())
    lm_seq_ser = lm_seq.seq.SerializeToString()
    seq_ae_seq_ser = seq_ae_seq.seq.SerializeToString()
    writer_lm.write(lm_seq_ser)
    writer_rev_lm.write(rev_lm_seq.seq.SerializeToString())
    writer_seq_ae.write(seq_ae_seq_ser)
    writer_lm_all.write(lm_seq_ser)
    writer_seq_ae_all.write(seq_ae_seq_ser)

  # Close test writers
  writer_lm.close()
  writer_rev_lm.close()
  writer_seq_ae.close()
  writer_class.close()
  writer_bd_class.close()


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info('Assigning vocabulary ids...')
  vocab_ids = make_vocab_ids(
      FLAGS.vocab_file or os.path.join(FLAGS.output_dir, 'vocab.txt'))

  with build_shuffling_tf_record_writer(data.ALL_LM) as writer_lm_all:
    with build_shuffling_tf_record_writer(data.ALL_SA) as writer_seq_ae_all:

      tf.logging.info('Generating training data...')
      generate_training_data(vocab_ids, writer_lm_all, writer_seq_ae_all)

      tf.logging.info('Generating test data...')
      generate_test_data(vocab_ids, writer_lm_all, writer_seq_ae_all)


if __name__ == '__main__':
  tf.app.run()
