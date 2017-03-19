# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Converts a set of text files to TFRecord format with Example protos.

Each Example proto in the output contains the following fields:

  decode_pre: list of int64 ids corresponding to the "previous" sentence.
  encode: list of int64 ids corresponding to the "current" sentence.
  decode_post: list of int64 ids corresponding to the "post" sentence.

In addition, the following files are generated:

  vocab.txt: List of "<word> <id>" pairs, where <id> is the integer
             encoding of <word> in the Example protos.
  word_counts.txt: List of "<word> <count>" pairs, where <count> is the number
                   of occurrences of <word> in the input files.

The vocabulary of word ids is constructed from the top --num_words by word
count. All other words get the <unk> word id.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os


import numpy as np
import tensorflow as tf

from skip_thoughts.data import special_words

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_files", None,
                       "Comma-separated list of globs matching the input "
                       "files. The format of the input files is assumed to be "
                       "a list of newline-separated sentences, where each "
                       "sentence is already tokenized.")

tf.flags.DEFINE_string("vocab_file", "",
                       "(Optional) existing vocab file. Otherwise, a new vocab "
                       "file is created and written to the output directory. "
                       "The file format is a list of newline-separated words, "
                       "where the word id is the corresponding 0-based index "
                       "in the file.")

tf.flags.DEFINE_string("output_dir", None, "Output directory.")

tf.flags.DEFINE_integer("train_output_shards", 100,
                        "Number of output shards for the training set.")

tf.flags.DEFINE_integer("validation_output_shards", 1,
                        "Number of output shards for the validation set.")

tf.flags.DEFINE_integer("num_validation_sentences", 50000,
                        "Number of output shards for the validation set.")

tf.flags.DEFINE_integer("num_words", 20000,
                        "Number of words to include in the output.")

tf.flags.DEFINE_integer("max_sentences", 0,
                        "If > 0, the maximum number of sentences to output.")

tf.flags.DEFINE_integer("max_sentence_length", 30,
                        "If > 0, exclude sentences whose encode, decode_pre OR"
                        "decode_post sentence exceeds this length.")

tf.flags.DEFINE_boolean("add_eos", True,
                        "Whether to add end-of-sentence ids to the output.")

tf.logging.set_verbosity(tf.logging.INFO)


def _build_vocabulary(input_files):
  """Loads or builds the model vocabulary.

  Args:
    input_files: List of pre-tokenized input .txt files.

  Returns:
    vocab: A dictionary of word to id.
  """
  if FLAGS.vocab_file:
    tf.logging.info("Loading existing vocab file.")
    vocab = collections.OrderedDict()
    with tf.gfile.GFile(FLAGS.vocab_file, mode="r") as f:
      for i, line in enumerate(f):
        word = line.decode("utf-8").strip()
        assert word not in vocab, "Attempting to add word twice: %s" % word
        vocab[word] = i
    tf.logging.info("Read vocab of size %d from %s",
                    len(vocab), FLAGS.vocab_file)
    return vocab

  tf.logging.info("Creating vocabulary.")
  num = 0
  wordcount = collections.Counter()
  for input_file in input_files:
    tf.logging.info("Processing file: %s", input_file)
    for sentence in tf.gfile.FastGFile(input_file):
      wordcount.update(sentence.split())

      num += 1
      if num % 1000000 == 0:
        tf.logging.info("Processed %d sentences", num)

  tf.logging.info("Processed %d sentences total", num)

  words = wordcount.keys()
  freqs = wordcount.values()
  sorted_indices = np.argsort(freqs)[::-1]

  vocab = collections.OrderedDict()
  vocab[special_words.EOS] = special_words.EOS_ID
  vocab[special_words.UNK] = special_words.UNK_ID
  for w_id, w_index in enumerate(sorted_indices[0:FLAGS.num_words - 2]):
    vocab[words[w_index]] = w_id + 2  # 0: EOS, 1: UNK.

  tf.logging.info("Created vocab with %d words", len(vocab))

  vocab_file = os.path.join(FLAGS.output_dir, "vocab.txt")
  with tf.gfile.FastGFile(vocab_file, "w") as f:
    f.write("\n".join(vocab.keys()))
  tf.logging.info("Wrote vocab file to %s", vocab_file)

  word_counts_file = os.path.join(FLAGS.output_dir, "word_counts.txt")
  with tf.gfile.FastGFile(word_counts_file, "w") as f:
    for i in sorted_indices:
      f.write("%s %d\n" % (words[i], freqs[i]))
  tf.logging.info("Wrote word counts file to %s", word_counts_file)

  return vocab


def _int64_feature(value):
  """Helper for creating an Int64 Feature."""
  return tf.train.Feature(int64_list=tf.train.Int64List(
      value=[int(v) for v in value]))


def _sentence_to_ids(sentence, vocab):
  """Helper for converting a sentence (list of words) to a list of ids."""
  ids = [vocab.get(w, special_words.UNK_ID) for w in sentence]
  if FLAGS.add_eos:
    ids.append(special_words.EOS_ID)
  return ids


def _create_serialized_example(predecessor, current, successor, vocab):
  """Helper for creating a serialized Example proto."""
  example = tf.train.Example(features=tf.train.Features(feature={
      "decode_pre": _int64_feature(_sentence_to_ids(predecessor, vocab)),
      "encode": _int64_feature(_sentence_to_ids(current, vocab)),
      "decode_post": _int64_feature(_sentence_to_ids(successor, vocab)),
  }))

  return example.SerializeToString()


def _process_input_file(filename, vocab, stats):
  """Processes the sentences in an input file.

  Args:
    filename: Path to a pre-tokenized input .txt file.
    vocab: A dictionary of word to id.
    stats: A Counter object for statistics.

  Returns:
    processed: A list of serialized Example protos
  """
  tf.logging.info("Processing input file: %s", filename)
  processed = []

  predecessor = None  # Predecessor sentence (list of words).
  current = None  # Current sentence (list of words).
  successor = None  # Successor sentence (list of words).

  for successor_str in tf.gfile.FastGFile(filename):
    stats.update(["sentences_seen"])
    successor = successor_str.split()

    # The first 2 sentences per file will be skipped.
    if predecessor and current and successor:
      stats.update(["sentences_considered"])

      # Note that we are going to insert <EOS> later, so we only allow
      # sentences with strictly less than max_sentence_length to pass.
      if FLAGS.max_sentence_length and (
          len(predecessor) >= FLAGS.max_sentence_length or len(current) >=
          FLAGS.max_sentence_length or len(successor) >=
          FLAGS.max_sentence_length):
        stats.update(["sentences_too_long"])
      else:
        serialized = _create_serialized_example(predecessor, current, successor,
                                                vocab)
        processed.append(serialized)
        stats.update(["sentences_output"])

    predecessor = current
    current = successor

    sentences_seen = stats["sentences_seen"]
    sentences_output = stats["sentences_output"]
    if sentences_seen and sentences_seen % 100000 == 0:
      tf.logging.info("Processed %d sentences (%d output)", sentences_seen,
                      sentences_output)
    if FLAGS.max_sentences and sentences_output >= FLAGS.max_sentences:
      break

  tf.logging.info("Completed processing file %s", filename)
  return processed


def _write_shard(filename, dataset, indices):
  """Writes a TFRecord shard."""
  with tf.python_io.TFRecordWriter(filename) as writer:
    for j in indices:
      writer.write(dataset[j])


def _write_dataset(name, dataset, indices, num_shards):
  """Writes a sharded TFRecord dataset.

  Args:
    name: Name of the dataset (e.g. "train").
    dataset: List of serialized Example protos.
    indices: List of indices of 'dataset' to be written.
    num_shards: The number of output shards.
  """
  tf.logging.info("Writing dataset %s", name)
  borders = np.int32(np.linspace(0, len(indices), num_shards + 1))
  for i in range(num_shards):
    filename = os.path.join(FLAGS.output_dir, "%s-%.5d-of-%.5d" % (name, i,
                                                                   num_shards))
    shard_indices = indices[borders[i]:borders[i + 1]]
    _write_shard(filename, dataset, shard_indices)
    tf.logging.info("Wrote dataset indices [%d, %d) to output shard %s",
                    borders[i], borders[i + 1], filename)
  tf.logging.info("Finished writing %d sentences in dataset %s.",
                  len(indices), name)


def main(unused_argv):
  if not FLAGS.input_files:
    raise ValueError("--input_files is required.")
  if not FLAGS.output_dir:
    raise ValueError("--output_dir is required.")

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  input_files = []
  for pattern in FLAGS.input_files.split(","):
    match = tf.gfile.Glob(FLAGS.input_files)
    if not match:
      raise ValueError("Found no files matching %s" % pattern)
    input_files.extend(match)
  tf.logging.info("Found %d input files.", len(input_files))

  vocab = _build_vocabulary(input_files)

  tf.logging.info("Generating dataset.")
  stats = collections.Counter()
  dataset = []
  for filename in input_files:
    dataset.extend(_process_input_file(filename, vocab, stats))
    if FLAGS.max_sentences and stats["sentences_output"] >= FLAGS.max_sentences:
      break

  tf.logging.info("Generated dataset with %d sentences.", len(dataset))
  for k, v in stats.items():
    tf.logging.info("%s: %d", k, v)

  tf.logging.info("Shuffling dataset.")
  np.random.seed(123)
  shuffled_indices = np.random.permutation(len(dataset))
  val_indices = shuffled_indices[:FLAGS.num_validation_sentences]
  train_indices = shuffled_indices[FLAGS.num_validation_sentences:]

  _write_dataset("train", dataset, train_indices, FLAGS.train_output_shards)
  _write_dataset("validation", dataset, val_indices,
                 FLAGS.validation_output_shards)


if __name__ == "__main__":
  tf.app.run()
