#!/usr/bin/env python
#
# Copyright 2016 Google Inc. All Rights Reserved.
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

"""Prepare a corpus for processing by swivel.

Creates a sharded word co-occurrence matrix from a text file input corpus.

Usage:

  prep.py --output_dir <output-dir> --input <text-file>

Options:

  --input <filename>
      The input text.

  --output_dir <directory>
      Specifies the output directory where the various Swivel data
      files should be placed.

  --shard_size <int>
      Specifies the shard size; default 4096.

  --min_count <int>
      Specifies the minimum number of times a word should appear
      to be included in the vocabulary; default 5.

  --max_vocab <int>
      Specifies the maximum vocabulary size; default shard size
      times 1024.

  --vocab <filename>
      Use the specified unigram vocabulary instead of generating
      it from the corpus.

  --window_size <int>
      Specifies the window size for computing co-occurrence stats;
      default 10.

  --bufsz <int>
      The number of co-occurrences that are buffered; default 16M.

"""

import itertools
import math
import os
import struct
import sys

from six.moves import xrange
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('input', '', 'The input text.')
flags.DEFINE_string('output_dir', '/tmp/swivel_data',
                    'Output directory for Swivel data')
flags.DEFINE_integer('shard_size', 4096, 'The size for each shard')
flags.DEFINE_integer('min_count', 5,
                     'The minimum number of times a word should occur to be '
                     'included in the vocabulary')
flags.DEFINE_integer('max_vocab', 4096 * 64, 'The maximum vocabulary size')
flags.DEFINE_string('vocab', '', 'Vocabulary to use instead of generating one')
flags.DEFINE_integer('window_size', 10, 'The window size')
flags.DEFINE_integer('bufsz', 16 * 1024 * 1024,
                     'The number of co-occurrences to buffer')

FLAGS = flags.FLAGS

shard_cooc_fmt = struct.Struct('iif')


def words(line):
  """Splits a line of text into tokens."""
  return line.strip().split()


def create_vocabulary(lines):
  """Reads text lines and generates a vocabulary."""
  lines.seek(0, os.SEEK_END)
  nbytes = lines.tell()
  lines.seek(0, os.SEEK_SET)

  vocab = {}
  for lineno, line in enumerate(lines, start=1):
    for word in words(line):
      vocab.setdefault(word, 0)
      vocab[word] += 1

    if lineno % 100000 == 0:
      pos = lines.tell()
      sys.stdout.write('\rComputing vocabulary: %0.1f%% (%d/%d)...' % (
          100.0 * pos / nbytes, pos, nbytes))
      sys.stdout.flush()

  sys.stdout.write('\n')

  vocab = [(tok, n) for tok, n in vocab.iteritems() if n >= FLAGS.min_count]
  vocab.sort(key=lambda kv: (-kv[1], kv[0]))

  num_words = min(len(vocab), FLAGS.max_vocab)
  if num_words % FLAGS.shard_size != 0:
    num_words -= num_words % FLAGS.shard_size

  if not num_words:
    raise Exception('empty vocabulary')

  print('vocabulary contains %d tokens' % num_words)

  vocab = vocab[:num_words]
  return [tok for tok, n in vocab]


def write_vocab_and_sums(vocab, sums, vocab_filename, sums_filename):
  """Writes vocabulary and marginal sum files."""
  with open(os.path.join(FLAGS.output_dir, vocab_filename), 'w') as vocab_out:
    with open(os.path.join(FLAGS.output_dir, sums_filename), 'w') as sums_out:
      for tok, cnt in itertools.izip(vocab, sums):
        print >> vocab_out, tok
        print >> sums_out, cnt


def compute_coocs(lines, vocab):
  """Compute the co-occurrence statistics from the text.

  This generates a temporary file for each shard that contains the intermediate
  counts from the shard: these counts must be subsequently sorted and collated.

  """
  word_to_id = {tok: idx for idx, tok in enumerate(vocab)}

  lines.seek(0, os.SEEK_END)
  nbytes = lines.tell()
  lines.seek(0, os.SEEK_SET)

  num_shards = len(vocab) / FLAGS.shard_size

  shardfiles = {}
  for row in range(num_shards):
    for col in range(num_shards):
      filename = os.path.join(
          FLAGS.output_dir, 'shard-%03d-%03d.tmp' % (row, col))

      shardfiles[(row, col)] = open(filename, 'w+')

  def flush_coocs():
    for (row_id, col_id), cnt in coocs.iteritems():
      row_shard = row_id % num_shards
      row_off = row_id / num_shards
      col_shard = col_id % num_shards
      col_off = col_id / num_shards

      # Since we only stored (a, b), we emit both (a, b) and (b, a).
      shardfiles[(row_shard, col_shard)].write(
          shard_cooc_fmt.pack(row_off, col_off, cnt))

      shardfiles[(col_shard, row_shard)].write(
          shard_cooc_fmt.pack(col_off, row_off, cnt))

  coocs = {}
  sums = [0.0] * len(vocab)

  for lineno, line in enumerate(lines, start=1):
    # Computes the word IDs for each word in the sentence.  This has the effect
    # of "stretching" the window past OOV tokens.
    wids = filter(
        lambda wid: wid is not None,
        (word_to_id.get(w) for w in words(line)))

    for pos in xrange(len(wids)):
      lid = wids[pos]
      window_extent = min(FLAGS.window_size + 1, len(wids) - pos)
      for off in xrange(1, window_extent):
        rid = wids[pos + off]
        pair = (min(lid, rid), max(lid, rid))
        count = 1.0 / off
        sums[lid] += count
        sums[rid] += count
        coocs.setdefault(pair, 0.0)
        coocs[pair] += count

      sums[lid] += 1.0
      pair = (lid, lid)
      coocs.setdefault(pair, 0.0)
      coocs[pair] += 0.5  # Only add 1/2 since we output (a, b) and (b, a)

    if lineno % 10000 == 0:
      pos = lines.tell()
      sys.stdout.write('\rComputing co-occurrences: %0.1f%% (%d/%d)...' % (
          100.0 * pos / nbytes, pos, nbytes))
      sys.stdout.flush()

      if len(coocs) > FLAGS.bufsz:
        flush_coocs()
        coocs = {}

  flush_coocs()
  sys.stdout.write('\n')

  return shardfiles, sums


def write_shards(vocab, shardfiles):
  """Processes the temporary files to generate the final shard data.

  The shard data is stored as a tf.Example protos using a TFRecordWriter. The
  temporary files are removed from the filesystem once they've been processed.

  """
  num_shards = len(vocab) / FLAGS.shard_size

  ix = 0
  for (row, col), fh in shardfiles.iteritems():
    ix += 1
    sys.stdout.write('\rwriting shard %d/%d' % (ix, len(shardfiles)))
    sys.stdout.flush()

    # Read the entire binary co-occurrence and unpack it into an array.
    fh.seek(0)
    buf = fh.read()
    os.unlink(fh.name)
    fh.close()

    coocs = [
        shard_cooc_fmt.unpack_from(buf, off)
        for off in range(0, len(buf), shard_cooc_fmt.size)]

    # Sort and merge co-occurrences for the same pairs.
    coocs.sort()

    if coocs:
      current_pos = 0
      current_row_col = (coocs[current_pos][0], coocs[current_pos][1])
      for next_pos in range(1, len(coocs)):
        next_row_col = (coocs[next_pos][0], coocs[next_pos][1])
        if current_row_col == next_row_col:
          coocs[current_pos] = (
              coocs[current_pos][0],
              coocs[current_pos][1],
              coocs[current_pos][2] + coocs[next_pos][2])
        else:
          current_pos += 1
          if current_pos < next_pos:
            coocs[current_pos] = coocs[next_pos]

          current_row_col = (coocs[current_pos][0], coocs[current_pos][1])

      coocs = coocs[:(1 + current_pos)]

    # Convert to a TF Example proto.
    def _int64s(xs):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=list(xs)))

    def _floats(xs):
      return tf.train.Feature(float_list=tf.train.FloatList(value=list(xs)))

    example = tf.train.Example(features=tf.train.Features(feature={
        'global_row': _int64s(
            row + num_shards * i for i in range(FLAGS.shard_size)),
        'global_col': _int64s(
            col + num_shards * i for i in range(FLAGS.shard_size)),

        'sparse_local_row': _int64s(cooc[0] for cooc in coocs),
        'sparse_local_col': _int64s(cooc[1] for cooc in coocs),
        'sparse_value': _floats(cooc[2] for cooc in coocs),
    }))

    filename = os.path.join(FLAGS.output_dir, 'shard-%03d-%03d.pb' % (row, col))
    with open(filename, 'w') as out:
      out.write(example.SerializeToString())

  sys.stdout.write('\n')


def main(_):
  # Create the output directory, if necessary
  if FLAGS.output_dir and not os.path.isdir(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  # Read the file onces to create the vocabulary.
  if FLAGS.vocab:
    with open(FLAGS.vocab, 'r') as lines:
      vocab = [line.strip() for line in lines]
  else:
    with open(FLAGS.input, 'r') as lines:
      vocab = create_vocabulary(lines)

  # Now read the file again to determine the co-occurrence stats.
  with open(FLAGS.input, 'r') as lines:
    shardfiles, sums = compute_coocs(lines, vocab)

  # Collect individual shards into the shards.recs file.
  write_shards(vocab, shardfiles)

  # Now write the marginals.  They're symmetric for this application.
  write_vocab_and_sums(vocab, sums, 'row_vocab.txt', 'row_sums.txt')
  write_vocab_and_sums(vocab, sums, 'col_vocab.txt', 'col_sums.txt')

  print('done!')


if __name__ == '__main__':
  tf.app.run()
