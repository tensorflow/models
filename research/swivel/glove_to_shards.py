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

"""Converts a Glove binary co-occurrence matrix into Swivel shards.

Usage:

  glove_to_shards.py --input <coocs> --vocab <vocab> --output_dir <output_dir>

Options

  --input <coocs>
      The Glove co-occurrence file.

  --vocab <vocab>
      Path to the vocabulary text file, one token per line.

  --output_dir <directory>
      Specifies the touput directory where the various Swivel data
      files sohuld be placed.

  --shard_size <int>
      Specifies the shard size; default 4096.
"""

from __future__ import print_function

import itertools
import os
import struct
import sys

import tensorflow as tf

from six.moves import xrange

flags = tf.app.flags

flags.DEFINE_string('input', 'coocurrences.bin', 'Vocabulary file')
flags.DEFINE_string('vocab', 'vocab.txt', 'Vocabulary file')
flags.DEFINE_string('output_dir', '/tmp/swivel_data', 'Output directory')
flags.DEFINE_integer('shard_size', 4096, 'Shard size')

FLAGS = tf.app.flags.FLAGS

glove_cooc_fmt = struct.Struct('iid')
shard_cooc_fmt = struct.Struct('if')


def make_shard_files(coocs, nshards, vocab_sz):
  """Chops the binary Glove co-occurrence matrix into shards.

  This reads the Glove binary co-occurrence file and assigns individual
  co-occurrence counts to the appropriate Swivel shard.

  Args:
    coocs: the co-occurrnece file to read
    nshards: the number of shards along one dimension of the square matrix
    vocab_sz: the vocabulary size

  Returns:
    A (shard_table, marginals) tuple.  The shard_table maps the row and column
    shard ID to a file handle containing the co-occurrences for that shard; the
    marginals contain the marginal sums.
  """
  row_sums = [0] * vocab_sz
  col_sums = [0] * vocab_sz

  coocs.seek(0, os.SEEK_END)
  ncoocs = coocs.tell() / glove_cooc_fmt.size
  coocs.seek(0, os.SEEK_SET)

  shard_files = {}

  for row in range(nshards):
    for col in range(nshards):
      filename = os.path.join(
          FLAGS.output_dir, 'shard-%03d-%03d.bin' % (row, col))

      shard_files[(row, col)] = open(filename, 'w+')

  for ix in xrange(ncoocs):
    if ix % 1000000 == 0:
      sys.stdout.write('\rsharding co-occurrences: %0.1f%% (%d/%d)' % (
          100.0 * ix / ncoocs, ix, ncoocs))

      sys.stdout.flush()

    bits = coocs.read(glove_cooc_fmt.size)
    if not bits:
      break

    # Glove has 1-indexed IDs.
    row_id, col_id, cnt = glove_cooc_fmt.unpack(bits)
    if row_id > vocab_sz or col_id > vocab_sz:
      continue

    row_id -= 1
    row_shard = row_id % nshards
    row_off = row_id / nshards

    col_id -= 1
    col_shard = col_id % nshards
    col_off = col_id / nshards

    shard_pos = row_off * FLAGS.shard_size + col_off  # row major

    shard_files[(row_shard, col_shard)].write(
        shard_cooc_fmt.pack(shard_pos, cnt))

    # Accumulate marginals.
    row_sums[row_id] += cnt
    col_sums[col_id] += cnt

  sys.stdout.write('\n')

  if any(abs(r - c) > 0.1 for r, c in itertools.izip(row_sums, col_sums)):
    print('WARNING! Row and column marginals differ; is your matrix symmetric?',
          file=sys.stderr)

  return (shard_files, row_sums)


def main(_):
  with open(FLAGS.vocab, 'r') as lines:
    orig_vocab_sz = sum(1 for _ in lines)

  shard_sz = FLAGS.shard_size
  vocab_sz = orig_vocab_sz - orig_vocab_sz % shard_sz
  nshards = vocab_sz / shard_sz

  print('vocab size is %d (originally %d), %d %dx%d-element shards' % (
      vocab_sz, orig_vocab_sz, nshards * nshards, shard_sz, shard_sz))

  # Create the output directory, if necessary
  if FLAGS.output_dir and not os.path.isdir(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  with open(FLAGS.input, 'r') as coocs:
    shard_files, marginals = make_shard_files(coocs, nshards, vocab_sz)

  # Now sort the shards and write the TFRecords.
  filename = os.path.join(FLAGS.output_dir, 'shards.recs')
  with tf.python_io.TFRecordWriter(filename) as writer:
    ix = 0
    for (row, col), fh in shard_files.iteritems():
      ix += 1
      sys.stdout.write('\rwriting shard %d/%d' % (ix, len(shard_files)))
      sys.stdout.flush()

      fh.seek(0)
      buf = fh.read()
      os.unlink(fh.name)
      fh.close()

      coocs = [
          shard_cooc_fmt.unpack_from(buf, off)
          for off in range(0, len(buf), shard_cooc_fmt.size)]

      # N.B. we assume that there aren't any duplicates here!
      coocs.sort(key=lambda kv: kv[0])

      def _int64s(xs):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(xs)))

      def _floats(xs):
        return tf.train.Feature(float_list=tf.train.FloatList(value=list(xs)))

      example = tf.train.Example(features=tf.train.Features(feature={
          'global_row': _int64s(row + nshards * i for i in range(shard_sz)),
          'global_col': _int64s(col + nshards * i for i in range(shard_sz)),
          'sparse_local_row': _int64s(pos / shard_sz for pos, _ in coocs),
          'sparse_local_col': _int64s(pos % shard_sz for pos, _ in coocs),
          'sparse_value': _floats(cnt for _, cnt in coocs)}))

      writer.write(example.SerializeToString())

  print('\nwriting marginals...')

  with open(os.path.join(FLAGS.output_dir, 'marginals.txt'), 'w') as fh:
    for cnt in marginals:
      fh.write('%0.1f\n' % cnt)

  print('done!')


if __name__ == '__main__':
  tf.app.run()
