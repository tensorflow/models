#!/usr/bin/env python
# Copyright 2017, 2018 Google, Inc. All Rights Reserved.
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
"""Takes as input a sorted, tab-separated of paths to produce tf.Examples."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import os
import sys
import tensorflow as tf

import lexnet_common

tf.flags.DEFINE_string('input', '', 'tab-separated input data')
tf.flags.DEFINE_string('vocab', '', 'a text file containing lemma vocabulary')
tf.flags.DEFINE_string('relations', '', 'a text file containing the relations')
tf.flags.DEFINE_string('output_dir', '', 'output directory')
tf.flags.DEFINE_string('splits', '', 'text file enumerating splits')
tf.flags.DEFINE_string('default_split', '', 'default split for unlabeled pairs')
tf.flags.DEFINE_string('compression', 'GZIP', 'compression for output records')
tf.flags.DEFINE_integer('max_paths', 100, 'maximum number of paths per record')
tf.flags.DEFINE_integer('max_pathlen', 8, 'maximum path length')
FLAGS = tf.flags.FLAGS


def _int64_features(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_features(value):
  value = [v.encode('utf-8') if isinstance(v, unicode) else v for v in value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class CreateExampleFn(object):

  def __init__(self):
    # Read the vocabulary.  N.B. that 0 = PAD, 1 = UNK, 2 = <X>, 3 = <Y>, hence
    # the enumeration starting at 4.
    with tf.gfile.GFile(FLAGS.vocab) as fh:
      self.vocab = {w: ix for ix, w in enumerate(fh.read().splitlines(), start=4)}

    self.vocab.update({'<PAD>': 0, '<UNK>': 1, '<X>': 2, '<Y>': 3})

    # Read the relations.
    with tf.gfile.GFile(FLAGS.relations) as fh:
      self.relations = {r: ix for ix, r in enumerate(fh.read().splitlines())}

    # Some hackery to map from SpaCy postags to Google's.
    lexnet_common.POSTAG_TO_ID['PROPN'] = lexnet_common.POSTAG_TO_ID['NOUN']
    lexnet_common.POSTAG_TO_ID['PRON'] = lexnet_common.POSTAG_TO_ID['NOUN']
    lexnet_common.POSTAG_TO_ID['CCONJ'] = lexnet_common.POSTAG_TO_ID['CONJ']
    #lexnet_common.DEPLABEL_TO_ID['relcl'] = lexnet_common.DEPLABEL_TO_ID['rel']
    #lexnet_common.DEPLABEL_TO_ID['compound'] = lexnet_common.DEPLABEL_TO_ID['xcomp']
    #lexnet_common.DEPLABEL_TO_ID['oprd'] = lexnet_common.DEPLABEL_TO_ID['UNK']

  def __call__(self, mod, head, rel, raw_paths):
    # Drop any really long paths.
    paths = []
    counts = []
    for raw, count in raw_paths.most_common(FLAGS.max_paths):
      path = raw.split('::')
      if len(path) <= FLAGS.max_pathlen:
        paths.append(path)
        counts.append(count)

    if not paths:
      return None

    # Compute the true length.
    pathlens = [len(path) for path in paths]

    # Pad each path out to max_pathlen so the LSTM can eat it.
    paths = (
        itertools.islice(
            itertools.chain(path, itertools.repeat('<PAD>/PAD/PAD/_')),
            FLAGS.max_pathlen)
        for path in paths)

    # Split the lemma, POS, dependency label, and direction each into a
    # separate feature.
    lemmas, postags, deplabels, dirs = zip(
        *(part.split('/') for part in itertools.chain(*paths)))

    lemmas = [self.vocab.get(lemma, 1) for lemma in lemmas]
    postags = [lexnet_common.POSTAG_TO_ID[pos] for pos in postags]
    deplabels = [lexnet_common.DEPLABEL_TO_ID.get(dep, 1) for dep in deplabels]
    dirs = [lexnet_common.DIR_TO_ID.get(d, 0) for d in dirs]

    return tf.train.Example(features=tf.train.Features(feature={
        'pair': _bytes_features(['::'.join((mod, head))]),
        'rel': _bytes_features([rel]),
        'rel_id': _int64_features([self.relations[rel]]),
        'reprs': _bytes_features(raw_paths),
        'pathlens': _int64_features(pathlens),
        'counts': _int64_features(counts),
        'lemmas': _int64_features(lemmas),
        'dirs': _int64_features(dirs),
        'deplabels': _int64_features(deplabels),
        'postags': _int64_features(postags),
        'x_embedding_id': _int64_features([self.vocab[mod]]),
        'y_embedding_id': _int64_features([self.vocab[head]]),
    }))


def main(_):
  # Read the splits file, if there is one.
  assignments = {}
  if FLAGS.splits:
    with tf.gfile.GFile(FLAGS.splits) as fh:
      parts = (line.split('\t') for line in fh.read().splitlines())
      assignments = {(mod, head): split for mod, head, split in parts}

  splits = set(assignments.itervalues())
  if FLAGS.default_split:
    default_split = FLAGS.default_split
    splits.add(FLAGS.default_split)
  elif splits:
    default_split = iter(splits).next()
  else:
    print('Please specify --splits, --default_split, or both', file=sys.stderr)
    return 1

  last_mod, last_head, last_label = None, None, None
  raw_paths = collections.Counter()

  # Keep track of pairs we've seen to ensure that we don't get unsorted data.
  seen_labeled_pairs = set()

  # Set up output compression
  compression_type = getattr(
      tf.python_io.TFRecordCompressionType, FLAGS.compression)
  options = tf.python_io.TFRecordOptions(compression_type=compression_type)

  writers = {
      split: tf.python_io.TFRecordWriter(
          os.path.join(FLAGS.output_dir, '%s.tfrecs.gz' % split),
          options=options)
      for split in splits}

  create_example = CreateExampleFn()

  in_fh = sys.stdin if not FLAGS.input else tf.gfile.GFile(FLAGS.input)
  for lineno, line in enumerate(in_fh, start=1):
    if lineno % 100 == 0:
      print('\rProcessed %d lines...' % lineno, end='', file=sys.stderr)

    parts = line.decode('utf-8').strip().split('\t')
    if len(parts) != 5:
      print('Skipping line %d: %d columns (expected 5)' % (
          lineno, len(parts)), file=sys.stderr)

      continue

    mod, head, label, raw_path, source = parts
    if mod == last_mod and head == last_head and label == last_label:
      raw_paths.update([raw_path])
      continue

    if last_mod and last_head and last_label and raw_paths:
      if (last_mod, last_head, last_label) in seen_labeled_pairs:
        print('It looks like the input data is not sorted; ignoring extra '
              'record for (%s::%s, %s) at line %d' % (
                  last_mod, last_head, last_label, lineno))
      else:
        ex = create_example(last_mod, last_head, last_label, raw_paths)
        if ex:
          split = assignments.get((last_mod, last_head), default_split)
          writers[split].write(ex.SerializeToString())

        seen_labeled_pairs.add((last_mod, last_head, last_label))

    last_mod, last_head, last_label = mod, head, label
    raw_paths = collections.Counter()

  if last_mod and last_head and last_label and raw_paths:
    ex = create_example(last_mod, last_head, last_label, raw_paths)
    if ex:
      split = assignments.get((last_mod, last_head), default_split)
      writers[split].write(ex.SerializeToString())

  for writer in writers.itervalues():
    writer.close()


if __name__ == '__main__':
  tf.app.run()
