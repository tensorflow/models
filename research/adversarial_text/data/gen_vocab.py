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
"""Generates vocabulary and term frequency files for datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

# Dependency imports

import tensorflow as tf

from adversarial_text.data import data_utils
from adversarial_text.data import document_generators

flags = tf.app.flags
FLAGS = flags.FLAGS

# Flags controlling input are in document_generators.py

flags.DEFINE_string('output_dir', '',
                    'Path to save vocab.txt and vocab_freq.txt.')

flags.DEFINE_boolean('use_unlabeled', True, 'Whether to use the '
                     'unlabeled sentiment dataset in the vocabulary.')
flags.DEFINE_boolean('include_validation', False, 'Whether to include the '
                     'validation set in the vocabulary.')
flags.DEFINE_integer('doc_count_threshold', 1, 'The minimum number of '
                     'documents a word or bigram should occur in to keep '
                     'it in the vocabulary.')

MAX_VOCAB_SIZE = 100 * 1000


def fill_vocab_from_doc(doc, vocab_freqs, doc_counts):
  """Fills vocabulary and doc counts with tokens from doc.

  Args:
    doc: Document to read tokens from.
    vocab_freqs: dict<token, frequency count>
    doc_counts: dict<token, document count>

  Returns:
    None
  """
  doc_seen = set()

  for token in document_generators.tokens(doc):
    if doc.add_tokens or token in vocab_freqs:
      vocab_freqs[token] += 1
    if token not in doc_seen:
      doc_counts[token] += 1
      doc_seen.add(token)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  vocab_freqs = defaultdict(int)
  doc_counts = defaultdict(int)

  # Fill vocabulary frequencies map and document counts map
  for doc in document_generators.documents(
      dataset='train',
      include_unlabeled=FLAGS.use_unlabeled,
      include_validation=FLAGS.include_validation):
    fill_vocab_from_doc(doc, vocab_freqs, doc_counts)

  # Filter out low-occurring terms
  vocab_freqs = dict((term, freq) for term, freq in vocab_freqs.iteritems()
                     if doc_counts[term] > FLAGS.doc_count_threshold)

  # Sort by frequency
  ordered_vocab_freqs = data_utils.sort_vocab_by_frequency(vocab_freqs)

  # Limit vocab size
  ordered_vocab_freqs = ordered_vocab_freqs[:MAX_VOCAB_SIZE]

  # Add EOS token
  ordered_vocab_freqs.append((data_utils.EOS_TOKEN, 1))

  # Write
  tf.gfile.MakeDirs(FLAGS.output_dir)
  data_utils.write_vocab_and_frequency(ordered_vocab_freqs, FLAGS.output_dir)


if __name__ == '__main__':
  tf.app.run()
