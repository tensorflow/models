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

"""Common stuff used with LexNET."""
# pylint: disable=bad-whitespace

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from sklearn import metrics
import tensorflow as tf

# Part of speech tags used in the paths.
POSTAGS = [
    'PAD',   'VERB',   'CONJ',   'NOUN',   'PUNCT',
    'ADP',   'ADJ',    'DET',    'ADV',    'PART',
    'NUM',   'X',      'INTJ',   'SYM',
]

POSTAG_TO_ID = {tag: tid for tid, tag in enumerate(POSTAGS)}

# Dependency labels used in the paths.
DEPLABELS = [
    'PAD',     'UNK',       'ROOT',    'abbrev',    'acomp', 'advcl',
    'advmod',  'agent',     'amod',    'appos',     'attr',  'aux',
    'auxpass', 'cc',        'ccomp',   'complm',    'conj',  'cop',
    'csubj',   'csubjpass', 'dep',     'det',       'dobj',  'expl',
    'infmod',  'iobj',      'mark',    'mwe',       'nc',    'neg',
    'nn',      'npadvmod',  'nsubj',   'nsubjpass', 'num',   'number',
    'p',       'parataxis', 'partmod', 'pcomp',     'pobj',  'poss',
    'preconj', 'predet',    'prep',    'prepc',     'prt',   'ps',
    'purpcl',  'quantmod',  'rcmod',   'ref',       'rel',   'suffix',
    'title',   'tmod',      'xcomp',   'xsubj',
]

DEPLABEL_TO_ID = {label: lid for lid, label in enumerate(DEPLABELS)}

# Direction codes used in the paths.
DIRS = '_^V<>'
DIR_TO_ID = {dir: did for did, dir in enumerate(DIRS)}


def load_word_embeddings(embedding_filename):
  """Loads pretrained word embeddings from a binary file and returns the matrix.

  Adds the <PAD>, <UNK>, <X>, and <Y> tokens to the beginning of the vocab.

  Args:
    embedding_filename: filename of the binary NPY data

  Returns:
    The word embeddings matrix
  """
  embeddings = np.load(embedding_filename)
  dim = embeddings.shape[1]

  # Four initially random vectors for the special tokens: <PAD>, <UNK>, <X>, <Y>
  special_embeddings = np.random.normal(0, 0.1, (4, dim))
  embeddings = np.vstack((special_embeddings, embeddings))
  embeddings = embeddings.astype(np.float32)

  return embeddings


def full_evaluation(model, session, instances, labels, set_name, classes):
  """Prints a full evaluation on the current set.

  Performance (recall, precision and F1), classification report (per
  class performance), and confusion matrix).

  Args:
    model: The currently trained path-based model.
    session: The current TensorFlow session.
    instances: The current set instances.
    labels: The current set labels.
    set_name: The current set name (train/validation/test).
    classes: The class label names.

  Returns:
    The model's prediction for the given instances.
  """

  # Predict the labels
  pred = model.predict(session, instances)

  # Print the performance
  precision, recall, f1, _ = metrics.precision_recall_fscore_support(
      labels, pred, average='weighted')

  print('%s set: Precision: %.3f, Recall: %.3f, F1: %.3f' % (
      set_name, precision, recall, f1))

  # Print a classification report
  print('%s classification report:' % set_name)
  print(metrics.classification_report(labels, pred, target_names=classes))

  # Print the confusion matrix
  print('%s confusion matrix:' % set_name)
  cm = metrics.confusion_matrix(labels, pred, labels=range(len(classes)))
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
  print_cm(cm, labels=classes)
  return pred


def print_cm(cm, labels):
  """Pretty print for confusion matrices.

  From: https://gist.github.com/zachguo/10296432.

  Args:
    cm: The confusion matrix.
    labels: The class names.
  """
  columnwidth = 10
  empty_cell = ' ' * columnwidth
  short_labels = [label[:12].rjust(10, ' ') for label in labels]

  # Print header
  header = empty_cell + ' '
  header += ''.join([' %{0}s '.format(columnwidth) % label
                     for label in short_labels])

  print(header)

  # Print rows
  for i, label1 in enumerate(short_labels):
    row = '%{0}s '.format(columnwidth) % label1[:10]
    for j in range(len(short_labels)):
      value = int(cm[i, j]) if not np.isnan(cm[i, j]) else 0
      cell = ' %{0}d '.format(10) % value
      row += cell + ' '
    print(row)


def load_all_labels(records):
  """Reads TensorFlow examples from a RecordReader and returns only the labels.

  Args:
    records: a record list with TensorFlow examples.

  Returns:
    The labels
  """
  curr_features = tf.parse_example(records, {
      'rel_id': tf.FixedLenFeature([1], dtype=tf.int64),
  })

  labels = tf.squeeze(curr_features['rel_id'], [-1])
  return labels


def load_all_pairs(records):
  """Reads TensorFlow examples from a RecordReader and returns the word pairs.

  Args:
    records: a record list with TensorFlow examples.

  Returns:
    The word pairs
  """
  curr_features = tf.parse_example(records, {
      'pair': tf.FixedLenFeature([1], dtype=tf.string)
  })

  word_pairs = curr_features['pair']
  return word_pairs


def write_predictions(pairs, labels, predictions, classes, predictions_file):
  """Write the predictions to a file.

  Args:
    pairs: the word pairs (list of tuple of two strings).
    labels: the gold-standard labels for these pairs (array of rel ID).
    predictions: the predicted labels for these pairs (array of rel ID).
    classes: a list of relation names.
    predictions_file: where to save the predictions.
  """
  with open(predictions_file, 'w') as f_out:
    for pair, label, pred in zip(pairs, labels, predictions):
      w1, w2 = pair
      f_out.write('\t'.join([w1, w2, classes[label], classes[pred]]) + '\n')
