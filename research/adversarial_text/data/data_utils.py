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
"""Utilities for generating/preprocessing data for adversarial text models."""

import operator
import os
import random
import re

# Dependency imports

import tensorflow as tf

EOS_TOKEN = '</s>'

# Data filenames
# Sequence Autoencoder
ALL_SA = 'all_sa.tfrecords'
TRAIN_SA = 'train_sa.tfrecords'
TEST_SA = 'test_sa.tfrecords'
# Language Model
ALL_LM = 'all_lm.tfrecords'
TRAIN_LM = 'train_lm.tfrecords'
TEST_LM = 'test_lm.tfrecords'
# Classification
TRAIN_CLASS = 'train_classification.tfrecords'
TEST_CLASS = 'test_classification.tfrecords'
VALID_CLASS = 'validate_classification.tfrecords'
# LM with bidirectional LSTM
TRAIN_REV_LM = 'train_reverse_lm.tfrecords'
TEST_REV_LM = 'test_reverse_lm.tfrecords'
# Classification with bidirectional LSTM
TRAIN_BD_CLASS = 'train_bidir_classification.tfrecords'
TEST_BD_CLASS = 'test_bidir_classification.tfrecords'
VALID_BD_CLASS = 'validate_bidir_classification.tfrecords'


class ShufflingTFRecordWriter(object):
  """Thin wrapper around TFRecordWriter that shuffles records."""

  def __init__(self, path):
    self._path = path
    self._records = []
    self._closed = False

  def write(self, record):
    assert not self._closed
    self._records.append(record)

  def close(self):
    assert not self._closed
    random.shuffle(self._records)
    with tf.python_io.TFRecordWriter(self._path) as f:
      for record in self._records:
        f.write(record)
    self._closed = True

  def __enter__(self):
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    self.close()


class Timestep(object):
  """Represents a single timestep in a SequenceWrapper."""

  def __init__(self, token, label, weight, multivalent_tokens=False):
    """Constructs Timestep from empty Features."""
    self._token = token
    self._label = label
    self._weight = weight
    self._multivalent_tokens = multivalent_tokens
    self._fill_with_defaults()

  @property
  def token(self):
    if self._multivalent_tokens:
      raise TypeError('Timestep may contain multiple values; use `tokens`')
    return self._token.int64_list.value[0]

  @property
  def tokens(self):
    return self._token.int64_list.value

  @property
  def label(self):
    return self._label.int64_list.value[0]

  @property
  def weight(self):
    return self._weight.float_list.value[0]

  def set_token(self, token):
    if self._multivalent_tokens:
      raise TypeError('Timestep may contain multiple values; use `add_token`')
    self._token.int64_list.value[0] = token
    return self

  def add_token(self, token):
    self._token.int64_list.value.append(token)
    return self

  def set_label(self, label):
    self._label.int64_list.value[0] = label
    return self

  def set_weight(self, weight):
    self._weight.float_list.value[0] = weight
    return self

  def copy_from(self, timestep):
    self.set_token(timestep.token).set_label(timestep.label).set_weight(
        timestep.weight)
    return self

  def _fill_with_defaults(self):
    if not self._multivalent_tokens:
      self._token.int64_list.value.append(0)
    self._label.int64_list.value.append(0)
    self._weight.float_list.value.append(0.0)


class SequenceWrapper(object):
  """Wrapper around tf.SequenceExample."""

  F_TOKEN_ID = 'token_id'
  F_LABEL = 'label'
  F_WEIGHT = 'weight'

  def __init__(self, multivalent_tokens=False):
    self._seq = tf.train.SequenceExample()
    self._flist = self._seq.feature_lists.feature_list
    self._timesteps = []
    self._multivalent_tokens = multivalent_tokens

  @property
  def seq(self):
    return self._seq

  @property
  def multivalent_tokens(self):
    return self._multivalent_tokens

  @property
  def _tokens(self):
    return self._flist[SequenceWrapper.F_TOKEN_ID].feature

  @property
  def _labels(self):
    return self._flist[SequenceWrapper.F_LABEL].feature

  @property
  def _weights(self):
    return self._flist[SequenceWrapper.F_WEIGHT].feature

  def add_timestep(self):
    timestep = Timestep(
        self._tokens.add(),
        self._labels.add(),
        self._weights.add(),
        multivalent_tokens=self._multivalent_tokens)
    self._timesteps.append(timestep)
    return timestep

  def __iter__(self):
    for timestep in self._timesteps:
      yield timestep

  def __len__(self):
    return len(self._timesteps)

  def __getitem__(self, idx):
    return self._timesteps[idx]


def build_reverse_sequence(seq):
  """Builds a sequence that is the reverse of the input sequence."""
  reverse_seq = SequenceWrapper()

  # Copy all but last timestep
  for timestep in reversed(seq[:-1]):
    reverse_seq.add_timestep().copy_from(timestep)

  # Copy final timestep
  reverse_seq.add_timestep().copy_from(seq[-1])

  return reverse_seq


def build_bidirectional_seq(seq, rev_seq):
  bidir_seq = SequenceWrapper(multivalent_tokens=True)
  for forward_ts, reverse_ts in zip(seq, rev_seq):
    bidir_seq.add_timestep().add_token(forward_ts.token).add_token(
        reverse_ts.token)

  return bidir_seq


def build_lm_sequence(seq):
  """Builds language model sequence from input sequence.

  Args:
    seq: SequenceWrapper.

  Returns:
    SequenceWrapper with `seq` tokens copied over to output sequence tokens and
    labels (offset by 1, i.e. predict next token) with weights set to 1.0,
    except for <eos> token.
  """
  lm_seq = SequenceWrapper()
  for i, timestep in enumerate(seq):
    if i == len(seq) - 1:
      lm_seq.add_timestep().set_token(timestep.token).set_label(
          seq[i].token).set_weight(0.0)
    else:
      lm_seq.add_timestep().set_token(timestep.token).set_label(
          seq[i + 1].token).set_weight(1.0)
  return lm_seq


def build_seq_ae_sequence(seq):
  """Builds seq_ae sequence from input sequence.

  Args:
    seq: SequenceWrapper.

  Returns:
    SequenceWrapper with `seq` inputs copied and concatenated, and with labels
    copied in on the right-hand (i.e. decoder) side with weights set to 1.0.
    The new sequence will have length `len(seq) * 2 - 1`, as the last timestep
    of the encoder section and the first step of the decoder section will
    overlap.
  """
  seq_ae_seq = SequenceWrapper()

  for i in range(len(seq) * 2 - 1):
    ts = seq_ae_seq.add_timestep()

    if i < len(seq) - 1:
      # Encoder
      ts.set_token(seq[i].token)
    elif i == len(seq) - 1:
      # Transition step
      ts.set_token(seq[i].token)
      ts.set_label(seq[0].token)
      ts.set_weight(1.0)
    else:
      # Decoder
      ts.set_token(seq[i % len(seq)].token)
      ts.set_label(seq[(i + 1) % len(seq)].token)
      ts.set_weight(1.0)

  return seq_ae_seq


def build_labeled_sequence(seq, class_label, label_gain=False):
  """Builds labeled sequence from input sequence.

  Args:
    seq: SequenceWrapper.
    class_label: integer, starting from 0.
    label_gain: bool. If True, class_label will be put on every timestep and
      weight will increase linearly from 0 to 1.

  Returns:
    SequenceWrapper with `seq` copied in and `class_label` added as label to
    final timestep.
  """
  label_seq = SequenceWrapper(multivalent_tokens=seq.multivalent_tokens)

  # Copy sequence without labels
  seq_len = len(seq)
  final_timestep = None
  for i, timestep in enumerate(seq):
    label_timestep = label_seq.add_timestep()
    if seq.multivalent_tokens:
      for token in timestep.tokens:
        label_timestep.add_token(token)
    else:
      label_timestep.set_token(timestep.token)
    if label_gain:
      label_timestep.set_label(int(class_label))
      weight = 1.0 if seq_len < 2 else float(i) / (seq_len - 1)
      label_timestep.set_weight(weight)
    if i == (seq_len - 1):
      final_timestep = label_timestep

  # Edit final timestep to have class label and weight = 1.
  final_timestep.set_label(int(class_label)).set_weight(1.0)

  return label_seq


def split_by_punct(segment):
  """Splits str segment by punctuation, filters our empties and spaces."""
  return [s for s in re.split(r'\W+', segment) if s and not s.isspace()]


def sort_vocab_by_frequency(vocab_freq_map):
  """Sorts vocab_freq_map by count.

  Args:
    vocab_freq_map: dict<str term, int count>, vocabulary terms with counts.

  Returns:
    list<tuple<str term, int count>> sorted by count, descending.
  """
  return sorted(
      vocab_freq_map.items(), key=operator.itemgetter(1), reverse=True)


def write_vocab_and_frequency(ordered_vocab_freqs, output_dir):
  """Writes ordered_vocab_freqs into vocab.txt and vocab_freq.txt."""
  tf.gfile.MakeDirs(output_dir)
  with open(os.path.join(output_dir, 'vocab.txt'), 'w', encoding='utf-8') as vocab_f:
    with open(os.path.join(output_dir, 'vocab_freq.txt'), 'w', encoding='utf-8') as freq_f:
      for word, freq in ordered_vocab_freqs:
        vocab_f.write('{}\n'.format(word))
        freq_f.write('{}\n'.format(freq))
