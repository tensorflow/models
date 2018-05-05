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
"""Tests for data_utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow as tf

from data import data_utils

data = data_utils


class SequenceWrapperTest(tf.test.TestCase):

  def testDefaultTimesteps(self):
    seq = data.SequenceWrapper()
    t1 = seq.add_timestep()
    _ = seq.add_timestep()
    self.assertEqual(len(seq), 2)

    self.assertEqual(t1.weight, 0.0)
    self.assertEqual(t1.label, 0)
    self.assertEqual(t1.token, 0)

  def testSettersAndGetters(self):
    ts = data.SequenceWrapper().add_timestep()
    ts.set_token(3)
    ts.set_label(4)
    ts.set_weight(2.0)
    self.assertEqual(ts.token, 3)
    self.assertEqual(ts.label, 4)
    self.assertEqual(ts.weight, 2.0)

  def testTimestepIteration(self):
    seq = data.SequenceWrapper()
    seq.add_timestep().set_token(0)
    seq.add_timestep().set_token(1)
    seq.add_timestep().set_token(2)
    for i, ts in enumerate(seq):
      self.assertEqual(ts.token, i)

  def testFillsSequenceExampleCorrectly(self):
    seq = data.SequenceWrapper()
    seq.add_timestep().set_token(1).set_label(2).set_weight(3.0)
    seq.add_timestep().set_token(10).set_label(20).set_weight(30.0)

    seq_ex = seq.seq
    fl = seq_ex.feature_lists.feature_list
    fl_token = fl[data.SequenceWrapper.F_TOKEN_ID].feature
    fl_label = fl[data.SequenceWrapper.F_LABEL].feature
    fl_weight = fl[data.SequenceWrapper.F_WEIGHT].feature
    _ = [self.assertEqual(len(f), 2) for f in [fl_token, fl_label, fl_weight]]
    self.assertAllEqual([f.int64_list.value[0] for f in fl_token], [1, 10])
    self.assertAllEqual([f.int64_list.value[0] for f in fl_label], [2, 20])
    self.assertAllEqual([f.float_list.value[0] for f in fl_weight], [3.0, 30.0])


class DataUtilsTest(tf.test.TestCase):

  def testSplitByPunct(self):
    output = data.split_by_punct(
        'hello! world, i\'ve been\nwaiting\tfor\ryou for.a long time')
    expected = [
        'hello', 'world', 'i', 've', 'been', 'waiting', 'for', 'you', 'for',
        'a', 'long', 'time'
    ]
    self.assertListEqual(output, expected)

  def _buildDummySequence(self):
    seq = data.SequenceWrapper()
    for i in range(10):
      seq.add_timestep().set_token(i)
    return seq

  def testBuildLMSeq(self):
    seq = self._buildDummySequence()
    lm_seq = data.build_lm_sequence(seq)
    for i, ts in enumerate(lm_seq):
      # For end of sequence, the token and label should be same, and weight
      # should be 0.0.
      if i == len(lm_seq) - 1:
        self.assertEqual(ts.token, i)
        self.assertEqual(ts.label, i)
        self.assertEqual(ts.weight, 0.0)
      else:
        self.assertEqual(ts.token, i)
        self.assertEqual(ts.label, i + 1)
        self.assertEqual(ts.weight, 1.0)

  def testBuildSAESeq(self):
    seq = self._buildDummySequence()
    sa_seq = data.build_seq_ae_sequence(seq)

    self.assertEqual(len(sa_seq), len(seq) * 2 - 1)

    # Tokens should be sequence twice, minus the EOS token at the end
    for i, ts in enumerate(sa_seq):
      self.assertEqual(ts.token, seq[i % 10].token)

    # Weights should be len-1 0.0's and len 1.0's.
    for i in range(len(seq) - 1):
      self.assertEqual(sa_seq[i].weight, 0.0)
    for i in range(len(seq) - 1, len(sa_seq)):
      self.assertEqual(sa_seq[i].weight, 1.0)

    # Labels should be len-1 0's, and then the sequence
    for i in range(len(seq) - 1):
      self.assertEqual(sa_seq[i].label, 0)
    for i in range(len(seq) - 1, len(sa_seq)):
      self.assertEqual(sa_seq[i].label, seq[i - (len(seq) - 1)].token)

  def testBuildLabelSeq(self):
    seq = self._buildDummySequence()
    eos_id = len(seq) - 1
    label_seq = data.build_labeled_sequence(seq, True)
    for i, ts in enumerate(label_seq[:-1]):
      self.assertEqual(ts.token, i)
      self.assertEqual(ts.label, 0)
      self.assertEqual(ts.weight, 0.0)

    final_timestep = label_seq[-1]
    self.assertEqual(final_timestep.token, eos_id)
    self.assertEqual(final_timestep.label, 1)
    self.assertEqual(final_timestep.weight, 1.0)

  def testBuildBidirLabelSeq(self):
    seq = self._buildDummySequence()
    reverse_seq = data.build_reverse_sequence(seq)
    bidir_seq = data.build_bidirectional_seq(seq, reverse_seq)
    label_seq = data.build_labeled_sequence(bidir_seq, True)

    for (i, ts), j in zip(
        enumerate(label_seq[:-1]), reversed(range(len(seq) - 1))):
      self.assertAllEqual(ts.tokens, [i, j])
      self.assertEqual(ts.label, 0)
      self.assertEqual(ts.weight, 0.0)

    final_timestep = label_seq[-1]
    eos_id = len(seq) - 1
    self.assertAllEqual(final_timestep.tokens, [eos_id, eos_id])
    self.assertEqual(final_timestep.label, 1)
    self.assertEqual(final_timestep.weight, 1.0)

  def testReverseSeq(self):
    seq = self._buildDummySequence()
    reverse_seq = data.build_reverse_sequence(seq)
    for i, ts in enumerate(reversed(reverse_seq[:-1])):
      self.assertEqual(ts.token, i)
      self.assertEqual(ts.label, 0)
      self.assertEqual(ts.weight, 0.0)

    final_timestep = reverse_seq[-1]
    eos_id = len(seq) - 1
    self.assertEqual(final_timestep.token, eos_id)
    self.assertEqual(final_timestep.label, 0)
    self.assertEqual(final_timestep.weight, 0.0)

  def testBidirSeq(self):
    seq = self._buildDummySequence()
    reverse_seq = data.build_reverse_sequence(seq)
    bidir_seq = data.build_bidirectional_seq(seq, reverse_seq)
    for (i, ts), j in zip(
        enumerate(bidir_seq[:-1]), reversed(range(len(seq) - 1))):
      self.assertAllEqual(ts.tokens, [i, j])
      self.assertEqual(ts.label, 0)
      self.assertEqual(ts.weight, 0.0)

    final_timestep = bidir_seq[-1]
    eos_id = len(seq) - 1
    self.assertAllEqual(final_timestep.tokens, [eos_id, eos_id])
    self.assertEqual(final_timestep.label, 0)
    self.assertEqual(final_timestep.weight, 0.0)

  def testLabelGain(self):
    seq = self._buildDummySequence()
    label_seq = data.build_labeled_sequence(seq, True, label_gain=True)
    for i, ts in enumerate(label_seq):
      self.assertEqual(ts.token, i)
      self.assertEqual(ts.label, 1)
      self.assertNear(ts.weight, float(i) / (len(seq) - 1), 1e-3)


if __name__ == '__main__':
  tf.test.main()
