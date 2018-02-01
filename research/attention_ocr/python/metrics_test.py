# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Tests for the metrics module."""
import contextlib
import numpy as np
import tensorflow as tf

import metrics


class AccuracyTest(tf.test.TestCase):
  def setUp(self):
    tf.test.TestCase.setUp(self)
    self.rng = np.random.RandomState([11, 23, 50])
    self.num_char_classes = 3
    self.batch_size = 4
    self.seq_length = 5
    self.rej_char = 42

  @contextlib.contextmanager
  def initialized_session(self):
    """Wrapper for test session context manager with required initialization.

    Yields:
      A session object that should be used as a context manager.
    """
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      yield sess

  def _fake_labels(self):
    return self.rng.randint(
        low=0,
        high=self.num_char_classes,
        size=(self.batch_size, self.seq_length),
        dtype='int32')

  def _incorrect_copy(self, values, bad_indexes):
    incorrect = np.copy(values)
    incorrect[bad_indexes] = values[bad_indexes] + 1
    return incorrect

  def test_sequence_accuracy_identical_samples(self):
    labels_tf = tf.convert_to_tensor(self._fake_labels())

    accuracy_tf = metrics.sequence_accuracy(labels_tf, labels_tf,
                                            self.rej_char)
    with self.initialized_session() as sess:
      accuracy_np = sess.run(accuracy_tf)

    self.assertAlmostEqual(accuracy_np, 1.0)

  def test_sequence_accuracy_one_char_difference(self):
    ground_truth_np = self._fake_labels()
    ground_truth_tf = tf.convert_to_tensor(ground_truth_np)
    prediction_tf = tf.convert_to_tensor(
        self._incorrect_copy(ground_truth_np, bad_indexes=((0, 0))))

    accuracy_tf = metrics.sequence_accuracy(prediction_tf, ground_truth_tf,
                                            self.rej_char)
    with self.initialized_session() as sess:
      accuracy_np = sess.run(accuracy_tf)

    # 1 of 4 sequences is incorrect.
    self.assertAlmostEqual(accuracy_np, 1.0 - 1.0 / self.batch_size)

  def test_char_accuracy_one_char_difference_with_padding(self):
    ground_truth_np = self._fake_labels()
    ground_truth_tf = tf.convert_to_tensor(ground_truth_np)
    prediction_tf = tf.convert_to_tensor(
        self._incorrect_copy(ground_truth_np, bad_indexes=((0, 0))))

    accuracy_tf = metrics.char_accuracy(prediction_tf, ground_truth_tf,
                                        self.rej_char)
    with self.initialized_session() as sess:
      accuracy_np = sess.run(accuracy_tf)

    chars_count = self.seq_length * self.batch_size
    self.assertAlmostEqual(accuracy_np, 1.0 - 1.0 / chars_count)


if __name__ == '__main__':
  tf.test.main()
