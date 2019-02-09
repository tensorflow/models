# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the keras_common module."""
from __future__ import absolute_import
from __future__ import print_function

from mock import Mock
import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.resnet.keras import keras_common

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class KerasCommonTests(tf.test.TestCase):
  """Tests for keras_common."""

  @classmethod
  def setUpClass(cls):  # pylint: disable=invalid-name
    super(KerasCommonTests, cls).setUpClass()

  def test_build_stats(self):

    history = self._build_history(1.145, cat_accuracy=.99988)
    eval_output = self._build_eval_output(.56432111, 5.990)
    th = keras_common.TimeHistory(128, 100)

    th.batch_start_timestamps = [1, 2, 3]
    th.batch_end_timestamps = [4, 5, 6]
    th.train_finish_time = 12345
    stats = keras_common.build_stats(history, eval_output, th)

    self.assertEqual(1.145, stats['loss'])
    self.assertEqual(.99988, stats['training_accuracy_top_1'])

    self.assertEqual(.56432111, stats['accuracy_top_1'])
    self.assertEqual(5.990, stats['eval_loss'])

    self.assertItemsEqual([1, 2, 3], stats['batch_start_timestamps'])
    self.assertItemsEqual([4, 5, 6], stats['batch_end_timestamps'])
    self.assertEqual(12345, stats['train_finish_time'])

  def test_build_stats_sparse(self):

    history = self._build_history(1.145, cat_accuracy_sparse=.99988)
    eval_output = self._build_eval_output(.928, 1.9844)
    stats = keras_common.build_stats(history, eval_output, None)

    self.assertEqual(1.145, stats['loss'])
    self.assertEqual(.99988, stats['training_accuracy_top_1'])

    self.assertEqual(.928, stats['accuracy_top_1'])
    self.assertEqual(1.9844, stats['eval_loss'])

  def test_time_history(self):
    th = keras_common.TimeHistory(batch_size=128, log_steps=3)

    th.on_train_begin()
    th.on_batch_begin(0)
    th.on_batch_end(0)
    th.on_batch_begin(1)
    th.on_batch_end(1)
    th.on_batch_begin(2)
    th.on_batch_end(2)
    th.on_batch_begin(3)
    th.on_batch_end(3)
    th.on_batch_begin(4)
    th.on_batch_end(4)
    th.on_batch_begin(5)
    th.on_batch_end(5)
    th.on_batch_begin(6)
    th.on_batch_end(6)
    th.on_train_end()

    self.assertEqual(3, len(th.batch_start_timestamps))
    self.assertEqual(2, len(th.batch_end_timestamps))

    self.assertEqual(0, th.batch_start_timestamps[0].batch_index)
    self.assertEqual(1, th.batch_start_timestamps[1].batch_index)
    self.assertEqual(4, th.batch_start_timestamps[2].batch_index)

    self.assertEqual(3, th.batch_end_timestamps[0].batch_index)
    self.assertEqual(6, th.batch_end_timestamps[1].batch_index)

  def _build_history(self, loss, cat_accuracy=None,
                     cat_accuracy_sparse=None):
    history_p = Mock()
    history = {}
    history_p.history = history
    history['loss'] = [np.float64(loss)]
    if cat_accuracy:
      history['categorical_accuracy'] = [np.float64(cat_accuracy)]
    if cat_accuracy_sparse:
      history['sparse_categorical_accuracy'] = [np.float64(cat_accuracy_sparse)]

    return history_p

  def _build_eval_output(self, top_1, eval_loss):
    eval_output = [np.float64(eval_loss), np.float64(top_1)]
    return eval_output
