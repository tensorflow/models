# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for callbacks."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import collections
import functools
import os

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import callbacks_test
from tensorflow.python.keras import keras_parameterized
from official.vision.image_classification import callbacks

_ObservedSummary = collections.namedtuple('_ObservedSummary', ('logdir', 'tag'))


def _trivial_function(a):
  return a


class UtilFunctionTests(tf.test.TestCase, parameterized.TestCase):
  """Tests to check utility functions provided in callbacks.py."""

  @parameterized.named_parameters(
      ('integer', 1),
      ('float', 1.),
      ('lambda', lambda: 1),
      ('partial', functools.partial(_trivial_function, 1)))
  def test_scalar_from_tensors(self, t):
    t = tf.Variable(t)
    value = callbacks.get_scalar_from_tensor(t)
    print (value)
    self.assertTrue(np.isscalar(value))


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class CustomTensorBoardTest(callbacks_test.TestTensorBoardV2):

  def test_custom_tb_learning_rate(self):
    os.chdir(self.get_temp_dir())
    model = self._get_model()
    x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
    tb_cbk = callbacks.CustomTensorBoard(log_dir=self.logdir,
                                         track_lr=True)

    model.fit(
        x,
        y,
        batch_size=2,
        epochs=2,
        validation_data=(x, y),
        callbacks=[tb_cbk])

    summary_file = callbacks_test.list_summaries(logdir=self.logdir)
    self.assertEqual(
        summary_file.scalars, {
            _ObservedSummary(logdir=self.train_dir, tag='epoch_loss'),
            _ObservedSummary(logdir=self.train_dir, tag='epoch_learning_rate'),
            _ObservedSummary(logdir=self.validation_dir, tag='epoch_loss'),
        })


if __name__ == '__main__':
  tf.test.main()
