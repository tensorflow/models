# Copyright 2025 The Orbit Authors. All Rights Reserved.
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

"""Tests for orbit.actions.new_best_metric."""

import os

from orbit import actions

import tensorflow as tf, tf_keras


class NewBestMetricTest(tf.test.TestCase):

  def test_new_best_metric_higher_is_better(self):
    new_best_metric = actions.NewBestMetric(
        lambda x: x['value'], higher_is_better=True)
    self.assertTrue(new_best_metric.test({'value': 0.0}))
    self.assertTrue(new_best_metric.commit({'value': 0.0}))
    self.assertFalse(new_best_metric.test({'value': 0.0}))
    self.assertTrue(new_best_metric.test({'value': 1.0}))

  def test_new_best_metric_lower_is_better(self):
    new_best_metric = actions.NewBestMetric('value', higher_is_better=False)
    self.assertTrue(new_best_metric.test({'value': 0.0}))
    self.assertTrue(new_best_metric.commit({'value': 0.0}))
    self.assertFalse(new_best_metric.test({'value': 0.0}))
    self.assertTrue(new_best_metric.test({'value': -1.0}))

  def test_new_best_metric_persistence(self):
    backing_file = self.create_tempfile()
    new_best_metric = actions.NewBestMetric(
        'value',
        higher_is_better=True,
        filename=backing_file.full_path,
        write_metric=False)
    self.assertTrue(new_best_metric.test({'value': 0.0}))
    self.assertTrue(new_best_metric.commit({'value': 0.0}))
    self.assertFalse(new_best_metric.test({'value': 0.0}))
    new_best_metric = actions.NewBestMetric(
        'value', higher_is_better=True, filename=backing_file.full_path)
    self.assertLess(new_best_metric.best_value, 0.0)
    self.assertTrue(new_best_metric.commit({'value': 5.0}))
    self.assertEqual(new_best_metric.best_value, 5.0)
    new_best_metric = actions.NewBestMetric(
        'value', higher_is_better=True, filename=backing_file.full_path)
    self.assertEqual(new_best_metric.best_value, 5.0)

  def test_json_persisted_value(self):
    tempfile = self.create_tempfile().full_path
    value = {'a': 1, 'b': 2}
    persisted_value = actions.JSONPersistedValue(value, tempfile)
    # The initial value is used since tempfile is empty.
    self.assertEqual(persisted_value.read(), value)
    persisted_value = actions.JSONPersistedValue('ignored', tempfile)
    # Initial value of 'ignored' is ignored, since there's a value in tempfile.
    self.assertEqual(persisted_value.read(), value)
    value = [1, 2, 3]
    persisted_value.write(value)
    # Now that a new value is written, it gets read on initialization.
    persisted_value = actions.JSONPersistedValue(['also ignored'], tempfile)
    self.assertEqual(persisted_value.read(), value)
    # Writes can be disabled.
    persisted_value = actions.JSONPersistedValue(
        'ignored', tempfile, write_value=False)
    self.assertEqual(persisted_value.read(), value)
    persisted_value.write("won't get persisted")
    persisted_value = actions.JSONPersistedValue(
        'ignored', tempfile, write_value=False)
    self.assertEqual(persisted_value.read(), value)

  def test_json_persisted_value_create_dirs(self):
    tempfile = os.path.join(self.create_tempdir().full_path, 'subdir/value')
    value = {'a': 1, 'b': 2}
    # The directory is not created if write_value=False.
    actions.JSONPersistedValue(value, tempfile, write_value=False)
    self.assertFalse(tf.io.gfile.exists(os.path.dirname(tempfile)))
    actions.JSONPersistedValue(value, tempfile)
    self.assertTrue(tf.io.gfile.exists(tempfile))


if __name__ == '__main__':
  tf.test.main()
