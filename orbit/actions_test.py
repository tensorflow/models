# Copyright 2021 The Orbit Authors. All Rights Reserved.
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

"""Tests for orbit.actions."""

import os

from orbit import actions

import tensorflow as tf


def _id_key(name):
  _, id_num = name.rsplit('-', maxsplit=1)
  return int(id_num)


def _id_sorted_file_base_names(dir_path):
  return sorted(tf.io.gfile.listdir(dir_path), key=_id_key)


class TestModel(tf.Module):

  def __init__(self):
    self.value = tf.Variable(0)

  @tf.function(input_signature=[])
  def __call__(self):
    return self.value


class ActionsTest(tf.test.TestCase):

  def test_conditional_action(self):
    # Define a function to raise an AssertionError, since we can't in a lambda.
    def raise_assertion(arg):
      raise AssertionError(str(arg))

    conditional_action = actions.ConditionalAction(
        condition=lambda x: x, action=raise_assertion)

    conditional_action(False)  # Nothing is raised.
    with self.assertRaises(AssertionError) as ctx:
      conditional_action(True)
      self.assertEqual(ctx.exception.message, 'True')

  def test_new_best_metric_higher_is_better(self):
    new_best_metric = actions.NewBestMetric(lambda x: x, higher_is_better=True)
    self.assertTrue(new_best_metric.test(0.0))
    self.assertTrue(new_best_metric.commit(0.0))
    self.assertFalse(new_best_metric.test(0.0))
    self.assertTrue(new_best_metric.test(1.0))

  def test_new_best_metric_lower_is_better(self):
    new_best_metric = actions.NewBestMetric(lambda x: x, higher_is_better=False)
    self.assertTrue(new_best_metric.test(0.0))
    self.assertTrue(new_best_metric.commit(0.0))
    self.assertFalse(new_best_metric.test(0.0))
    self.assertTrue(new_best_metric.test(-1.0))

  def test_new_best_metric_persistence(self):
    backing_file = self.create_tempfile()
    new_best_metric = actions.NewBestMetric(
        lambda x: x,
        higher_is_better=True,
        filename=backing_file.full_path,
        write_metric=False)
    self.assertTrue(new_best_metric.test(0.0))
    self.assertTrue(new_best_metric.commit(0.0))
    self.assertFalse(new_best_metric.test(0.0))
    new_best_metric = actions.NewBestMetric(
        lambda x: x, higher_is_better=True, filename=backing_file.full_path)
    self.assertLess(new_best_metric.best_value, 0.0)
    self.assertTrue(new_best_metric.commit(5.0))
    self.assertEqual(new_best_metric.best_value, 5.0)
    new_best_metric = actions.NewBestMetric(
        lambda x: x, higher_is_better=True, filename=backing_file.full_path)
    self.assertEqual(new_best_metric.best_value, 5.0)

  def test_json_persisted_value(self):
    tempfile = self.create_tempfile().full_path
    value = {'a': 1, 'b': 2}
    persisted_value = actions.JSONPersistedValue(value, tempfile)
    # The inital value is used since tempfile is empty.
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

  def test_export_file_manager_default_ids(self):
    directory = self.create_tempdir()
    base_name = os.path.join(directory.full_path, 'basename')
    manager = actions.ExportFileManager(base_name, max_to_keep=3)
    self.assertLen(tf.io.gfile.listdir(directory.full_path), 0)
    directory.create_file(manager.next_name())
    manager.clean_up()  # Shouldn't do anything...
    self.assertLen(tf.io.gfile.listdir(directory.full_path), 1)
    directory.create_file(manager.next_name())
    manager.clean_up()  # Shouldn't do anything...
    self.assertLen(tf.io.gfile.listdir(directory.full_path), 2)
    directory.create_file(manager.next_name())
    manager.clean_up()  # Shouldn't do anything...
    self.assertLen(tf.io.gfile.listdir(directory.full_path), 3)
    directory.create_file(manager.next_name())
    self.assertLen(tf.io.gfile.listdir(directory.full_path), 4)
    self.assertEqual(
        _id_sorted_file_base_names(directory.full_path),
        ['basename-0', 'basename-1', 'basename-2', 'basename-3'])
    manager.clean_up()  # Should delete file with lowest ID.
    self.assertEqual(
        _id_sorted_file_base_names(directory.full_path),
        ['basename-1', 'basename-2', 'basename-3'])
    manager = actions.ExportFileManager(base_name, max_to_keep=3)
    self.assertEqual(os.path.basename(manager.next_name()), 'basename-4')

  def test_export_file_manager_custom_ids(self):
    directory = self.create_tempdir()
    base_name = os.path.join(directory.full_path, 'basename')

    id_num = 0

    def next_id():
      return id_num

    manager = actions.ExportFileManager(
        base_name, max_to_keep=2, next_id_fn=next_id)
    self.assertLen(tf.io.gfile.listdir(directory.full_path), 0)
    id_num = 30
    directory.create_file(manager.next_name())
    self.assertLen(tf.io.gfile.listdir(directory.full_path), 1)
    manager.clean_up()  # Shouldn't do anything...
    self.assertEqual(
        _id_sorted_file_base_names(directory.full_path), ['basename-30'])
    id_num = 200
    directory.create_file(manager.next_name())
    self.assertLen(tf.io.gfile.listdir(directory.full_path), 2)
    manager.clean_up()  # Shouldn't do anything...
    self.assertEqual(
        _id_sorted_file_base_names(directory.full_path),
        ['basename-30', 'basename-200'])
    id_num = 1000
    directory.create_file(manager.next_name())
    self.assertLen(tf.io.gfile.listdir(directory.full_path), 3)
    self.assertEqual(
        _id_sorted_file_base_names(directory.full_path),
        ['basename-30', 'basename-200', 'basename-1000'])
    manager.clean_up()  # Should delete file with lowest ID.
    self.assertLen(tf.io.gfile.listdir(directory.full_path), 2)
    self.assertEqual(
        _id_sorted_file_base_names(directory.full_path),
        ['basename-200', 'basename-1000'])

  def test_export_saved_model(self):
    directory = self.create_tempdir()
    base_name = os.path.join(directory.full_path, 'basename')
    file_manager = actions.ExportFileManager(base_name, max_to_keep=2)
    model = TestModel()
    export_action = actions.ExportSavedModel(
        model, file_manager=file_manager, signatures=model.__call__)

    model.value.assign(3)
    self.assertEqual(model(), 3)
    self.assertEmpty(file_manager.managed_files)
    export_action({})
    self.assertLen(file_manager.managed_files, 1)
    reloaded_model = tf.saved_model.load(file_manager.managed_files[-1])
    self.assertEqual(reloaded_model(), 3)

    model.value.assign(5)
    self.assertEqual(model(), 5)
    export_action({})
    self.assertLen(file_manager.managed_files, 2)
    reloaded_model = tf.saved_model.load(file_manager.managed_files[-1])
    self.assertEqual(reloaded_model(), 5)

    model.value.assign(7)
    self.assertEqual(model(), 7)
    export_action({})
    self.assertLen(file_manager.managed_files, 2)  # Still 2, due to clean up.
    reloaded_model = tf.saved_model.load(file_manager.managed_files[-1])
    self.assertEqual(reloaded_model(), 7)


if __name__ == '__main__':
  tf.test.main()
