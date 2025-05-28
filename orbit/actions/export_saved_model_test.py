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

"""Tests for orbit.actions.export_saved_model."""

import os

from orbit import actions
from orbit.actions import export_saved_model

import tensorflow as tf, tf_keras


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


class ExportSavedModelTest(tf.test.TestCase):

  def test_export_file_manager_default_ids(self):
    directory = self.create_tempdir()
    base_name = os.path.join(directory.full_path, 'basename')
    manager = actions.ExportFileManager(base_name, max_to_keep=3)
    self.assertEmpty(tf.io.gfile.listdir(directory.full_path))
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
    self.assertEmpty(tf.io.gfile.listdir(directory.full_path))
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

  def test_export_file_manager_with_suffix(self):
    directory = self.create_tempdir()
    base_name = os.path.join(directory.full_path, 'basename')

    id_num = 0

    def next_id():
      return id_num

    subdirectory = 'sub'

    manager = actions.ExportFileManager(
        base_name, max_to_keep=2, next_id_fn=next_id, subdirectory=subdirectory
    )
    self.assertEmpty(tf.io.gfile.listdir(directory.full_path))
    id_num = 30
    directory.create_file(manager.next_name())
    self.assertLen(tf.io.gfile.listdir(directory.full_path), 1)
    manager.clean_up()  # Shouldn't do anything...
    self.assertEqual(
        _id_sorted_file_base_names(directory.full_path), ['basename-30']
    )
    id_num = 200
    directory.create_file(manager.next_name())
    self.assertLen(tf.io.gfile.listdir(directory.full_path), 2)
    manager.clean_up()  # Shouldn't do anything...
    self.assertEqual(
        _id_sorted_file_base_names(directory.full_path),
        ['basename-30', 'basename-200'],
    )
    id_num = 1000
    directory.create_file(manager.next_name())
    self.assertLen(tf.io.gfile.listdir(directory.full_path), 3)
    self.assertEqual(
        _id_sorted_file_base_names(directory.full_path),
        ['basename-30', 'basename-200', 'basename-1000'],
    )
    manager.clean_up()  # Should delete file with lowest ID.
    self.assertLen(tf.io.gfile.listdir(directory.full_path), 3)
    # Note that the base folder is intact, only the suffix folder is deleted.
    self.assertEqual(
        _id_sorted_file_base_names(directory.full_path),
        ['basename-30', 'basename-200', 'basename-1000'],
    )

    step_folder = os.path.join(directory.full_path, 'basename-1000')
    self.assertIn(subdirectory, tf.io.gfile.listdir(step_folder))

  def test_export_file_manager_with_suffix_second_cleanup_succeeds(self):
    directory = self.create_tempdir()
    base_name = os.path.join(directory.full_path, 'basename')

    id_num = 0

    def next_id():
      return id_num

    subdirectory = 'sub'

    manager = actions.ExportFileManager(
        base_name, max_to_keep=2, next_id_fn=next_id, subdirectory=subdirectory
    )
    id_num = 30
    directory.create_file(manager.next_name())
    id_num = 200
    directory.create_file(manager.next_name())
    id_num = 1000
    directory.create_file(manager.next_name())
    manager.clean_up()  # Should delete file with lowest ID.
    # Note that the base folder is intact, only the suffix folder is deleted.
    self.assertEqual(
        _id_sorted_file_base_names(directory.full_path),
        ['basename-30', 'basename-200', 'basename-1000'],
    )
    # Verify that the suffix folder has been deleted from the lowest ID
    # but not from the others.
    self.assertEmpty(
        tf.io.gfile.listdir(os.path.join(directory.full_path, 'basename-30'))
    )
    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(directory.full_path, 'basename-200'))
    )
    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(directory.full_path, 'basename-1000'))
    )
    # Add another ID, run clean_up again and verify that it worked.
    id_num = 2000
    directory.create_file(manager.next_name())
    manager.clean_up()  # Should delete file with lowest ID.
    # Verify that the suffix folder has been deleted from the two lowest ID
    # directories but not from the others.
    self.assertEmpty(
        tf.io.gfile.listdir(os.path.join(directory.full_path, 'basename-30'))
    )
    self.assertEmpty(
        tf.io.gfile.listdir(os.path.join(directory.full_path, 'basename-200'))
    )
    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(directory.full_path, 'basename-1000'))
    )
    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(directory.full_path, 'basename-2000'))
    )

  def test_export_file_manager_managed_files(self):
    directory = self.create_tempdir()
    directory.create_file('basename-5')
    directory.create_file('basename-10')
    directory.create_file('basename-50')
    directory.create_file('basename-1000')
    directory.create_file('basename-9')
    directory.create_file('basename-10-suffix')
    base_name = os.path.join(directory.full_path, 'basename')
    manager = actions.ExportFileManager(base_name, max_to_keep=3)
    self.assertLen(manager.managed_files, 5)
    self.assertEqual(manager.next_name(), f'{base_name}-1001')
    manager.clean_up()
    self.assertEqual(
        manager.managed_files,
        [f'{base_name}-10', f'{base_name}-50', f'{base_name}-1000'])

  def test_export_file_manager_managed_files_double_slash(self):
    directory = self.create_tempdir('foo//bar')
    directory.create_file('basename-5')
    directory.create_file('basename-10')
    directory.create_file('basename-50')
    directory.create_file('basename-1000')
    directory.create_file('basename-9')
    directory.create_file('basename-10-suffix')
    base_name = os.path.join(directory.full_path, 'basename')
    expected_base_name = os.path.normpath(base_name)
    self.assertNotEqual(base_name, expected_base_name)
    manager = actions.ExportFileManager(base_name, max_to_keep=3)
    self.assertLen(manager.managed_files, 5)
    self.assertEqual(manager.next_name(), f'{expected_base_name}-1001')
    manager.clean_up()
    self.assertEqual(manager.managed_files, [
        f'{expected_base_name}-10', f'{expected_base_name}-50',
        f'{expected_base_name}-1000'
    ])

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

  def test_safe_normpath_gs(self):
    path = export_saved_model.safe_normpath('gs://foo//bar')
    self.assertEqual(path, 'gs://foo/bar')


if __name__ == '__main__':
  tf.test.main()
