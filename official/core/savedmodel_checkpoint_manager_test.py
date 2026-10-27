# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

import os
import time
from typing import Any, Iterable, Optional
from unittest import mock

import tensorflow as tf, tf_keras

from official.core import savedmodel_checkpoint_manager


_DEFAULT_MODULES = object()


def _models_exist(checkpoint_path: str, models: Iterable[str]) -> bool:
  for model_name in models:
    if not tf.io.gfile.isdir(
        os.path.join(
            savedmodel_checkpoint_manager.make_saved_modules_directory_name(
                checkpoint_path), model_name)):
      return False
  return True


class _ModelForTest(tf_keras.Model):
  def __init__(self, hidden_size: int = 8):
    super().__init__()
    self.dense = tf_keras.layers.Dense(hidden_size)

  @tf.function(input_signature=[tf.TensorSpec([None, 16])])
  def call(self, inputs):
    return self.dense(inputs)

  @property
  def saved_model_signatures(self):
    # Build SavedModel signatures.
    return dict(serving_default=self.call)


class CheckpointManagerTest(tf.test.TestCase):

  def _create_manager(
      self,
      max_to_keep: int = 1,
      step_counter: Optional[tf.Variable] = None,
      checkpoint_interval: Optional[int] = None,
      directory: Optional[str] = None,
      modules_to_export: Any = _DEFAULT_MODULES,
  ) -> savedmodel_checkpoint_manager.SavedModelCheckpointManager:
    """Sets up SavedModelCheckpointManager object.

    Args:
      max_to_keep: max number of savedmodels to keep.
      step_counter: Optional step counter Variable for checkpoint interval.
      checkpoint_interval: Optional minimum step interval between checkpoints.
      directory: Optional directory path. Defaults to self.get_temp_dir().
      modules_to_export: Optional mapping of modules to export. Defaults to
        sample models with saved_model_signatures.

    Returns:
      Created SavedModelCheckpointManager instance.
    """
    if modules_to_export is _DEFAULT_MODULES:
      modules_to_export = {
          'model_1': _ModelForTest(12),
          'model_2': _ModelForTest(14),
      }
    if step_counter is not None:
      checkpoint = tf.train.Checkpoint(step=step_counter)
    else:
      checkpoint = tf.train.Checkpoint()
    manager = savedmodel_checkpoint_manager.SavedModelCheckpointManager(
        checkpoint=checkpoint,
        directory=directory or self.get_temp_dir(),
        max_to_keep=max_to_keep,
        modules_to_export=modules_to_export,
        step_counter=step_counter,
        checkpoint_interval=checkpoint_interval,
    )
    return manager

  def test_max_to_keep(self):
    manager = self._create_manager()
    models = manager.modules_to_export
    first_path = manager.save()
    second_path = manager.save()

    savedmodel = savedmodel_checkpoint_manager.make_saved_modules_directory_name(
        manager.latest_checkpoint)
    self.assertEqual(savedmodel, manager.latest_savedmodel)
    self.assertTrue(_models_exist(second_path, models.keys()))
    self.assertFalse(_models_exist(first_path, models.keys()))

  def test_returns_none_after_timeout(self):
    manager = self._create_manager()
    start = time.time()
    ret = manager.wait_for_new_savedmodel(
        None, timeout=1.0, seconds_to_sleep=0.5)
    end = time.time()
    self.assertIsNone(ret)
    # We've waited 0.5 second.
    self.assertGreater(end, start + 0.5)
    # The timeout kicked in.
    self.assertLess(end, start + 0.6)

  def test_saved_model_iterator(self):
    manager = self._create_manager(max_to_keep=2)
    self.assertIsNotNone(manager.save(checkpoint_number=1))
    self.assertIsNotNone(manager.save(checkpoint_number=2))
    self.assertIsNotNone(manager.save(checkpoint_number=3))

    # Savedmodels are in time order.
    expected_savedmodels = manager.savedmodels
    # Order not guaranteed.
    existing_savedmodels = manager.get_existing_savedmodels()
    savedmodels = list(manager.savedmodels_iterator(timeout=3.0))
    self.assertEqual(savedmodels, expected_savedmodels)
    self.assertEqual(set(savedmodels), set(existing_savedmodels))

  def test_saved_model_iterator_timeout_fn(self):
    manager = self._create_manager()
    timeout_fn_calls = [0]

    def timeout_fn():
      timeout_fn_calls[0] += 1
      return timeout_fn_calls[0] > 3

    results = list(
        manager.savedmodels_iterator(timeout=0.1, timeout_fn=timeout_fn))
    self.assertEqual([], results)
    self.assertEqual(4, timeout_fn_calls[0])

  def test_recovers_missing_saved_model_when_checkpoint_already_exists(self):
    step_counter = tf.Variable(10, dtype=tf.int64)
    manager = self._create_manager(
        max_to_keep=2,
        step_counter=step_counter,
        checkpoint_interval=10,
    )
    models = manager.modules_to_export
    ckpt_path = manager.save(checkpoint_number=10)
    self.assertIsNotNone(ckpt_path)

    # Save checkpoint 20.
    step_counter.assign(20)
    ckpt_path_20 = manager.save(checkpoint_number=20)
    self.assertIsNotNone(ckpt_path_20)
    saved_modules_dir = (
        savedmodel_checkpoint_manager.make_saved_modules_directory_name(
            ckpt_path_20
        )
    )
    self.assertTrue(_models_exist(ckpt_path_20, models.keys()))

    # Simulate crash during export: remove the exported saved_modules and
    # leave a dirty _temp directory in its place.
    tf.io.gfile.rmtree(saved_modules_dir)
    temp_dir = saved_modules_dir + '_temp'
    stale_file = os.path.join(temp_dir, 'stale_leftover.txt')
    tf.io.gfile.makedirs(temp_dir)
    with tf.io.gfile.GFile(stale_file, 'w') as f:
      f.write('dirty data')
    self.assertTrue(tf.io.gfile.exists(stale_file))
    self.assertFalse(tf.io.gfile.exists(saved_modules_dir))

    # At step 20, super().save() returns None because step 20 was already saved.
    # Calling save(check_interval=False) should detect that the SavedModel is
    # missing, wipe _temp, and re-export it.
    ret_path = manager.save(checkpoint_number=20, check_interval=False)
    self.assertEqual(ret_path, ckpt_path_20)
    self.assertFalse(tf.io.gfile.exists(temp_dir))
    self.assertTrue(tf.io.gfile.exists(saved_modules_dir))
    self.assertFalse(
        tf.io.gfile.exists(
            os.path.join(saved_modules_dir, 'stale_leftover.txt')
        )
    )
    self.assertTrue(_models_exist(ckpt_path_20, models.keys()))

  def test_recovers_missing_saved_model_with_matching_step(self):
    step_counter = tf.Variable(10, dtype=tf.int64)
    manager = self._create_manager(
        max_to_keep=2,
        step_counter=step_counter,
        checkpoint_interval=10,
    )
    models = manager.modules_to_export
    ckpt_path = manager.save(checkpoint_number=10)
    self.assertIsNotNone(ckpt_path)
    saved_modules_dir = (
        savedmodel_checkpoint_manager.make_saved_modules_directory_name(
            ckpt_path
        )
    )
    # Remove saved_modules to simulate missing export.
    tf.io.gfile.rmtree(saved_modules_dir)
    self.assertFalse(tf.io.gfile.exists(saved_modules_dir))

    # At step 10, super().save() returns None because step 10 was already saved.
    # With check_interval=True, since checkpoint_number matches latest
    # checkpoint and the SavedModel is missing, it should re-export.
    ret_path = manager.save(checkpoint_number=10, check_interval=True)
    self.assertEqual(ret_path, ckpt_path)
    self.assertTrue(tf.io.gfile.exists(saved_modules_dir))
    self.assertTrue(_models_exist(ckpt_path, models.keys()))

  def test_recovers_missing_saved_model_after_process_restart(self):
    ckpt_dir = os.path.join(self.get_temp_dir(), 'restart')
    step_counter1 = tf.Variable(10, dtype=tf.int64)
    # Simulate Process 1: saves checkpoint and is interrupted midway during
    # export.
    manager1 = self._create_manager(
        max_to_keep=2,
        step_counter=step_counter1,
        checkpoint_interval=10,
        directory=ckpt_dir,
    )
    models = manager1.modules_to_export
    ckpt_path = manager1.save(checkpoint_number=10)
    self.assertIsNotNone(ckpt_path)
    saved_modules_dir = (
        savedmodel_checkpoint_manager.make_saved_modules_directory_name(
            ckpt_path
        )
    )
    # Simulate preemption midway through export:
    tf.io.gfile.rmtree(saved_modules_dir)
    temp_dir = saved_modules_dir + '_temp'
    tf.io.gfile.makedirs(os.path.join(temp_dir, 'partial_data'))
    self.assertTrue(tf.io.gfile.exists(temp_dir))
    self.assertFalse(tf.io.gfile.exists(saved_modules_dir))

    # Simulate Process 2: Borg restarts the worker.
    # A fresh manager instance is initialized on the same directory.
    step_counter2 = tf.Variable(0, dtype=tf.int64)
    manager2 = self._create_manager(
        max_to_keep=2,
        step_counter=step_counter2,
        checkpoint_interval=10,
        directory=ckpt_dir,
    )
    self.assertEqual(manager2.latest_checkpoint, ckpt_path)
    self.assertNotIn(saved_modules_dir, manager2.savedmodels)

    # Restoring checkpoint sets step_counter2 to 10 and updates
    # manager2._last_checkpoint_step to 10.
    manager2.restore_or_initialize()
    self.assertEqual(step_counter2.numpy(), 10)

    # Calling save on the new manager: super().save() returns None because
    # step 10 is already the last checkpoint step. Manager should recover
    # and export the missing SavedModel.
    ret_path = manager2.save(check_interval=True)
    self.assertEqual(ret_path, ckpt_path)
    self.assertFalse(tf.io.gfile.exists(temp_dir))
    self.assertTrue(tf.io.gfile.exists(saved_modules_dir))
    self.assertIn(saved_modules_dir, manager2.savedmodels)
    self.assertTrue(_models_exist(ckpt_path, models.keys()))

  def test_does_not_reexport_when_step_does_not_match_with_check_interval(self):
    step_counter = tf.Variable(10, dtype=tf.int64)
    manager = self._create_manager(
        max_to_keep=2,
        step_counter=step_counter,
        checkpoint_interval=10,
    )
    ckpt_path = manager.save(checkpoint_number=10)
    self.assertIsNotNone(ckpt_path)
    saved_modules_dir = (
        savedmodel_checkpoint_manager.make_saved_modules_directory_name(
            ckpt_path
        )
    )
    # Remove saved_modules to simulate missing export.
    tf.io.gfile.rmtree(saved_modules_dir)
    self.assertFalse(tf.io.gfile.exists(saved_modules_dir))

    # Advance step_counter to 15 (less than interval 10 + 10 = 20).
    # super().save() returns None because interval has not elapsed.
    # Because step 15 does not match latest checkpoint (ckpt-10),
    # save(check_interval=True) must return None and NOT re-export.
    step_counter.assign(15)
    ret_path = manager.save(check_interval=True)
    self.assertIsNone(ret_path)
    self.assertFalse(tf.io.gfile.exists(saved_modules_dir))

  def test_does_not_reexport_when_saved_model_already_exists(self):
    step_counter = tf.Variable(10, dtype=tf.int64)
    manager = self._create_manager(
        max_to_keep=2,
        step_counter=step_counter,
        checkpoint_interval=10,
    )
    ckpt_path = manager.save(checkpoint_number=10)
    self.assertIsNotNone(ckpt_path)
    saved_modules_dir = (
        savedmodel_checkpoint_manager.make_saved_modules_directory_name(
            ckpt_path
        )
    )
    self.assertTrue(tf.io.gfile.exists(saved_modules_dir))

    # At step 10, super().save() returns None because step 10 was already saved.
    # Since saved_modules already exists, save() must return None and not
    # re-export.
    ret_path = manager.save(checkpoint_number=10, check_interval=True)
    self.assertIsNone(ret_path)

  def test_does_not_recover_when_modules_to_export_is_empty(self):
    step_counter = tf.Variable(10, dtype=tf.int64)
    manager = self._create_manager(
        max_to_keep=2,
        step_counter=step_counter,
        checkpoint_interval=10,
        modules_to_export=None,
    )
    ckpt_path = manager.save(checkpoint_number=10)
    self.assertIsNotNone(ckpt_path)

    # Calling save again at step 10: super().save() returns None.
    # Since modules_to_export is None, save() must return None instead of
    # returning latest_checkpoint.
    ret_path = manager.save(checkpoint_number=10, check_interval=False)
    self.assertIsNone(ret_path)

  def test_does_not_recover_when_no_module_defines_signatures(self):
    step_counter = tf.Variable(10, dtype=tf.int64)
    dummy_module = tf.Module()
    manager = self._create_manager(
        max_to_keep=2,
        step_counter=step_counter,
        checkpoint_interval=10,
        modules_to_export={'dummy': dummy_module},
    )
    ckpt_path = manager.save(checkpoint_number=10)
    self.assertIsNotNone(ckpt_path)

    # Calling save again at step 10: super().save() returns None.
    # Since dummy_module has no saved_model_signatures, save() must return None
    # instead of returning latest_checkpoint.
    ret_path = manager.save(checkpoint_number=10, check_interval=False)
    self.assertIsNone(ret_path)

  def test_cleans_up_temp_directory_on_export_failure(self):
    manager = self._create_manager()
    saved_modules_dir = (
        savedmodel_checkpoint_manager.make_saved_modules_directory_name(
            os.path.join(self.get_temp_dir(), 'ckpt-1')
        )
    )
    temp_dir = saved_modules_dir + '_temp'

    def failing_save(obj, export_dir, signatures):
      del obj, signatures
      tf.io.gfile.makedirs(export_dir)
      raise RuntimeError('Simulated export failure')

    with mock.patch.object(tf.saved_model, 'save', side_effect=failing_save):
      with self.assertRaises(RuntimeError):
        manager.save(checkpoint_number=1)

    self.assertFalse(tf.io.gfile.exists(temp_dir))
    self.assertFalse(tf.io.gfile.exists(saved_modules_dir))


if __name__ == '__main__':
  tf.test.main()
