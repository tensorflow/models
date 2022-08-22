# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
from typing import Iterable

import tensorflow as tf

from official.core import savedmodel_checkpoint_manager


def _models_exist(checkpoint_path: str, models: Iterable[str]) -> bool:
  for model_name in models:
    if not tf.io.gfile.isdir(
        os.path.join(
            savedmodel_checkpoint_manager.make_saved_modules_directory_name(
                checkpoint_path), model_name)):
      return False
  return True


class CheckpointManagerTest(tf.test.TestCase):

  def _create_manager(self, max_to_keep: int = 1) -> tf.train.CheckpointManager:
    """Sets up SavedModelCheckpointManager object.

    Args:
      max_to_keep: max number of savedmodels to keep.

    Returns:
      created savedmodel manager.
    """
    models = {
        'model_1':
            tf.keras.Sequential(
                layers=[tf.keras.layers.Dense(8, input_shape=(16,))]),
        'model_2':
            tf.keras.Sequential(
                layers=[tf.keras.layers.Dense(16, input_shape=(32,))]),
    }
    checkpoint = tf.train.Checkpoint()
    manager = savedmodel_checkpoint_manager.SavedModelCheckpointManager(
        checkpoint=checkpoint,
        directory=self.get_temp_dir(),
        max_to_keep=max_to_keep,
        modules_to_export=models)
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


if __name__ == '__main__':
  tf.test.main()
