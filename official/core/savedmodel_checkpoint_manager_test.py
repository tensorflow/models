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

  def testSimpleTest(self):
    models = {
        "model_1":
            tf.keras.Sequential(
                layers=[tf.keras.layers.Dense(8, input_shape=(16,))]),
        "model_2":
            tf.keras.Sequential(
                layers=[tf.keras.layers.Dense(16, input_shape=(32,))]),
    }
    checkpoint = tf.train.Checkpoint()
    manager = savedmodel_checkpoint_manager.SavedModelCheckpointManager(
        checkpoint=checkpoint,
        directory=self.get_temp_dir(),
        max_to_keep=1,
        modules_to_export=models)

    first_path = manager.save()
    second_path = manager.save()

    savedmodel = savedmodel_checkpoint_manager.make_saved_modules_directory_name(
        manager.latest_checkpoint)
    self.assertEqual(savedmodel, manager.latest_savedmodel)
    self.assertTrue(_models_exist(second_path, models.keys()))
    self.assertFalse(_models_exist(first_path, models.keys()))


if __name__ == "__main__":
  tf.test.main()
