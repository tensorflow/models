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
"""Test Transformer's schedule manager."""

import tensorflow.compat.v1 as tf

from official.r1.transformer import schedule


class ScheduleBaseTester(tf.test.TestCase):
  def test_mutual_exclusivity(self):
    with self.assertRaises(ValueError):
      schedule.Manager(
          train_steps=100, steps_between_evals=100, train_epochs=2,
          epochs_between_evals=1, default_train_epochs=None, batch_size=2048,
          max_length=256)

  def test_step_basis(self):
    manager = schedule.Manager(
        train_steps=1000, steps_between_evals=100, train_epochs=None,
        epochs_between_evals=None, default_train_epochs=None, batch_size=2048,
        max_length=256)

    self.assertEqual(manager.single_iteration_train_steps, 100)

    # Evaluation uses the full set
    self.assertIsNone(manager.single_iteration_eval_steps)

    self.assertIsNone(manager.repeat_dataset)

  def test_epoch_basis(self):
    manager = schedule.Manager(
        train_steps=None, steps_between_evals=None, train_epochs=10,
        epochs_between_evals=2, default_train_epochs=None, batch_size=2048,
        max_length=256)

    # For non-TPU, estimator relies on dataset exhausion
    self.assertIsNone(manager.single_iteration_train_steps)
    self.assertIsNone(manager.single_iteration_eval_steps)

    self.assertEqual(manager.repeat_dataset, 2)

  def test_step_basis_tpu(self):
    manager = schedule.Manager(
        train_steps=1000, steps_between_evals=100, train_epochs=None,
        epochs_between_evals=None, default_train_epochs=None, batch_size=2048,
        max_length=256, use_tpu=True)

    self.assertEqual(manager.single_iteration_train_steps, 100)
    # num_eval_examples / (batch_size / max_length) == 3000 / (2048 / 256)
    self.assertEqual(manager.single_iteration_eval_steps, 375)
    self.assertIsNone(manager.repeat_dataset)

  def test_epoch_basis_tpu(self):
    manager = schedule.Manager(
        train_steps=None, steps_between_evals=None, train_epochs=10,
        epochs_between_evals=2, default_train_epochs=None, batch_size=2048,
        max_length=256, use_tpu=True)

    self.assertEqual(
        manager.single_iteration_train_steps,
        schedule.NUM_EXAMPLES[tf.estimator.ModeKeys.TRAIN] * 2 // (2048 / 256)
    )

    # num_eval_examples / (batch_size / max_length) == 3000 / (2048 / 256)
    self.assertEqual(manager.single_iteration_eval_steps, 375)

    self.assertEqual(manager.repeat_dataset, 2)


if __name__ == "__main__":
  tf.test.main()
