# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow_models.core.base_task."""

import functools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.utils.testing import mock_task


def all_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.tpu_strategy,
          strategy_combinations.one_device_strategy_gpu,
      ],
      mode='eager',
  )


class TaskKerasTest(tf.test.TestCase, parameterized.TestCase):

  @combinations.generate(all_strategy_combinations())
  def test_task_with_step_override(self, distribution):
    with distribution.scope():
      task = mock_task.MockTask()
      model = task.build_model()
      model = task.compile_model(
          model,
          optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
          metrics=task.build_metrics(),
          train_step=task.train_step,
          validation_step=task.validation_step)

    dataset = task.build_inputs(params=None)
    logs = model.fit(dataset, epochs=1, steps_per_epoch=2)
    self.assertIn('loss', logs.history)
    self.assertIn('acc', logs.history)

    # Without specifying metrics through compile.
    with distribution.scope():
      train_metrics = task.build_metrics(training=True)
      val_metrics = task.build_metrics(training=False)
      model = task.build_model()
      model = task.compile_model(
          model,
          optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
          train_step=functools.partial(task.train_step, metrics=train_metrics),
          validation_step=functools.partial(
              task.validation_step, metrics=val_metrics))
    logs = model.fit(dataset, epochs=1, steps_per_epoch=2)
    self.assertIn('loss', logs.history)
    self.assertIn('acc', logs.history)

  def test_task_with_fit(self):
    task = mock_task.MockTask()
    model = task.build_model()
    model = task.compile_model(
        model,
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=task.build_metrics())
    dataset = task.build_inputs(params=None)
    logs = model.fit(dataset, epochs=1, steps_per_epoch=2)
    self.assertIn('loss', logs.history)
    self.assertIn('acc', logs.history)
    self.assertLen(model.evaluate(dataset, steps=1), 2)

  def test_task_invalid_compile(self):
    task = mock_task.MockTask()
    model = task.build_model()
    with self.assertRaises(ValueError):
      _ = task.compile_model(
          model,
          optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
          loss=tf.keras.losses.CategoricalCrossentropy(),
          metrics=task.build_metrics(),
          train_step=task.train_step)


if __name__ == '__main__':
  tf.test.main()
