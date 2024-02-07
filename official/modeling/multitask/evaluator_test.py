# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for multitask.evaluator."""
from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.core import base_task
from official.core import config_definitions as cfg
from official.modeling.multitask import evaluator


def all_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.cloud_tpu_strategy,
          strategy_combinations.one_device_strategy_gpu,
      ],
      mode="eager",
  )


class MockModel(tf_keras.Model):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.dense = tf_keras.layers.Dense(1)

  def call(self, inputs):
    print(inputs, type(inputs))
    if "y" in inputs:
      self.add_loss(tf.zeros((1,), dtype=tf.float32))
    else:
      self.add_loss(tf.ones((1,), dtype=tf.float32))
    return self.dense(inputs["x"])


class MockTask(base_task.Task):
  """Mock task object for testing."""

  def build_metrics(self, training: bool = True):
    del training
    return [tf_keras.metrics.Accuracy(name="acc")]

  def build_inputs(self, params):

    def generate_data(_):
      x = tf.zeros(shape=(2,), dtype=tf.float32)
      label = tf.zeros([1], dtype=tf.int32)
      if self.name == "bar":
        return dict(x=x, y=x), label
      else:
        return dict(x=x), label

    dataset = tf.data.Dataset.range(1)
    dataset = dataset.repeat()
    dataset = dataset.map(
        generate_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset.prefetch(buffer_size=1).batch(2, drop_remainder=True)

  def validation_step(self, inputs, model: tf_keras.Model, metrics=None):
    logs = super().validation_step(inputs, model, metrics)
    logs["counter"] = tf.ones((1,), dtype=tf.float32)
    return logs

  def aggregate_logs(self, state, step_outputs):
    if state is None:
      state = {}
    for key, value in step_outputs.items():
      if key not in state:
        state[key] = []
      state[key].append(
          np.concatenate([np.expand_dims(v.numpy(), axis=0) for v in value]))
    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    for k, v in aggregated_logs.items():
      aggregated_logs[k] = np.sum(np.stack(v, axis=0))
    return aggregated_logs


class EvaluatorTest(tf.test.TestCase, parameterized.TestCase):

  @combinations.generate(all_strategy_combinations())
  def test_multitask_evaluator(self, distribution):
    with distribution.scope():
      tasks = [
          MockTask(params=cfg.TaskConfig(), name="bar"),
          MockTask(params=cfg.TaskConfig(), name="foo")
      ]
      model = MockModel()
      test_evaluator = evaluator.MultiTaskEvaluator(
          eval_tasks=tasks, model=model)
      results = test_evaluator.evaluate(tf.convert_to_tensor(1, dtype=tf.int32))
    self.assertContainsSubset(["validation_loss", "acc"], results["bar"].keys())
    self.assertContainsSubset(["validation_loss", "acc"], results["foo"].keys())
    self.assertEqual(results["bar"]["validation_loss"], 0.0)
    self.assertEqual(results["foo"]["validation_loss"], 1.0)

  @combinations.generate(all_strategy_combinations())
  def test_multitask_evaluator_numpy_metrics(self, distribution):
    with distribution.scope():
      tasks = [
          MockTask(params=cfg.TaskConfig(), name="bar"),
          MockTask(params=cfg.TaskConfig(), name="foo")
      ]
      model = MockModel()
      test_evaluator = evaluator.MultiTaskEvaluator(
          eval_tasks=tasks, model=model)
      results = test_evaluator.evaluate(tf.convert_to_tensor(5, dtype=tf.int32))
    self.assertEqual(results["bar"]["counter"],
                     5. * distribution.num_replicas_in_sync)
    self.assertEqual(results["foo"]["counter"],
                     5. * distribution.num_replicas_in_sync)


if __name__ == "__main__":
  tf.test.main()
