# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for multitask.interleaving_trainer."""
from absl.testing import parameterized
import tensorflow as tf, tf_keras

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.modeling.multitask import configs
from official.modeling.multitask import interleaving_trainer
from official.modeling.multitask import multitask
from official.modeling.multitask import task_sampler
from official.modeling.multitask import test_utils


def all_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.cloud_tpu_strategy,
          strategy_combinations.one_device_strategy_gpu,
      ],
      mode="eager",
  )


class InterleavingTrainerTest(tf.test.TestCase, parameterized.TestCase):

  @combinations.generate(all_strategy_combinations())
  def test_multitask_interleaving_trainer(self, distribution):
    with distribution.scope():
      tasks = [
          test_utils.MockFooTask(params=test_utils.FooConfig(), name="foo"),
          test_utils.MockBarTask(params=test_utils.BarConfig(), name="bar")
      ]
      test_multitask = multitask.MultiTask(tasks=tasks)
      test_optimizer = tf_keras.optimizers.SGD(0.1)
      model = test_utils.MockMultiTaskModel()
      sampler = task_sampler.UniformTaskSampler(
          task_weights=test_multitask.task_weights)
      test_trainer = interleaving_trainer.MultiTaskInterleavingTrainer(
          multi_task=test_multitask,
          multi_task_model=model,
          optimizer=test_optimizer,
          task_sampler=sampler)
      results = test_trainer.train(tf.convert_to_tensor(5, dtype=tf.int32))
      self.assertContainsSubset(["training_loss", "bar_acc"],
                                results["bar"].keys())
      self.assertContainsSubset(["training_loss", "foo_acc"],
                                results["foo"].keys())
      self.assertNotIn("total_loss", results)

  @combinations.generate(all_strategy_combinations())
  def test_trainer_with_configs(self, distribution):
    config = configs.MultiTaskConfig(
        task_routines=(configs.TaskRoutine(
            task_name="foo",
            task_config=test_utils.FooConfig(),
            task_weight=3.0),
                       configs.TaskRoutine(
                           task_name="bar",
                           task_config=test_utils.BarConfig(),
                           task_weight=1.0)))
    with distribution.scope():
      test_multitask = multitask.MultiTask.from_config(config)
    test_optimizer = tf_keras.optimizers.SGD(0.1)
    model = test_utils.MockMultiTaskModel()
    num_step = 1000
    sampler = task_sampler.AnnealingTaskSampler(
        task_weights=test_multitask.task_weights,
        steps_per_epoch=num_step/5,
        total_steps=num_step)
    test_trainer = interleaving_trainer.MultiTaskInterleavingTrainer(
        multi_task=test_multitask,
        multi_task_model=model,
        optimizer=test_optimizer,
        task_sampler=sampler)
    results = test_trainer.train(tf.convert_to_tensor(num_step, dtype=tf.int32))
    self.assertContainsSubset(["training_loss", "bar_acc"],
                              results["bar"].keys())
    self.assertContainsSubset(["training_loss", "foo_acc"],
                              results["foo"].keys())
    self.assertEqual(test_trainer.global_step.numpy(), num_step)
    bar_sampled_step = test_trainer.task_step_counter("bar").numpy()
    foo_sampled_step = test_trainer.task_step_counter("foo").numpy()
    self.assertEqual(bar_sampled_step + foo_sampled_step, num_step)


if __name__ == "__main__":
  tf.test.main()
