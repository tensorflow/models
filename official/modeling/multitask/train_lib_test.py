# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for multitask.train_lib."""
from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.core import task_factory
from official.modeling.hyperparams import params_dict
from official.modeling.multitask import configs
from official.modeling.multitask import multitask
from official.modeling.multitask import test_utils
from official.modeling.multitask import train_lib


class TrainLibTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._test_config = {
        'trainer': {
            'checkpoint_interval': 10,
            'steps_per_loop': 10,
            'summary_interval': 10,
            'train_steps': 10,
            'validation_steps': 5,
            'validation_interval': 10,
            'continuous_eval_timeout': 1,
            'optimizer_config': {
                'optimizer': {
                    'type': 'sgd',
                },
                'learning_rate': {
                    'type': 'constant'
                }
            }
        },
    }

  @combinations.generate(
      combinations.combine(
          distribution_strategy=[
              strategy_combinations.default_strategy,
              strategy_combinations.cloud_tpu_strategy,
              strategy_combinations.one_device_strategy_gpu,
          ],
          mode='eager',
          optimizer=['sgd_experimental', 'sgd'],
          flag_mode=['train', 'eval', 'train_and_eval']))
  def test_end_to_end(self, distribution_strategy, optimizer, flag_mode):
    model_dir = self.get_temp_dir()
    experiment_config = configs.MultiTaskExperimentConfig(
        task=configs.MultiTaskConfig(
            task_routines=(
                configs.TaskRoutine(
                    task_name='foo', task_config=test_utils.FooConfig()),
                configs.TaskRoutine(
                    task_name='bar', task_config=test_utils.BarConfig()))))
    experiment_config = params_dict.override_params_dict(
        experiment_config, self._test_config, is_strict=False)
    experiment_config.trainer.optimizer_config.optimizer.type = optimizer
    with distribution_strategy.scope():
      test_multitask = multitask.MultiTask.from_config(experiment_config.task)
      model = test_utils.MockMultiTaskModel()
    train_lib.run_experiment(
        distribution_strategy=distribution_strategy,
        task=test_multitask,
        model=model,
        mode=flag_mode,
        params=experiment_config,
        model_dir=model_dir)

  @combinations.generate(
      combinations.combine(
          distribution_strategy=[
              strategy_combinations.default_strategy,
              strategy_combinations.cloud_tpu_strategy,
              strategy_combinations.one_device_strategy_gpu,
          ],
          mode='eager',
          flag_mode=['train', 'eval', 'train_and_eval']))
  def test_end_to_end_multi_eval(self, distribution_strategy, flag_mode):
    model_dir = self.get_temp_dir()
    experiment_config = configs.MultiEvalExperimentConfig(
        task=test_utils.FooConfig(),
        eval_tasks=(configs.TaskRoutine(
            task_name='foo', task_config=test_utils.FooConfig(), eval_steps=2),
                    configs.TaskRoutine(
                        task_name='bar',
                        task_config=test_utils.BarConfig(),
                        eval_steps=3)))
    experiment_config = params_dict.override_params_dict(
        experiment_config, self._test_config, is_strict=False)
    with distribution_strategy.scope():
      train_task = task_factory.get_task(experiment_config.task)
      eval_tasks = [
          task_factory.get_task(config.task_config, name=config.task_name)
          for config in experiment_config.eval_tasks
      ]
    train_lib.run_experiment_with_multitask_eval(
        distribution_strategy=distribution_strategy,
        train_task=train_task,
        eval_tasks=eval_tasks,
        mode=flag_mode,
        params=experiment_config,
        model_dir=model_dir)


if __name__ == '__main__':
  tf.test.main()
