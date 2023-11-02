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

"""Tests for the progressive train_lib."""
import os

from absl import flags
from absl.testing import parameterized
import dataclasses
import orbit
import tensorflow as tf, tf_keras

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.common import flags as tfm_flags
# pylint: disable=unused-import
from official.common import registry_imports
# pylint: enable=unused-import
from official.core import config_definitions as cfg
from official.core import task_factory
from official.modeling import optimization
from official.modeling.hyperparams import params_dict
from official.modeling.fast_training.progressive import policies
from official.modeling.fast_training.progressive import train_lib
from official.modeling.fast_training.progressive import trainer as prog_trainer_lib
from official.utils.testing import mock_task

FLAGS = flags.FLAGS

tfm_flags.define_flags()


@dataclasses.dataclass
class ProgTaskConfig(cfg.TaskConfig):
  pass


@task_factory.register_task_cls(ProgTaskConfig)
class ProgMockTask(policies.ProgressivePolicy, mock_task.MockTask):
  """Progressive task for testing."""

  def __init__(self, params: cfg.TaskConfig, logging_dir: str = None):
    mock_task.MockTask.__init__(
        self, params=params, logging_dir=logging_dir)
    policies.ProgressivePolicy.__init__(self)

  def num_stages(self):
    return 2

  def num_steps(self, stage_id):
    return 2 if stage_id == 0 else 4

  def get_model(self, stage_id, old_model=None):
    del stage_id, old_model
    return self.build_model()

  def get_optimizer(self, stage_id):
    """Build optimizer for each stage."""
    params = optimization.OptimizationConfig({
        'optimizer': {
            'type': 'adamw',
        },
        'learning_rate': {
            'type': 'polynomial',
            'polynomial': {
                'initial_learning_rate': 0.01,
                'end_learning_rate': 0.0,
                'power': 1.0,
                'decay_steps': 10,
            },
        },
        'warmup': {
            'polynomial': {
                'power': 1,
                'warmup_steps': 2,
            },
            'type': 'polynomial',
        }
    })
    opt_factory = optimization.OptimizerFactory(params)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())

    return optimizer

  def get_train_dataset(self, stage_id):
    del stage_id
    strategy = tf.distribute.get_strategy()
    return orbit.utils.make_distributed_dataset(
        strategy, self.build_inputs, None)

  def get_eval_dataset(self, stage_id):
    del stage_id
    strategy = tf.distribute.get_strategy()
    return orbit.utils.make_distributed_dataset(
        strategy, self.build_inputs, None)


class TrainTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(TrainTest, self).setUp()
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
          flag_mode=['train', 'eval', 'train_and_eval'],
          run_post_eval=[True, False]))
  def test_end_to_end(self, distribution_strategy, flag_mode, run_post_eval):
    model_dir = self.get_temp_dir()
    experiment_config = cfg.ExperimentConfig(
        trainer=prog_trainer_lib.ProgressiveTrainerConfig(),
        task=ProgTaskConfig())
    experiment_config = params_dict.override_params_dict(
        experiment_config, self._test_config, is_strict=False)

    with distribution_strategy.scope():
      task = task_factory.get_task(experiment_config.task,
                                   logging_dir=model_dir)

    _, logs = train_lib.run_experiment(
        distribution_strategy=distribution_strategy,
        task=task,
        mode=flag_mode,
        params=experiment_config,
        model_dir=model_dir,
        run_post_eval=run_post_eval)

    if run_post_eval:
      self.assertNotEmpty(logs)
    else:
      self.assertEmpty(logs)

    if flag_mode == 'eval':
      return
    self.assertNotEmpty(
        tf.io.gfile.glob(os.path.join(model_dir, 'checkpoint')))
    # Tests continuous evaluation.
    _, logs = train_lib.run_experiment(
        distribution_strategy=distribution_strategy,
        task=task,
        mode='continuous_eval',
        params=experiment_config,
        model_dir=model_dir,
        run_post_eval=run_post_eval)
    print(logs)


if __name__ == '__main__':
  tf.test.main()
