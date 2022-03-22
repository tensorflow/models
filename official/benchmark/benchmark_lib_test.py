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
"""Tests for tensorflow_models.official.benchmark.benchmark_lib."""
# pylint: disable=g-direct-tensorflow-import

from absl.testing import parameterized
import gin
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.common import registry_imports  # pylint: disable=unused-import
from official.benchmark import benchmark_lib
from official.core import exp_factory
from official.modeling import hyperparams


def all_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.cloud_tpu_strategy,
          strategy_combinations.one_device_strategy_gpu,
      ],)


class BenchmarkLibTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(BenchmarkLibTest, self).setUp()
    self._test_config = {
        'trainer': {
            'steps_per_loop': 10,
            'optimizer_config': {
                'optimizer': {
                    'type': 'sgd'
                },
                'learning_rate': {
                    'type': 'constant'
                }
            },
            'continuous_eval_timeout': 5,
            'train_steps': 20,
            'validation_steps': 10
        },
    }

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.default_strategy,
              strategy_combinations.cloud_tpu_strategy,
              strategy_combinations.one_device_strategy_gpu,
          ],
          execution_mode=['performance', 'accuracy'],
      ))
  def test_benchmark(self, distribution, execution_mode):

    model_dir = self.get_temp_dir()
    params = exp_factory.get_exp_config('mock')
    params = hyperparams.override_params_dict(
        params, self._test_config, is_strict=True)

    benchmark_data = benchmark_lib.run_benchmark(execution_mode,
                                                 params,
                                                 model_dir,
                                                 distribution)

    self.assertIn('examples_per_second', benchmark_data)
    self.assertIn('wall_time', benchmark_data)
    self.assertIn('startup_time', benchmark_data)

    if execution_mode == 'accuracy':
      self.assertIn('metrics', benchmark_data)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.default_strategy,
              strategy_combinations.cloud_tpu_strategy,
              strategy_combinations.one_device_strategy_gpu,
          ],
          execution_mode=['performance', 'accuracy'],
      ))
  def test_fast_training_benchmark(self, distribution, execution_mode):

    model_dir = self.get_temp_dir()
    with gin.unlock_config():
      gin.parse_config_files_and_bindings(
          None,
          "get_initialize_fn.stacking_pattern = 'dense_{:layer_id}/'\n"
          "StageParamProgressor.stage_overrides = ("
          "    {'trainer': {'train_steps': 1}},"
          "    {'trainer': {'train_steps': 2}},"
          ")")
    params = exp_factory.get_exp_config('mock')
    params = hyperparams.override_params_dict(
        params, self._test_config, is_strict=True)

    benchmark_data = benchmark_lib.run_fast_training_benchmark(execution_mode,
                                                               params,
                                                               model_dir,
                                                               distribution)

    if execution_mode == 'performance':
      self.assertEqual(dict(examples_per_second=0.0,
                            wall_time=0.0,
                            startup_time=0.0),
                       benchmark_data)
    else:
      self.assertIn('wall_time', benchmark_data)
      self.assertIn('metrics', benchmark_data)

if __name__ == '__main__':
  tf.test.main()
