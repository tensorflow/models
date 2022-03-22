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
"""Common benchmark class for model garden models."""

import os
import pprint

# Import libraries

from absl import logging
import gin
import tensorflow as tf

from tensorflow.python.platform import benchmark  # pylint: disable=unused-import
from official.common import registry_imports  # pylint: disable=unused-import
from official.benchmark import benchmark_lib
from official.benchmark import benchmark_definitions
from official.benchmark import config_utils
from official.core import exp_factory
from official.modeling import hyperparams


def _get_benchmark_params(benchmark_models, eval_tflite=False):
  """Formats benchmark params into a list."""
  parameterized_benchmark_params = []
  for _, benchmarks in benchmark_models.items():
    for name, params in benchmarks.items():
      if eval_tflite:
        execution_modes = ['performance', 'tflite_accuracy']
      else:
        execution_modes = ['performance', 'accuracy']
      for execution_mode in execution_modes:
        benchmark_name = '{}.{}'.format(name, execution_mode)
        benchmark_params = (
            benchmark_name,  # First arg is used by ParameterizedBenchmark.
            benchmark_name,
            params.get('benchmark_function') or benchmark_lib.run_benchmark,
            params['experiment_type'],
            execution_mode,
            params['platform'],
            params['precision'],
            params['metric_bounds'],
            params.get('config_files') or [],
            params.get('params_override') or None,
            params.get('gin_file') or [])
        parameterized_benchmark_params.append(benchmark_params)
  return parameterized_benchmark_params


class BaseBenchmark(  # pylint: disable=undefined-variable
    tf.test.Benchmark, metaclass=benchmark.ParameterizedBenchmark):
  """Common Benchmark.

     benchmark.ParameterizedBenchmark is used to auto create benchmarks from
     benchmark method according to the benchmarks defined in
     benchmark_definitions. The name of the new benchmark methods is
     benchmark__{benchmark_name}. _get_benchmark_params is used to generate the
     benchmark name and args.
  """

  _benchmark_parameters = _get_benchmark_params(
      benchmark_definitions.VISION_BENCHMARKS) + _get_benchmark_params(
          benchmark_definitions.NLP_BENCHMARKS) + _get_benchmark_params(
              benchmark_definitions.QAT_BENCHMARKS, True)

  def __init__(self,
               output_dir=None,
               tpu=None):
    """Initialize class.

    Args:
      output_dir: Base directory to store all output for the test.
      tpu: (optional) TPU name to use in a TPU benchmark.
    """

    if os.getenv('BENCHMARK_OUTPUT_DIR'):
      self.output_dir = os.getenv('BENCHMARK_OUTPUT_DIR')
    elif output_dir:
      self.output_dir = output_dir
    else:
      self.output_dir = '/tmp'

    if os.getenv('BENCHMARK_TPU'):
      self._resolved_tpu = os.getenv('BENCHMARK_TPU')
    elif tpu:
      self._resolved_tpu = tpu
    else:
      self._resolved_tpu = None

  def _get_model_dir(self, folder_name):
    """Returns directory to store info, e.g. saved model and event log."""
    return os.path.join(self.output_dir, folder_name)

  def benchmark(self,
                benchmark_name,
                benchmark_function,
                experiment_type,
                execution_mode,
                platform,
                precision,
                metric_bounds,
                config_files,
                params_override,
                gin_file):

    with gin.unlock_config():
      gin.parse_config_files_and_bindings(
          [config_utils.get_config_path(g) for g in gin_file], None)

    params = exp_factory.get_exp_config(experiment_type)

    for config_file in config_files:
      file_path = config_utils.get_config_path(config_file)
      params = hyperparams.override_params_dict(
          params, file_path, is_strict=True)

    if params_override:
      params = hyperparams.override_params_dict(
          params, params_override, is_strict=True)
    # platform in format tpu.[n]x[n] or gpu.[n]
    if 'tpu' in platform:
      params.runtime.distribution_strategy = 'tpu'
      params.runtime.tpu = self._resolved_tpu
    elif 'gpu' in platform:
      params.runtime.num_gpus = int(platform.split('.')[-1])
      params.runtime.distribution_strategy = 'mirrored'
    else:
      NotImplementedError('platform :{} is not supported'.format(platform))

    params.runtime.mixed_precision_dtype = precision

    params.validate()
    params.lock()

    tf.io.gfile.makedirs(self._get_model_dir(benchmark_name))
    hyperparams.save_params_dict_to_yaml(
        params,
        os.path.join(self._get_model_dir(benchmark_name), 'params.yaml'))

    pp = pprint.PrettyPrinter()
    logging.info('Final experiment parameters: %s',
                 pp.pformat(params.as_dict()))

    benchmark_data = benchmark_function(
        execution_mode, params, self._get_model_dir(benchmark_name))

    metrics = []
    if execution_mode in ['accuracy', 'tflite_accuracy']:
      for metric_bound in metric_bounds:
        metric = {
            'name': metric_bound['name'],
            'value': benchmark_data['metrics'][metric_bound['name']],
            'min_value': metric_bound['min_value'],
            'max_value': metric_bound['max_value']
        }
        metrics.append(metric)

    metrics.append({'name': 'startup_time',
                    'value': benchmark_data['startup_time']})
    metrics.append({'name': 'exp_per_second',
                    'value': benchmark_data['examples_per_second']})

    self.report_benchmark(
        iters=-1,
        wall_time=benchmark_data['wall_time'],
        metrics=metrics,
        extras={'model_name': benchmark_name.split('.')[0],
                'platform': platform,
                'implementation': 'orbit.ctl',
                'parameters': precision})


if __name__ == '__main__':
  tf.test.main()
