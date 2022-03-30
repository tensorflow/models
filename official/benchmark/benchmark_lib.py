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
"""TFM common benchmark training driver."""
import os
import time
from typing import Any, Mapping

from absl import logging
import orbit
import tensorflow as tf
from official.benchmark import tflite_utils
from official.common import distribute_utils
from official.core import config_definitions
from official.core import task_factory
from official.core import train_utils
from official.modeling import performance
from official.modeling.fast_training import stage_lib
from official.projects.token_dropping import experiment_configs  # pylint: disable=unused-import


def run_benchmark(
    execution_mode: str,
    params: config_definitions.ExperimentConfig,
    model_dir: str,
    distribution_strategy: tf.distribute.Strategy = None
) -> Mapping[str, Any]:
  """Runs benchmark for a specific experiment.

  Args:
    execution_mode: A 'str', specifying the mode. Can be 'accuracy',
      'performance', or 'tflite_accuracy'.
    params: ExperimentConfig instance.
    model_dir: A 'str', a path to store model checkpoints and summaries.
    distribution_strategy: A tf.distribute.Strategy to use. If specified,
     it will be used instead of inferring the strategy from params.

  Returns:
    benchmark_data: returns benchmark data in dict format.

  Raises:
    NotImplementedError: If try to use unsupported setup.
  """

  # For GPU runs, allow option to set thread mode
  if params.runtime.gpu_thread_mode:
    os.environ['TF_GPU_THREAD_MODE'] = params.runtime.gpu_thread_mode
    logging.info('TF_GPU_THREAD_MODE: %s', os.environ['TF_GPU_THREAD_MODE'])

  # Sets mixed_precision policy. Using 'mixed_float16' or 'mixed_bfloat16'
  # can have significant impact on model speeds by utilizing float16 in case of
  # GPUs, and bfloat16 in the case of TPUs. loss_scale takes effect only when
  # dtype is float16
  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype)

  strategy = distribution_strategy or distribute_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      tpu_address=params.runtime.tpu)

  with strategy.scope():
    task = task_factory.get_task(params.task, logging_dir=model_dir)
    trainer = train_utils.create_trainer(
        params,
        task,
        train=True,
        evaluate=(execution_mode == 'accuracy'))
    # Initialize the model if possible, e.g., from a pre-trained checkpoint.
    trainer.initialize()

  steps_per_loop = params.trainer.steps_per_loop if (
      execution_mode in ['accuracy', 'tflite_accuracy']) else 100
  controller = orbit.Controller(
      strategy=strategy,
      trainer=trainer,
      evaluator=trainer if (execution_mode == 'accuracy') else None,
      global_step=trainer.global_step,
      steps_per_loop=steps_per_loop)

  logging.info('Starts to execute execution mode: %s', execution_mode)
  with strategy.scope():

    # Training for one loop, first loop time includes warmup time.
    first_loop_start_time = time.time()
    controller.train(steps=steps_per_loop)
    first_loop_time = time.time() - first_loop_start_time
    # Training for second loop.
    second_loop_start_time = time.time()
    controller.train(steps=2*steps_per_loop)
    second_loop_time = time.time() - second_loop_start_time

    if execution_mode == 'accuracy':
      controller.train(steps=params.trainer.train_steps)
      wall_time = time.time() - first_loop_time
      eval_logs = trainer.evaluate(
          tf.convert_to_tensor(params.trainer.validation_steps))
      benchmark_data = {'metrics': eval_logs}
    elif execution_mode == 'performance':
      benchmark_data = {}
    elif execution_mode == 'tflite_accuracy':
      eval_logs = tflite_utils.train_and_evaluate(
          params, task, trainer, controller)
      benchmark_data = {'metrics': eval_logs}
    else:
      raise NotImplementedError(
          'The benchmark execution mode is not implemented: %s' %
          execution_mode)

    # First training loop time contains startup time plus training time, while
    # second training loop time is purely training time. Startup time can be
    # recovered by subtracting second trianing loop time from first training
    # loop time.
    startup_time = first_loop_time - second_loop_time
    wall_time = time.time() - first_loop_start_time
    examples_per_second = steps_per_loop * params.task.train_data.global_batch_size / second_loop_time
    benchmark_data.update(
        dict(
            examples_per_second=examples_per_second,
            wall_time=wall_time,
            startup_time=startup_time))

    return benchmark_data


def run_fast_training_benchmark(
    execution_mode: str,
    params: config_definitions.ExperimentConfig,
    model_dir: str,
    distribution_strategy: tf.distribute.Strategy = None
) -> Mapping[str, Any]:
  """Runs benchmark for a fast training experiment.

  This benchmark tests and only tests the binary
  tensorflow_models/official/modeling/fast_training/train.py

  Args:
    execution_mode: A 'str', specifying the mode. Can be 'accuracy',
      'performance', or 'tflite_accuracy'.
    params: ExperimentConfig instance.
    model_dir: A 'str', a path to store model checkpoints and summaries.
    distribution_strategy: A tf.distribute.Strategy to use. If specified,
     it will be used instead of inferring the strategy from params.

  Returns:
    benchmark_data: returns benchmark data in dict format.

  Raises:
    NotImplementedError: If try to use unsupported setup.
  """
  if execution_mode == 'performance':
    logging.warn('Fast training benchmark does not support execution_mode == '
                 'performance. This benchmark run will be skipped..')
    return dict(examples_per_second=0.0,
                wall_time=0.0,
                startup_time=0.0)

  strategy = distribution_strategy or distribute_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      tpu_address=params.runtime.tpu)

  first_loop_start_time = time.time()
  _, eval_logs = stage_lib.run_progressive_experiment(
      distribution_strategy=strategy,
      mode='train',
      params=params,
      model_dir=model_dir,
      run_post_eval=True)
  wall_time = time.time() - first_loop_start_time

  return dict(metrics=eval_logs, wall_time=wall_time,
              startup_time=0.0, examples_per_second=0.0)
