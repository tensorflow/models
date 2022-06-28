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
from typing import Any, Mapping, Optional

from absl import logging
import orbit
import tensorflow as tf
from official.benchmark import tflite_utils
from official.common import distribute_utils
from official.core import config_definitions
from official.core import task_factory
from official.core import train_utils
from official.modeling import performance
from official.projects.token_dropping import experiment_configs  # pylint: disable=unused-import


class _OutputRecorderAction:
  """Simple `Action` that saves the outputs passed to `__call__`."""

  def __init__(self):
    self.train_output = {}

  def __call__(
      self,
      output: Optional[Mapping[str, tf.Tensor]] = None) -> Mapping[str, Any]:
    self.train_output = {k: v.numpy() for k, v in output.items()
                        } if output else {}


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

  train_output_recorder = _OutputRecorderAction()
  controller = orbit.Controller(
      strategy=strategy,
      trainer=trainer,
      evaluator=trainer if (execution_mode == 'accuracy') else None,
      train_actions=[train_output_recorder],
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
      if train_output_recorder.train_output:
        benchmark_data = {'metrics': train_output_recorder.train_output}
      else:
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
