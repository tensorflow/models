# Lint as: python3
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
"""Executes Keras benchmarks and accuracy tests."""
# pylint: disable=line-too-long
from __future__ import print_function

import json
import os
import time

from typing import Any, MutableMapping, Optional

from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.benchmark import benchmark_wrappers
from official.benchmark import keras_benchmark
from official.benchmark.models import resnet_imagenet_main
from official.vision.image_classification import classifier_trainer

MIN_TOP_1_ACCURACY = 0.76
MAX_TOP_1_ACCURACY = 0.77

MOBILENET_V1_MIN_TOP_1_ACCURACY = 0.65
MOBILENET_V1_MAX_TOP_1_ACCURACY = 0.68

# Range of top-1 accracies for model optimization techniques.
# Each item indicates (MIN_TOP_1_ACCURACY, MAX_TOP_1_ACCURACY).
MODEL_OPTIMIZATION_TOP_1_ACCURACY = {
    'RESNET50_FINETUNE_PRUNING': (0.76, 0.77),
    'MOBILENET_V1_FINETUNE_PRUNING': (0.67, 0.68),
    'MOBILENET_V1_FINETUNE_CLUSTERING': (0.68, 0.70)
}

FLAGS = flags.FLAGS


def _get_classifier_parameters(
    model_variant: Optional[str] = None,
    num_gpus: int = 0,
    builder: str = 'records',
    skip_eval: bool = False,
    distribution_strategy: str = 'mirrored',
    per_replica_batch_size: int = 128,
    epochs: int = 90,
    steps: int = 0,
    epochs_between_evals: int = 1,
    dtype: str = 'float32',
    enable_xla: bool = False,
    run_eagerly: bool = False,
    gpu_thread_mode: Optional[str] = None,
    dataset_num_private_threads: Optional[int] = None,
    loss_scale: Optional[str] = None,
    report_metrics: bool = True,
    batchnorm_spatial_persistent: bool = False) -> MutableMapping[str, Any]:
  """Gets classifier trainer's ResNet parameters."""
  params = {
      'runtime': {
          'num_gpus': num_gpus,
          'distribution_strategy': distribution_strategy,
          'run_eagerly': run_eagerly,
          'enable_xla': enable_xla,
          'dataset_num_private_threads': dataset_num_private_threads,
          'gpu_thread_mode': gpu_thread_mode,
          'loss_scale': loss_scale,
          'batchnorm_spatial_persistent': batchnorm_spatial_persistent,
      },
      'train_dataset': {
          'builder': builder,
          'use_per_replica_batch_size': True,
          'batch_size': per_replica_batch_size,
          'image_size': 224,
          'dtype': dtype,
      },
      'validation_dataset': {
          'builder': builder,
          'batch_size': per_replica_batch_size,
          'use_per_replica_batch_size': True,
          'image_size': 224,
          'dtype': dtype,
      },
      'train': {
          'epochs': epochs,
          'steps': steps,
          'callbacks': {
              'enable_tensorboard': False,
              'enable_checkpoint_and_export': False,
              'enable_time_history': True,
          },
          'metrics': ['accuracy'] if report_metrics else [],
      },
      'model': {
          'loss': {
              'label_smoothing': 0.1,
          },
      },
      'evaluation': {
          'epochs_between_evals': epochs_between_evals,
          'skip_eval': skip_eval,
      },
  }
  if model_variant is not None:
    params['model']['model_params'] = {
        'model_name': model_variant,
    }
  return params


class Resnet50KerasAccuracy(keras_benchmark.KerasBenchmark):
  """Benchmark accuracy tests for ResNet50 in Keras."""

  def __init__(self,
               output_dir: Optional[str] = None,
               root_data_dir: Optional[str] = None,
               **kwargs):
    """A benchmark class.

    Args:
      output_dir: directory where to output e.g. log files
      root_data_dir: directory under which to look for dataset
      **kwargs: arbitrary named arguments. This is needed to make the
                constructor forward compatible in case PerfZero provides more
                named arguments before updating the constructor.
    """

    flag_methods = [classifier_trainer.define_classifier_flags]

    self.data_dir = os.path.join(root_data_dir, 'imagenet')
    super(Resnet50KerasAccuracy, self).__init__(
        output_dir=output_dir, flag_methods=flag_methods)

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(
      self,
      experiment_name: str,
      top_1_min: float = MIN_TOP_1_ACCURACY,
      top_1_max: float = MAX_TOP_1_ACCURACY,
      num_gpus: int = 0,
      distribution_strategy: str = 'mirrored',
      per_replica_batch_size: int = 128,
      epochs: int = 90,
      steps: int = 0,
      epochs_between_evals: int = 1,
      dtype: str = 'float32',
      enable_xla: bool = False,
      run_eagerly: bool = False,
      gpu_thread_mode: Optional[str] = None,
      dataset_num_private_threads: Optional[int] = None,
      loss_scale: Optional[str] = None):
    """Runs and reports the benchmark given the provided configuration."""
    FLAGS.model_type = 'resnet'
    FLAGS.dataset = 'imagenet'
    FLAGS.mode = 'train_and_eval'
    FLAGS.data_dir = self.data_dir
    FLAGS.model_dir = self._get_model_dir(experiment_name)
    parameters = _get_classifier_parameters(
        num_gpus=num_gpus,
        distribution_strategy=distribution_strategy,
        per_replica_batch_size=per_replica_batch_size,
        epochs=epochs,
        steps=steps,
        epochs_between_evals=epochs_between_evals,
        dtype=dtype,
        enable_xla=enable_xla,
        run_eagerly=run_eagerly,
        gpu_thread_mode=gpu_thread_mode,
        dataset_num_private_threads=dataset_num_private_threads,
        report_metrics=True,
        loss_scale=loss_scale,
        batchnorm_spatial_persistent=True)
    FLAGS.params_override = json.dumps(parameters)
    total_batch_size = num_gpus * per_replica_batch_size

    start_time_sec = time.time()
    stats = classifier_trainer.run(flags.FLAGS)
    wall_time_sec = time.time() - start_time_sec

    super(Resnet50KerasAccuracy, self)._report_benchmark(
        stats,
        wall_time_sec,
        top_1_min=top_1_min,
        top_1_max=top_1_max,
        total_batch_size=total_batch_size,
        log_steps=100)

  def benchmark_8_gpu(self):
    """Tests Keras model with eager, dist_strat and 8 GPUs."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_8_gpu',
        num_gpus=8,
        per_replica_batch_size=128,
        epochs=90,
        epochs_between_evals=10,
        dtype='float32')

  def benchmark_8_gpu_fp16(self):
    """Tests Keras model with eager, dist_strat, 8 GPUs, and fp16."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_8_gpu_fp16',
        num_gpus=8,
        per_replica_batch_size=256,
        epochs=90,
        epochs_between_evals=10,
        dtype='float16')

  def benchmark_xla_8_gpu_fp16(self):
    """Tests Keras model with XLA, eager, dist_strat, 8 GPUs and fp16."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_xla_8_gpu_fp16',
        num_gpus=8,
        per_replica_batch_size=256,
        epochs=90,
        epochs_between_evals=10,
        dtype='float16',
        enable_xla=True)

  def benchmark_xla_8_gpu_fp16_dynamic(self):
    """Tests Keras model with XLA, eager, dist_strat, 8 GPUs, dynamic fp16."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_xla_8_gpu_fp16_dynamic',
        top_1_min=0.736,
        num_gpus=8,
        per_replica_batch_size=256,
        epochs=90,
        epochs_between_evals=10,
        dtype='float16',
        loss_scale='dynamic')

  def _get_model_dir(self, folder_name):
    return os.path.join(self.output_dir, folder_name)


class MobilenetV1KerasAccuracy(keras_benchmark.KerasBenchmark):
  """Benchmark accuracy tests for MobilenetV1 in Keras."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    """A benchmark class.

    Args:
      output_dir: directory where to output e.g. log files
      root_data_dir: directory under which to look for dataset
      **kwargs: arbitrary named arguments. This is needed to make the
                constructor forward compatible in case PerfZero provides more
                named arguments before updating the constructor.
    """

    flag_methods = [resnet_imagenet_main.define_imagenet_keras_flags]

    self.data_dir = os.path.join(root_data_dir, 'imagenet')
    super(MobilenetV1KerasAccuracy, self).__init__(
        output_dir=output_dir,
        flag_methods=flag_methods,
        default_flags={
            'model': 'mobilenet',
            'optimizer': 'mobilenet_default',
            'initial_learning_rate_per_sample': 0.00039,
        })

  def benchmark_8_gpu(self):
    """Test Keras model with eager, dist_strat and 8 GPUs."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128 * 8
    FLAGS.train_epochs = 90
    FLAGS.epochs_between_evals = 10
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu')
    FLAGS.dtype = 'fp32'
    FLAGS.enable_eager = True
    self._run_and_report_benchmark()

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self,
                                top_1_min=MOBILENET_V1_MIN_TOP_1_ACCURACY,
                                top_1_max=MOBILENET_V1_MAX_TOP_1_ACCURACY):
    start_time_sec = time.time()
    stats = resnet_imagenet_main.run(flags.FLAGS)
    wall_time_sec = time.time() - start_time_sec

    super(MobilenetV1KerasAccuracy, self)._report_benchmark(
        stats,
        wall_time_sec,
        top_1_min=top_1_min,
        top_1_max=top_1_max,
        total_batch_size=FLAGS.batch_size,
        log_steps=100)

  def _get_model_dir(self, folder_name):
    return os.path.join(self.output_dir, folder_name)


class KerasClassifierBenchmarkBase(keras_benchmark.KerasBenchmark):
  """Classifier Trainer benchmarks."""

  def __init__(self, model, output_dir=None, default_flags=None,
               tpu=None, dataset_builder='records', train_epochs=1,
               train_steps=110, data_dir=None):
    flag_methods = [classifier_trainer.define_classifier_flags]

    self.model = model
    self.dataset_builder = dataset_builder
    self.train_epochs = train_epochs
    self.train_steps = train_steps
    self.data_dir = data_dir

    super(KerasClassifierBenchmarkBase, self).__init__(
        output_dir=output_dir,
        flag_methods=flag_methods,
        default_flags=default_flags,
        tpu=tpu)

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(
      self,
      experiment_name: str,
      model_variant: Optional[str] = None,
      skip_steps: Optional[int] = None,
      top_1_min: float = MIN_TOP_1_ACCURACY,
      top_1_max: float = MAX_TOP_1_ACCURACY,
      num_gpus: int = 0,
      num_tpus: int = 0,
      distribution_strategy: str = 'mirrored',
      per_replica_batch_size: int = 128,
      epochs_between_evals: int = 1,
      dtype: str = 'float32',
      enable_xla: bool = False,
      run_eagerly: bool = False,
      gpu_thread_mode: Optional[str] = None,
      dataset_num_private_threads: Optional[int] = None,
      loss_scale: Optional[str] = None):
    """Runs and reports the benchmark given the provided configuration."""
    FLAGS.model_type = self.model
    FLAGS.dataset = 'imagenet'
    FLAGS.mode = 'train_and_eval'
    FLAGS.data_dir = self.data_dir
    FLAGS.model_dir = self._get_model_dir(experiment_name)
    parameters = _get_classifier_parameters(
        model_variant=model_variant,
        builder=self.dataset_builder,
        skip_eval=True,
        num_gpus=num_gpus,
        distribution_strategy=distribution_strategy,
        per_replica_batch_size=per_replica_batch_size,
        epochs=self.train_epochs,
        steps=self.train_steps,
        epochs_between_evals=epochs_between_evals,
        dtype=dtype,
        enable_xla=enable_xla,
        gpu_thread_mode=gpu_thread_mode,
        dataset_num_private_threads=dataset_num_private_threads,
        loss_scale=loss_scale,
        report_metrics=False,
        batchnorm_spatial_persistent=True)
    FLAGS.params_override = json.dumps(parameters)
    if distribution_strategy == 'tpu':
      total_batch_size = num_tpus * per_replica_batch_size
    else:
      total_batch_size = num_gpus * per_replica_batch_size

    start_time_sec = time.time()
    stats = classifier_trainer.run(flags.FLAGS)
    wall_time_sec = time.time() - start_time_sec
    # Number of logged step time entries that are excluded in performance
    # report. We keep results from last 100 batches, or skip the steps based on
    # input skip_steps.
    warmup = (skip_steps or (self.train_steps - 100)) // FLAGS.log_steps

    super(KerasClassifierBenchmarkBase, self)._report_benchmark(
        stats,
        wall_time_sec,
        total_batch_size=total_batch_size,
        log_steps=FLAGS.log_steps,
        warmup=warmup,
        start_time_sec=start_time_sec)

  def benchmark_1_gpu_no_dist_strat(self):
    """Tests Keras model with 1 GPU, no distribution strategy."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_1_gpu_no_dist_strat',
        num_gpus=1,
        distribution_strategy='off',
        per_replica_batch_size=128)

  def benchmark_1_gpu_no_dist_strat_run_eagerly(self):
    """Tests Keras model with 1 GPU, no distribution strategy, run eagerly."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_1_gpu_no_dist_strat_run_eagerly',
        num_gpus=1,
        run_eagerly=True,
        distribution_strategy='off',
        per_replica_batch_size=64)

  def benchmark_1_gpu_no_dist_strat_run_eagerly_fp16(self):
    """Tests with 1 GPU, no distribution strategy, fp16, run eagerly."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_1_gpu_no_dist_strat_run_eagerly_fp16',
        num_gpus=1,
        run_eagerly=True,
        distribution_strategy='off',
        dtype='float16',
        per_replica_batch_size=128)

  def benchmark_1_gpu(self):
    """Tests Keras model with 1 GPU."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_1_gpu',
        num_gpus=1,
        distribution_strategy='one_device',
        per_replica_batch_size=128)

  def benchmark_xla_1_gpu(self):
    """Tests Keras model with XLA and 1 GPU."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_xla_1_gpu',
        num_gpus=1,
        enable_xla=True,
        distribution_strategy='one_device',
        per_replica_batch_size=128)

  def benchmark_1_gpu_fp16(self):
    """Tests Keras model with 1 GPU and fp16."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_1_gpu_fp16',
        num_gpus=1,
        distribution_strategy='one_device',
        dtype='float16',
        per_replica_batch_size=256)

  def benchmark_1_gpu_fp16_dynamic(self):
    """Tests Keras model with 1 GPU, fp16, and dynamic loss scaling."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_1_gpu_fp16_dynamic',
        num_gpus=1,
        distribution_strategy='one_device',
        dtype='float16',
        per_replica_batch_size=256,
        loss_scale='dynamic')

  def benchmark_xla_1_gpu_fp16(self):
    """Tests Keras model with XLA, 1 GPU and fp16."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_xla_1_gpu_fp16',
        num_gpus=1,
        enable_xla=True,
        distribution_strategy='one_device',
        dtype='float16',
        per_replica_batch_size=256)

  def benchmark_xla_1_gpu_fp16_tweaked(self):
    """Tests Keras model with XLA, 1 GPU, fp16, and manual config tuning."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_xla_1_gpu_fp16_tweaked',
        num_gpus=1,
        enable_xla=True,
        distribution_strategy='one_device',
        dtype='float16',
        per_replica_batch_size=256,
        gpu_thread_mode='gpu_private')

  def benchmark_xla_1_gpu_fp16_dynamic(self):
    """Tests Keras model with XLA, 1 GPU, fp16, and dynamic loss scaling."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_xla_1_gpu_fp16_dynamic',
        num_gpus=1,
        enable_xla=True,
        distribution_strategy='one_device',
        dtype='float16',
        per_replica_batch_size=256,
        loss_scale='dynamic')

  def benchmark_8_gpu(self):
    """Tests Keras model with 8 GPUs."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_8_gpu',
        num_gpus=8,
        distribution_strategy='mirrored',
        per_replica_batch_size=128)

  def benchmark_8_gpu_tweaked(self):
    """Tests Keras model with manual config tuning and 8 GPUs."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_8_gpu_tweaked',
        num_gpus=8,
        distribution_strategy='mirrored',
        per_replica_batch_size=128,
        dataset_num_private_threads=14)

  def benchmark_xla_8_gpu(self):
    """Tests Keras model with XLA and 8 GPUs."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_xla_8_gpu',
        num_gpus=8,
        enable_xla=True,
        distribution_strategy='mirrored',
        per_replica_batch_size=128)

  def benchmark_xla_8_gpu_tweaked(self):
    """Tests Keras model with manual config tuning, 8 GPUs, and XLA."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_xla_8_gpu_tweaked',
        num_gpus=8,
        enable_xla=True,
        distribution_strategy='mirrored',
        per_replica_batch_size=128,
        gpu_thread_mode='gpu_private',
        dataset_num_private_threads=24)

  def benchmark_8_gpu_fp16(self):
    """Tests Keras model with 8 GPUs and fp16."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_8_gpu_fp16',
        num_gpus=8,
        dtype='float16',
        distribution_strategy='mirrored',
        per_replica_batch_size=256)

  def benchmark_8_gpu_fp16_tweaked(self):
    """Tests Keras model with 8 GPUs, fp16, and manual config tuning."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_8_gpu_fp16_tweaked',
        num_gpus=8,
        dtype='float16',
        distribution_strategy='mirrored',
        per_replica_batch_size=256,
        gpu_thread_mode='gpu_private',
        dataset_num_private_threads=40)

  def benchmark_8_gpu_fp16_dynamic_tweaked(self):
    """Tests Keras model with 8 GPUs, fp16, dynamic loss scaling, and tuned."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_8_gpu_fp16_dynamic_tweaked',
        num_gpus=8,
        dtype='float16',
        distribution_strategy='mirrored',
        per_replica_batch_size=256,
        loss_scale='dynamic',
        gpu_thread_mode='gpu_private',
        dataset_num_private_threads=40)

  def benchmark_xla_8_gpu_fp16(self):
    """Tests Keras model with XLA, 8 GPUs and fp16."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_xla_8_gpu_fp16',
        dtype='float16',
        num_gpus=8,
        enable_xla=True,
        distribution_strategy='mirrored',
        per_replica_batch_size=256)

  def benchmark_xla_8_gpu_fp16_tweaked(self):
    """Test Keras model with manual config tuning, XLA, 8 GPUs and fp16."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_xla_8_gpu_fp16_tweaked',
        dtype='float16',
        num_gpus=8,
        enable_xla=True,
        distribution_strategy='mirrored',
        per_replica_batch_size=256,
        gpu_thread_mode='gpu_private',
        dataset_num_private_threads=48)

  def benchmark_xla_8_gpu_fp16_tweaked_delay_measure(self):
    """Tests with manual config tuning, XLA, 8 GPUs and fp16.

    Delay performance measurement for stable performance on 96 vCPU platforms.
    """
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_xla_8_gpu_fp16_tweaked_delay_measure',
        dtype='float16',
        num_gpus=8,
        enable_xla=True,
        distribution_strategy='mirrored',
        per_replica_batch_size=256,
        gpu_thread_mode='gpu_private',
        dataset_num_private_threads=48)

  def benchmark_xla_8_gpu_fp16_dynamic_tweaked(self):
    """Tests Keras model with config tuning, XLA, 8 GPUs and dynamic fp16."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_xla_8_gpu_fp16_dynamic_tweaked',
        dtype='float16',
        num_gpus=8,
        enable_xla=True,
        distribution_strategy='mirrored',
        per_replica_batch_size=256,
        gpu_thread_mode='gpu_private',
        loss_scale='dynamic',
        dataset_num_private_threads=48)

  def benchmark_2x2_tpu_bf16(self):
    """Test Keras model with 2x2 TPU, bf16."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_2x2_tpu_bf16',
        dtype='bfloat16',
        num_tpus=8,
        distribution_strategy='tpu',
        per_replica_batch_size=128)

  def benchmark_2x2_tpu(self):
    """Test Keras model with 2x2 TPU."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_2x2_tpu',
        num_tpus=8,
        distribution_strategy='tpu',
        per_replica_batch_size=128)

  def benchmark_4x4_tpu_bf16(self):
    """Test Keras model with 4x4 TPU, bf16."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_4x4_tpu_bf16',
        dtype='bfloat16',
        num_tpus=32,
        distribution_strategy='tpu',
        per_replica_batch_size=128)

  def benchmark_4x4_tpu(self):
    """Test Keras model with 4x4 TPU."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_4x4_tpu',
        num_tpus=32,
        distribution_strategy='tpu',
        per_replica_batch_size=128)

  def benchmark_2x2_tpu_bf16_mlir(self):
    """Test Keras model with 2x2 TPU, bf16."""
    self._setup()
    tf.config.experimental.enable_mlir_bridge()
    self._run_and_report_benchmark(
        experiment_name='benchmark_2x2_tpu_bf16_mlir',
        dtype='bfloat16',
        num_tpus=8,
        distribution_strategy='tpu',
        per_replica_batch_size=128)

  def benchmark_4x4_tpu_bf16_mlir(self):
    """Test Keras model with 4x4 TPU, bf16."""
    self._setup()
    tf.config.experimental.enable_mlir_bridge()
    self._run_and_report_benchmark(
        experiment_name='benchmark_4x4_tpu_bf16_mlir',
        dtype='bfloat16',
        num_tpus=32,
        distribution_strategy='tpu',
        per_replica_batch_size=128)

  def benchmark_8x8_tpu_bf16(self):
    """Test Keras model with 8x8 TPU, bf16."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_8x8_tpu_bf16',
        dtype='bfloat16',
        num_tpus=128,
        distribution_strategy='tpu',
        per_replica_batch_size=64)

  def benchmark_8x8_tpu(self):
    """Test Keras model with 8x8 TPU."""
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_8x8_tpu',
        num_tpus=128,
        distribution_strategy='tpu',
        per_replica_batch_size=64)

  def fill_report_object(self, stats):
    super(KerasClassifierBenchmarkBase, self).fill_report_object(
        stats,
        total_batch_size=FLAGS.batch_size,
        log_steps=FLAGS.log_steps)


class Resnet50KerasBenchmarkBase(keras_benchmark.KerasBenchmark):
  """Resnet50 benchmarks."""

  def __init__(self, output_dir=None, default_flags=None, tpu=None):
    flag_methods = [resnet_imagenet_main.define_imagenet_keras_flags]

    super(Resnet50KerasBenchmarkBase, self).__init__(
        output_dir=output_dir,
        flag_methods=flag_methods,
        default_flags=default_flags,
        tpu=tpu)

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self, skip_steps=None):
    start_time_sec = time.time()
    stats = resnet_imagenet_main.run(FLAGS)
    wall_time_sec = time.time() - start_time_sec
    # Number of logged step time entries that are excluded in performance
    # report. We keep results from last 100 batches, or skip the steps based on
    # input skip_steps.
    warmup = (skip_steps or (FLAGS.train_steps - 100)) // FLAGS.log_steps

    super(Resnet50KerasBenchmarkBase, self)._report_benchmark(
        stats,
        wall_time_sec,
        total_batch_size=FLAGS.batch_size,
        log_steps=FLAGS.log_steps,
        warmup=warmup,
        start_time_sec=start_time_sec)

  def benchmark_1_gpu_no_dist_strat(self):
    """Test Keras model with 1 GPU, no distribution strategy."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_no_dist_strat')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_run_eagerly(self):
    """Test Keras model with 1 GPU, no distribution strategy, run eagerly."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.run_eagerly = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu_no_dist_strat_run_eagerly')
    FLAGS.batch_size = 64
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_run_eagerly_tweaked(self):
    """Test Keras model with 1 GPU, no distribution strategy, run eagerly."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.run_eagerly = True
    FLAGS.explicit_gpu_placement = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu_no_dist_strat_run_eagerly_tweaked')
    FLAGS.batch_size = 64
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_run_eagerly_fp16(self):
    """Test with 1 GPU, no distribution strategy, fp16, run eagerly."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.run_eagerly = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu_no_dist_strat_run_eagerly_fp16')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_run_eagerly_fp16_tweaked(self):
    """Test with 1 GPU, no distribution strategy, fp16, run eagerly."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.run_eagerly = True
    FLAGS.explicit_gpu_placement = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu_no_dist_strat_run_eagerly_fp16_tweaked')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_1_gpu(self):
    """Test Keras model with 1 GPU."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_1_gpu_amp(self):
    """Test Keras model with 1 GPU with automatic mixed precision."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.dtype = 'fp16'
    FLAGS.fp16_implementation = 'graph_rewrite'
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_amp')
    FLAGS.batch_size = 256
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu(self):
    """Test Keras model with XLA and 1 GPU."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_amp(self):
    """Test Keras model with XLA and 1 GPU with automatic mixed precision."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.dtype = 'fp16'
    FLAGS.fp16_implementation = 'graph_rewrite'
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_amp')
    FLAGS.batch_size = 256
    self._run_and_report_benchmark()

  def benchmark_1_gpu_fp16(self):
    """Test Keras model with 1 GPU and fp16."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_fp16')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    self._run_and_report_benchmark()

  def benchmark_1_gpu_fp16_dynamic(self):
    """Test Keras model with 1 GPU, fp16, and dynamic loss scaling."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_fp16_dynamic')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    FLAGS.loss_scale = 'dynamic'
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_fp16(self):
    """Test Keras model with XLA, 1 GPU and fp16."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_fp16')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_fp16_tweaked(self):
    """Test Keras model with XLA, 1 GPU, fp16, and manual config tuning."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_fp16_tweaked')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_fp16_dynamic(self):
    """Test Keras model with XLA, 1 GPU, fp16, and dynamic loss scaling."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_fp16_dynamic')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    FLAGS.loss_scale = 'dynamic'
    self._run_and_report_benchmark()

  def benchmark_8_gpu(self):
    """Test Keras model with 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu')
    FLAGS.batch_size = 128 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_8_gpu_fp32_no_tf32(self):
    """Test Keras model with 8 GPUs.Runs in FP32 by disabling TF32 execution."""
    self._setup()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_fp32_no_tf32')
    FLAGS.batch_size = 128 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_8_gpu_amp(self):
    """Test Keras model with 8 GPUs with automatic mixed precision."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.dtype = 'fp16'
    FLAGS.fp16_implementation = 'graph_rewrite'
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_amp')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_8_gpu_tweaked(self):
    """Test Keras model with manual config tuning and 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_tweaked')
    FLAGS.batch_size = 128 * 8  # 8 GPUs
    FLAGS.datasets_num_private_threads = 14
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu(self):
    """Test Keras model with XLA and 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu')
    FLAGS.batch_size = 128 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_amp(self):
    """Test Keras model with XLA and 8 GPUs with automatic mixed precision."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.dtype = 'fp16'
    FLAGS.fp16_implementation = 'graph_rewrite'
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu_amp')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_tweaked(self):
    """Test Keras model with manual config tuning, 8 GPUs, and XLA."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu_tweaked')
    FLAGS.batch_size = 128 * 8
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 24
    self._run_and_report_benchmark()

  def benchmark_8_gpu_fp16(self):
    """Test Keras model with 8 GPUs and fp16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_fp16')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_8_gpu_fp16_tweaked(self):
    """Test Keras model with 8 GPUs, fp16, and manual config tuning."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_fp16_tweaked')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 40
    self._run_and_report_benchmark()

  def benchmark_8_gpu_fp16_dynamic_tweaked(self):
    """Test Keras model with 8 GPUs, fp16, dynamic loss scaling, and tuned."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_8_gpu_fp16_dynamic_tweaked')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.loss_scale = 'dynamic'
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 40
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_fp16(self):
    """Test Keras model with XLA, 8 GPUs and fp16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu_fp16')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_fp16_tweaked(self):
    """Test Keras model with manual config tuning, XLA, 8 GPUs and fp16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu_fp16_tweaked')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 48
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_fp16_tweaked_delay_measure(self):
    """Test with manual config tuning, XLA, 8 GPUs and fp16.

    Delay performance measurement for stable performance on 96 vCPU platforms.
    """
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_xla_8_gpu_fp16_tweaked_delay_measure')
    FLAGS.batch_size = 256 * 8
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 48
    FLAGS.train_steps = 310
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_fp16_dynamic_tweaked(self):
    """Test Keras model with config tuning, XLA, 8 GPUs and dynamic fp16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_xla_8_gpu_fp16_dynamic_tweaked')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.loss_scale = 'dynamic'
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 48
    self._run_and_report_benchmark()

  def benchmark_2x2_tpu_bf16(self):
    """Test Keras model with 2x2 TPU, bf16."""
    self._setup()

    FLAGS.dtype = 'bf16'
    FLAGS.distribution_strategy = 'tpu'
    FLAGS.model_dir = self._get_model_dir('benchmark_2x2_tpu_bf16')
    FLAGS.batch_size = 1024
    self._run_and_report_benchmark()

  def benchmark_4x4_tpu_bf16(self):
    """Test Keras model with 4x4 TPU, bf16."""
    self._setup()

    FLAGS.dtype = 'bf16'
    FLAGS.distribution_strategy = 'tpu'
    FLAGS.model_dir = self._get_model_dir('benchmark_4x4_tpu_bf16')
    FLAGS.batch_size = 4096
    self._run_and_report_benchmark()

  def benchmark_8x8_tpu_bf16(self):
    """Test Keras model with 8x8 TPU, bf16."""
    self._setup()

    FLAGS.dtype = 'bf16'
    FLAGS.distribution_strategy = 'tpu'
    FLAGS.model_dir = self._get_model_dir('benchmark_8x8_tpu_bf16')
    FLAGS.batch_size = 8192
    self._run_and_report_benchmark()

  def fill_report_object(self, stats):
    super(Resnet50KerasBenchmarkBase, self).fill_report_object(
        stats,
        total_batch_size=FLAGS.batch_size,
        log_steps=FLAGS.log_steps)


class Resnet50KerasBenchmarkSynth(KerasClassifierBenchmarkBase):
  """Resnet50 synthetic benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, tpu=None, **kwargs):
    def_flags = {}
    def_flags['log_steps'] = 10

    super(Resnet50KerasBenchmarkSynth, self).__init__(
        model='resnet', output_dir=output_dir, default_flags=def_flags, tpu=tpu,
        dataset_builder='synthetic', train_epochs=1, train_steps=110)


class Resnet50KerasBenchmarkReal(KerasClassifierBenchmarkBase):
  """Resnet50 real data benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, tpu=None, **kwargs):
    data_dir = os.path.join(root_data_dir, 'imagenet')
    def_flags = {}
    def_flags['log_steps'] = 10

    super(Resnet50KerasBenchmarkReal, self).__init__(
        model='resnet', output_dir=output_dir, default_flags=def_flags, tpu=tpu,
        dataset_builder='records', train_epochs=1, train_steps=110,
        data_dir=data_dir)


class EfficientNetKerasBenchmarkReal(KerasClassifierBenchmarkBase):
  """EfficientNet real data benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, tpu=None, **kwargs):
    data_dir = os.path.join(root_data_dir, 'imagenet')
    def_flags = {}
    def_flags['log_steps'] = 10

    super(EfficientNetKerasBenchmarkReal, self).__init__(
        model='efficientnet', output_dir=output_dir, default_flags=def_flags,
        tpu=tpu, dataset_builder='records', train_epochs=1, train_steps=110,
        data_dir=data_dir)

  def benchmark_2x2_tpu_b7_bf16(self):
    self._setup()
    self._run_and_report_benchmark(
        experiment_name='benchmark_b7_2x2_tpu_bf16',
        model_variant='efficientnet-b7',
        dtype='bfloat16',
        num_tpus=8,
        distribution_strategy='tpu',
        per_replica_batch_size=128)


class Resnet50KerasBenchmarkRemoteData(Resnet50KerasBenchmarkBase):
  """Resnet50 real data (stored in remote storage) benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    def_flags = {}
    def_flags['skip_eval'] = True
    def_flags['report_accuracy_metrics'] = False
    def_flags['data_dir'] = os.path.join(root_data_dir, 'imagenet')
    # Defining multiple epochs overrides the train_steps setting in benchmarks.
    def_flags['train_epochs'] = 2
    # Cache dataset so performance is stable after the first epoch.
    def_flags['training_dataset_cache'] = True
    def_flags['log_steps'] = 100
    # Note that for single GPU and pure eager tests which are less likely to be
    # input bound and more stable, these tests will run for shorter time by
    # overriding FLAGS.train_epochs, train_seteps, log_steps in benchmark
    # methods, and skip_steps in _run_and_report_benchmark().

    super(Resnet50KerasBenchmarkRemoteData, self).__init__(
        output_dir=output_dir, default_flags=def_flags)

  def _override_flags_to_run_test_shorter(self):
    FLAGS.train_epochs = 1
    FLAGS.train_steps = 300
    FLAGS.log_steps = 10

  def benchmark_1_gpu_no_dist_strat(self):
    """Test Keras model with 1 GPU, no distribution strategy."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_no_dist_strat')
    FLAGS.batch_size = 128
    self._override_flags_to_run_test_shorter()
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_run_eagerly(self):
    """Test Keras model with 1 GPU, no distribution strategy, run eagerly."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.run_eagerly = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu_no_dist_strat_run_eagerly')
    FLAGS.batch_size = 64
    self._override_flags_to_run_test_shorter()
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_run_eagerly_tweaked(self):
    """Test Keras model with 1 GPU, no distribution strategy, run eagerly."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.run_eagerly = True
    FLAGS.explicit_gpu_placement = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu_no_dist_strat_run_eagerly_tweaked')
    FLAGS.batch_size = 64
    self._override_flags_to_run_test_shorter()
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_run_eagerly_fp16(self):
    """Test with 1 GPU, no distribution strategy, fp16, run eagerly."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.run_eagerly = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu_no_dist_strat_run_eagerly_fp16')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 128
    self._override_flags_to_run_test_shorter()
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_run_eagerly_fp16_tweaked(self):
    """Test with 1 GPU, no distribution strategy, fp16, run eagerly."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.run_eagerly = True
    FLAGS.explicit_gpu_placement = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu_no_dist_strat_run_eagerly_fp16_tweaked')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 128
    self._override_flags_to_run_test_shorter()
    self._run_and_report_benchmark()

  def benchmark_1_gpu(self):
    """Test Keras model with 1 GPU."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu')
    FLAGS.batch_size = 128
    self._override_flags_to_run_test_shorter()
    self._run_and_report_benchmark()

  def benchmark_1_gpu_amp(self):
    """Test Keras model with 1 GPU with automatic mixed precision."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.dtype = 'fp16'
    FLAGS.fp16_implementation = 'graph_rewrite'
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_amp')
    FLAGS.batch_size = 256
    self._override_flags_to_run_test_shorter()
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu(self):
    """Test Keras model with XLA and 1 GPU."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu')
    FLAGS.batch_size = 128
    self._override_flags_to_run_test_shorter()
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_amp(self):
    """Test Keras model with XLA and 1 GPU with automatic mixed precision."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.dtype = 'fp16'
    FLAGS.fp16_implementation = 'graph_rewrite'
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_amp')
    FLAGS.batch_size = 256
    self._override_flags_to_run_test_shorter()
    self._run_and_report_benchmark()

  def benchmark_1_gpu_fp16(self):
    """Test Keras model with 1 GPU and fp16."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_fp16')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    self._override_flags_to_run_test_shorter()
    self._run_and_report_benchmark()

  def benchmark_1_gpu_fp16_dynamic(self):
    """Test Keras model with 1 GPU, fp16, and dynamic loss scaling."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_fp16_dynamic')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    FLAGS.loss_scale = 'dynamic'
    self._override_flags_to_run_test_shorter()
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_fp16(self):
    """Test Keras model with XLA, 1 GPU and fp16."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_fp16')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    self._override_flags_to_run_test_shorter()
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_fp16_tweaked(self):
    """Test Keras model with XLA, 1 GPU, fp16, and manual config tuning."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_fp16_tweaked')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    self._override_flags_to_run_test_shorter()
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_fp16_dynamic(self):
    """Test Keras model with XLA, 1 GPU, fp16, and dynamic loss scaling."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_fp16_dynamic')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    FLAGS.loss_scale = 'dynamic'
    self._override_flags_to_run_test_shorter()
    self._run_and_report_benchmark()

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self):
    if FLAGS.num_gpus == 1 or FLAGS.run_eagerly:
      # For single GPU and pure eager tests which are less likely to be input
      # bound and more stable, run for shorter time and use the default
      # skip_steps.
      skip_steps = None
    else:
      # skip the first epoch for performance measurement.
      skip_steps = 600
    super(Resnet50KerasBenchmarkRemoteData,
          self)._run_and_report_benchmark(skip_steps=skip_steps)


class TrivialKerasBenchmarkReal(keras_benchmark.KerasBenchmark):
  """Trivial model with real data benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    flag_methods = [resnet_imagenet_main.define_imagenet_keras_flags]

    def_flags = {}
    def_flags['use_trivial_model'] = True
    def_flags['skip_eval'] = True
    def_flags['report_accuracy_metrics'] = False
    def_flags['dtype'] = 'fp16'
    def_flags['data_dir'] = os.path.join(root_data_dir, 'imagenet')
    def_flags['train_steps'] = 600
    def_flags['log_steps'] = 100
    def_flags['distribution_strategy'] = 'mirrored'

    super(TrivialKerasBenchmarkReal, self).__init__(
        output_dir=output_dir,
        flag_methods=flag_methods,
        default_flags=def_flags)

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = resnet_imagenet_main.run(FLAGS)
    wall_time_sec = time.time() - start_time_sec

    super(TrivialKerasBenchmarkReal, self)._report_benchmark(
        stats,
        wall_time_sec,
        total_batch_size=FLAGS.batch_size,
        log_steps=FLAGS.log_steps)

  def benchmark_8_gpu_warmup(self):
    """Dummy test that runs over an epoch to warmup the machine."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_warmup')
    FLAGS.batch_size = 256 * 8
    FLAGS.train_steps = 700
    self._run_and_report_benchmark()

  def fill_report_object(self, stats):
    super(TrivialKerasBenchmarkReal, self).fill_report_object(
        stats,
        total_batch_size=FLAGS.batch_size,
        log_steps=FLAGS.log_steps)


class Resnet50MultiWorkerKerasAccuracy(keras_benchmark.KerasBenchmark):
  """Resnet50 distributed accuracy tests with multiple workers."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    flag_methods = [classifier_trainer.define_imagenet_keras_flags]
    self.data_dir = os.path.join(root_data_dir, 'imagenet')
    super(Resnet50MultiWorkerKerasAccuracy, self).__init__(
        output_dir=output_dir, flag_methods=flag_methods)

  def _benchmark_common(self, eager, num_workers, all_reduce_alg):
    """Common to all benchmarks in this class."""
    self._setup()

    num_gpus = 8
    FLAGS.num_gpus = num_gpus
    FLAGS.data_dir = self.data_dir
    FLAGS.train_epochs = 90
    FLAGS.epochs_between_evals = 10
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = eager
    FLAGS.enable_xla = False
    FLAGS.distribution_strategy = 'multi_worker_mirrored'
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 32
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_{}_8_gpu_{}_worker_fp16_{}_tweaked'.format(
            'eager' if eager else 'graph', num_workers, all_reduce_alg))
    FLAGS.batch_size = 256 * num_gpus * num_workers
    FLAGS.all_reduce_alg = all_reduce_alg

    self._run_and_report_benchmark()

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self,
                                top_1_min=MIN_TOP_1_ACCURACY,
                                top_1_max=MAX_TOP_1_ACCURACY):
    start_time_sec = time.time()
    stats = classifier_trainer.run(flags.FLAGS)
    wall_time_sec = time.time() - start_time_sec

    super(Resnet50MultiWorkerKerasAccuracy, self)._report_benchmark(
        stats,
        wall_time_sec,
        top_1_min=top_1_min,
        top_1_max=top_1_max,
        total_batch_size=FLAGS.batch_size,
        log_steps=100)

  def _get_model_dir(self, folder_name):
    return os.path.join(self.output_dir, folder_name)

  def benchmark_eager_8_gpu_2_workers_fp16_ring_tweaked(self):
    """Eager, 8 GPUs per worker, 2 workers, fp16, ring all-reduce."""
    self._benchmark_common(eager=True, num_workers=2, all_reduce_alg='ring')

  def benchmark_eager_8_gpu_2_workers_fp16_nccl_tweaked(self):
    """Eager, 8 GPUs per worker, 2 workers, fp16, nccl all-reduce."""
    self._benchmark_common(eager=True, num_workers=2, all_reduce_alg='nccl')

  def benchmark_eager_8_gpu_8_workers_fp16_ring_tweaked(self):
    """Eager, 8 GPUs per worker, 8 workers, fp16, ring all-reduce."""
    self._benchmark_common(eager=True, num_workers=8, all_reduce_alg='ring')

  def benchmark_eager_8_gpu_8_workers_fp16_nccl_tweaked(self):
    """Eager, 8 GPUs per worker, 8 workers, fp16, nccl all-reduce."""
    self._benchmark_common(eager=True, num_workers=8, all_reduce_alg='nccl')


class Resnet50MultiWorkerKerasBenchmark(Resnet50KerasBenchmarkBase):
  """Resnet50 distributed benchmark tests with multiple workers."""

  def __init__(self, output_dir=None, default_flags=None):
    super(Resnet50MultiWorkerKerasBenchmark, self).__init__(
        output_dir=output_dir, default_flags=default_flags)

  def _benchmark_common(self, eager, num_workers, all_reduce_alg):
    """Common to all benchmarks in this class."""
    self._setup()

    num_gpus = 8
    FLAGS.num_gpus = num_gpus
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = eager
    FLAGS.enable_xla = False
    FLAGS.distribution_strategy = 'multi_worker_mirrored'
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 32
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_{}_8_gpu_{}_worker_fp16_{}_tweaked'.format(
            'eager' if eager else 'graph', num_workers, all_reduce_alg))
    FLAGS.batch_size = 256 * num_gpus * num_workers
    FLAGS.all_reduce_alg = all_reduce_alg

    self._run_and_report_benchmark()

  def benchmark_eager_8_gpu_1_worker_fp16_ring_tweaked(self):
    """Eager, 8 GPUs per worker, 1 worker, fp16, ring all-reduce."""
    self._benchmark_common(eager=True, num_workers=1, all_reduce_alg='ring')

  def benchmark_eager_8_gpu_1_worker_fp16_nccl_tweaked(self):
    """Eager, 8 GPUs per worker, 1 worker, fp16, nccl all-reduce."""
    self._benchmark_common(eager=True, num_workers=1, all_reduce_alg='nccl')

  def benchmark_eager_8_gpu_2_workers_fp16_ring_tweaked(self):
    """Eager, 8 GPUs per worker, 2 workers, fp16, ring all-reduce."""
    self._benchmark_common(eager=True, num_workers=2, all_reduce_alg='ring')

  def benchmark_eager_8_gpu_2_workers_fp16_nccl_tweaked(self):
    """Eager, 8 GPUs per worker, 2 workers, fp16, nccl all-reduce."""
    self._benchmark_common(eager=True, num_workers=2, all_reduce_alg='nccl')

  def benchmark_eager_8_gpu_8_workers_fp16_ring_tweaked(self):
    """Eager, 8 GPUs per worker, 8 workers, fp16, ring all-reduce."""
    self._benchmark_common(eager=True, num_workers=8, all_reduce_alg='ring')

  def benchmark_eager_8_gpu_8_workers_fp16_nccl_tweaked(self):
    """Eager, 8 GPUs per worker, 8 workers, fp16, nccl all-reduce."""
    self._benchmark_common(eager=True, num_workers=8, all_reduce_alg='nccl')


class Resnet50MultiWorkerKerasBenchmarkSynth(Resnet50MultiWorkerKerasBenchmark):
  """Resnet50 multi-worker synthetic data benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    def_flags = {}
    def_flags['skip_eval'] = True
    def_flags['report_accuracy_metrics'] = False
    def_flags['use_synthetic_data'] = True
    def_flags['train_steps'] = 110
    def_flags['log_steps'] = 10

    super(Resnet50MultiWorkerKerasBenchmarkSynth, self).__init__(
        output_dir=output_dir, default_flags=def_flags)


class Resnet50MultiWorkerKerasBenchmarkReal(Resnet50MultiWorkerKerasBenchmark):
  """Resnet50 multi-worker real data benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    def_flags = {}
    def_flags['skip_eval'] = True
    def_flags['report_accuracy_metrics'] = False
    def_flags['data_dir'] = os.path.join(root_data_dir, 'imagenet')
    def_flags['train_steps'] = 110
    def_flags['log_steps'] = 10

    super(Resnet50MultiWorkerKerasBenchmarkReal, self).__init__(
        output_dir=output_dir, default_flags=def_flags)


# TODO(kimjaehong): It also should be also cover other metheods of model
# optimization techniques. In that time, this class will change to something
# like 'KerasModelOptimizationAccuracyBase'.
class KerasPruningAccuracyBase(keras_benchmark.KerasBenchmark):
  """Benchmark accuracy tests for pruning method."""

  def __init__(self,
               output_dir=None,
               root_data_dir=None,
               default_flags=None,
               **kwargs):
    """A accuracy benchmark class for pruning method.

    Args:
      output_dir: directory where to output e.g. log files
      root_data_dir: directory under which to look for dataset
      default_flags: default flags
      **kwargs: arbitrary named arguments. This is needed to make the
                constructor forward compatible in case PerfZero provides more
                named arguments before updating the constructor.
    """
    if default_flags is None:
      default_flags = {}
    default_flags['pruning_method'] = 'polynomial_decay'
    default_flags['data_dir'] = os.path.join(root_data_dir, 'imagenet')

    flag_methods = [resnet_imagenet_main.define_imagenet_keras_flags]

    super(KerasPruningAccuracyBase, self).__init__(
        output_dir=output_dir,
        flag_methods=flag_methods,
        default_flags=default_flags,
        **kwargs)

  def benchmark_8_gpu(self):
    """Test Keras model with eager, dist_strat and 8 GPUs."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.batch_size = 32 * 8
    FLAGS.train_epochs = 90
    FLAGS.epochs_between_evals = 10
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu')
    FLAGS.dtype = 'fp32'
    FLAGS.enable_eager = True
    self._run_and_report_benchmark()

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self,
                                top_1_min=MODEL_OPTIMIZATION_TOP_1_ACCURACY[
                                    'RESNET50_FINETUNE_PRUNING'][0],
                                top_1_max=MODEL_OPTIMIZATION_TOP_1_ACCURACY[
                                    'RESNET50_FINETUNE_PRUNING'][1]):
    start_time_sec = time.time()
    stats = resnet_imagenet_main.run(flags.FLAGS)
    wall_time_sec = time.time() - start_time_sec

    super(KerasPruningAccuracyBase, self)._report_benchmark(
        stats,
        wall_time_sec,
        top_1_min=top_1_min,
        top_1_max=top_1_max,
        total_batch_size=FLAGS.batch_size,
        log_steps=100)


class MobilenetV1KerasPruningAccuracy(KerasPruningAccuracyBase):
  """Benchmark accuracy tests for MobilenetV1 with pruning method."""

  def __init__(self, root_data_dir=None, **kwargs):
    default_flags = {
        'model': 'mobilenet',
        'optimizer': 'mobilenet_default',
        'initial_learning_rate_per_sample': 0.00007,
        'pretrained_filepath': tf.train.latest_checkpoint(
            os.path.join(root_data_dir, 'mobilenet_v1')),
        'pruning_begin_step': 0,
        'pruning_end_step': 100000,
        'pruning_initial_sparsity': 0.0,
        'pruning_final_sparsity': 0.5,
        'pruning_frequency': 100,
    }
    super(MobilenetV1KerasPruningAccuracy, self).__init__(
        root_data_dir=root_data_dir,
        default_flags=default_flags,
        **kwargs)

  def _run_and_report_benchmark(self):
    super(MobilenetV1KerasPruningAccuracy, self)._run_and_report_benchmark(
        top_1_min=\
        MODEL_OPTIMIZATION_TOP_1_ACCURACY['MOBILENET_V1_FINETUNE_PRUNING'][0],
        top_1_max=\
        MODEL_OPTIMIZATION_TOP_1_ACCURACY['MOBILENET_V1_FINETUNE_PRUNING'][1])


class Resnet50KerasPruningAccuracy(KerasPruningAccuracyBase):
  """Benchmark accuracy tests for resnet50 with pruning method."""

  def __init__(self, root_data_dir=None, **kwargs):
    default_flags = {
        'model': 'resnet50_v1.5',
        'optimizer': 'mobilenet_default',
        'initial_learning_rate_per_sample': 0.0000039,
        'pretrained_filepath': tf.train.latest_checkpoint(
            os.path.join(root_data_dir, 'resnet50')),
        'pruning_begin_step': 0,
        'pruning_end_step': 50000,
        'pruning_initial_sparsity': 0.0,
        'pruning_final_sparsity': 0.5,
        'pruning_frequency': 100,
    }
    super(Resnet50KerasPruningAccuracy, self).__init__(
        root_data_dir=root_data_dir,
        default_flags=default_flags,
        **kwargs)

  def _run_and_report_benchmark(self):
    super(Resnet50KerasPruningAccuracy, self)._run_and_report_benchmark(
        top_1_min=\
        MODEL_OPTIMIZATION_TOP_1_ACCURACY['RESNET50_FINETUNE_PRUNING'][0],
        top_1_max=\
        MODEL_OPTIMIZATION_TOP_1_ACCURACY['RESNET50_FINETUNE_PRUNING'][1])


class KerasPruningBenchmarkRealBase(Resnet50KerasBenchmarkBase):
  """Pruning method benchmarks."""

  def __init__(self, root_data_dir=None, default_flags=None, **kwargs):
    if default_flags is None:
      default_flags = {}
    default_flags.update({
        'skip_eval': True,
        'report_accuracy_metrics': False,
        'data_dir': os.path.join(root_data_dir, 'imagenet'),
        'train_steps': 110,
        'log_steps': 10,
        'pruning_method': 'polynomial_decay',
        'pruning_begin_step': 0,
        'pruning_end_step': 50000,
        'pruning_initial_sparsity': 0,
        'pruning_final_sparsity': 0.5,
        'pruning_frequency': 100,
    })
    super(KerasPruningBenchmarkRealBase, self).__init__(
        default_flags=default_flags, **kwargs)


class MobilenetV1KerasPruningBenchmarkReal(KerasPruningBenchmarkRealBase):
  """Pruning method benchmarks for MobilenetV1."""

  def __init__(self, **kwargs):
    default_flags = {
        'model': 'mobilenet',
        'optimizer': 'mobilenet_default',
    }
    super(MobilenetV1KerasPruningBenchmarkReal, self).__init__(
        default_flags=default_flags, **kwargs)


class Resnet50KerasPruningBenchmarkReal(KerasPruningBenchmarkRealBase):
  """Pruning method benchmarks for resnet50."""

  def __init__(self, **kwargs):
    default_flags = {
        'model': 'resnet50_v1.5',
        'optimizer': 'mobilenet_default',
    }
    super(Resnet50KerasPruningBenchmarkReal, self).__init__(
        default_flags=default_flags, **kwargs)


class KerasClusteringAccuracyBase(keras_benchmark.KerasBenchmark):
  """Benchmark accuracy tests for clustering method."""

  def __init__(self,
               output_dir=None,
               root_data_dir=None,
               default_flags=None,
               **kwargs):
    """An accuracy benchmark class for clustering method.

    Args:
      output_dir: directory where to output e.g. log files
      root_data_dir: directory under which to look for dataset
      default_flags: default flags
      **kwargs: arbitrary named arguments. This is needed to make the
                constructor forward compatible in case PerfZero provides more
                named arguments before updating the constructor.
    """
    if default_flags is None:
      default_flags = {}
    default_flags['clustering_method'] = 'selective_clustering'
    default_flags['data_dir'] = os.path.join(root_data_dir, 'imagenet')
    default_flags['model'] = 'mobilenet_pretrained'
    default_flags['optimizer'] = 'mobilenet_fine_tune'

    flag_methods = [resnet_imagenet_main.define_imagenet_keras_flags]

    super(KerasClusteringAccuracyBase, self).__init__(
        output_dir=output_dir,
        flag_methods=flag_methods,
        default_flags=default_flags,
        **kwargs)

  def benchmark_8_gpu(self):
    """Test Keras model with eager, dist_strat and 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.batch_size = 32 * 8
    FLAGS.train_epochs = 1
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu')
    FLAGS.dtype = 'fp32'
    FLAGS.enable_eager = True
    self._run_and_report_benchmark()

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self,
                                top_1_min=MODEL_OPTIMIZATION_TOP_1_ACCURACY[
                                    'MOBILENET_V1_FINETUNE_CLUSTERING'][0],
                                top_1_max=MODEL_OPTIMIZATION_TOP_1_ACCURACY[
                                    'MOBILENET_V1_FINETUNE_CLUSTERING'][1]):
    start_time_sec = time.time()
    stats = resnet_imagenet_main.run(flags.FLAGS)
    wall_time_sec = time.time() - start_time_sec

    super(KerasClusteringAccuracyBase, self)._report_benchmark(
        stats,
        wall_time_sec,
        top_1_min=top_1_min,
        top_1_max=top_1_max,
        total_batch_size=FLAGS.batch_size,
        log_steps=100)


class MobilenetV1KerasClusteringAccuracy(KerasClusteringAccuracyBase):
  """Benchmark accuracy tests for MobilenetV1 with clustering method."""

  def __init__(self, root_data_dir=None, **kwargs):
    default_flags = {
        'model': 'mobilenet_pretrained',
        'optimizer': 'mobilenet_fine_tune',
    }
    super(MobilenetV1KerasClusteringAccuracy, self).__init__(
        root_data_dir=root_data_dir,
        default_flags=default_flags,
        **kwargs)

  def _run_and_report_benchmark(self):
    super(MobilenetV1KerasClusteringAccuracy, self)._run_and_report_benchmark(
        top_1_min=\
        MODEL_OPTIMIZATION_TOP_1_ACCURACY['MOBILENET_V1_FINETUNE_CLUSTERING'][0],
        top_1_max=\
        MODEL_OPTIMIZATION_TOP_1_ACCURACY['MOBILENET_V1_FINETUNE_CLUSTERING'][1])


class KerasClusteringBenchmarkRealBase(Resnet50KerasBenchmarkBase):
  """Clustering method benchmarks."""

  def __init__(self, root_data_dir=None, default_flags=None, **kwargs):
    if default_flags is None:
      default_flags = {}
    default_flags.update({
        'skip_eval': True,
        'report_accuracy_metrics': False,
        'data_dir': os.path.join(root_data_dir, 'imagenet'),
        'clustering_method': 'selective_clustering',
        'train_steps': 110,
        'log_steps': 10,
    })
    super(KerasClusteringBenchmarkRealBase, self).__init__(
        default_flags=default_flags, **kwargs)


class MobilenetV1KerasClusteringBenchmarkReal(KerasClusteringBenchmarkRealBase):
  """Clustering method benchmarks for MobilenetV1."""

  def __init__(self, **kwargs):
    default_flags = {
        'model': 'mobilenet_pretrained',
        'optimizer': 'mobilenet_fine_tune',
    }
    super(MobilenetV1KerasClusteringBenchmarkReal, self).__init__(
        default_flags=default_flags, **kwargs)


if __name__ == '__main__':
  tf.test.main()
