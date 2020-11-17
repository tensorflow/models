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
"""Executes benchmark testing for 3D Unet model."""
# pylint: disable=line-too-long
from __future__ import print_function

import functools
import os
import time
from typing import Optional

from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.benchmark import benchmark_wrappers
from official.benchmark import keras_benchmark
from official.benchmark import owner_utils
from official.vision.segmentation import unet_main as unet_training_lib
from official.vision.segmentation import unet_model as unet_model_lib

UNET3D_MIN_ACCURACY = 0.94
UNET3D_MAX_ACCURACY = 0.98
UNET_TRAINING_FILES = 'gs://mlcompass-data/unet3d/train_data/*'
UNET_EVAL_FILES = 'gs://mlcompass-data/unet3d/eval_data/*'
UNET_MODEL_CONFIG_FILE = 'gs://mlcompass-data/unet3d/config/unet_config.yaml'

FLAGS = flags.FLAGS


class Unet3DAccuracyBenchmark(keras_benchmark.KerasBenchmark):
  """Benchmark accuracy tests for UNet3D model in Keras."""

  def __init__(self,
               output_dir: Optional[str] = None,
               root_data_dir: Optional[str] = None,
               **kwargs):
    """A benchmark class.

    Args:
      output_dir: directory where to output e.g. log files
      root_data_dir: directory under which to look for dataset
      **kwargs: arbitrary named arguments. This is needed to make the
        constructor forward compatible in case PerfZero provides more named
        arguments before updating the constructor.
    """

    flag_methods = [unet_training_lib.define_unet3d_flags]

    # UNet3D model in Keras."""
    self.training_file_pattern = UNET_TRAINING_FILES
    self.eval_file_pattern = UNET_EVAL_FILES

    # TODO(hongjunchoi): Create and use shared config file instead.
    self.config_file = UNET_MODEL_CONFIG_FILE
    super(Unet3DAccuracyBenchmark, self).__init__(
        output_dir=output_dir, flag_methods=flag_methods)

  def _set_benchmark_parameters(self, experiment_name):
    """Overrides training parameters for benchmark tests."""
    FLAGS.model_dir = self._get_model_dir(experiment_name)
    FLAGS.mode = 'train'
    FLAGS.training_file_pattern = self.training_file_pattern
    FLAGS.eval_file_pattern = self.eval_file_pattern
    FLAGS.config_file = self.config_file
    FLAGS.lr_init_value = 0.00005
    FLAGS.lr_decay_rate = 0.5
    FLAGS.epochs = 3

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self,
                                experiment_name: str,
                                min_accuracy: float = UNET3D_MIN_ACCURACY,
                                max_accuracy: float = UNET3D_MAX_ACCURACY,
                                distribution_strategy: str = 'tpu',
                                epochs: int = 10,
                                steps: int = 0,
                                epochs_between_evals: int = 1,
                                dtype: str = 'float32',
                                enable_xla: bool = False,
                                run_eagerly: bool = False):
    """Runs and reports the benchmark given the provided configuration."""
    params = unet_training_lib.extract_params(FLAGS)
    strategy = unet_training_lib.create_distribution_strategy(params)

    input_dtype = params.dtype
    if input_dtype == 'float16' or input_dtype == 'bfloat16':
      policy = tf.keras.mixed_precision.experimental.Policy(
          'mixed_bfloat16' if input_dtype == 'bfloat16' else 'mixed_float16')
      tf.keras.mixed_precision.experimental.set_policy(policy)

    stats = {}
    start_time_sec = time.time()
    with strategy.scope():
      unet_model = unet_model_lib.build_unet_model(params)
      history = unet_training_lib.train(
          params, strategy, unet_model,
          functools.partial(unet_training_lib.get_train_dataset, params),
          functools.partial(unet_training_lib.get_eval_dataset, params))

      stats['accuracy_top_1'] = history.history['val_metric_accuracy'][-1]
      stats['training_accuracy_top_1'] = history.history['metric_accuracy'][-1]
    wall_time_sec = time.time() - start_time_sec

    super(Unet3DAccuracyBenchmark, self)._report_benchmark(
        stats,
        wall_time_sec,
        top_1_min=min_accuracy,
        top_1_max=max_accuracy,
        total_batch_size=params.train_batch_size)

  def _get_model_dir(self, folder_name):
    return os.path.join(self.output_dir, folder_name)

  @owner_utils.Owner('tf-model-garden')
  def benchmark_4x4_tpu_bf16(self):
    """Test Keras model with 4x4 TPU, fp16."""
    experiment_name = 'benchmark_4x4_tpu_fp16'
    self._setup()
    self._set_benchmark_parameters(experiment_name)
    self._run_and_report_benchmark(
        experiment_name=experiment_name,
        dtype='bfloat16',
        distribution_strategy='tpu')

  @owner_utils.Owner('tf-graph-compiler')
  def benchmark_4x4_tpu_bf16_mlir(self):
    """Test Keras model with 4x4 TPU, fp16 and MLIR enabled."""
    experiment_name = 'benchmark_4x4_tpu_fp16_mlir'
    tf.config.experimental.enable_mlir_bridge()
    self._setup()
    self._set_benchmark_parameters(experiment_name)
    self._run_and_report_benchmark(
        experiment_name=experiment_name,
        dtype='bfloat16',
        distribution_strategy='tpu')


if __name__ == '__main__':
  tf.test.main()
