# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Executes Keras MobileNetV2 benchmarks and accuracy tests."""
from __future__ import print_function

import os
import time

from absl import flags
from absl.testing import flagsaver
import tensorflow as tf

from official.keras_application_models.v2.benchmark import benchmark_datasets
from official.keras_application_models.v2 import train_mobilenetv2
from official.keras_application_models.v2 import utils


FLAGS = flags.FLAGS


class MobileNetV2Benchmark(tf.test.Benchmark):
  """Benchmarks tf.keras.application.MobileNetV2."""

  local_flags = None

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    if output_dir is None:
      output_dir = FLAGS.benchmark_output_dir
    if root_data_dir is None:
      root_data_dir = FLAGS.benchmark_data_dir
    self._output_dir = output_dir
    self._data_dir = root_data_dir

  def _prepare_dataset_builder(self, data_spec):
    if data_spec is None:
      return benchmark_datasets.ImageNetDatasetBuilder(self._data_dir)

  def _setup(self):
    if MobileNetV2Benchmark.local_flags is None:
      utils.define_flags()
      # Loads flags to get defaults.
      flags.FLAGS(["foo"])
      saved_flag_values = flagsaver.save_flag_values()
      MobileNetV2Benchmark.local_flags = saved_flag_values
    else:
      flagsaver.restore_flag_values(MobileNetV2Benchmark.local_flags)

  def _run_and_report(self, data_spec=None):
    start_time_sec = time.time()
    result = train_mobilenetv2.run(
        self._prepare_dataset_builder(data_spec), flags.FLAGS)
    wall_time_sec = time.time() - start_time_sec
    self.report_benchmark(
        iters=result["iters"],
        wall_time=wall_time_sec,
        extras={
          "accuracy": result["history"]["val_acc"][-1],
          "accuracy_top_5": {
            "value": result["history"]["val_top_k_categorical_accuracy"][-1],
          },
        })

  def benchmark_no_dist_strat_sanity(self):
    self._setup()
    FLAGS.no_pretrained_weights=True
    FLAGS.batch_size=32
    FLAGS.train_epochs=2
    FLAGS.limit_train_num=960
    self._run_and_report()

  def benchmark_1_gpu_sanity(self):
    self._setup()
    FLAGS.no_pretrained_weights=True
    FLAGS.batch_size=32
    FLAGS.train_epochs=2
    FLAGS.limit_train_num=960
    FLAGS.dist_strat=True
    FLAGS.num_gpus=1
    self._run_and_report()

  def benchmark_2_gpus_sanity(self):
    self._setup()
    FLAGS.no_pretrained_weights=True
    FLAGS.batch_size=32
    FLAGS.train_epochs=2
    FLAGS.limit_train_num=960
    FLAGS.dist_strat=True
    FLAGS.num_gpus=2
    self._run_and_report()


if __name__ == "__main__":
  flags.DEFINE_string(
      name="benchmark_output_dir", default="", help=
          "Output dir for benchmarking. If the benchmark is triggered by "
          "perfzero, don't set it.")
  flags.DEFINE_string(
      name="benchmark_data_dir", default="", help=
          "Data dir for benchmarking. If the benchmark is triggered by perfzero, "
          "don't set it.")
  tf.test.main()

