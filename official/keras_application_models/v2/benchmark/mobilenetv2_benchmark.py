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
import tensorflow as tf

from official.keras_application_models.v2 import datasets
from official.keras_application_models.v2 import train_mobilenetv2
from official.keras_application_models.v2 import utils


FLAGS = flags.FLAGS


class MobileNetV2Benchmark(tf.test.Benchmark):
  """Benchmarks tf.keras.application.MobileNetV2."""

  def __init__(self, output_dir=None, data_dir=None, **kwargs):
    self._output_dir = output_dir
    self._data_dir = data_dir
    utils.define_flags()

  def _prepare_dataset_builder(data_spec):
    if data_spec is None:
      return datasets.ImageNetDatasetBuilder()

  def _run_and_report(self, data_spec):
    start_time_sec = time.time()
    result = train_mobilenetv2.run(self._prepare_dataset_builder(data_spec))
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

  def benchmark_1_gpu_sanity(self):
    FLAGS.no_pretrained_weights=True
    FLAGS.no_eager=False
    FLAGS.batch_size=32
    FLAGS.train_epochs=2
    FLAGS.limit_train_num=960
    FLAGS.dist_strat=True
    FLAGS.num_gpus=1
    self._run_and_report()

  def benchmark_2_gpus_sanity(self):
    FLAGS.no_pretrained_weights=True
    FLAGS.no_eager=False
    FLAGS.batch_size=32
    FLAGS.train_epochs=2
    FLAGS.limit_train_num=960
    FLAGS.dist_strat=True
    FLAGS.num_gpus=2
    self._run_and_report()


if __name__ == "__main__":
  tf.test.main()

