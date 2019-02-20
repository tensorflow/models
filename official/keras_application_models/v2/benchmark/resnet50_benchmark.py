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
"""Executes benchmarks for ResNet50 in Keras Application models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from absl import flags
import tensorflow as tf

from official.keras_application_models.v2 import train_resnet50_on_cifar10
from official.keras_application_models.v2 import utils


FLAGS = flags.FLAGS


class KerasResnet50Benchmark(tf.test.Benchmark):
  """Benchmarks tf.keras.application.Resnet50.

  TODO(xunkai): Consider abstract it for other models or combine
  `official.resnet.keras.keras_benchmark`.
  """

  def __init__(self):
    utils.define_flags()

  def _run_and_report(self):
    start_time_sec = time.time()
    history = train_resnet50_on_cifar10.run(FLAGS)
    print(history.history)
    # TODO(xunkai): Convert the history to statistics.
    wall_time_sec = time.time() - start_time_sec
    '''
    super(KerasResnet50Benchmark, self).report_benchmark(
        stats,
        wall_time=wall_time_sec,
        extras=stats)
    '''

  def benchmark_1_gpu_finetune_no_dist_strat(self):
    FLAGS.no_pretrained_weights=False
    FLAGS.no_eager=False
    FLAGS.batch_size=32
    FLAGS.train_epochs=1
    FLAGS.dist_strat=False
    FLAGS.num_gpus=1
    self._run_and_report()


if __name__ == "__main__":
  tf.test.main()
