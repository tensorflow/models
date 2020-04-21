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
"""Runs a memory usage benchmark for a Tensorflow Hub model.

Loads a SavedModel and records memory usage.
"""
import functools
import time

from absl import flags
import tensorflow as tf
import tensorflow_hub as hub

from official.benchmark.perfzero_benchmark import PerfZeroBenchmark

FLAGS = flags.FLAGS


class TfHubMemoryUsageBenchmark(PerfZeroBenchmark):
  """A benchmark measuring memory usage for a given TF Hub SavedModel."""

  def __init__(self,
               hub_model_handle_list=None,
               output_dir=None,
               default_flags=None,
               root_data_dir=None,
               **kwargs):
    super(TfHubMemoryUsageBenchmark, self).__init__(
        output_dir=output_dir, default_flags=default_flags, **kwargs)
    if hub_model_handle_list:
      for hub_model_handle in hub_model_handle_list.split(';'):
        # Converts a model handle of the form
        # https://tfhub.dev/google/nnlm-en-dim128/1 to valid python method name
        # like google_nnlm_en_dim128_1.
        hub_model_method_name = hub_model_handle.replace(
            'https://tfhub.dev',
            '').replace('/', '_').replace('-', '_').strip('_')
        setattr(
            self, 'benchmark_' + hub_model_method_name,
            functools.partial(self.benchmark_memory_usage, hub_model_handle))

  def benchmark_memory_usage(
      self, hub_model_handle='https://tfhub.dev/google/nnlm-en-dim128/1'):
    start_time_sec = time.time()
    self.load_model(hub_model_handle)
    wall_time_sec = time.time() - start_time_sec

    metrics = []
    self.report_benchmark(iters=-1, wall_time=wall_time_sec, metrics=metrics)

  def load_model(self, hub_model_handle):
    """Loads a TF Hub module."""
    hub.load(hub_model_handle)


if __name__ == '__main__':
  tf.test.main()
