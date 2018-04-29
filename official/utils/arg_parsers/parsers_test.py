# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

import argparse
import unittest

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.arg_parsers import parsers


class TestParser(argparse.ArgumentParser):
  """Class to test canned parser functionality."""

  def __init__(self):
    super(TestParser, self).__init__(parents=[
        parsers.BaseParser(),
        parsers.PerformanceParser(num_parallel_calls=True, inter_op=True,
                                  intra_op=True, use_synthetic_data=True),
        parsers.ImageModelParser(data_format=True),
        parsers.BenchmarkParser(benchmark_log_dir=True, bigquery_uploader=True)
    ])


class BaseTester(unittest.TestCase):

  def test_default_setting(self):
    """Test to ensure fields exist and defaults can be set.
    """

    defaults = dict(
        data_dir="dfgasf",
        model_dir="dfsdkjgbs",
        train_epochs=534,
        epochs_between_evals=15,
        batch_size=256,
        hooks=["LoggingTensorHook"],
        num_parallel_calls=18,
        inter_op_parallelism_threads=5,
        intra_op_parallelism_thread=10,
        data_format="channels_first"
    )

    parser = TestParser()
    parser.set_defaults(**defaults)

    namespace_vars = vars(parser.parse_args([]))
    for key, value in defaults.items():
      assert namespace_vars[key] == value

  def test_benchmark_setting(self):
    defaults = dict(
        hooks=["LoggingMetricHook"],
        benchmark_log_dir="/tmp/12345",
        gcp_project="project_abc",
    )

    parser = TestParser()
    parser.set_defaults(**defaults)

    namespace_vars = vars(parser.parse_args([]))
    for key, value in defaults.items():
      assert namespace_vars[key] == value

  def test_booleans(self):
    """Test to ensure boolean flags trigger as expected.
    """

    parser = TestParser()
    namespace = parser.parse_args(["--multi_gpu", "--use_synthetic_data"])

    assert namespace.multi_gpu
    assert namespace.use_synthetic_data

  def test_parse_dtype_info(self):
    parser = TestParser()
    for dtype_str, tf_dtype, loss_scale in [["fp16", tf.float16, 128],
                                            ["fp32", tf.float32, 1]]:
      args = parser.parse_args(["--dtype", dtype_str])
      parsers.parse_dtype_info(args)

      assert args.dtype == tf_dtype
      assert args.loss_scale == loss_scale

      args = parser.parse_args(["--dtype", dtype_str, "--loss_scale", "5"])
      parsers.parse_dtype_info(args)

      assert args.loss_scale == 5

    with self.assertRaises(SystemExit):
      parser.parse_args(["--dtype", "int8"])


if __name__ == "__main__":
  unittest.main()
