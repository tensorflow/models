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

"""Collection of parsers which are shared among the official models.

The parsers in this module are intended to be used as parents to all arg
parsers in official models. For instance, one might define a new class:

class ExampleParser(argparse.ArgumentParser):
  def __init__(self):
    super(ExampleParser, self).__init__(parents=[
      arg_parsers.LocationParser(data_dir=True, model_dir=True),
      arg_parsers.DummyParser(use_synthetic_data=True),
    ])

    self.add_argument(
      "--application_specific_arg", "-asa", type=int, default=123,
      help="[default: %(default)s] This arg is application specific.",
      metavar="<ASA>"
    )

Notes about add_argument():
    Argparse will automatically template in default values in help messages if
  the "%(default)s" string appears in the message. Using the example above:

    parser = ExampleParser()
    parser.set_defaults(application_specific_arg=3141592)
    parser.parse_args(["-h"])

    When the help text is generated, it will display 3141592 to the user. (Even
  though the default was 123 when the flag was created.)


    The metavar variable determines how the flag will appear in help text. If
  not specified, the convention is to use name.upper(). Thus rather than:

    --app_specific_arg APP_SPECIFIC_ARG, -asa APP_SPECIFIC_ARG

  if metavar="<ASA>" is set, the user sees:

    --app_specific_arg <ASA>, -asa <ASA>

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf


# Map string to (TensorFlow dtype, default loss scale)
DTYPE_MAP = {
    "fp16": (tf.float16, 128),
    "fp32": (tf.float32, 1),
}


def parse_dtype_info(flags):
  """Convert dtype string to tf dtype, and set loss_scale default as needed.

  Args:
    flags: namespace object returned by arg parser.

  Raises:
    ValueError: If an invalid dtype is provided.
  """
  if flags.dtype in (i[0] for i in DTYPE_MAP.values()):
    return  # Make function idempotent

  try:
    flags.dtype, default_loss_scale = DTYPE_MAP[flags.dtype]
  except KeyError:
    raise ValueError("Invalid dtype: {}".format(flags.dtype))

  flags.loss_scale = flags.loss_scale or default_loss_scale


class BaseParser(argparse.ArgumentParser):
  """Parser to contain flags which will be nearly universal across models.

  Args:
    add_help: Create the "--help" flag. False if class instance is a parent.
    data_dir: Create a flag for specifying the input data directory.
    model_dir: Create a flag for specifying the model file directory.
    train_epochs: Create a flag to specify the number of training epochs.
    epochs_between_evals: Create a flag to specify the frequency of testing.
    stop_threshold: Create a flag to specify a threshold accuracy or other
      eval metric which should trigger the end of training.
    batch_size: Create a flag to specify the batch size.
    multi_gpu: Create a flag to allow the use of all available GPUs.
    hooks: Create a flag to specify hooks for logging.
  """

  def __init__(self, add_help=False, data_dir=True, model_dir=True,
               train_epochs=True, epochs_between_evals=True,
               stop_threshold=True, batch_size=True, multi_gpu=True,
               hooks=True):
    super(BaseParser, self).__init__(add_help=add_help)

    if data_dir:
      self.add_argument(
          "--data_dir", "-dd", default="/tmp",
          help="[default: %(default)s] The location of the input data.",
          metavar="<DD>",
      )

    if model_dir:
      self.add_argument(
          "--model_dir", "-md", default="/tmp",
          help="[default: %(default)s] The location of the model checkpoint "
               "files.",
          metavar="<MD>",
      )

    if train_epochs:
      self.add_argument(
          "--train_epochs", "-te", type=int, default=1,
          help="[default: %(default)s] The number of epochs used to train.",
          metavar="<TE>"
      )

    if epochs_between_evals:
      self.add_argument(
          "--epochs_between_evals", "-ebe", type=int, default=1,
          help="[default: %(default)s] The number of training epochs to run "
               "between evaluations.",
          metavar="<EBE>"
      )

    if stop_threshold:
      self.add_argument(
          "--stop_threshold", "-st", type=float, default=None,
          help="[default: %(default)s] If passed, training will stop at "
          "the earlier of train_epochs and when the evaluation metric is "
          "greater than or equal to stop_threshold.",
          metavar="<ST>"
      )

    if batch_size:
      self.add_argument(
          "--batch_size", "-bs", type=int, default=32,
          help="[default: %(default)s] Batch size for training and evaluation.",
          metavar="<BS>"
      )

    if multi_gpu:
      self.add_argument(
          "--multi_gpu", action="store_true",
          help="If set, run across all available GPUs."
      )

    if hooks:
      self.add_argument(
          "--hooks", "-hk", nargs="+", default=["LoggingTensorHook"],
          help="[default: %(default)s] A list of strings to specify the names "
               "of train hooks. "
               "Example: --hooks LoggingTensorHook ExamplesPerSecondHook. "
               "Allowed hook names (case-insensitive): LoggingTensorHook, "
               "ProfilerHook, ExamplesPerSecondHook, LoggingMetricHook."
               "See official.utils.logs.hooks_helper for details.",
          metavar="<HK>"
      )


class PerformanceParser(argparse.ArgumentParser):
  """Default parser for specifying performance tuning arguments.

  Args:
    add_help: Create the "--help" flag. False if class instance is a parent.
    num_parallel_calls: Create a flag to specify parallelism of data loading.
    inter_op: Create a flag to allow specification of inter op threads.
    intra_op: Create a flag to allow specification of intra op threads.
  """

  def __init__(self, add_help=False, num_parallel_calls=True, inter_op=True,
               intra_op=True, use_synthetic_data=True, max_train_steps=True,
               dtype=True):
    super(PerformanceParser, self).__init__(add_help=add_help)

    if num_parallel_calls:
      self.add_argument(
          "--num_parallel_calls", "-npc",
          type=int, default=5,
          help="[default: %(default)s] The number of records that are "
               "processed in parallel  during input processing. This can be "
               "optimized per data set but for generally homogeneous data "
               "sets, should be approximately the number of available CPU "
               "cores.",
          metavar="<NPC>"
      )

    if inter_op:
      self.add_argument(
          "--inter_op_parallelism_threads", "-inter",
          type=int, default=0,
          help="[default: %(default)s Number of inter_op_parallelism_threads "
               "to use for CPU. See TensorFlow config.proto for details.",
          metavar="<INTER>"
      )

    if intra_op:
      self.add_argument(
          "--intra_op_parallelism_threads", "-intra",
          type=int, default=0,
          help="[default: %(default)s Number of intra_op_parallelism_threads "
               "to use for CPU. See TensorFlow config.proto for details.",
          metavar="<INTRA>"
      )

    if use_synthetic_data:
      self.add_argument(
          "--use_synthetic_data", "-synth",
          action="store_true",
          help="If set, use fake data (zeroes) instead of a real dataset. "
               "This mode is useful for performance debugging, as it removes "
               "input processing steps, but will not learn anything."
      )

    if max_train_steps:
      self.add_argument(
          "--max_train_steps", "-mts", type=int, default=None,
          help="[default: %(default)s] The model will stop training if the "
               "global_step reaches this value. If not set, training will run"
               "until the specified number of epochs have run as usual. It is"
               "generally recommended to set --train_epochs=1 when using this"
               "flag.",
          metavar="<MTS>"
      )

    if dtype:
      self.add_argument(
          "--dtype", "-dt",
          default="fp32",
          choices=list(DTYPE_MAP.keys()),
          help="[default: %(default)s] {%(choices)s} The TensorFlow datatype "
               "used for calculations. Variables may be cast to a higher"
               "precision on a case-by-case basis for numerical stability.",
          metavar="<DT>"
      )

      self.add_argument(
          "--loss_scale", "-ls",
          type=int,
          help="[default: %(default)s] The amount to scale the loss by when "
               "the model is run. Before gradients are computed, the loss is "
               "multiplied by the loss scale, making all gradients loss_scale "
               "times larger. To adjust for this, gradients are divided by the "
               "loss scale before being applied to variables. This is "
               "mathematically equivalent to training without a loss scale, "
               "but the loss scale helps avoid some intermediate gradients "
               "from underflowing to zero. If not provided the default for "
               "fp16 is 128 and 1 for all other dtypes.",
      )


class ImageModelParser(argparse.ArgumentParser):
  """Default parser for specification image specific behavior.

  Args:
    add_help: Create the "--help" flag. False if class instance is a parent.
    data_format: Create a flag to specify image axis convention.
  """

  def __init__(self, add_help=False, data_format=True):
    super(ImageModelParser, self).__init__(add_help=add_help)
    if data_format:
      self.add_argument(
          "--data_format", "-df",
          default=None,
          choices=["channels_first", "channels_last"],
          help="A flag to override the data format used in the model. "
               "channels_first provides a performance boost on GPU but is not "
               "always compatible with CPU. If left unspecified, the data "
               "format will be chosen automatically based on whether TensorFlow"
               "was built for CPU or GPU.",
          metavar="<CF>"
      )


class ExportParser(argparse.ArgumentParser):
  """Parsing options for exporting saved models or other graph defs.

  This is a separate parser for now, but should be made part of BaseParser
  once all models are brought up to speed.

  Args:
    add_help: Create the "--help" flag. False if class instance is a parent.
    export_dir: Create a flag to specify where a SavedModel should be exported.
  """

  def __init__(self, add_help=False, export_dir=True):
    super(ExportParser, self).__init__(add_help=add_help)
    if export_dir:
      self.add_argument(
          "--export_dir", "-ed",
          help="[default: %(default)s] If set, a SavedModel serialization of "
               "the model will be exported to this directory at the end of "
               "training. See the README for more details and relevant links.",
          metavar="<ED>"
      )


class BenchmarkParser(argparse.ArgumentParser):
  """Default parser for benchmark logging.

  Args:
    add_help: Create the "--help" flag. False if class instance is a parent.
    benchmark_log_dir: Create a flag to specify location for benchmark logging.
  """

  def __init__(self, add_help=False, benchmark_log_dir=True,
               bigquery_uploader=True):
    super(BenchmarkParser, self).__init__(add_help=add_help)
    if benchmark_log_dir:
      self.add_argument(
          "--benchmark_log_dir", "-bld", default=None,
          help="[default: %(default)s] The location of the benchmark logging.",
          metavar="<BLD>"
      )
    if bigquery_uploader:
      self.add_argument(
          "--gcp_project", "-gp", default=None,
          help="[default: %(default)s] The GCP project name where the benchmark"
               " will be uploaded.",
          metavar="<GP>"
      )
      self.add_argument(
          "--bigquery_data_set", "-bds", default="test_benchmark",
          help="[default: %(default)s] The Bigquery dataset name where the"
               " benchmark will be uploaded.",
          metavar="<BDS>"
      )
      self.add_argument(
          "--bigquery_run_table", "-brt", default="benchmark_run",
          help="[default: %(default)s] The Bigquery table name where the"
               " benchmark run information will be uploaded.",
          metavar="<BRT>"
      )
      self.add_argument(
          "--bigquery_metric_table", "-bmt", default="benchmark_metric",
          help="[default: %(default)s] The Bigquery table name where the"
               " benchmark metric information will be uploaded.",
          metavar="<BMT>"
      )


class EagerParser(BaseParser):
  """Remove options not relevant for Eager from the BaseParser."""

  def __init__(self, add_help=False, data_dir=True, model_dir=True,
               train_epochs=True, batch_size=True):
    super(EagerParser, self).__init__(
        add_help=add_help, data_dir=data_dir, model_dir=model_dir,
        train_epochs=train_epochs, epochs_between_evals=False,
        stop_threshold=False, batch_size=batch_size, multi_gpu=False,
        hooks=False)
