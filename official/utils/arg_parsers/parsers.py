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


class BaseParser(argparse.ArgumentParser):
  """Parser to contain flags which will be nearly universal across models.

  Args:
    add_help: Create the "--help" flag. False if class instance is a parent.
    data_dir: Create a flag for specifying the input data directory.
    model_dir: Create a flag for specifying the model file directory.
    train_epochs: Create a flag to specify the number of training epochs.
    epochs_between_evals: Create a flag to specify the frequency of testing.
    batch_size: Create a flag to specify the batch size.
    multi_gpu: Create a flag to allow the use of all available GPUs.
    hooks: Create a flag to specify hooks for logging.
  """

  def __init__(self, add_help=False, data_dir=True, model_dir=True,
               train_epochs=True, epochs_between_evals=True, batch_size=True,
               multi_gpu=True, hooks=True):
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
               "See official.utils.logging.hooks_helper for details.",
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
               intra_op=True, use_synthetic_data=True, max_train_steps=True):
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


class BenchmarkParser(argparse.ArgumentParser):
  """Default parser for benchmark logging.

  Args:
    add_help: Create the "--help" flag. False if class instance is a parent.
    benchmark_log_dir: Create a flag to specify location for benchmark logging.
  """

  def __init__(self, add_help=False, benchmark_log_dir=True):
    super(BenchmarkParser, self).__init__(add_help=add_help)
    if benchmark_log_dir:
      self.add_argument(
          "--benchmark_log_dir", "-bld", default=None,
          help="[default: %(default)s] The location of the benchmark logging.",
          metavar="<BLD>"
      )
