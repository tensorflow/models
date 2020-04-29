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
"""Flags which will be nearly universal across models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf
from official.utils.flags._conventions import help_wrap


def define_base(data_dir=True, model_dir=True, clean=False, train_epochs=False,
                epochs_between_evals=False, stop_threshold=False,
                batch_size=True, num_gpu=False, hooks=False, export_dir=False,
                distribution_strategy=False, run_eagerly=False):
  """Register base flags.

  Args:
    data_dir: Create a flag for specifying the input data directory.
    model_dir: Create a flag for specifying the model file directory.
    clean: Create a flag for removing the model_dir.
    train_epochs: Create a flag to specify the number of training epochs.
    epochs_between_evals: Create a flag to specify the frequency of testing.
    stop_threshold: Create a flag to specify a threshold accuracy or other
      eval metric which should trigger the end of training.
    batch_size: Create a flag to specify the batch size.
    num_gpu: Create a flag to specify the number of GPUs used.
    hooks: Create a flag to specify hooks for logging.
    export_dir: Create a flag to specify where a SavedModel should be exported.
    distribution_strategy: Create a flag to specify which Distribution Strategy
      to use.
    run_eagerly: Create a flag to specify to run eagerly op by op.
  Returns:
    A list of flags for core.py to marks as key flags.
  """
  key_flags = []

  if data_dir:
    flags.DEFINE_string(
        name="data_dir", short_name="dd", default="/tmp",
        help=help_wrap("The location of the input data."))
    key_flags.append("data_dir")

  if model_dir:
    flags.DEFINE_string(
        name="model_dir", short_name="md", default="/tmp",
        help=help_wrap("The location of the model checkpoint files."))
    key_flags.append("model_dir")

  if clean:
    flags.DEFINE_boolean(
        name="clean", default=False,
        help=help_wrap("If set, model_dir will be removed if it exists."))
    key_flags.append("clean")

  if train_epochs:
    flags.DEFINE_integer(
        name="train_epochs", short_name="te", default=1,
        help=help_wrap("The number of epochs used to train."))
    key_flags.append("train_epochs")

  if epochs_between_evals:
    flags.DEFINE_integer(
        name="epochs_between_evals", short_name="ebe", default=1,
        help=help_wrap("The number of training epochs to run between "
                       "evaluations."))
    key_flags.append("epochs_between_evals")

  if stop_threshold:
    flags.DEFINE_float(
        name="stop_threshold", short_name="st",
        default=None,
        help=help_wrap("If passed, training will stop at the earlier of "
                       "train_epochs and when the evaluation metric is  "
                       "greater than or equal to stop_threshold."))

  if batch_size:
    flags.DEFINE_integer(
        name="batch_size", short_name="bs", default=32,
        help=help_wrap("Batch size for training and evaluation. When using "
                       "multiple gpus, this is the global batch size for "
                       "all devices. For example, if the batch size is 32 "
                       "and there are 4 GPUs, each GPU will get 8 examples on "
                       "each step."))
    key_flags.append("batch_size")

  if num_gpu:
    flags.DEFINE_integer(
        name="num_gpus", short_name="ng",
        default=1,
        help=help_wrap(
            "How many GPUs to use at each worker with the "
            "DistributionStrategies API. The default is 1."))

  if run_eagerly:
    flags.DEFINE_boolean(
        name="run_eagerly", default=False,
        help="Run the model op by op without building a model function.")

  if hooks:
    flags.DEFINE_list(
        name="hooks", short_name="hk", default="LoggingTensorHook",
        help=help_wrap(
            u"A list of (case insensitive) strings to specify the names of "
            u"training hooks. Example: `--hooks ProfilerHook,"
            u"ExamplesPerSecondHook`\n See hooks_helper "
            u"for details.")
    )
    key_flags.append("hooks")

  if export_dir:
    flags.DEFINE_string(
        name="export_dir", short_name="ed", default=None,
        help=help_wrap("If set, a SavedModel serialization of the model will "
                       "be exported to this directory at the end of training. "
                       "See the README for more details and relevant links.")
    )
    key_flags.append("export_dir")

  if distribution_strategy:
    flags.DEFINE_string(
        name="distribution_strategy", short_name="ds", default="mirrored",
        help=help_wrap("The Distribution Strategy to use for training. "
                       "Accepted values are 'off', 'one_device', "
                       "'mirrored', 'parameter_server', 'collective', "
                       "case insensitive. 'off' means not to use "
                       "Distribution Strategy; 'default' means to choose "
                       "from `MirroredStrategy` or `OneDeviceStrategy` "
                       "according to the number of GPUs.")
    )


  return key_flags


def get_num_gpus(flags_obj):
  """Treat num_gpus=-1 as 'use all'."""
  if flags_obj.num_gpus != -1:
    return flags_obj.num_gpus

  from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top
  local_device_protos = device_lib.list_local_devices()
  return sum([1 for d in local_device_protos if d.device_type == "GPU"])
