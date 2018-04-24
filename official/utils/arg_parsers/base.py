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
"""Base parser class for all official model parsers."""

import argparse
import sys

import tensorflow as tf


# Map string to (TensorFlow dtype, default loss scale)
DTYPE_MAP = {
    "fp16": (tf.float16, 128),
    "fp32": (tf.float32, 1),
}


class Parser(argparse.ArgumentParser):
  """Base class for all official model parsers.

  Implements the following features:
    1)  --helpful for detailed command line flag help
    2)  Enables secondary parsing for additional logic
  """

  def __init__(self, parents=None):  # pylint: disable=unused-argument
    # this forces --helpful below --help in the flag summary
    preemptive_parser = argparse.ArgumentParser(add_help=False)
    preemptive_parser.add_argument("--helpful", "-H", action="store_true",
                                   help="Detailed list of arguments.")
    parents = [preemptive_parser] +(parents or [])

    self.verbose_flags = self.aggregate_verbose_flags(parents=parents)
    super(Parser, self).__init__(add_help=True, parents=parents)

  def aggregate_verbose_flags(self, parents):
    output = []
    for parent in parents:
      if "verbose_flags" in vars(parent):
        output.extend(parent.verbose_flags)
    return output

  def parse_args(self, args=None, namespace=None):
    args = args or []

    if "--helpful" in args or "-H" in args:
      self.print_help()
      sys.exit()

    with TemporarySilence(self.verbose_flags):
      args = super(Parser, self).parse_args(args=args, namespace=namespace)
    self.secondary_arg_parsing(args=args)
    return args

  def secondary_arg_parsing(self, args):
    arg_dict = vars(args)
    self.parse_dtype_info(arg_dict)

  @staticmethod
  def parse_dtype_info(flag_dict):
    """Convert dtype string to tf dtype, and set loss_scale default as needed.

    Args:
      flag_dict: dictionary representing the underlying data of a namespace
        object returned by the arg parser.
    Raises:
      ValueError: If an invalid dtype is provided.
    """

    if ("dtype" not in flag_dict or
        flag_dict["dtype"] in (i[0] for i in DTYPE_MAP.values())):
      return  # Make function safe without dtype flag, as well as idempotent

    try:
      flag_dict["dtype"], default_loss_scale = DTYPE_MAP[flag_dict["dtype"]]
    except KeyError:
      raise ValueError("Invalid dtype: {}".format(flag_dict["dtype"]))

    flag_dict["loss_scale"] = flag_dict["loss_scale"] or default_loss_scale


class TemporarySilence(object):
  """Set help messages to argparse.SUPPRESS, and then restore them."""

  def __init__(self, verbose_flags):
    self.verbose_flags = verbose_flags
    self._help_messages = [flag.help for flag in self.verbose_flags]

  def __enter__(self):
    for flag in self.verbose_flags:
      flag.help = argparse.SUPPRESS

  def __exit__(self, *args):
    for i in range(len(self.verbose_flags)):
      self.verbose_flags[i].help = self._help_messages[i]
