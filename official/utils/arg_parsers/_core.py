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
import itertools as it
import sys
from typing import Union


class ArgManager(argparse.ArgumentParser):
  """Base class for all official model parsers.

  Implements the following features:
    1)  --helpful for detailed command line flag help
    2)  Enables secondary parsing for additional logic
  """

  def __init__(self, parents=None):
    # this forces --helpful below --help in the flag summary
    preemptive_parser = argparse.ArgumentParser(add_help=False)
    preemptive_parser.add_argument(
        "--helpful", "-H", action="store_true",
        help="Extended list of arguments and flags for running this script.")
    self.parents = [preemptive_parser] + (parents or [])

    self.verbose_flags = self.aggregate_parent_field("verbose_flags")
    self.extra_parse_fns = self.aggregate_parent_field("extra_parse_fns")
    super(ArgManager, self).__init__(add_help=True, parents=self.parents)

  def aggregate_parent_field(self, field):
    """Join and flatten field from parent parsers."""
    return list(it.chain.from_iterable([vars(p).get(field, [])
                                        for p in self.parents]))

  def parse_args(self, args=None,  # type: Union[None, list]
                 namespace=None    # type: Union[None, argparse.Namespace]
                ):
    # type: (...) -> argparse.Namespace
    """Parse arguments with additional logic.

    Args:
      args: List of strings to parse. The default is taken from sys.argv.
      namespace: An object to take the attributes. The default is a new empty
        argparse.Namespace object.

    Returns:
      A Namespace object containing parsed arguments.
    """
    args = sys.argv[1:] if args is None else args

    if "--helpful" in args or "-H" in args:
      self.print_help()
      sys.exit()

    # mark verbose flags with argparse.SUPPRESS
    with TemporarySilence(self.verbose_flags):
      flags = super(ArgManager, self).parse_args(args=args, namespace=namespace)
    self.secondary_arg_parsing(flags=flags)
    return args

  def secondary_arg_parsing(self, flags):
    """Perform arbitrary secondary processing of arguments.

    Args:
      flags: The parsed result of self.parse_args()
    """

    flag_dict = vars(flags)
    [fn(flag_dict) for fn in self.extra_parse_fns]  # pylint: disable=expression-not-assigned


class TemporarySilence(object):
  """Set help messages to argparse.SUPPRESS, and then restore them."""

  def __init__(self, verbose_flags):
    """Store information for flags which are to be silenced.

    Args:
      verbose_flags: A list of flags (argparse action containers) to be
        manipulated in context.
    """
    self.verbose_flags = verbose_flags
    self._help_messages = [flag.help for flag in self.verbose_flags]

  def __enter__(self):
    # Silence flags upon entry.
    for flag in self.verbose_flags:
      flag.help = argparse.SUPPRESS

  def __exit__(self, *args):
    # Restore original help messages upon exit.
    for i in range(len(self.verbose_flags)):
      self.verbose_flags[i].help = self._help_messages[i]
