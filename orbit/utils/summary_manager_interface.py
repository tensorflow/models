# Copyright 2024 The Orbit Authors. All Rights Reserved.
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

"""Provides a utility class for managing summary writing."""

import abc


class SummaryManagerInterface(abc.ABC):
  """A utility interface for managing summary writing."""

  @abc.abstractmethod
  def flush(self):
    """Flushes the the recorded summaries."""
    raise NotImplementedError

  @abc.abstractmethod
  def summary_writer(self, relative_path=""):
    """Returns the underlying summary writer for scoped writers."""
    raise NotImplementedError

  @abc.abstractmethod
  def write_summaries(self, summary_dict):
    """Writes summaries for the given dictionary of values.

    The summary_dict can be any nested dict. The SummaryManager should
    recursively creates summaries, yielding a hierarchy of summaries which will
    then be reflected in the corresponding UIs.

    For example, users may evaluate on multiple datasets and return
    `summary_dict` as a nested dictionary:

        {
            "dataset1": {
                "loss": loss1,
                "accuracy": accuracy1
            },
            "dataset2": {
                "loss": loss2,
                "accuracy": accuracy2
            },
        }

    This will create two set of summaries, "dataset1" and "dataset2". Each
    summary dict will contain summaries including both "loss" and "accuracy".

    Args:
      summary_dict: A dictionary of values. If any value in `summary_dict` is
        itself a dictionary, then the function will create a new summary_dict
        with name given by the corresponding key. This is performed recursively.
        Leaf values are then summarized using the parent relative path.
    """
    raise NotImplementedError
