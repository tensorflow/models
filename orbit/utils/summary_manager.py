# Copyright 2020 The Orbit Authors. All Rights Reserved.
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

import os

import tensorflow as tf


class SummaryManager:
  """A utility class for managing summary writing."""

  def __init__(self, summary_dir, summary_fn, global_step=None):
    """Initializes the `SummaryManager` instance.

    Args:
      summary_dir: The directory in which to write summaries. If `None`, all
        summary writing operations provided by this class are no-ops.
      summary_fn: A callable defined accepting `name`, `value`, and `step`
        parameters, making calls to `tf.summary` functions to write summaries.
      global_step: A `tf.Variable` containing the global step value.
    """
    self._enabled = summary_dir is not None
    self._summary_dir = summary_dir
    self._summary_fn = summary_fn
    self._summary_writers = {}

    if global_step is None:
      self._global_step = tf.summary.experimental.get_step()
    else:
      self._global_step = global_step

  def summary_writer(self, relative_path=""):
    """Returns the underlying summary writer for a specific subdirectory.

    Args:
      relative_path: The current path in which to write summaries, relative to
        the summary directory. By default it is empty, which corresponds to the
        root directory.
    """
    if self._summary_writers and relative_path in self._summary_writers:
      return self._summary_writers[relative_path]
    if self._enabled:
      self._summary_writers[relative_path] = tf.summary.create_file_writer(
          os.path.join(self._summary_dir, relative_path))
    else:
      self._summary_writers[relative_path] = tf.summary.create_noop_writer()
    return self._summary_writers[relative_path]

  def flush(self):
    """Flushes the underlying summary writers."""
    if self._enabled:
      tf.nest.map_structure(tf.summary.flush, self._summary_writers)

  def write_summaries(self, summary_dict):
    """Writes summaries for the given dictionary of values.

    This recursively creates subdirectories for any nested dictionaries
    provided in `summary_dict`, yielding a hierarchy of directories which will
    then be reflected in the TensorBoard UI as different colored curves.

    For example, users may evaluate on muliple datasets and return
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

    This will create two subdirectories, "dataset1" and "dataset2", inside the
    summary root directory. Each directory will contain event files including
    both "loss" and "accuracy" summaries.

    Args:
      summary_dict: A dictionary of values. If any value in `summary_dict` is
        itself a dictionary, then the function will create a subdirectory with
        name given by the corresponding key. This is performed recursively. Leaf
        values are then summarized using the summary writer instance specific to
        the parent relative path.
    """
    if not self._enabled:
      return
    self._write_summaries(summary_dict)

  def _write_summaries(self, summary_dict, relative_path=""):
    for name, value in summary_dict.items():
      if isinstance(value, dict):
        self._write_summaries(
            value, relative_path=os.path.join(relative_path, name))
      else:
        with self.summary_writer(relative_path).as_default():
          self._summary_fn(name, value, step=self._global_step)
