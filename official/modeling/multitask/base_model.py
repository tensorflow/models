# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Abstraction of multi-task model."""
from typing import Text, Dict

import tensorflow as tf


class MultiTaskBaseModel(tf.Module):
  """Base class that holds multi-task model computation."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._sub_tasks = self._instantiate_sub_tasks()

  def _instantiate_sub_tasks(self) -> Dict[Text, tf.keras.Model]:
    """Abstract function that sets up the computation for each sub-task.

    Returns:
      A map from task name (as string) to a tf.keras.Model object that
        represents the sub-task in the multi-task pool.
    """
    raise NotImplementedError(
        "_instantiate_sub_task_models() is not implemented.")

  @property
  def sub_tasks(self):
    """Fetch a map of task name (string) to task model (tf.keras.Model)."""
    return self._sub_tasks

  def initialize(self):
    """Optional function that loads a pre-train checkpoint."""
    return

  def build(self):
    """Builds the networks for tasks to make sure variables are created."""
    # Try to build all sub tasks.
    for task_model in self._sub_tasks.values():
      # Assumes all the tf.Module models are built because we don't have any
      # way to check them.
      if isinstance(task_model, tf.keras.Model) and not task_model.built:
        _ = task_model(task_model.inputs)
