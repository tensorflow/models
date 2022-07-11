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

"""Custom checkpoint manager that also exports saved models."""

import os
from typing import Callable, Mapping, Optional

from absl import logging
import tensorflow as tf

_SAVED_MODULES_PATH_SUFFIX = 'saved_modules'


def make_saved_modules_directory_name(checkpoint_name: str) -> str:
  return f'{checkpoint_name}_{_SAVED_MODULES_PATH_SUFFIX}'


class SavedModelCheckpointManager(tf.train.CheckpointManager):
  """A CheckpointManager that also exports `SavedModel`s."""

  def __init__(self,
               checkpoint: tf.train.Checkpoint,
               directory: str,
               max_to_keep: int,
               modules_to_export: Optional[Mapping[str, tf.Module]] = None,
               keep_checkpoint_every_n_hours: Optional[int] = None,
               checkpoint_name: str = 'ckpt',
               step_counter: Optional[tf.Variable] = None,
               checkpoint_interval: Optional[int] = None,
               init_fn: Optional[Callable[[], None]] = None):
    """See base class."""
    super().__init__(
        checkpoint=checkpoint,
        directory=directory,
        max_to_keep=max_to_keep,
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
        checkpoint_name=checkpoint_name,
        step_counter=step_counter,
        checkpoint_interval=checkpoint_interval,
        init_fn=init_fn)
    self._modules_to_export = modules_to_export
    self._savedmodels = self._get_existing_savedmodels()

  def save(self,
           checkpoint_number=None,
           check_interval: bool = True,
           options: Optional[tf.train.CheckpointOptions] = None):
    """See base class."""
    checkpoint_path = super().save(
        checkpoint_number=checkpoint_number,
        check_interval=check_interval,
        options=options)
    if not checkpoint_path:  # Nothing got written.
      return
    if not self._modules_to_export:  # No modules to export.
      logging.info('Skip saving SavedModel due to empty modules_to_export.')
      return checkpoint_path

    # Save the models for the checkpoint that just got written.
    saved_modules_directory = make_saved_modules_directory_name(checkpoint_path)
    for model_name, model in self._modules_to_export.items():
      tf.saved_model.save(
          obj=model,
          export_dir=os.path.join(saved_modules_directory, model_name))

    saved_modules_directories_to_keep = [
        make_saved_modules_directory_name(ckpt) for ckpt in self.checkpoints
    ]
    existing_saved_modules_dirs = self._get_existing_savedmodels()

    self._savedmodels = []
    # Keep savedmodels in the same order as checkpoints (from oldest to newest).
    for saved_modules_dir_to_keep in saved_modules_directories_to_keep:
      if saved_modules_dir_to_keep in existing_saved_modules_dirs:
        self._savedmodels.append(saved_modules_dir_to_keep)

    for existing_saved_modules_dir in existing_saved_modules_dirs:
      if existing_saved_modules_dir not in self._savedmodels:
        tf.io.gfile.rmtree(existing_saved_modules_dir)

    return checkpoint_path

  def _get_existing_savedmodels(self):
    """Gets a list of all existing SavedModel paths in `directory`.

    Returns:
      A list of all existing SavedModel paths.
    """
    saved_modules_glob = make_saved_modules_directory_name(
        self._checkpoint_prefix + '-*')
    return tf.io.gfile.glob(saved_modules_glob)

  @property
  def latest_savedmodel(self):
    """The path of the most recent SavedModel in `directory`.

    Returns:
      The latest SavedModel path. If there are no SavedModels, returns `None`.
    """
    if self._savedmodels:
      return self._savedmodels[-1]
    return None

  @property
  def savedmodels(self):
    """A list of managed SavedModels.

    Returns:
      A list of SavedModel paths, sorted from oldest to newest.
    """
    return self._savedmodels
