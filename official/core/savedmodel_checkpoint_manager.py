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
import re
from typing import Callable, Mapping, Optional

from absl import logging
import tensorflow as tf


def make_saved_modules_directory_name(checkpoint_name: str) -> str:
  return f'{checkpoint_name}_saved_modules'


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

    # `checkpoint_path` ends in `-[\d]+`.  We want to glob for all existing
    # checkpoints, and we use the .index file for that.
    checkpoint_glob = re.sub(r'\d+$', '*.index', checkpoint_path)
    existing_checkpoint_files = tf.io.gfile.glob(checkpoint_glob)

    saved_modules_directories_to_keep = [
        make_saved_modules_directory_name(os.path.splitext(ckpt_index)[0])
        for ckpt_index in existing_checkpoint_files
    ]
    saved_modules_glob = re.sub(r'\d+_saved_modules$', '*_saved_modules',
                                saved_modules_directory)

    for existing_saved_modules_dir in tf.io.gfile.glob(saved_modules_glob):
      if (existing_saved_modules_dir not in saved_modules_directories_to_keep
          and tf.io.gfile.isdir(existing_saved_modules_dir)):
        tf.io.gfile.rmtree(existing_saved_modules_dir)

    return checkpoint_path
