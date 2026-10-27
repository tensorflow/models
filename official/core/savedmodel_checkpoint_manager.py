# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
import time
from typing import Callable, List, Mapping, Optional, Union

from absl import logging
import tensorflow as tf, tf_keras

SAVED_MODULES_PATH_SUFFIX = 'saved_modules'


def make_saved_modules_directory_name(checkpoint_name: str) -> str:
  return f'{checkpoint_name}_{SAVED_MODULES_PATH_SUFFIX}'


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
    self._savedmodels = self.get_existing_savedmodels()

  def save(
      self,
      checkpoint_number: Optional[int] = None,
      check_interval: bool = True,
      options: Optional[tf.train.CheckpointOptions] = None,
  ) -> Optional[str]:
    """Creates a new checkpoint or recovers a missing SavedModel export.

    Unlike the base class `tf.train.CheckpointManager.save()`, which only saves
    new checkpoints and returns None when a checkpoint is not written (e.g. if
    the step was already saved or the checkpoint interval has not elapsed), this
    method also checks if the latest existing checkpoint is missing its exported
    SavedModel (e.g. due to process preemption during export). If missing and
    eligible, it recovers and exports the SavedModel and returns the existing
    `latest_checkpoint` path.

    Args:
      checkpoint_number: An optional integer, or an integer-dtype `Variable` or
        `Tensor`, used to number the checkpoint. If `None` (default),
        checkpoints are numbered using `checkpoint.save_counter`.
      check_interval: An optional boolean. Effective when `checkpoint_interval`
        is passed into the manager. If `True`, only saves or recovers if the
        interval condition is met or the step matches `latest_checkpoint`.
      options: Optional `tf.train.CheckpointOptions` object.

    Returns:
      The path to the new checkpoint if one was saved, or the path to
      `latest_checkpoint` if an unexported SavedModel was recovered and
      exported.
      Returns `None` if no checkpoint was saved and no missing SavedModel was
      recovered (e.g. interval has not elapsed, step mismatch with
      `check_interval=True`, SavedModel already exists, or `modules_to_export`
      is empty / contains no exportable signatures).
    """
    checkpoint_path = super().save(
        checkpoint_number=checkpoint_number,
        check_interval=check_interval,
        options=options,
    )
    if not checkpoint_path:
      if not self._modules_to_export:
        return None

      # If no new checkpoint was written (e.g. because it was already saved
      # prior to a preemption/restart), check if the latest checkpoint is
      # missing its exported SavedModel.
      checkpoint_path = self._find_unexported_checkpoint(
          checkpoint_number=checkpoint_number, check_interval=check_interval
      )
      if not checkpoint_path or not any(
          getattr(m, 'saved_model_signatures', None) is not None
          for m in self._modules_to_export.values()
      ):
        return None

    if not self._modules_to_export:
      logging.info('Skip saving SavedModel due to empty modules_to_export.')
      return checkpoint_path

    self._export_saved_model(checkpoint_path)
    self._prune_saved_models()
    return checkpoint_path

  def _find_unexported_checkpoint(
      self,
      checkpoint_number: Optional[int] = None,
      check_interval: bool = True,
  ) -> Optional[str]:
    """Returns latest_checkpoint if it exists but its SavedModel is missing.

    Args:
      checkpoint_number: An optional integer specifying the step number being
        saved. If None, falls back to `_step_counter` if available.
      check_interval: If True, the checkpoint is only eligible for re-export if
        `checkpoint_number` (or `_step_counter`) matches the step suffix of
        `latest_checkpoint`. If False, step matching is bypassed.

    Returns:
      The path to `latest_checkpoint` if `latest_checkpoint` exists, step
      matching criteria are met (or `check_interval=False`), and its SavedModel
      directory does not exist on disk. Returns `None` if there is no latest
      checkpoint, if `check_interval=True` and the step does not match the
      latest checkpoint, or if the SavedModel directory already exists.
    """
    if not self.latest_checkpoint:
      return None

    step = checkpoint_number
    if step is None and self._step_counter is not None:
      step = int(self._step_counter.numpy())

    step_matches = step is not None and self.latest_checkpoint.endswith(
        f'-{step}'
    )
    if check_interval and not step_matches:
      return None

    target_saved_modules = make_saved_modules_directory_name(
        self.latest_checkpoint
    )
    if not tf.io.gfile.exists(target_saved_modules):
      logging.info(
          'Checkpoint %s already exists but %s is missing. Exporting'
          ' SavedModel.',
          self.latest_checkpoint,
          target_saved_modules,
      )
      return self.latest_checkpoint
    return None

  def _export_saved_model(self, checkpoint_path: str) -> None:
    """Atomically exports modules for the given checkpoint."""
    if not self._modules_to_export:
      return

    saved_modules_directory = make_saved_modules_directory_name(checkpoint_path)
    saved_modules_directory_tmp = saved_modules_directory + '_temp'

    # Clean up any leftover temporary directory from an interrupted export.
    if tf.io.gfile.exists(saved_modules_directory_tmp):
      tf.io.gfile.rmtree(saved_modules_directory_tmp)

    try:
      for model_name, model in self._modules_to_export.items():
        signatures = getattr(model, 'saved_model_signatures', None)
        if signatures is not None:
          tf.saved_model.save(
              obj=model,
              export_dir=os.path.join(saved_modules_directory_tmp, model_name),
              signatures=signatures,
          )
      if tf.io.gfile.exists(saved_modules_directory_tmp):
        if tf.io.gfile.exists(saved_modules_directory):
          tf.io.gfile.rmtree(saved_modules_directory)
        tf.io.gfile.rename(saved_modules_directory_tmp, saved_modules_directory)
    except Exception:
      if tf.io.gfile.exists(saved_modules_directory_tmp):
        tf.io.gfile.rmtree(saved_modules_directory_tmp)
      raise

  def _prune_saved_models(self) -> None:
    """Prunes SavedModels that no longer correspond to managed checkpoints."""
    saved_modules_directories_to_keep = [
        make_saved_modules_directory_name(ckpt) for ckpt in self.checkpoints
    ]
    existing_saved_modules_dirs = self.get_existing_savedmodels()

    self._savedmodels = []
    # Keep savedmodels in the same order as checkpoints (from oldest to newest).
    for saved_modules_dir_to_keep in saved_modules_directories_to_keep:
      if saved_modules_dir_to_keep in existing_saved_modules_dirs:
        self._savedmodels.append(saved_modules_dir_to_keep)

    for existing_saved_modules_dir in existing_saved_modules_dirs:
      if existing_saved_modules_dir not in self._savedmodels:
        tf.io.gfile.rmtree(existing_saved_modules_dir)

  def get_existing_savedmodels(self) -> List[str]:
    """Gets a list of all existing SavedModel paths in `directory`.

    Returns:
      A list of all existing SavedModel paths.
    """
    saved_modules_glob = make_saved_modules_directory_name(
        self._checkpoint_prefix + '-*')
    savedmodels = tf.io.gfile.glob(saved_modules_glob)
    # Filter out temporary savedmodel.
    savedmodels = [
        savedmodel
        for savedmodel in savedmodels
        if savedmodel.endswith(SAVED_MODULES_PATH_SUFFIX)
    ]
    return savedmodels

  @property
  def latest_savedmodel(self) -> Union[str, None]:
    """The path of the most recent SavedModel in `directory`.

    Returns:
      The latest SavedModel path. If there are no SavedModels, returns `None`.
    """
    if self._savedmodels:
      return self._savedmodels[-1]
    return None

  @property
  def savedmodels(self) -> List[str]:
    """A list of managed SavedModels.

    Returns:
      A list of SavedModel paths, sorted from oldest to newest.
    """
    return self._savedmodels

  @property
  def modules_to_export(self) -> Union[Mapping[str, tf.Module], None]:
    return self._modules_to_export

  def get_savedmodel_number_from_path(self,
                                      savedmodel_path: str) -> Union[int, None]:
    """Gets the savedmodel_number/checkpoint_number from savedmodel filepath.

    The savedmodel_number is global step when using with orbit controller.

    Args:
      savedmodel_path: savedmodel directory path.

    Returns:
      Savedmodel number or None if no matched pattern found in savedmodel path.
    """
    pattern = rf'\d+_{SAVED_MODULES_PATH_SUFFIX}$'
    savedmodel_number = re.search(pattern, savedmodel_path)
    if savedmodel_number:
      savedmodel_number = savedmodel_number.group()
      return int(savedmodel_number[:-len(SAVED_MODULES_PATH_SUFFIX) - 1])
    return None

  def savedmodels_iterator(self,
                           min_interval_secs: float = 0,
                           timeout: Optional[float] = None,
                           timeout_fn: Optional[Callable[[], bool]] = None):
    """Continuously yield new SavedModel files as they appear.

    The iterator only checks for new savedmodels when control flow has been
    reverted to it. The logic is same to the `train.checkpoints_iterator`.

    Args:
      min_interval_secs: The minimum number of seconds between yielding
        savedmodels.
      timeout: The maximum number of seconds to wait between savedmodels. If
        left as `None`, then the process will wait indefinitely.
      timeout_fn: Optional function to call after a timeout.  If the function
        returns True, then it means that no new savedmodels will be generated
        and the iterator will exit.  The function is called with no arguments.

    Yields:
      String paths to latest SavedModel files as they arrive.
    """
    savedmodel_path = None
    while True:
      new_savedmodel_path = self.wait_for_new_savedmodel(
          savedmodel_path, timeout=timeout)
      if new_savedmodel_path is None:
        if not timeout_fn:
          # timed out
          logging.info('Timed-out waiting for a savedmodel.')
          return
        if timeout_fn():
          # The timeout_fn indicated that we are truly done.
          return
        else:
          # The timeout_fn indicated that more savedmodels may come.
          continue
      start = time.time()
      savedmodel_path = new_savedmodel_path
      yield savedmodel_path
      time_to_next_eval = start + min_interval_secs - time.time()
      if time_to_next_eval > 0:
        time.sleep(time_to_next_eval)

  def wait_for_new_savedmodel(
      self,
      last_savedmodel: Optional[str] = None,
      seconds_to_sleep: float = 1.0,
      timeout: Optional[float] = None) -> Union[str, None]:
    """Waits until a new savedmodel file is found.

    Args:
      last_savedmodel: The last savedmodel path used or `None` if we're
        expecting a savedmodel for the first time.
      seconds_to_sleep: The number of seconds to sleep for before looking for a
        new savedmodel.
      timeout: The maximum number of seconds to wait. If left as `None`, then
        the process will wait indefinitely.

    Returns:
      A new savedmodel path, or None if the timeout was reached.
    """
    logging.info('Waiting for new savedmodel at %s', self._directory)
    stop_time = time.time() + timeout if timeout is not None else None

    last_savedmodel_number = -1
    if last_savedmodel:
      last_savedmodel_number = self.get_savedmodel_number_from_path(
          last_savedmodel)

    while True:
      if stop_time is not None and time.time() + seconds_to_sleep > stop_time:
        return None

      existing_savedmodels = {}
      for savedmodel_path in self.get_existing_savedmodels():
        savedmodel_number = self.get_savedmodel_number_from_path(
            savedmodel_path)
        if savedmodel_number is not None:
          existing_savedmodels[savedmodel_number] = savedmodel_path

      # Find the first savedmodel with larger step number as next savedmodel.
      savedmodel_path = None
      existing_savedmodels = dict(sorted(existing_savedmodels.items()))
      for savedmodel_number in existing_savedmodels:
        if savedmodel_number > last_savedmodel_number:
          savedmodel_path = existing_savedmodels[savedmodel_number]
          break

      if savedmodel_path:
        logging.info('Found new savedmodel at %s', savedmodel_path)
        return savedmodel_path
      else:
        time.sleep(seconds_to_sleep)
