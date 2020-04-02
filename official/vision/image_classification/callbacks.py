# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Common modules for callbacks."""
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import os
from absl import logging

import tensorflow as tf
from typing import Any, List, MutableMapping

from official.utils.misc import keras_utils


def get_callbacks(model_checkpoint: bool = True,
                  include_tensorboard: bool = True,
                  time_history: bool = True,
                  track_lr: bool = True,
                  write_model_weights: bool = True,
                  initial_step: int = 0,
                  batch_size: int = 0,
                  log_steps: int = 0,
                  model_dir: str = None) -> List[tf.keras.callbacks.Callback]:
  """Get all callbacks."""
  model_dir = model_dir or ''
  callbacks = []
  if model_checkpoint:
    ckpt_full_path = os.path.join(model_dir, 'model.ckpt-{epoch:04d}')
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_full_path, save_weights_only=True, verbose=1))
  if include_tensorboard:
    callbacks.append(
        CustomTensorBoard(
            log_dir=model_dir,
            track_lr=track_lr,
            initial_step=initial_step,
            write_images=write_model_weights))
  if time_history:
    callbacks.append(
        keras_utils.TimeHistory(
            batch_size,
            log_steps,
            logdir=model_dir if include_tensorboard else None))
  return callbacks


def get_scalar_from_tensor(t: tf.Tensor) -> int:
  """Utility function to convert a Tensor to a scalar."""
  t = tf.keras.backend.get_value(t)
  if callable(t):
    return t()
  else:
    return t


class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
  """A customized TensorBoard callback that tracks additional datapoints.

  Metrics tracked:
  - Global learning rate

  Attributes:
    log_dir: the path of the directory where to save the log files to be parsed
      by TensorBoard.
    track_lr: `bool`, whether or not to track the global learning rate.
    initial_step: the initial step, used for preemption recovery.
    **kwargs: Additional arguments for backwards compatibility. Possible key is
      `period`.
  """

  # TODO(b/146499062): track params, flops, log lr, l2 loss,
  # classification loss

  def __init__(self,
               log_dir: str,
               track_lr: bool = False,
               initial_step: int = 0,
               **kwargs):
    super(CustomTensorBoard, self).__init__(log_dir=log_dir, **kwargs)
    self.step = initial_step
    self._track_lr = track_lr

  def on_batch_begin(self,
                     epoch: int,
                     logs: MutableMapping[str, Any] = None) -> None:
    self.step += 1
    if logs is None:
      logs = {}
    logs.update(self._calculate_metrics())
    super(CustomTensorBoard, self).on_batch_begin(epoch, logs)

  def on_epoch_begin(self,
                     epoch: int,
                     logs: MutableMapping[str, Any] = None) -> None:
    if logs is None:
      logs = {}
    metrics = self._calculate_metrics()
    logs.update(metrics)
    for k, v in metrics.items():
      logging.info('Current %s: %f', k, v)
    super(CustomTensorBoard, self).on_epoch_begin(epoch, logs)

  def on_epoch_end(self,
                   epoch: int,
                   logs: MutableMapping[str, Any] = None) -> None:
    if logs is None:
      logs = {}
    metrics = self._calculate_metrics()
    logs.update(metrics)
    super(CustomTensorBoard, self).on_epoch_end(epoch, logs)

  def _calculate_metrics(self) -> MutableMapping[str, Any]:
    logs = {}
    # TODO(b/149030439): disable LR reporting.
    # if self._track_lr:
    #   logs['learning_rate'] = self._calculate_lr()
    return logs

  def _calculate_lr(self) -> int:
    """Calculates the learning rate given the current step."""
    return get_scalar_from_tensor(
        self._get_base_optimizer()._decayed_lr(var_dtype=tf.float32))

  def _get_base_optimizer(self) -> tf.keras.optimizers.Optimizer:
    """Get the base optimizer used by the current model."""

    optimizer = self.model.optimizer

    # The optimizer might be wrapped by another class, so unwrap it
    while hasattr(optimizer, '_optimizer'):
      optimizer = optimizer._optimizer  # pylint:disable=protected-access

    return optimizer
