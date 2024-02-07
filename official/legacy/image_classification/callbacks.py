# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Common modules for callbacks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, List, MutableMapping, Optional, Text

from absl import logging
import tensorflow as tf, tf_keras

from official.modeling import optimization
from official.utils.misc import keras_utils


def get_callbacks(
    model_checkpoint: bool = True,
    include_tensorboard: bool = True,
    time_history: bool = True,
    track_lr: bool = True,
    write_model_weights: bool = True,
    apply_moving_average: bool = False,
    initial_step: int = 0,
    batch_size: int = 0,
    log_steps: int = 0,
    model_dir: Optional[str] = None,
    backup_and_restore: bool = False) -> List[tf_keras.callbacks.Callback]:
  """Get all callbacks."""
  model_dir = model_dir or ''
  callbacks = []
  if model_checkpoint:
    ckpt_full_path = os.path.join(model_dir, 'model.ckpt-{epoch:04d}')
    callbacks.append(
        tf_keras.callbacks.ModelCheckpoint(
            ckpt_full_path, save_weights_only=True, verbose=1))
  if backup_and_restore:
    backup_dir = os.path.join(model_dir, 'tmp')
    callbacks.append(
        tf_keras.callbacks.experimental.BackupAndRestore(backup_dir))
  if include_tensorboard:
    callbacks.append(
        CustomTensorBoard(
            log_dir=model_dir,
            track_lr=track_lr,
            initial_step=initial_step,
            write_images=write_model_weights,
            profile_batch=0))
  if time_history:
    callbacks.append(
        keras_utils.TimeHistory(
            batch_size,
            log_steps,
            logdir=model_dir if include_tensorboard else None))
  if apply_moving_average:
    # Save moving average model to a different file so that
    # we can resume training from a checkpoint
    ckpt_full_path = os.path.join(model_dir, 'average',
                                  'model.ckpt-{epoch:04d}')
    callbacks.append(
        AverageModelCheckpoint(
            update_weights=False,
            filepath=ckpt_full_path,
            save_weights_only=True,
            verbose=1))
    callbacks.append(MovingAverageCallback())
  return callbacks


def get_scalar_from_tensor(t: tf.Tensor) -> int:
  """Utility function to convert a Tensor to a scalar."""
  t = tf_keras.backend.get_value(t)
  if callable(t):
    return t()
  else:
    return t


class CustomTensorBoard(tf_keras.callbacks.TensorBoard):
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
                     logs: Optional[MutableMapping[str, Any]] = None) -> None:
    self.step += 1
    if logs is None:
      logs = {}
    logs.update(self._calculate_metrics())
    super(CustomTensorBoard, self).on_batch_begin(epoch, logs)

  def on_epoch_begin(self,
                     epoch: int,
                     logs: Optional[MutableMapping[str, Any]] = None) -> None:
    if logs is None:
      logs = {}
    metrics = self._calculate_metrics()
    logs.update(metrics)
    for k, v in metrics.items():
      logging.info('Current %s: %f', k, v)
    super(CustomTensorBoard, self).on_epoch_begin(epoch, logs)

  def on_epoch_end(self,
                   epoch: int,
                   logs: Optional[MutableMapping[str, Any]] = None) -> None:
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
        self._get_base_optimizer()._decayed_lr(var_dtype=tf.float32))  # pylint:disable=protected-access

  def _get_base_optimizer(self) -> tf_keras.optimizers.Optimizer:
    """Get the base optimizer used by the current model."""

    optimizer = self.model.optimizer

    # The optimizer might be wrapped by another class, so unwrap it
    while hasattr(optimizer, '_optimizer'):
      optimizer = optimizer._optimizer  # pylint:disable=protected-access

    return optimizer


class MovingAverageCallback(tf_keras.callbacks.Callback):
  """A Callback to be used with a `ExponentialMovingAverage` optimizer.

  Applies moving average weights to the model during validation time to test
  and predict on the averaged weights rather than the current model weights.
  Once training is complete, the model weights will be overwritten with the
  averaged weights (by default).

  Attributes:
    overwrite_weights_on_train_end: Whether to overwrite the current model
      weights with the averaged weights from the moving average optimizer.
    **kwargs: Any additional callback arguments.
  """

  def __init__(self, overwrite_weights_on_train_end: bool = False, **kwargs):
    super(MovingAverageCallback, self).__init__(**kwargs)
    self.overwrite_weights_on_train_end = overwrite_weights_on_train_end

  def set_model(self, model: tf_keras.Model):
    super(MovingAverageCallback, self).set_model(model)
    assert isinstance(self.model.optimizer,
                      optimization.ExponentialMovingAverage)
    self.model.optimizer.shadow_copy(self.model)

  def on_test_begin(self, logs: Optional[MutableMapping[Text, Any]] = None):
    self.model.optimizer.swap_weights()

  def on_test_end(self, logs: Optional[MutableMapping[Text, Any]] = None):
    self.model.optimizer.swap_weights()

  def on_train_end(self, logs: Optional[MutableMapping[Text, Any]] = None):
    if self.overwrite_weights_on_train_end:
      self.model.optimizer.assign_average_vars(self.model.variables)


class AverageModelCheckpoint(tf_keras.callbacks.ModelCheckpoint):
  """Saves and, optionally, assigns the averaged weights.

  Taken from tfa.callbacks.AverageModelCheckpoint.

  Attributes:
    update_weights: If True, assign the moving average weights to the model, and
      save them. If False, keep the old non-averaged weights, but the saved
      model uses the average weights. See `tf_keras.callbacks.ModelCheckpoint`
      for the other args.
  """

  def __init__(self,
               update_weights: bool,
               filepath: str,
               monitor: str = 'val_loss',
               verbose: int = 0,
               save_best_only: bool = False,
               save_weights_only: bool = False,
               mode: str = 'auto',
               save_freq: str = 'epoch',
               **kwargs):
    self.update_weights = update_weights
    super().__init__(filepath, monitor, verbose, save_best_only,
                     save_weights_only, mode, save_freq, **kwargs)

  def set_model(self, model):
    if not isinstance(model.optimizer, optimization.ExponentialMovingAverage):
      raise TypeError('AverageModelCheckpoint is only used when training'
                      'with MovingAverage')
    return super().set_model(model)

  def _save_model(self, epoch, logs):
    assert isinstance(self.model.optimizer,
                      optimization.ExponentialMovingAverage)

    if self.update_weights:
      self.model.optimizer.assign_average_vars(self.model.variables)
      return super()._save_model(epoch, logs)  # pytype: disable=attribute-error  # typed-keras
    else:
      # Note: `model.get_weights()` gives us the weights (non-ref)
      # whereas `model.variables` returns references to the variables.
      non_avg_weights = self.model.get_weights()
      self.model.optimizer.assign_average_vars(self.model.variables)
      # result is currently None, since `super._save_model` doesn't
      # return anything, but this may change in the future.
      result = super()._save_model(epoch, logs)  # pytype: disable=attribute-error  # typed-keras
      self.model.set_weights(non_avg_weights)
      return result
