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
"""Callbacks for Keras built-in application models.

Note that, Keras Callbacks provide two levels (batch and epoch level) to
record the internal states and statistics during model training. So in the
following customized callbacks, we also provide two levels for benchmark:
`batch_based` and `epoch_based`. Users can specify the `--benchmark_level` to
choose benchmark level. This is similar to the `every_n_seconds` and
`every_n_steps` options in tensorflow Estimator hooks.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

# pylint: disable=g-bad-import-order
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.utils.logs import logger

CALLBACKS = ["examplespersecondcallback", "loggingmetriccallback"]

_METRICS_TO_LOG = {
    "loss": "train_loss",
    "acc": "train_accuracy",
    "val_loss": "loss",
    "val_acc": "accuracy"
}


class ExamplesPerSecondCallback(tf.keras.callbacks.Callback):
  """ExamplesPerSecond callback.

  This callback records the average examples per second during training.
  """

  def __init__(self, batch_size=None, epoch_size=None, batch_based=None,
               epoch_based=None, metric_logger=None):
    if (batch_based is None) == (epoch_based is None):
      raise ValueError("Exactly one of batch_based "
                       "and epoch_based should be provided.")

    if batch_based and (batch_size is None):
      raise ValueError("batch_size should be provided for "
                       "batch_based benchmark.")

    if epoch_based and (epoch_size is None):
      raise ValueError("epoch_size should be provided for "
                       "epoch_based benchmark.")

    self._batch_based = batch_based
    self._epoch_based = epoch_based
    self._batch_size = batch_size
    self._epoch_size = epoch_size

    self._logger = metric_logger or logger.BaseBenchmarkLogger()

  def on_train_begin(self, logs=None):
    self._global_step = 0
    # For batch_based
    if self._batch_based:
      self._train_time_batch_based = 0
    else:  # For epoch_based
      self._epochs = 0
      self._train_time_epoch_based = 0

  def on_batch_begin(self, batch, logs=None):
    if self._batch_based:
      self._time_start_batch_based = time.time()

  def on_batch_end(self, batch, logs=None):
    self._global_step += 1
    if self._batch_based:
      self._train_time_batch_based += time.time() - self._time_start_batch_based
      examples_per_sec_batch_based = self._batch_size * (
          self._global_step / self._train_time_batch_based)
      self._logger.log_metric(
          "examples_per_sec_batch_based",
          examples_per_sec_batch_based,
          global_step=self._global_step)

  def on_epoch_begin(self, epoch, logs=None):
    if self._epoch_based:
      self._time_start_epoch_based = time.time()

  def on_epoch_end(self, epoch, logs=None):
    if self._epoch_based:
      self._epochs += 1
      self._train_time_epoch_based += time.time() - self._time_start_epoch_based
      examples_per_sec_epoch_based = self._epoch_size * (
          self._epochs / self._train_time_epoch_based)
      self._logger.log_metric(
          "examples_per_sec_epoch_based",
          examples_per_sec_epoch_based,
          global_step=self._global_step)


class LoggingMetricCallback(tf.keras.callbacks.Callback):
  """LoggingMetric callback.

  By default, four metrics are logged: train accuracy, train loss, evaluation
  accuracy and evaluation loss. Check _METRICS_TO_LOG for details.
  """

  def __init__(self, metrics=None, batch_based=None, epoch_based=None,
               metric_logger=None):
    if (batch_based is None) == (epoch_based is None):
      raise ValueError("Exactly one of batch_based "
                       "and epoch_based should be provided.")

    self._batch_based = batch_based
    self._epoch_based = epoch_based

    self._logger = metric_logger or logger.BaseBenchmarkLogger()
    self._metrics = metrics or _METRICS_TO_LOG
    for metric in self._metrics:
      if metric.strip().lower() not in _METRICS_TO_LOG.keys():
        raise ValueError("Unrecognized metric requested: {}".format(metric))

  def on_train_begin(self, logs=None):
    self._global_step = 0

  def on_batch_end(self, batch, logs=None):
    """Log metrics after each batch."""
    self._global_step += 1
    if self._batch_based:
      for metric in self._metrics.keys():
        # `val_acc` and `val_loss` can only be obtained after each epoch.
        metric = metric.strip().lower()
        if metric not in ["val_acc", "val_loss"]:
          self._logger.log_metric(
              _METRICS_TO_LOG[metric],
              logs.get(metric),
              global_step=self._global_step)

  def on_epoch_end(self, epoch, logs=None):
    """Log metrics after each epoch."""
    if self._epoch_based:
      for metric in self._metrics.keys():
        metric = metric.strip().lower()
        self._logger.log_metric(
            _METRICS_TO_LOG[metric],
            logs.get(metric),
            global_step=self._global_step)
