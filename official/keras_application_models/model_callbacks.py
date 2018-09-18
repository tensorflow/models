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

Note that, in the callbacks, the global_step is initialized in the __init__ of
each callback rather than on_train_begin. As on_train_begin gets called in
the fit_loop, and it will be reset with each call to fit(). To keep the
global_step persistent across all training sessions, it should be initialized in
the __init__.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.logs import logger

# Metrics to log after each batch and epoch
_PER_BATCH_METRICS = {
    "loss": "train_loss",
    "acc": "train_accuracy",
}
_PER_EPOCH_METRICS = {
    "loss": "train_loss",
    "acc": "train_accuracy",
    "val_loss": "loss",
    "val_acc": "accuracy"
}


class ExamplesPerSecondCallback(tf.keras.callbacks.Callback):
  """ExamplesPerSecond callback.

  This callback records the average_examples_per_sec and
  current_examples_per_sec during training.
  """

  def __init__(self, batch_size, every_n_steps=1, metric_logger=None):
    self._batch_size = batch_size
    self._every_n_steps = every_n_steps
    self._logger = metric_logger or logger.BaseBenchmarkLogger()
    self._global_step = 0  # Initialize it in __init__
    super(ExamplesPerSecondCallback, self).__init__()

  def on_train_begin(self, logs=None):
    self._train_start_time = time.time()
    self._last_recorded_time = time.time()

  def on_batch_end(self, batch, logs=None):
    """Log the examples_per_sec metric every_n_steps."""
    self._global_step += 1
    current_time = time.time()

    if self._global_step % self._every_n_steps == 0:
      average_examples_per_sec = self._batch_size * (
          self._global_step / (current_time - self._train_start_time))
      self._logger.log_metric(
          "average_examples_per_sec", average_examples_per_sec,
          global_step=self._global_step)

      current_examples_per_sec = self._batch_size * (
          self._every_n_steps / (current_time - self._last_recorded_time))
      self._logger.log_metric(
          "current_examples_per_sec", current_examples_per_sec,
          global_step=self._global_step)
      self._last_recorded_time = current_time  # Update last_recorded_time


class LoggingMetricCallback(tf.keras.callbacks.Callback):
  """LoggingMetric callback.

  Log the predefined _PER_BATCH_METRICS after each batch, and log the predefined
  _PER_EPOCH_METRICS after each epoch.
  """

  def __init__(self, metric_logger=None):
    self._logger = metric_logger or logger.BaseBenchmarkLogger()
    self._per_batch_metrics = _PER_BATCH_METRICS
    self._per_epoch_metrics = _PER_EPOCH_METRICS
    self._global_step = 0  # Initialize it in __init__
    super(LoggingMetricCallback, self).__init__()

  def on_batch_end(self, batch, logs=None):
    """Log metrics after each batch."""
    self._global_step += 1
    for metric in _PER_BATCH_METRICS:
      self._logger.log_metric(
          _PER_BATCH_METRICS[metric],
          logs.get(metric),
          global_step=self._global_step)

  def on_epoch_end(self, epoch, logs=None):
    """Log metrics after each epoch."""
    for metric in _PER_EPOCH_METRICS:
      self._logger.log_metric(
          _PER_EPOCH_METRICS[metric],
          logs.get(metric),
          global_step=self._global_step)


def get_model_callbacks(name_list, **kwargs):
  """Factory for getting a list of TensorFlow hooks for training by name.

  Args:
    name_list: a list of strings to name desired callback classes. Allowed:
      ExamplesPerSecondCallback, LoggingMetricCallback, which are defined
      as keys in CALLBACKS.
    **kwargs: a dictionary of arguments to the callbacks.

  Returns:
    list of instantiated callbacks, ready to be used in a classifier.train call.

  Raises:
    ValueError: if an unrecognized name is passed.
  """

  if not name_list:
    return []

  callbacks = []
  for name in name_list:
    callback_name = CALLBACKS.get(name.strip().lower())
    if callback_name is None:
      raise ValueError(
          "Unrecognized training callback requested: {}".format(name))
    else:
      callbacks.append(callback_name(**kwargs))

  return callbacks


def get_examples_per_second_callback(
    every_n_steps=1, batch_size=32, metric_logger=None, **kwargs):  # pylint: disable=unused-argument
  """Function to get ExamplesPerSecondCallback."""
  return ExamplesPerSecondCallback(
      batch_size=batch_size, every_n_steps=every_n_steps,
      metric_logger=metric_logger or logger.get_benchmark_logger())


def get_logging_metric_callback(metric_logger=None, **kwargs):  # pylint: disable=unused-argument
  """Function to get LoggingMetricCallback."""
  return LoggingMetricCallback(
      metric_logger=metric_logger or logger.get_benchmark_logger())


# A dictionary to map the callback name and its corresponding function
CALLBACKS = {
    "examplespersecondcallback": get_examples_per_second_callback,
    "loggingmetriccallback": get_logging_metric_callback,
}
