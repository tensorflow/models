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
"""Common util functions and classes used by both keras cifar and imagenet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os

import numpy as np

# pylint: disable=g-bad-import-order
from absl import flags
import tensorflow as tf

from official.utils.misc import keras_utils
# pylint: disable=ungrouped-imports
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.keras.optimizer_v2 import (gradient_descent as
                                                  gradient_descent_v2)

FLAGS = flags.FLAGS
BASE_LEARNING_RATE = 0.1  # This matches Jing's version.
TRAIN_TOP_1 = 'training_accuracy_top_1'


class LearningRateBatchScheduler(tf.keras.callbacks.Callback):
  """Callback to update learning rate on every batch (not epoch boundaries).

  N.B. Only support Keras optimizers, not TF optimizers.

  Args:
      schedule: a function that takes an epoch index and a batch index as input
          (both integer, indexed from 0) and returns a new learning rate as
          output (float).
  """

  def __init__(self, schedule, batch_size, num_images):
    super(LearningRateBatchScheduler, self).__init__()
    self.schedule = schedule
    self.batches_per_epoch = num_images / batch_size
    self.batch_size = batch_size
    self.epochs = -1
    self.prev_lr = -1

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'learning_rate'):
      raise ValueError('Optimizer must have a "learning_rate" attribute.')
    self.epochs += 1

  def on_batch_begin(self, batch, logs=None):
    """Executes before step begins."""
    lr = self.schedule(self.epochs,
                       batch,
                       self.batches_per_epoch,
                       self.batch_size)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function should be float.')
    if lr != self.prev_lr:
      self.model.optimizer.learning_rate = lr  # lr should be a float here
      self.prev_lr = lr
      tf.compat.v1.logging.debug(
          'Epoch %05d Batch %05d: LearningRateBatchScheduler '
          'change learning rate to %s.', self.epochs, batch, lr)


def get_config_proto_v1():
  """Return config proto according to flag settings, or None to use default."""
  config = None
  if FLAGS.enable_xla:
    # TODO(haoyuzhang): Remove this monkey patch when XLA OOM issue is fixed.
    _monkey_patch_org_assert_broadcastable()

    config = tf.compat.v1.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_2)
    # Disable PinToHostOptimizer in grappler when enabling XLA because it causes
    # OOM and performance regression.
    config.graph_options.rewrite_options.pin_to_host_optimization = (
        rewriter_config_pb2.RewriterConfig.OFF)
  return config


def set_config_v2():
  """Config eager context according to flag values using TF 2.0 API."""
  if FLAGS.enable_xla:
    # TODO(haoyuzhang): Remove this monkey patch when XLA OOM issue is fixed.
    _monkey_patch_org_assert_broadcastable()

    tf.config.optimizer.set_jit(True)
    # Disable PinToHostOptimizer in grappler when enabling XLA because it
    # causes OOM and performance regression.
    tf.config.optimizer.set_experimental_options(
        {"pin_to_host_optimization": False}
    )


def set_gpu_thread_mode_and_count(flags_obj):
  """Set GPU thread mode and count, and adjust dataset threads count."""
  cpu_count = multiprocessing.cpu_count()
  tf.compat.v1.logging.info('Logical CPU cores: %s', cpu_count)

  # Allocate private thread pool for each GPU to schedule and launch kernels
  per_gpu_thread_count = flags_obj.per_gpu_thread_count or 1
  os.environ['TF_GPU_THREAD_MODE'] = flags_obj.tf_gpu_thread_mode
  os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
  tf.compat.v1.logging.info('TF_GPU_THREAD_COUNT: %s',
                            os.environ['TF_GPU_THREAD_COUNT'])
  tf.compat.v1.logging.info('TF_GPU_THREAD_MODE: %s',
                            os.environ['TF_GPU_THREAD_MODE'])

  # Limit data preprocessing threadpool to CPU cores minus number of total GPU
  # private threads and memory copy threads.
  total_gpu_thread_count = per_gpu_thread_count * flags_obj.num_gpus
  num_mem_copy_threads = flags_obj.num_gpus
  if not flags_obj.datasets_num_private_threads:
    flags_obj.datasets_num_private_threads = (cpu_count - total_gpu_thread_count
                                              - num_mem_copy_threads)
    tf.compat.v1.logging.info('Set datasets_num_private_threads to %s',
                              flags_obj.datasets_num_private_threads)


def get_optimizer():
  """Returns optimizer to use."""
  # The learning_rate is overwritten at the beginning of each step by callback.
  return gradient_descent_v2.SGD(learning_rate=0.1, momentum=0.9)


def get_callbacks(learning_rate_schedule_fn, num_images):
  """Returns common callbacks."""
  time_callback = keras_utils.TimeHistory(FLAGS.batch_size, FLAGS.log_steps)

  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=FLAGS.model_dir)

  lr_callback = LearningRateBatchScheduler(
      learning_rate_schedule_fn,
      batch_size=FLAGS.batch_size,
      num_images=num_images)

  return time_callback, tensorboard_callback, lr_callback


def build_stats(history, eval_output, time_callback):
  """Normalizes and returns dictionary of stats.

  Args:
    history: Results of the training step. Supports both categorical_accuracy
      and sparse_categorical_accuracy.
    eval_output: Output of the eval step. Assumes first value is eval_loss and
      second value is accuracy_top_1.
    time_callback: Time tracking callback likely used during keras.fit.

  Returns:
    Dictionary of normalized results.
  """
  stats = {}
  if eval_output:
    stats['accuracy_top_1'] = eval_output[1].item()
    stats['eval_loss'] = eval_output[0].item()

  if history and history.history:
    train_hist = history.history
    # Gets final loss from training.
    stats['loss'] = train_hist['loss'][-1].item()
    # Gets top_1 training accuracy.
    if 'categorical_accuracy' in train_hist:
      stats[TRAIN_TOP_1] = train_hist['categorical_accuracy'][-1].item()
    elif 'sparse_categorical_accuracy' in train_hist:
      stats[TRAIN_TOP_1] = train_hist['sparse_categorical_accuracy'][-1].item()

  if time_callback:
    timestamp_log = time_callback.timestamp_log
    stats['step_timestamp_log'] = timestamp_log
    stats['train_finish_time'] = time_callback.train_finish_time
    if len(timestamp_log) > 1:
      stats['avg_exp_per_second'] = (
          time_callback.batch_size * time_callback.log_steps *
          (len(time_callback.timestamp_log)-1) /
          (timestamp_log[-1].timestamp - timestamp_log[0].timestamp))

  return stats


def define_keras_flags():
  """Define flags for Keras models."""

  flags.DEFINE_boolean(name='enable_eager', default=False, help='Enable eager?')
  flags.DEFINE_boolean(name='skip_eval', default=False, help='Skip evaluation?')
  flags.DEFINE_boolean(name='use_trivial_model', default=False,
                       help='Whether to use a trivial Keras model.')
  flags.DEFINE_boolean(
      name='enable_xla', default=False,
      help='Whether to enable XLA auto jit compilation. This is still an '
      'experimental feature, and is not yet effective with TF 2.0.')
  flags.DEFINE_boolean(
      name='enable_tensorboard', default=False,
      help='Whether to enable Tensorboard callback.')
  flags.DEFINE_integer(
      name='train_steps', default=None,
      help='The number of steps to run for training. If it is larger than '
      '# batches per epoch, then use # batches per epoch. When this flag is '
      'set, only one epoch is going to run for training.')


def get_synth_input_fn(height, width, num_channels, num_classes,
                       dtype=tf.float32):
  """Returns an input function that returns a dataset with random data.

  This input_fn returns a data set that iterates over a set of random data and
  bypasses all preprocessing, e.g. jpeg decode and copy. The host to device
  copy is still included. This used to find the upper throughput bound when
  tuning the full input pipeline.

  Args:
    height: Integer height that will be used to create a fake image tensor.
    width: Integer width that will be used to create a fake image tensor.
    num_channels: Integer depth that will be used to create a fake image tensor.
    num_classes: Number of classes that should be represented in the fake labels
      tensor
    dtype: Data type for features/images.

  Returns:
    An input_fn that can be used in place of a real one to return a dataset
    that can be used for iteration.
  """
  # pylint: disable=unused-argument
  def input_fn(is_training, data_dir, batch_size, *args, **kwargs):
    """Returns dataset filled with random data."""
    # Synthetic input should be within [0, 255].
    inputs = tf.random.truncated_normal([height, width, num_channels],
                                        dtype=dtype,
                                        mean=127,
                                        stddev=60,
                                        name='synthetic_inputs')

    labels = tf.random.uniform([1],
                               minval=0,
                               maxval=num_classes - 1,
                               dtype=tf.int32,
                               name='synthetic_labels')
    # Cast to float32 for Keras model.
    labels = tf.cast(labels, dtype=tf.float32)

    data = tf.data.Dataset.from_tensors((inputs, labels)).repeat()

    # `drop_remainder` will make dataset produce outputs with known shapes.
    data = data.batch(batch_size, drop_remainder=True)
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return data

  return input_fn


def is_v2_0():
  """Returns true if using tf 2.0."""
  return tf.__version__.startswith('2')


def get_strategy_scope(strategy):
  if strategy:
    strategy_scope = strategy.scope()
  else:
    strategy_scope = DummyContextManager()

  return strategy_scope


class DummyContextManager(object):

  def __enter__(self):
    pass

  def __exit__(self, *args):
    pass


def _monkey_patch_org_assert_broadcastable():
  """Monkey-patch `assert_broadcast` op to avoid OOM when enabling XLA."""
  def no_op_assert_broadcastable(weights, values):
    del weights, values
    tf.compat.v1.logging.info(
        'Using monkey-patched version of assert_broadcastable op, which always '
        'returns an no_op. It should be removed after XLA OOM issue is fixed.')
    return tf.constant([], dtype=tf.float32)

  from tensorflow.python.ops import weights_broadcast_ops  # pylint: disable=g-import-not-at-top
  if not hasattr(weights_broadcast_ops, 'org_assert_broadcastable'):
    weights_broadcast_ops.org_assert_broadcastable = (
        weights_broadcast_ops.assert_broadcastable)
  weights_broadcast_ops.assert_broadcastable = no_op_assert_broadcastable


def _undo_monkey_patch_org_assert_broadcastable():
  from tensorflow.python.ops import weights_broadcast_ops  # pylint: disable=g-import-not-at-top
  if hasattr(weights_broadcast_ops, 'org_assert_broadcastable'):
    weights_broadcast_ops.assert_broadcastable = (
        weights_broadcast_ops.org_assert_broadcastable)
