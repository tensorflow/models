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

from official.utils.flags import core as flags_core
from official.utils.misc import keras_utils
# pylint: disable=ungrouped-imports
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


class PiecewiseConstantDecayWithWarmup(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Piecewise constant decay with warmup schedule."""

  def __init__(self, batch_size, epoch_size, warmup_epochs, boundaries,
               multipliers, compute_lr_on_cpu=True, name=None):
    super(PiecewiseConstantDecayWithWarmup, self).__init__()
    if len(boundaries) != len(multipliers) - 1:
      raise ValueError('The length of boundaries must be 1 less than the '
                       'length of multipliers')

    base_lr_batch_size = 256
    num_batches_per_epoch = epoch_size // batch_size

    self.rescaled_lr = BASE_LEARNING_RATE * batch_size / base_lr_batch_size
    self.step_boundaries = [float(num_batches_per_epoch) * x
                            for x in boundaries]
    self.lr_values = [self.rescaled_lr * m for m in multipliers]
    self.warmup_steps = warmup_epochs * num_batches_per_epoch
    self.compute_lr_on_cpu = compute_lr_on_cpu
    self.name = name

    self.learning_rate_ops_cache = {}

  def __call__(self, step):
    if tf.executing_eagerly():
      return self._get_learning_rate(step)

    # In an eager function or graph, the current implementation of optimizer
    # repeatedly call and thus create ops for the learning rate schedule. To
    # avoid this, we cache the ops if not executing eagerly.
    graph = tf.compat.v1.get_default_graph()
    if graph not in self.learning_rate_ops_cache:
      if self.compute_lr_on_cpu:
        with tf.device('/device:CPU:0'):
          self.learning_rate_ops_cache[graph] = self._get_learning_rate(step)
      else:
        self.learning_rate_ops_cache[graph] = self._get_learning_rate(step)
    return self.learning_rate_ops_cache[graph]

  def _get_learning_rate(self, step):
    """Compute learning rate at given step."""
    with tf.compat.v1.name_scope(self.name, 'PiecewiseConstantDecayWithWarmup',
                                 [self.rescaled_lr, self.step_boundaries,
                                  self.lr_values, self.warmup_steps,
                                  self.compute_lr_on_cpu]):
      def warmup_lr(step):
        return self.rescaled_lr * (
            tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32))
      def piecewise_lr(step):
        return tf.compat.v1.train.piecewise_constant(
            step, self.step_boundaries, self.lr_values)
      return tf.cond(step < self.warmup_steps,
                     lambda: warmup_lr(step),
                     lambda: piecewise_lr(step))

  def get_config(self):
    return {
        'rescaled_lr': self.rescaled_lr,
        'step_boundaries': self.step_boundaries,
        'lr_values': self.lr_values,
        'warmup_steps': self.warmup_steps,
        'compute_lr_on_cpu': self.compute_lr_on_cpu,
        'name': self.name
    }


def set_gpu_thread_mode_and_count(flags_obj):
  """Set GPU thread mode and count, and adjust dataset threads count."""
  cpu_count = multiprocessing.cpu_count()
  tf.compat.v1.logging.info('Logical CPU cores: %s', cpu_count)

  # Allocate private thread pool for each GPU to schedule and launch kernels
  per_gpu_thread_count = flags_obj.per_gpu_thread_count or 2
  os.environ['TF_GPU_THREAD_MODE'] = flags_obj.tf_gpu_thread_mode
  os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
  tf.compat.v1.logging.info('TF_GPU_THREAD_COUNT: %s',
                            os.environ['TF_GPU_THREAD_COUNT'])
  tf.compat.v1.logging.info('TF_GPU_THREAD_MODE: %s',
                            os.environ['TF_GPU_THREAD_MODE'])

  # Limit data preprocessing threadpool to CPU cores minus number of total GPU
  # private threads and memory copy threads.
  total_gpu_thread_count = per_gpu_thread_count * flags_obj.num_gpus
  num_runtime_threads = flags_obj.num_gpus
  if not flags_obj.datasets_num_private_threads:
    flags_obj.datasets_num_private_threads = min(
        cpu_count - total_gpu_thread_count - num_runtime_threads,
        flags_obj.num_gpus * 8)
    tf.compat.v1.logging.info('Set datasets_num_private_threads to %s',
                              flags_obj.datasets_num_private_threads)


def get_optimizer(learning_rate=0.1):
  """Returns optimizer to use."""
  # The learning_rate is overwritten at the beginning of each step by callback.
  return gradient_descent_v2.SGD(learning_rate=learning_rate, momentum=0.9)


def get_callbacks(learning_rate_schedule_fn, num_images):
  """Returns common callbacks."""
  time_callback = keras_utils.TimeHistory(FLAGS.batch_size, FLAGS.log_steps)
  callbacks = [time_callback]

  if not FLAGS.use_tensor_lr:
    lr_callback = LearningRateBatchScheduler(
        learning_rate_schedule_fn,
        batch_size=FLAGS.batch_size,
        num_images=num_images)
    callbacks.append(lr_callback)

  if FLAGS.enable_tensorboard:
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=FLAGS.model_dir)
    callbacks.append(tensorboard_callback)

  if FLAGS.profile_steps:
    profiler_callback = keras_utils.get_profiler_callback(
        FLAGS.model_dir,
        FLAGS.profile_steps,
        FLAGS.enable_tensorboard)
    callbacks.append(profiler_callback)

  return callbacks


def build_stats(history, eval_output, callbacks):
  """Normalizes and returns dictionary of stats.

  Args:
    history: Results of the training step. Supports both categorical_accuracy
      and sparse_categorical_accuracy.
    eval_output: Output of the eval step. Assumes first value is eval_loss and
      second value is accuracy_top_1.
    callbacks: a list of callbacks which might include a time history callback
      used during keras.fit.

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

  if not callbacks:
    return stats

  # Look for the time history callback which was used during keras.fit
  for callback in callbacks:
    if isinstance(callback, keras_utils.TimeHistory):
      timestamp_log = callback.timestamp_log
      stats['step_timestamp_log'] = timestamp_log
      stats['train_finish_time'] = callback.train_finish_time
      if len(timestamp_log) > 1:
        stats['avg_exp_per_second'] = (
            callback.batch_size * callback.log_steps *
            (len(callback.timestamp_log)-1) /
            (timestamp_log[-1].timestamp - timestamp_log[0].timestamp))
  return stats


def define_keras_flags(dynamic_loss_scale=True):
  """Define flags for Keras models."""
  flags_core.define_base(run_eagerly=True)
  flags_core.define_performance(num_parallel_calls=False,
                                tf_gpu_thread_mode=True,
                                datasets_num_private_threads=True,
                                dynamic_loss_scale=dynamic_loss_scale,
                                loss_scale=True,
                                tf_data_experimental_slack=True,
                                enable_xla=True)
  flags_core.define_image()
  flags_core.define_benchmark()
  flags.adopt_module_key_flags(flags_core)

  flags.DEFINE_boolean(name='enable_eager', default=False, help='Enable eager?')
  flags.DEFINE_boolean(name='skip_eval', default=False, help='Skip evaluation?')
  # TODO(b/135607288): Remove this flag once we understand the root cause of
  # slowdown when setting the learning phase in Keras backend.
  flags.DEFINE_boolean(
      name='set_learning_phase_to_train', default=True,
      help='If skip eval, also set Keras learning phase to 1 (training).')
  flags.DEFINE_boolean(
      name='explicit_gpu_placement', default=False,
      help='If not using distribution strategy, explicitly set device scope '
      'for the Keras training loop.')
  flags.DEFINE_boolean(name='use_trivial_model', default=False,
                       help='Whether to use a trivial Keras model.')
  flags.DEFINE_boolean(name='report_accuracy_metrics', default=True,
                       help='Report metrics during training and evaluation.')
  flags.DEFINE_boolean(name='use_tensor_lr', default=False,
                       help='Use learning rate tensor instead of a callback.')
  flags.DEFINE_boolean(
      name='enable_tensorboard', default=False,
      help='Whether to enable Tensorboard callback.')
  flags.DEFINE_integer(
      name='train_steps', default=None,
      help='The number of steps to run for training. If it is larger than '
      '# batches per epoch, then use # batches per epoch. When this flag is '
      'set, only one epoch is going to run for training.')
  flags.DEFINE_string(
      name='profile_steps', default=None,
      help='Save profiling data to model dir at given range of steps. The '
      'value must be a comma separated pair of positive integers, specifying '
      'the first and last step to profile. For example, "--profile_steps=2,4" '
      'triggers the profiler to process 3 steps, starting from the 2nd step. '
      'Note that profiler has a non-trivial performance overhead, and the '
      'output file can be gigantic if profiling many steps.')
  flags.DEFINE_boolean(
      name='data_delay_prefetch', default=False,
      help='Add a small delay in tf.data prefetch to prioritize memory copy of '
      'other tensors over the data minibatch for the (T+1)th step. It should '
      'help improve performance using EagerIterator and function. The codepath '
      'when enabling this feature is experimental and will be removed once the '
      'corresponding performance features are fully supported in TensorFlow.')
  flags.DEFINE_boolean(
      name='batchnorm_spatial_persistent', default=True,
      help='Enable the spacial persistent mode for CuDNN batch norm kernel.')
  flags.DEFINE_boolean(
      name='clone_model_in_keras_dist_strat', default=None,
      help='If False, then the experimental code path is used that doesn\'t '
           'clone models for distribution.')
  flags.DEFINE_boolean(
      name='enable_get_next_as_optional', default=False,
      help='Enable get_next_as_optional behavior in DistributedIterator.')
  # TODO(b/76028325): Remove when generic layout optimizer is ready.
  flags.DEFINE_boolean(
      name='enable_grappler_layout_optimizer',
      default=True,
      help='Enable Grappler layout optimizer. Currently Grappler can '
           'de-optimize fp16 graphs byt forcing NCHW layout for all '
           'convolutions and batch normalizations, and this flag allows to '
           'disable it.'
  )

def get_synth_input_fn(height, width, num_channels, num_classes,
                       dtype=tf.float32, drop_remainder=True):
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
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.

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
    data = data.batch(batch_size, drop_remainder=drop_remainder)
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return data

  return input_fn


def data_delay_prefetch():
  """Use unstable code for perf tuning purposes."""
  if not FLAGS.use_synthetic_data:
    _monkey_patch_org_create_device_dataset()


def set_cudnn_batchnorm_mode():
  """Set CuDNN batchnorm mode for better performance.

     Note: Spatial Persistent mode may lead to accuracy losses for certain
     models.
  """
  if FLAGS.batchnorm_spatial_persistent:
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
  else:
    os.environ.pop('TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT', None)


# TODO(haoyuzhang): remove this monkey patch when the "prefetch with slack"
# feature is available in tf.data.
def _monkey_patch_org_create_device_dataset():
  """Monkey-patch `_create_device_dataset` method with delayed prefetch."""

  import ast  # pylint: disable=g-import-not-at-top
  import inspect  # pylint: disable=g-import-not-at-top
  from tensorflow.python.data.ops import multi_device_iterator_ops  # pylint: disable=g-import-not-at-top

  tf.compat.v1.logging.info(
      'Using monkey-patched version of MultiDeviceIterator. It should be '
      'removed when the prefetch with slack feature is implemented in tf.data.')
  cls_multi_device_iterator = ast.parse(
      inspect.getsource(multi_device_iterator_ops.MultiDeviceIterator))
  org_create_device_dataset_code = inspect.getsource(
      multi_device_iterator_ops.MultiDeviceIterator._create_device_dataset)  # pylint: disable=protected-access
  code_lines = org_create_device_dataset_code.split('\n')
  # Insert in reverse order to avoid line number shift by previous insertions
  code_lines.insert(5, '      ds = ds.apply(sleep_ops.sleep(11000))')  # 11ms
  code_lines.insert(2, '    from tensorflow.python.data.experimental.ops import sleep as sleep_ops')  # pylint: disable=line-too-long
  patched_code = '\n'.join(line[2:] for line in code_lines)
  cls_multi_device_iterator.body[0].body[2] = ast.parse(patched_code).body[0]
  exec(compile(cls_multi_device_iterator, '<string>', 'exec'),  # pylint: disable=exec-used
       multi_device_iterator_ops.__dict__)
