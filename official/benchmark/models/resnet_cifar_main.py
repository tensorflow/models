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
"""Runs a ResNet model on the Cifar-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import libraries
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
from official.benchmark.models import cifar_preprocessing
from official.benchmark.models import resnet_cifar_model
from official.benchmark.models import synthetic_util
from official.common import distribute_utils
from official.utils.flags import core as flags_core
from official.utils.misc import keras_utils
from official.vision.image_classification.resnet import common


LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
    (0.1, 91), (0.01, 136), (0.001, 182)
]


def learning_rate_schedule(current_epoch,
                           current_batch,
                           batches_per_epoch,
                           batch_size):
  """Handles linear scaling rule and LR decay.

  Scale learning rate at epoch boundaries provided in LR_SCHEDULE by the
  provided scaling factor.

  Args:
    current_epoch: integer, current epoch indexed from 0.
    current_batch: integer, current batch in the current epoch, indexed from 0.
    batches_per_epoch: integer, number of steps in an epoch.
    batch_size: integer, total batch sized.

  Returns:
    Adjusted learning rate.
  """
  del current_batch, batches_per_epoch  # not used
  initial_learning_rate = common.BASE_LEARNING_RATE * batch_size / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if current_epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  return learning_rate


class LearningRateBatchScheduler(tf.keras.callbacks.Callback):
  """Callback to update learning rate on every batch (not epoch boundaries).

  N.B. Only support Keras optimizers, not TF optimizers.

  Attributes:
      schedule: a function that takes an epoch index and a batch index as input
          (both integer, indexed from 0) and returns a new learning rate as
          output (float).
  """

  def __init__(self, schedule, batch_size, steps_per_epoch):
    super(LearningRateBatchScheduler, self).__init__()
    self.schedule = schedule
    self.steps_per_epoch = steps_per_epoch
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
                       self.steps_per_epoch,
                       self.batch_size)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function should be float.')
    if lr != self.prev_lr:
      self.model.optimizer.learning_rate = lr  # lr should be a float here
      self.prev_lr = lr
      logging.debug(
          'Epoch %05d Batch %05d: LearningRateBatchScheduler '
          'change learning rate to %s.', self.epochs, batch, lr)


def run(flags_obj):
  """Run ResNet Cifar-10 training and eval loop using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.

  Returns:
    Dictionary of training and eval stats.
  """
  keras_utils.set_session_config(
      enable_xla=flags_obj.enable_xla)

  # Execute flag override logic for better model performance
  if flags_obj.tf_gpu_thread_mode:
    keras_utils.set_gpu_thread_mode_and_count(
        per_gpu_thread_count=flags_obj.per_gpu_thread_count,
        gpu_thread_mode=flags_obj.tf_gpu_thread_mode,
        num_gpus=flags_obj.num_gpus,
        datasets_num_private_threads=flags_obj.datasets_num_private_threads)
  common.set_cudnn_batchnorm_mode()

  dtype = flags_core.get_tf_dtype(flags_obj)
  if dtype == 'fp16':
    raise ValueError('dtype fp16 is not supported in Keras. Use the default '
                     'value(fp32).')

  data_format = flags_obj.data_format
  if data_format is None:
    data_format = ('channels_first' if tf.config.list_physical_devices('GPU')
                   else 'channels_last')
  tf.keras.backend.set_image_data_format(data_format)

  strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus,
      all_reduce_alg=flags_obj.all_reduce_alg,
      num_packs=flags_obj.num_packs)

  if strategy:
    # flags_obj.enable_get_next_as_optional controls whether enabling
    # get_next_as_optional behavior in DistributedIterator. If true, last
    # partial batch can be supported.
    strategy.extended.experimental_enable_get_next_as_optional = (
        flags_obj.enable_get_next_as_optional
    )

  strategy_scope = distribute_utils.get_strategy_scope(strategy)

  if flags_obj.use_synthetic_data:
    synthetic_util.set_up_synthetic_data()
    input_fn = common.get_synth_input_fn(
        height=cifar_preprocessing.HEIGHT,
        width=cifar_preprocessing.WIDTH,
        num_channels=cifar_preprocessing.NUM_CHANNELS,
        num_classes=cifar_preprocessing.NUM_CLASSES,
        dtype=flags_core.get_tf_dtype(flags_obj),
        drop_remainder=True)
  else:
    synthetic_util.undo_set_up_synthetic_data()
    input_fn = cifar_preprocessing.input_fn

  train_input_dataset = input_fn(
      is_training=True,
      data_dir=flags_obj.data_dir,
      batch_size=flags_obj.batch_size,
      parse_record_fn=cifar_preprocessing.parse_record,
      datasets_num_private_threads=flags_obj.datasets_num_private_threads,
      dtype=dtype,
      # Setting drop_remainder to avoid the partial batch logic in normalization
      # layer, which triggers tf.where and leads to extra memory copy of input
      # sizes between host and GPU.
      drop_remainder=(not flags_obj.enable_get_next_as_optional))

  eval_input_dataset = None
  if not flags_obj.skip_eval:
    eval_input_dataset = input_fn(
        is_training=False,
        data_dir=flags_obj.data_dir,
        batch_size=flags_obj.batch_size,
        parse_record_fn=cifar_preprocessing.parse_record)

  steps_per_epoch = (
      cifar_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size)
  lr_schedule = 0.1
  if flags_obj.use_tensor_lr:
    initial_learning_rate = common.BASE_LEARNING_RATE * flags_obj.batch_size / 128
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=list(p[1] * steps_per_epoch for p in LR_SCHEDULE),
        values=[initial_learning_rate] +
        list(p[0] * initial_learning_rate for p in LR_SCHEDULE))

  with strategy_scope:
    optimizer = common.get_optimizer(lr_schedule)
    model = resnet_cifar_model.resnet56(classes=cifar_preprocessing.NUM_CLASSES)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=(['sparse_categorical_accuracy']
                 if flags_obj.report_accuracy_metrics else None),
        run_eagerly=flags_obj.run_eagerly)

  train_epochs = flags_obj.train_epochs

  callbacks = common.get_callbacks()

  if not flags_obj.use_tensor_lr:
    lr_callback = LearningRateBatchScheduler(
        schedule=learning_rate_schedule,
        batch_size=flags_obj.batch_size,
        steps_per_epoch=steps_per_epoch)
    callbacks.append(lr_callback)

  # if mutliple epochs, ignore the train_steps flag.
  if train_epochs <= 1 and flags_obj.train_steps:
    steps_per_epoch = min(flags_obj.train_steps, steps_per_epoch)
    train_epochs = 1

  num_eval_steps = (cifar_preprocessing.NUM_IMAGES['validation'] //
                    flags_obj.batch_size)

  validation_data = eval_input_dataset
  if flags_obj.skip_eval:
    if flags_obj.set_learning_phase_to_train:
      # TODO(haoyuzhang): Understand slowdown of setting learning phase when
      # not using distribution strategy.
      tf.keras.backend.set_learning_phase(1)
    num_eval_steps = None
    validation_data = None

  if not strategy and flags_obj.explicit_gpu_placement:
    # TODO(b/135607227): Add device scope automatically in Keras training loop
    # when not using distribition strategy.
    no_dist_strat_device = tf.device('/device:GPU:0')
    no_dist_strat_device.__enter__()

  history = model.fit(train_input_dataset,
                      epochs=train_epochs,
                      steps_per_epoch=steps_per_epoch,
                      callbacks=callbacks,
                      validation_steps=num_eval_steps,
                      validation_data=validation_data,
                      validation_freq=flags_obj.epochs_between_evals,
                      verbose=2)
  eval_output = None
  if not flags_obj.skip_eval:
    eval_output = model.evaluate(eval_input_dataset,
                                 steps=num_eval_steps,
                                 verbose=2)

  if not strategy and flags_obj.explicit_gpu_placement:
    no_dist_strat_device.__exit__()

  stats = common.build_stats(history, eval_output, callbacks)
  return stats


def define_cifar_flags():
  common.define_keras_flags(dynamic_loss_scale=False)

  flags_core.set_defaults(data_dir='/tmp/cifar10_data/cifar-10-batches-bin',
                          model_dir='/tmp/cifar10_model',
                          epochs_between_evals=10,
                          batch_size=128)


def main(_):
  return run(flags.FLAGS)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  define_cifar_flags()
  app.run(main)
