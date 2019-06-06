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
"""Runs a ResNet model on the ImageNet dataset using custom training loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import datetime
import time

from absl import app as absl_app
from absl import flags
# from absl import logging

# TODO(anj-s): Identify why this import does not work
import tensorflow.compat.v2 as tf  # pylint: disable=g-bad-import-order
import tensorflow.compat.v1.logging as logging
# import tensorflow as tf
import numpy as np

from official.resnet import imagenet_main
from official.resnet.keras import keras_common
from official.resnet.keras import resnet_model
from official.resnet.keras import trivial_model
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers
from official.resnet.ctl import ctl_common


# Imagenet training and test data sets.
APPROX_IMAGENET_TRAINING_IMAGES = 1280000  # Approximate number of images.
IMAGENET_VALIDATION_IMAGES = 50000  # Number of images.

LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]


def learning_rate_schedule(current_epoch,
                           current_batch,
                           batches_per_epoch,
                           batch_size):
  """Handles linear scaling rule, gradual warmup, and LR decay.

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
  initial_lr = keras_common.BASE_LEARNING_RATE * batch_size / 256
  epoch = current_epoch + float(current_batch) / batches_per_epoch
  warmup_lr_multiplier, warmup_end_epoch = LR_SCHEDULE[0]
  if epoch < warmup_end_epoch:
    # Learning rate increases linearly per step.
    return initial_lr * warmup_lr_multiplier * epoch / warmup_end_epoch
  for mult, start_epoch in LR_SCHEDULE:
    if epoch >= start_epoch:
      learning_rate = initial_lr * mult
    else:
      break
  return learning_rate


def parse_record_keras(raw_record, is_training, dtype):
  """Adjust the shape of label."""
  image, label = imagenet_main.parse_record(raw_record, is_training, dtype)

  # Subtract one so that labels are in [0, 1000), and cast to float32 for
  # Keras model.
  label = tf.cast(tf.cast(tf.reshape(label, shape=[1]), dtype=tf.int32) - 1,
                  dtype=tf.float32)
  return image, label

# Learning rate schedule
_LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]
_BASE_LEARNING_RATE = 0.4


# TODO(anj-s): This is different than the learning rate schedule
# used in Keras+DS.
def compute_learning_rate(lr_epoch):
  """Learning rate for each step."""
  warmup_lr_multiplier, warmup_end_epoch = _LR_SCHEDULE[0]
  if lr_epoch < warmup_end_epoch:
    # Learning rate increases linearly per step.
    return (_BASE_LEARNING_RATE * warmup_lr_multiplier *
            lr_epoch / warmup_end_epoch)
  for mult, start_epoch in _LR_SCHEDULE:
    if lr_epoch >= start_epoch:
      learning_rate = _BASE_LEARNING_RATE * mult
    else:
      break
  return learning_rate


def run(flags_obj):
  """Run ResNet ImageNet training and eval loop using custom training loops.

  Args:
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.

  Returns:
    Dictionary of training and eval stats.
  """
  dtype = flags_core.get_tf_dtype(flags_obj)

  # TODO(anj-s): Set data_format without using Keras.
  data_format = flags_obj.data_format
  if data_format is None:
    data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')
  tf.keras.backend.set_image_data_format(data_format)

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus,
      num_workers=distribution_utils.configure_cluster(),
      all_reduce_alg=flags_obj.all_reduce_alg,
      num_packs=flags_obj.num_packs)

  strategy_scope = distribution_utils.get_strategy_scope(strategy)

  # pylint: disable=protected-access
  if flags_obj.use_synthetic_data:
    distribution_utils.set_up_synthetic_data()
    input_fn = keras_common.get_synth_input_fn(
        height=imagenet_main.DEFAULT_IMAGE_SIZE,
        width=imagenet_main.DEFAULT_IMAGE_SIZE,
        num_channels=imagenet_main.NUM_CHANNELS,
        num_classes=imagenet_main.NUM_CLASSES,
        dtype=dtype,
        drop_remainder=True)
  else:
    distribution_utils.undo_set_up_synthetic_data()
    input_fn = imagenet_main.input_fn

  # When `enable_xla` is True, we always drop the remainder of the batches
  # in the dataset, as XLA-GPU doesn't support dynamic shapes.
  drop_remainder = flags_obj.enable_xla

  train_ds = input_fn(
      is_training=True,
      data_dir=flags_obj.data_dir,
      batch_size=flags_obj.batch_size,
      num_epochs=flags_obj.train_epochs,
      parse_record_fn=parse_record_keras,
      datasets_num_private_threads=flags_obj.datasets_num_private_threads,
      dtype=dtype,
      drop_remainder=drop_remainder)

  test_ds = None
  if not flags_obj.skip_eval:
    test_ds = input_fn(
        is_training=False,
        data_dir=flags_obj.data_dir,
        batch_size=flags_obj.batch_size,
        num_epochs=flags_obj.train_epochs,
        parse_record_fn=parse_record_keras,
        dtype=dtype,
        drop_remainder=drop_remainder)
    
    test_ds = strategy.experimental_distribute_dataset(test_ds)
    steps_per_eval = IMAGENET_VALIDATION_IMAGES // flags_obj.batch_size

  train_ds = strategy.experimental_distribute_dataset(train_ds)
  steps_per_epoch = APPROX_IMAGENET_TRAINING_IMAGES // flags_obj.batch_size
  train_epochs = flags_obj.train_epochs

  if flags_obj.train_steps:
    train_steps = min(flags_obj.train_steps, steps_per_epoch)
    train_epochs = 1

  strategy_scope = distribution_utils.get_strategy_scope(strategy)
  with strategy_scope():
    logging.info('Building Keras ResNet-50 model')
    model = resnet_model.resnet50(num_classes=imagenet_main.NUM_CLASSES,
                                  dtype=dtype, batch_size=flags_obj.batch_size)

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=_BASE_LEARNING_RATE, momentum=0.9, nesterov=True)

    training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'training_accuracy', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'test_accuracy', dtype=tf.float32)
    logging.info('Finished building Keras ResNet-50 model')

    def train_step(train_ds_inputs):
      """Training StepFn."""
      def step_fn(inputs):
        """Per-Replica StepFn."""
        images, labels = inputs
        with tf.GradientTape() as tape:
          logits = model(images, training=True)

          prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
              labels, logits)
          loss1 = tf.reduce_sum(prediction_loss) * (1.0/ flags_obj.batch_size)
          loss2 = tf.reduce_sum(model.losses) / strategy.num_replicas_in_sync
          loss = loss1 + loss2

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        training_accuracy.update_state(labels, logits)
        return loss

      per_replica_losses = strategy.experimental_run_v2(
          step_fn, args=(train_ds_inputs,))
      return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                             axis=None)

    def test_step(test_ds_inputs):
      """Evaluation StepFn."""
      def step_fn(inputs):
        images, labels = inputs
        logits = model(images, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                               logits)
        loss = tf.reduce_sum(loss) * (1.0/ flags_obj.batch_size)
        test_loss.update_state(loss)
        test_accuracy.update_state(labels, logits)

      strategy.experimental_run_v2(step_fn, args=(test_ds_inputs,))

    if flags_obj.enable_function:
      train_step = tf.function(train_step)
      test_step = tf.function(test_step)

    epoch_exp_per_sec = []
    stats = {}
    for epoch in range(train_epochs):
      logging.info('Starting to run epoch: %s', epoch)
      train_iterator = iter(train_ds)

      step = 0
      total_loss = 0.0
      batch_exp_per_sec = []
      for step in range(train_steps):
        start_time = time.time()
        learning_rate = compute_learning_rate(
            epoch + 1 + (float(step) / steps_per_epoch))
        optimizer.lr = learning_rate
        start_time = time.time()
        total_loss += train_step(next(train_iterator))
        end_time = time.time()
        elapsed_time = end_time - start_time
        samples_per_sec = flags_obj.batch_size / elapsed_time
        # We skip the first step for warmup purposes. We can
        # add a flag for tuning this.
        if step > 0:
          batch_exp_per_sec.append(samples_per_sec)

        step += 1
      train_loss = total_loss / step
      # calculate average examples per second for a given epoch
      epoch_exp_per_sec.append(np.mean(batch_exp_per_sec))
      logging.info('Learning rate at epoch %s is %s',
                   epoch, optimizer.lr.numpy())
      logging.info('Training loss: %s, accuracy: %s%%',
                   train_loss.numpy(),
                   training_accuracy.result().numpy())
      logging.info(
          "Training Metric: {'epoch':%d, 'examples_per_second': %f}" %
          (epoch, epoch_exp_per_sec[epoch]))

      # Store the last train loss and accuracy calculated
      stats['train_loss'] = train_loss.numpy()
      stats['train_acc'] = training_accuracy.result().numpy()
      training_accuracy.reset_states()

      if (not flags_obj.skip_eval and
          epoch % flags_obj.epochs_between_eval == 0):
        test_iterator = iter(test_ds)
        for step in range(steps_per_eval):
          test_step(next(test_iterator))
        logging.info('Test loss: %s, accuracy: %s%%',
                     test_loss.result().numpy(),
                     test_accuracy.result().numpy())
        stats['accuracy_top_1'] = test_accuracy.result().numpy()
        stats['eval_loss'] = test_loss.result().numpy()
        logging.info(
            "Testing Metric: {'epoch':%d, 'test accuracy': %f}" %
            (epoch, test_accuracy.result().numpy()))
        test_loss.reset_states()
        test_accuracy.reset_states()

  stats['exp_per_second'] = np.mean(epoch_exp_per_sec)
  return stats


def main(_):
  model_helpers.apply_clean(flags.FLAGS)
  with logger.benchmark_context(flags.FLAGS):
    return run(flags.FLAGS)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.enable_v2_behavior()
  imagenet_main.define_imagenet_flags()
  # TODO(anj-s): Do we need this?
  ctl_common.define_ctl_flags()
  absl_app.run(main)
