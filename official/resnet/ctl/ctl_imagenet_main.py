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
from absl import logging

import tensorflow as tf
import numpy as np

from official.resnet import imagenet_main
from official.resnet.keras import keras_common
from official.resnet.keras import resnet_model
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers
from official.resnet.ctl import ctl_common
from official.utils.misc import keras_utils


def parse_record_keras(raw_record, is_training, dtype):
  """Adjust the shape of label."""
  image, label = imagenet_main.parse_record(raw_record, is_training, dtype)

  # Subtract one so that labels are in [0, 1000), and cast to float32 for
  # Keras model.
  label = tf.cast(tf.cast(tf.reshape(label, shape=[1]), dtype=tf.int32) - 1,
                  dtype=tf.float32)
  return image, label


def build_stats(loss, eval_result, time_callback, warmup=1):
  """Normalizes and returns dictionary of stats.
  Args:
    loss: The final loss at training time.
    eval_result: Output of the eval step. Assumes first value is eval_loss and
      second value is accuracy_top_1.
    time_callback: Time tracking callback likely used during keras.fit.
    warmup: Number of initial steps to skip when calculating steps/s.
  Returns:
    Dictionary of normalized results.
  """
  stats = {}
  if loss:
    stats["loss"] = loss

  if eval_result:
    stats["eval_loss"] = eval_result[0]
    stats["eval_acc"] = eval_result[1]

  if time_callback:
    timestamp_log = time_callback.timestamp_log
    stats["step_timestamp_log"] = timestamp_log
    stats["train_finish_time"] = time_callback.train_finish_time
    if len(timestamp_log) > warmup + 1:
      stats["avg_exp_per_second"] = (
          time_callback.batch_size * time_callback.log_steps *
          (len(time_callback.timestamp_log)- warmup - 1) /
          (timestamp_log[-1].timestamp - timestamp_log[warmup].timestamp))

  return stats


def get_input_dataset(flags_obj, strategy):
  dtype = flags_core.get_tf_dtype(flags_obj)
  if flags_obj.use_synthetic_data:
    input_fn = keras_common.get_synth_input_fn(
        height=imagenet_main.DEFAULT_IMAGE_SIZE,
        width=imagenet_main.DEFAULT_IMAGE_SIZE,
        num_channels=imagenet_main.NUM_CHANNELS,
        num_classes=imagenet_main.NUM_CLASSES,
        dtype=dtype,
        drop_remainder=True)
  else:
    input_fn = imagenet_main.input_fn

  train_ds = input_fn(
      is_training=True,
      data_dir=flags_obj.data_dir,
      batch_size=flags_obj.batch_size,
      parse_record_fn=parse_record_keras,
      datasets_num_private_threads=flags_obj.datasets_num_private_threads,
      dtype=dtype)

  if strategy:
    train_ds = strategy.experimental_distribute_dataset(train_ds)
  
  test_ds = None
  if not flags_obj.skip_eval:
    test_ds = input_fn(
        is_training=False,
        data_dir=flags_obj.data_dir,
        batch_size=flags_obj.batch_size,
        parse_record_fn=parse_record_keras,
        dtype=dtype)

    if strategy:
      test_ds = strategy.experimental_distribute_dataset(test_ds)

  return train_ds, test_ds


def get_num_train_iterations(flags_obj):
  steps_per_epoch = imagenet_main.NUM_IMAGES['train'] // flags_obj.batch_size
  train_epochs = flags_obj.train_epochs

  if flags_obj.train_steps:
    train_steps = min(flags_obj.train_steps, steps_per_epoch)
    train_epochs = 1

  steps_per_eval = imagenet_main.NUM_IMAGES['validation'] // flags_obj.batch_size
  
  return train_steps, train_epochs, steps_per_eval

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

  train_ds, test_ds = get_input_dataset(flags_obj, strategy)
  train_steps, train_epochs, steps_per_eval = get_num_train_iterations(flags_obj)

  time_callback = keras_utils.TimeHistory(flags_obj.batch_size, 1)

  strategy_scope = distribution_utils.get_strategy_scope(strategy)
  with strategy_scope:
    model = resnet_model.resnet50(num_classes=imagenet_main.NUM_CLASSES,
                                  dtype=dtype, batch_size=flags_obj.batch_size)

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=keras_common.BASE_LEARNING_RATE, momentum=0.9, nesterov=True)

    training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'training_accuracy', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'test_accuracy', dtype=tf.float32)
    
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
          loss2 = (tf.reduce_sum(model.losses) /
                   tf.distribute.get_strategy().num_replicas_in_sync)
          loss = loss1 + loss2

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        training_accuracy.update_state(labels, logits)
        return loss

      if strategy:
        per_replica_losses = strategy.experimental_run_v2(
            step_fn, args=(train_ds_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)
      else:
        return step_fn(train_ds_inputs)

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

      if strategy:
        strategy.experimental_run_v2(step_fn, args=(test_ds_inputs,))
      else:
        step_fn(test_ds_inputs)

    if flags_obj.use_tf_function:
      train_step = tf.function(train_step)
      test_step = tf.function(test_step)

    time_callback.on_train_begin()
    for epoch in range(train_epochs):
      train_iterator = iter(train_ds)

      step = 0
      total_loss = 0.0
      for step in range(train_steps):
        training_accuracy.reset_states()
        optimizer.lr = keras_common.learning_rate_schedule(
            epoch, step, train_steps, flags_obj.batch_size)

        time_callback.on_batch_begin(step+epoch*train_steps)
        total_loss += train_step(next(train_iterator))
        time_callback.on_batch_end(step+epoch*train_steps)
        step += 1
      train_loss = total_loss / step

      if (not flags_obj.skip_eval and epoch != 0 and
          epoch % flags_obj.epochs_between_eval == 0):
        test_loss.reset_states()
        test_accuracy.reset_states()

        test_iterator = iter(test_ds)
        for step in range(steps_per_eval):
          test_step(next(test_iterator))
    
    time_callback.on_train_end()
    eval_result = None
    if not flags_obj.skip_eval:
      eval_result = [test_loss.result().numpy(), 
                     test_accuracy.result().numpy()]

    stats = build_stats(train_loss.numpy(), eval_result, time_callback)
    return stats


def main(_):
  model_helpers.apply_clean(flags.FLAGS)
  with logger.benchmark_context(flags.FLAGS):
    return run(flags.FLAGS)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.enable_v2_behavior()
  imagenet_main.define_imagenet_flags()
  ctl_common.define_ctl_flags()
  absl_app.run(main)
