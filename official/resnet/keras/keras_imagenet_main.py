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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.resnet import imagenet_main
from official.resnet.keras import keras_common
from official.resnet.keras import resnet_model
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils


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


def run(flags_obj):
  """Run ResNet ImageNet training and eval loop using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.
  """
  if flags_obj.enable_eager:
    tf.compat.v1.enable_eager_execution()

  dtype = flags_core.get_tf_dtype(flags_obj)
  if dtype == 'fp16':
    raise ValueError('dtype fp16 is not supported in Keras. Use the default '
                     'value(fp32).')

  data_format = flags_obj.data_format
  if data_format is None:
    data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')
  tf.keras.backend.set_image_data_format(data_format)

  # pylint: disable=protected-access
  if flags_obj.use_synthetic_data:
    distribution_utils.set_up_synthetic_data()
    input_fn = keras_common.get_synth_input_fn(
        height=imagenet_main.DEFAULT_IMAGE_SIZE,
        width=imagenet_main.DEFAULT_IMAGE_SIZE,
        num_channels=imagenet_main.NUM_CHANNELS,
        num_classes=imagenet_main.NUM_CLASSES,
        dtype=flags_core.get_tf_dtype(flags_obj))
  else:
    distribution_utils.undo_set_up_synthetic_data()
    input_fn = imagenet_main.input_fn

  train_input_dataset = input_fn(is_training=True,
                                 data_dir=flags_obj.data_dir,
                                 batch_size=flags_obj.batch_size,
                                 num_epochs=flags_obj.train_epochs,
                                 parse_record_fn=parse_record_keras)

  eval_input_dataset = input_fn(is_training=False,
                                data_dir=flags_obj.data_dir,
                                batch_size=flags_obj.batch_size,
                                num_epochs=flags_obj.train_epochs,
                                parse_record_fn=parse_record_keras)

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus)

  strategy_scope = keras_common.get_strategy_scope(strategy)

  with strategy_scope:
    optimizer = keras_common.get_optimizer()
    model = resnet_model.resnet50(num_classes=imagenet_main.NUM_CLASSES)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['sparse_categorical_accuracy'])

  time_callback, tensorboard_callback, lr_callback = keras_common.get_callbacks(
      learning_rate_schedule, imagenet_main.NUM_IMAGES['train'])

  train_steps = imagenet_main.NUM_IMAGES['train'] // flags_obj.batch_size
  train_epochs = flags_obj.train_epochs

  if flags_obj.train_steps:
    train_steps = min(flags_obj.train_steps, train_steps)
    train_epochs = 1

  num_eval_steps = (imagenet_main.NUM_IMAGES['validation'] //
                    flags_obj.batch_size)

  validation_data = eval_input_dataset
  if flags_obj.skip_eval:
    # Only build the training graph. This reduces memory usage introduced by
    # control flow ops in layers that have different implementations for
    # training and inference (e.g., batch norm).
    tf.keras.backend.set_learning_phase(1)
    num_eval_steps = None
    validation_data = None

  history = model.fit(train_input_dataset,
                      epochs=train_epochs,
                      steps_per_epoch=train_steps,
                      callbacks=[
                          time_callback,
                          lr_callback,
                          tensorboard_callback
                      ],
                      validation_steps=num_eval_steps,
                      validation_data=validation_data,
                      validation_freq=flags_obj.epochs_between_evals,
                      verbose=2)

  eval_output = None
  if not flags_obj.skip_eval:
    eval_output = model.evaluate(eval_input_dataset,
                                 steps=num_eval_steps,
                                 verbose=2)
  stats = keras_common.build_stats(history, eval_output, time_callback)
  return stats


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    return run(flags.FLAGS)


if __name__ == '__main__':
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  imagenet_main.define_imagenet_flags()
  keras_common.define_keras_flags()
  absl_app.run(main)
