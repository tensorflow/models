# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

import time

from absl import app as absl_app
from absl import flags
import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.resnet import cifar10_main as cifar_main
from official.resnet import resnet_run_loop
from official.resnet.keras import keras_resnet_model
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2

IMAGENET_DATASET = "imagenet"
CIFAR_DATASET = "cifar"


class TimeHistory(tf.keras.callbacks.Callback):
  """Callback for Keras models."""

  def __init__(self, batch_size):
    """Callback for Keras models.

    Args:
      batch_size: Total batch size.

    """
    self._batch_size = batch_size
    self.last_exp_per_sec = 0
    super(TimeHistory, self).__init__()

  def on_train_begin(self, logs=None):
    self.epoch_times_secs = []
    self.batch_times_secs = []
    self.record_batch = True

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch_time_start = time.time()

  def on_epoch_end(self, epoch, logs=None):
    self.epoch_times_secs.append(time.time() - self.epoch_time_start)

  def on_batch_begin(self, batch, logs=None):
    if self.record_batch:
      self.batch_time_start = time.time()
      self.record_batch = False

  def on_batch_end(self, batch, logs=None):
    n = 100
    if batch % n == 0:
      last_n_batches = time.time() - self.batch_time_start
      examples_per_second = (self._batch_size * n) / last_n_batches
      self.batch_times_secs.append(last_n_batches)
      self.last_exp_per_sec = examples_per_second
      self.record_batch = True
      # TODO(anjalisridhar): add timestamp as well.
      if batch != 0:
        tf.logging.info("BenchmarkMetric: {'num_batches':%d, 'time_taken': %f,"
                        "'images_per_second': %f}" %
                        (batch, last_n_batches, examples_per_second))


# LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
#     (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
# ]
LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
    (0.1, 91), (0.01, 136), (0.001, 182)
]


BASE_LEARNING_RATE = 0.1

def learning_rate_schedule(current_epoch, current_batch, batches_per_epoch, batch_size):
  """Handles linear scaling rule, gradual warmup, and LR decay.

  The learning rate starts at 0, then it increases linearly per step.
  After 5 epochs we reach the base learning rate (scaled to account
    for batch size).
  After 30, 60 and 80 epochs the learning rate is divided by 10.
  After 90 epochs training stops and the LR is set to 0. This ensures
    that we train for exactly 90 epochs for reproducibility.

  Args:
    current_epoch: integer, current epoch indexed from 0.
    current_batch: integer, current batch in the current epoch, indexed from 0.

  Returns:
    Adjusted learning rate.
  """
  # epoch = current_epoch + float(current_batch) / batches_per_epoch
  # warmup_lr_multiplier, warmup_end_epoch = LR_SCHEDULE[0]
  # if epoch < warmup_end_epoch:
  #   # Learning rate increases linearly per step.
  #   return BASE_LEARNING_RATE * warmup_lr_multiplier * epoch / warmup_end_epoch
  # for mult, start_epoch in LR_SCHEDULE:
  #   if epoch >= start_epoch:
  #     learning_rate = BASE_LEARNING_RATE * mult
  #   else:
  #     break
  # return learning_rate

  initial_learning_rate = BASE_LEARNING_RATE * batch_size / 128
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
    #if not hasattr(self.model.optimizer, 'learning_rate'):
    #  raise ValueError('Optimizer must have a "learning_rate" attribute.')
    self.epochs += 1

  def on_batch_begin(self, batch, logs=None):
    lr = self.schedule(self.epochs, batch, self.batches_per_epoch, self.batch_size)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function should be float.')
    if lr != self.prev_lr:
      tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
      self.prev_lr = lr
      tf.logging.debug('Epoch %05d Batch %05d: LearningRateBatchScheduler change '
                   'learning rate to %s.', self.epochs, batch, lr)


def parse_record_keras(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: Data type to use for input images.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  if shining.dataset == IMAGENET_DATASET:
    image_buffer, label, bbox = imagenet_main._parse_example_proto(raw_record)

    image = imagenet_preprocessing.preprocess_image(
        image_buffer=image_buffer,
        bbox=bbox,
        output_height=imagenet_main._DEFAULT_IMAGE_SIZE,
        output_width=imagenet_main._DEFAULT_IMAGE_SIZE,
        num_channels=imagenet_main._NUM_CHANNELS,
        is_training=is_training)
    image = tf.cast(image, dtype)
    label = tf.sparse_to_dense(label, (imagenet_main._NUM_CLASSES,), 1)

  elif shining.dataset == CIFAR_DATASET:
    image, label = cifar_main.parse_record(raw_record, is_training, dtype)
    label = tf.sparse_to_dense(label, (cifar_main._NUM_CLASSES,), 1)
  else:
    raise ValueError("Unknown dataset: {%s}".format(shining.dataset))

  return image, label

def run_imagenet_with_keras(flags_obj):
  """Run ResNet ImageNet training and eval loop using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.
  """
  if flags_obj.enable_eager:
    tf.enable_eager_execution()

  dtype = flags_core.get_tf_dtype(flags_obj)
  if dtype == 'fp16':
    raise ValueError('dtype fp16 is not supported in Keras. Use the default '
                     'value(fp32).')

  per_device_batch_size = distribution_utils.per_device_batch_size(
      flags_obj.batch_size, flags_core.get_num_gpus(flags_obj))

  train_input_dataset, eval_input_dataset = get_data(
      shining.dataset, flags_obj.use_synthetic_data)

  # Use Keras ResNet50 applications model and native keras APIs
  # initialize RMSprop optimizer
  # TODO(anjalisridhar): Move to using MomentumOptimizer.
  # opt = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
  # I am setting an initial LR of 0.001 since this will be reset
  # at the beginning of the training loop.
  opt = gradient_descent_v2.SGD(learning_rate=0.1, momentum=0.9)

  # TF Optimizer:
  # learning_rate = BASE_LEARNING_RATE * flags_obj.batch_size / 256
  # opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

  strategy = distribution_utils.get_distribution_strategy(
      num_gpus=flags_obj.num_gpus)

  if shining.dataset == IMAGENET_DATASET:
    model = resnet_model_tpu.ResNet50(num_classes=imagenet_main._NUM_CLASSES)
    steps_per_epoch = imagenet_main._NUM_IMAGES['train'] // flags_obj.batch_size

    lr_callback = LearningRateBatchScheduler(
      learning_rate_schedule,
      batch_size=flags_obj.batch_size,
      num_images=imagenet_main._NUM_IMAGES['train'])

    num_eval_steps = (imagenet_main._NUM_IMAGES['validation'] //
                    flags_obj.batch_size)
  elif shining.dataset = CIFAR_DATASET:
    model = keras_resnet_model.ResNet56(input_shape=(32, 32, 3),
                                        include_top=True,
                                        classes=cifar_main._NUM_CLASSES,
                                        weights=None)

    steps_per_epoch = cifar_main._NUM_IMAGES['train'] // flags_obj.batch_size

    lr_callback = LearningRateBatchScheduler(
        learning_rate_schedule,
        batch_size=flags_obj.batch_size,
        num_images=cifar_main._NUM_IMAGES['train'])

    num_eval_steps = (cifar_main._NUM_IMAGES['validation'] //
                      flags_obj.batch_size)
  else:
    raise ValueError("Unknown dataset: {%s}".format(shining.dataset))

  loss = 'categorical_crossentropy'
  accuracy = 'categorical_accuracy'

  if flags_obj.num_gpus == 1 and flags_obj.dist_strat_off:
    print('Not using distribution strategies.')
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])
  else:
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy],
                  distribute=strategy)

  time_callback = TimeHistory(flags_obj.batch_size)

  tesorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=flags_obj.model_dir)
    #  update_freq="batch")  # Add this if want per batch logging.


  print('Executing eagerly?:', tf.executing_eagerly())
  history = model.fit(train_input_dataset,
                      epochs=flags_obj.train_epochs,
                      steps_per_epoch=steps_per_epoch,
                      callbacks=[
                          time_callback,
                          lr_callback,
                          tesorboard_callback
                      ],
                      validation_steps=num_eval_steps,
                      validation_data=eval_input_dataset,
                      verbose=1)

  eval_output = model.evaluate(eval_input_dataset,
                               steps=num_eval_steps,
                               verbose=1)

  print('Test loss:', eval_output[0])

  stats = {}
  stats['accuracy_top_1'] = eval_output[1]
  stats['eval_loss'] = eval_output[0]
  stats['training_loss'] = history.history['loss'][-1]
  stats['training_accuracy_top_1'] = history.history['categorical_accuracy'][-1]

  print('top_1 accuracy:{}'.format(stats['accuracy_top_1']))
  print('top_1_training_accuracy:{}'.format(stats['training_accuracy_top_1']))

  return stats

def get_data(dataset, use_synthetic_data):

  if dataset == IMAGENET_DATASET:
    if use_synthetic_data:
      synth_input_fn = resnet_run_loop.get_synth_input_fn(
          imagenet_main._DEFAULT_IMAGE_SIZE, imagenet_main._DEFAULT_IMAGE_SIZE,
          imagenet_main._NUM_CHANNELS, imagenet_main._NUM_CLASSES,
          dtype=flags_core.get_tf_dtype(flags_obj))
      train_input_dataset = synth_input_fn(
          batch_size=per_device_batch_size,
          height=imagenet_main._DEFAULT_IMAGE_SIZE,
          width=imagenet_main._DEFAULT_IMAGE_SIZE,
          num_channels=imagenet_main._NUM_CHANNELS,
          num_classes=imagenet_main._NUM_CLASSES,
          dtype=dtype)
      eval_input_dataset = synth_input_fn(
          batch_size=per_device_batch_size,
          height=imagenet_main._DEFAULT_IMAGE_SIZE,
          width=imagenet_main._DEFAULT_IMAGE_SIZE,
          num_channels=imagenet_main._NUM_CHANNELS,
          num_classes=imagenet_main._NUM_CLASSES,
          dtype=dtype)
    # pylint: enable=protected-access

    else:
      train_input_dataset = imagenet_main.input_fn(
            True,
            flags_obj.data_dir,
            batch_size=per_device_batch_size,
            num_epochs=flags_obj.train_epochs,
            parse_record_fn=parse_record_keras)

      eval_input_dataset = imagenet_main.input_fn(
            False,
            flags_obj.data_dir,
            batch_size=per_device_batch_size,
            num_epochs=flags_obj.train_epochs,
            parse_record_fn=parse_record_keras)
  elif dataset == CIFAR_DATASET:
    if use_synthetic_data:
      if flags_obj.use_synthetic_data:
        synth_input_fn = resnet_run_loop.get_synth_input_fn(
            cifar_main._HEIGHT, cifar_main._WIDTH,
            cifar_main._NUM_CHANNELS, cifar_main._NUM_CLASSES,
            dtype=flags_core.get_tf_dtype(flags_obj))
        train_input_dataset = synth_input_fn(
            True,
            flags_obj.data_dir,
            batch_size=per_device_batch_size,
            height=cifar_main._HEIGHT,
            width=cifar_main._WIDTH,
            num_channels=cifar_main._NUM_CHANNELS,
            num_classes=cifar_main._NUM_CLASSES,
            dtype=dtype)
        eval_input_dataset = synth_input_fn(
            False,
            flags_obj.data_dir,
            batch_size=per_device_batch_size,
            height=cifar_main._HEIGHT,
            width=cifar_main._WIDTH,
            num_channels=cifar_main._NUM_CHANNELS,
            num_classes=cifar_main._NUM_CLASSES,
            dtype=dtype)
      # pylint: enable=protected-access

      else:
        train_input_dataset = cifar_main.input_fn(
            True,
            flags_obj.data_dir,
            batch_size=per_device_batch_size,
            num_epochs=flags_obj.train_epochs,
            parse_record_fn=parse_record_keras)

        eval_input_dataset = cifar_main.input_fn(
            False,
            flags_obj.data_dir,
            batch_size=per_device_batch_size,
            num_epochs=flags_obj.train_epochs,
            parse_record_fn=parse_record_keras)

  return train_input_dataset, eval_input_dataset


def define_keras_flags():
  flags.DEFINE_boolean(name='enable_eager', default=False, help='Enable eager?')
  flags.DEFINE_string(name='dataset', default=IMAGENET_DATASET,
      help='Which dataset, ImageNet or Cifar?')


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_imagenet_with_keras(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_keras_flags()

  if shining.dataset == IMAGENET_DATASET:
    imagenet_main.define_imagenet_flags()
  elif shining.dataset == CIFAR_DATASET:
    cifar_main.define_cifar_flags()

  absl_app.run(main)
