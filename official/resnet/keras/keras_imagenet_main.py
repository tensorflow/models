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

from official.resnet import imagenet_main
from official.resnet import imagenet_preprocessing
from official.resnet import resnet_run_loop
from official.resnet.keras import keras_common
from official.resnet.keras import keras_resnet_model
from official.resnet.keras import resnet_model_tpu
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2


LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]
BASE_LEARNING_RATE = 0.1  # This matches Jing's version.

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
  initial_learning_rate = BASE_LEARNING_RATE * batch_size / 256
  epoch = current_epoch + float(current_batch) / batches_per_epoch
  warmup_lr_multiplier, warmup_end_epoch = LR_SCHEDULE[0]
  if epoch < warmup_end_epoch:
    # Learning rate increases linearly per step.
    return initial_learning_rate * warmup_lr_multiplier * epoch / warmup_end_epoch
  for mult, start_epoch in LR_SCHEDULE:
    if epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  return learning_rate


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

  # pylint: disable=protected-access
  if flags_obj.use_synthetic_data:
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


  opt, loss, accuracy = keras_common.get_optimizer_loss_and_metrics()
  strategy = keras_common.get_dist_strategy()

  model = resnet_model_tpu.ResNet50(num_classes=imagenet_main._NUM_CLASSES)

  model.compile(loss=loss,
                optimizer=opt,
                metrics=[accuracy],
                distribute=strategy)

  time_callback, tensorboard_callback, lr_callback = keras_common.get_fit_callbacks(
      learning_rate_schedule)

  steps_per_epoch = imagenet_main._NUM_IMAGES['train'] // flags_obj.batch_size
  num_eval_steps = (imagenet_main._NUM_IMAGES['validation'] //
                  flags_obj.batch_size)

  model.fit(train_input_dataset,
            epochs=flags_obj.train_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[
              time_callback,
              lr_callback,
              tensorboard_callback
            ],
            validation_steps=num_eval_steps,
            validation_data=eval_input_dataset,
            verbose=1)

  eval_output = model.evaluate(eval_input_dataset,
                               steps=num_eval_steps,
                               verbose=1)
  print('Test loss:', eval_output[0])
  stats = keras_common.analyze_eval_result(eval_output)

  return stats

def define_keras_imagenet_flags():
  flags.DEFINE_boolean(name='enable_eager', default=False, help='Enable eager?')


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_imagenet_with_keras(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_keras_imagenet_flags()
  imagenet_main.define_imagenet_flags()
  absl_app.run(main)
