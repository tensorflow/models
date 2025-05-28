# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Runs a simple model on the MNIST dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf, tf_keras
import tensorflow_datasets as tfds
from official.common import distribute_utils
from official.legacy.image_classification.resnet import common
from official.utils.flags import core as flags_core
from official.utils.misc import model_helpers

FLAGS = flags.FLAGS


def build_model():
  """Constructs the ML model used to predict handwritten digits."""

  image = tf_keras.layers.Input(shape=(28, 28, 1))

  y = tf_keras.layers.Conv2D(filters=32,
                             kernel_size=5,
                             padding='same',
                             activation='relu')(image)
  y = tf_keras.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2),
                                   padding='same')(y)
  y = tf_keras.layers.Conv2D(filters=32,
                             kernel_size=5,
                             padding='same',
                             activation='relu')(y)
  y = tf_keras.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2),
                                   padding='same')(y)
  y = tf_keras.layers.Flatten()(y)
  y = tf_keras.layers.Dense(1024, activation='relu')(y)
  y = tf_keras.layers.Dropout(0.4)(y)

  probs = tf_keras.layers.Dense(10, activation='softmax')(y)

  model = tf_keras.models.Model(image, probs, name='mnist')

  return model


@tfds.decode.make_decoder(output_dtype=tf.float32)
def decode_image(example, feature):
  """Convert image to float32 and normalize from [0, 255] to [0.0, 1.0]."""
  return tf.cast(feature.decode_example(example), dtype=tf.float32) / 255


def run(flags_obj, datasets_override=None, strategy_override=None):
  """Run MNIST model training and eval loop using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.
    datasets_override: A pair of `tf.data.Dataset` objects to train the model,
                       representing the train and test sets.
    strategy_override: A `tf.distribute.Strategy` object to use for model.

  Returns:
    Dictionary of training and eval stats.
  """
  # Start TF profiler server.
  tf.profiler.experimental.server.start(flags_obj.profiler_port)

  strategy = strategy_override or distribute_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus,
      tpu_address=flags_obj.tpu)

  strategy_scope = distribute_utils.get_strategy_scope(strategy)

  mnist = tfds.builder('mnist', data_dir=flags_obj.data_dir)
  if flags_obj.download:
    mnist.download_and_prepare()

  mnist_train, mnist_test = datasets_override or mnist.as_dataset(
      split=['train', 'test'],
      decoders={'image': decode_image()},  # pylint: disable=no-value-for-parameter
      as_supervised=True)
  train_input_dataset = mnist_train.cache().repeat().shuffle(
      buffer_size=50000).batch(flags_obj.batch_size)
  eval_input_dataset = mnist_test.cache().repeat().batch(flags_obj.batch_size)

  with strategy_scope:
    lr_schedule = tf_keras.optimizers.schedules.ExponentialDecay(
        0.05, decay_steps=100000, decay_rate=0.96)
    optimizer = tf_keras.optimizers.SGD(learning_rate=lr_schedule)

    model = build_model()
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])

  num_train_examples = mnist.info.splits['train'].num_examples
  train_steps = num_train_examples // flags_obj.batch_size
  train_epochs = flags_obj.train_epochs

  ckpt_full_path = os.path.join(flags_obj.model_dir, 'model.ckpt-{epoch:04d}')
  callbacks = [
      tf_keras.callbacks.ModelCheckpoint(
          ckpt_full_path, save_weights_only=True),
      tf_keras.callbacks.TensorBoard(log_dir=flags_obj.model_dir),
  ]

  num_eval_examples = mnist.info.splits['test'].num_examples
  num_eval_steps = num_eval_examples // flags_obj.batch_size

  history = model.fit(
      train_input_dataset,
      epochs=train_epochs,
      steps_per_epoch=train_steps,
      callbacks=callbacks,
      validation_steps=num_eval_steps,
      validation_data=eval_input_dataset,
      validation_freq=flags_obj.epochs_between_evals)

  export_path = os.path.join(flags_obj.model_dir, 'saved_model')
  model.save(export_path, include_optimizer=False)

  eval_output = model.evaluate(
      eval_input_dataset, steps=num_eval_steps, verbose=2)

  stats = common.build_stats(history, eval_output, callbacks)
  return stats


def define_mnist_flags():
  """Define command line flags for MNIST model."""
  flags_core.define_base(
      clean=True,
      num_gpu=True,
      train_epochs=True,
      epochs_between_evals=True,
      distribution_strategy=True)
  flags_core.define_device()
  flags_core.define_distribution()
  flags.DEFINE_bool('download', True,
                    'Whether to download data to `--data_dir`.')
  flags.DEFINE_integer('profiler_port', 9012,
                       'Port to start profiler server on.')
  FLAGS.set_default('batch_size', 1024)


def main(_):
  model_helpers.apply_clean(FLAGS)
  stats = run(flags.FLAGS)
  logging.info('Run stats:\n%s', stats)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  define_mnist_flags()
  app.run(main)
