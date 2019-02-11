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
"""Retrain the keras built-in ResNet50 on CIFAR-10."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-bad-import-order
import os
import numpy as np
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.keras_application_models import dataset
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils


class FakeScope():
  def __enter__(self):
    pass
  def __exit__(self, ext_type, ext_val, tb):
    pass


class NotReallyADistributionStrategy():

  def scope(self):
    return FakeScope()


def train_resnet50(_):
  """Train Resnet50 on CIFAR from the scratch."""
  # Ensure a valid model name was supplied via command line argument

  # Check if eager execution is enabled
  if FLAGS.eager:
    tf.logging.info("Eager execution is enabled...")
    tf.enable_eager_execution()

  strategy = None
  if FLAGS.dist_strat:
    strategy = distribution_utils.get_distribution_strategy(
        num_gpus=FLAGS.num_gpus)
  else:
    strategy = NotReallyADistributionStrategy()

  with strategy.scope():

    tf.logging.info("Using CIFAR-10 dataset...")
    dataset_name = "CIFAR-10"
    ds = dataset.Cifar10Dataset(FLAGS.batch_size)
    x_train, y_train = ds.x_train, ds.y_train
    x_test, y_test = ds.x_test, ds.y_test
    # Subtract pixel mean from data
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    # Apply data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True)

    base_model = tf.keras.applications.ResNet50(
        # Use imagenet pretrained weights require input_shape and pooling.
        weights='imagenet',
        input_shape=ds.input_shape,
        pooling='avg',
        # When include_top is False, we need manually add FC layers.
        include_top=False)
    # Manually add FC layer
    x = base_model.output
    x = tf.keras.layers.Dense(10, activation='softmax', name='fc10')(x)
    model = tf.keras.Model(inputs=base_model.inputs, outputs=x)

    # Adam optimizer and some other optimizers doesn't work well with
    # distribution strategy (b/113076709)
    # Use keras.SGD (SGD + Momentum) according to ResNet paper.
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=tf.keras.backend.variable(1e-3), momentum=0.9)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda x: 1e-3 if x <= 10 else 1e-4 if x <= 30 else 1e-5,
        verbose=1)

    # Add L1L2 regularization to avoid overfitting
    l1 = 0.
    l2 = 0.0001
    for layer in model.layers:
      if hasattr(layer, "kernel_regularizer"):
        layer.kerner_regularizer = tf.keras.regularizers.l1_l2(l1, l2)

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cifar10_resnet50_model.{epoch:03d}.h5'
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)

    callbacks = [lr_scheduler, checkpoint]
    datagen.fit(x_train)

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])

    # Train and evaluate the model
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=FLAGS.batch_size),
        epochs=FLAGS.train_epochs,
        callbacks=callbacks,
        validation_data=(x_test, y_test),
        steps_per_epoch=int(np.ceil(FLAGS.num_train_images / FLAGS.batch_size)),
    )

  # Clear the session explicitly to avoid session delete error
  tf.keras.backend.clear_session()


def define_keras_benchmark_flags():
  """Add flags for keras built-in application models."""
  flags_core.define_base(hooks=False)
  flags_core.define_performance()
  flags_core.define_image()
  flags_core.define_benchmark()
  flags.adopt_module_key_flags(flags_core)

  flags_core.set_defaults(
      data_format="channels_last",
      batch_size=32,
      train_epochs=2)

  flags.DEFINE_integer(
      name="num_train_images", default=50000,
      help=flags_core.help_wrap(
          "The number of images for training. The default value is 50000."))

  flags.DEFINE_integer(
      name="num_eval_images", default=10000,
      help=flags_core.help_wrap(
          "The number of images for evaluation. The default value is 10000."))

  flags.DEFINE_boolean(
      name="eager", default=False, help=flags_core.help_wrap(
          "To enable eager execution. Note that if eager execution is enabled, "
          "only one GPU is utilized even if multiple GPUs are provided and "
          "multi_gpu_model is used."))

  flags.DEFINE_boolean(
      name="dist_strat", default=False, help=flags_core.help_wrap(
          "To enable distribution strategy for model training and evaluation. "
          "Number of GPUs used for distribution strategy can be set by the "
          "argument --num_gpus."))

  flags.DEFINE_list(
      name="callbacks",
      default=[],
      help=flags_core.help_wrap(
          "A list of (case insensitive) strings to specify the names of "
          "callbacks. For example: `--callbacks ExamplesPerSecondCallback,"
          "LoggingMetricCallback`"))

  # pylint: disable=unused-variable
  def _check_eager_dist_strat(flag_dict):
    return not(flag_dict["eager"] and flag_dict["dist_strat"])


def main(_):
  with logger.benchmark_context(FLAGS):
    train_resnet50(FLAGS)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  define_keras_benchmark_flags()
  FLAGS = flags.FLAGS
  absl_app.run(main)
