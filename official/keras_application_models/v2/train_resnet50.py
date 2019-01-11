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
import numpy as np
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.keras_application_models import dataset
from official.keras_application_models import model_callbacks
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils

# Define a dictionary that maps model names to their model classes inside Keras
MODEL = tf.keras.applications.ResNet50


def train_resnet50(_):
  """Train Resnet50 on CIFAR from the scratch."""
  # Ensure a valid model name was supplied via command line argument

  # Check if eager execution is enabled
  if FLAGS.eager:
    tf.logging.info("Eager execution is enabled...")
    tf.enable_eager_execution()

  # Load the model
  model = "resnet50"
  keras_model = MODEL

  tf.logging.info("Using CIFAR-10 dataset...")
  dataset_name = "CIFAR-10"
  ds = dataset.Cifar10Dataset(FLAGS.batch_size)
  x_train, y_train, x_test, y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
  # Apply data augmentation
  datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

  model = keras_model(
      weights=None, input_shape=ds.input_shape, classes=ds.num_classes)

  num_gpus = flags_core.get_num_gpus(FLAGS)

  distribution = None
  # Use distribution strategy
  if FLAGS.dist_strat:
    distribution = distribution_utils.get_distribution_strategy(
        num_gpus=num_gpus)
  elif num_gpus > 1:
    # Run with multi_gpu_model
    # If eager execution is enabled, only one GPU is utilized even if multiple
    # GPUs are provided.
    if FLAGS.eager:
      tf.logging.warning(
          "{} GPUs are provided, but only one GPU is utilized as "
          "eager execution is enabled.".format(num_gpus))
    model = tf.keras.utils.multi_gpu_model(model, gpus=num_gpus)

  # Adam optimizer and some other optimizers doesn't work well with
  # distribution strategy (b/113076709)
  # Use keras.SGD (SGD + Momentum) according to ResNet paper.
  optimizer = tf.keras.optimizers.SGD(lr=0.1, momentum=0.9)
  optimization_callbacks = [
    tf.keras.callbacks.LearningRateScheduler(
        lambda x: 0.1 if x <= 80 else 0.01 if x <= 120 else 0.001,
        verbose=1)
  ]
  # Add weight decay based on ResNet paper
  weight_decay = 0.001
  for layer in model.layers:
    if hasattr(layer, "kernel_regularizer"):
      layer.kerner_regularizer = tf.keras.regularizers.l2(weight_decay)
  model.compile(loss="categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
                distribute=distribution)

  # Create benchmark logger for benchmark logging
  run_params = {
      "batch_size": FLAGS.batch_size,
      "synthetic_data": False,
      "train_epochs": FLAGS.train_epochs,
      "num_train_images": FLAGS.num_train_images,
      "num_eval_images": FLAGS.num_eval_images,
  }

  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info(
      model_name=model,
      dataset_name=dataset_name,
      run_params=run_params,
      test_id=FLAGS.benchmark_test_id)

  # Create callbacks that log metric values about the training and evaluation
  callbacks = model_callbacks.get_model_callbacks(
      FLAGS.callbacks,
      batch_size=FLAGS.batch_size,
      metric_logger=benchmark_logger)
  callbacks.extend(optimization_callbacks)
  # Train and evaluate the model
  history = model.fit(
      datagen.flow(x_train, y_train, batch_size=FLAGS.batch_size),
      epochs=FLAGS.train_epochs,
      callbacks=callbacks,
      validation_data=(x_test, y_test),
      steps_per_epoch=int(np.ceil(FLAGS.num_train_images / FLAGS.batch_size)),
      validation_steps=int(np.ceil(FLAGS.num_eval_images / FLAGS.batch_size))
  )

  tf.logging.info("Logging the evaluation results...")
  for epoch in range(FLAGS.train_epochs):
    eval_results = {
        "accuracy": history.history["val_acc"][epoch],
        "loss": history.history["val_loss"][epoch],
        tf.GraphKeys.GLOBAL_STEP: (epoch + 1) * np.ceil(
            FLAGS.num_eval_images/FLAGS.batch_size)
    }
    benchmark_logger.log_evaluation_result(eval_results)

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
      default=["LoggingEpochMetricCallback"],
      help=flags_core.help_wrap(
          "A list of (case insensitive) strings to specify the names of "
          "callbacks. For example: `--callbacks ExamplesPerSecondCallback,"
          "LoggingMetricCallback`"))

  @flags.multi_flags_validator(
      ["eager", "dist_strat"],
      message="Both --eager and --dist_strat were set. Only one can be "
              "defined, as DistributionStrategy is not supported in Eager "
              "execution currently.")
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
