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
"""Retrain the keras built-in MobileNetV2 on CIFAR-10."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-bad-import-order
import os
import numpy as np
import absl.logging
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.keras_application_models.v2 import dataset
from official.keras_application_models.v2 import utils


def get_model(input_shape, classes, no_pretrained_weights):
  if no_pretrained_weights:
    return tf.keras.applications.MobileNetV2(
        weights=None,
        alpha=0.35,
        input_shape=input_shape,
        include_top=True,
        classes=classes)
  else:
    base_model = tf.keras.applications.MobileNetV2(
        # Use imagenet pretrained weights require input_shape and pooling.
        weights="imagenet",
        alpha=0.35,
        input_shape=input_shape,
        include_top=False,
        pooling="avg")
    x = base_model.output
    x = tf.keras.layers.Dense(10, activation='softmax', name='fc10')(x)
    return tf.keras.Model(inputs=base_model.inputs, outputs=x)


def train_mobilenetv2(_):
  """Train MobileNetV2 on CIFAR from the scratch."""

  # Enable/Disable eager based on flags. It's enabled by default.
  utils.init_eager_execution(FLAGS.no_eager)

  # Initialize distribution strategy.
  # TODO(xunkai55): make it work when FLAGS.dist_strat == True.
  # If FLAGS.dist_strat == False, a fake placeholder will be used.
  if FLAGS.dist_strat:
    raise NotImplemented("Distribution strategy is not supported yet.")
  else:
    strategy = utils.NotReallyADistributionStrategy()

  if FLAGS.num_gpus > 1:
    raise NotImplemented("Multiple GPU is not supported yet.")

  with strategy.scope():
    ds = dataset.Cifar10Dataset(FLAGS.batch_size, dsize=(96, 96))
    x_train, y_train, x_test, y_test = ds.get_normalized_data()
    datagen = ds.get_data_augmentor()

    model = get_model(
        ds.input_shape, ds.num_classes, FLAGS.no_pretrained_weights)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=tf.keras.backend.variable(0.1))

    if FLAGS.no_pretrained_weights:
      lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
          lambda x: 1e-2 if x < 50 else 1e-3 if x < 70 else 1e-4,
          verbose=1)
    else:
      lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
          lambda x: 0.1 - 0.001 * x, verbose=1)

    # Add L1L2 regularization to avoid overfitting
    utils.add_global_regularization(model, l2=0.00004)

    checkpoint = utils.prepare_model_saving("mobilenetv2")
    callbacks = [lr_scheduler, checkpoint]

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])

    # Train and evaluate the model
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=FLAGS.batch_size),
        epochs=FLAGS.train_epochs,
        callbacks=callbacks,
        validation_data=(x_test, y_test),
        steps_per_epoch=int(np.ceil(len(x_train) / FLAGS.batch_size)),
    )

  # Clear the session explicitly to avoid session delete error
  tf.keras.backend.clear_session()


def main(_):
  train_mobilenetv2(FLAGS)


if __name__ == "__main__":
  absl.logging.set_verbosity(absl.logging.INFO)
  utils.define_flags()
  FLAGS = flags.FLAGS
  absl_app.run(main)
