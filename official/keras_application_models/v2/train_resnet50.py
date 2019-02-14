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
"""Retrain the keras built-in ResNet50 on CIFAR-10."""
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
    return tf.keras.applications.ResNet50(
        weights=None,
        input_shape=input_shape,
        include_top=True,
        classes=classes)
  else:
    base_model = tf.keras.applications.ResNet50(
        # Use imagenet pretrained weights require input_shape and pooling.
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg',
        # When include_top is False, we need manually add FC layers.
        include_top=False)
    # Manually add FC layer
    x = base_model.output
    x = tf.keras.layers.Dense(10, activation='softmax', name='fc10')(x)
    return tf.keras.Model(inputs=base_model.inputs, outputs=x)


def get_cifar_model(input_shape=(32, 32, 3),
                    classes=10,
                    no_pretrained_weights=False,
                    up_sampling=1):
  if up_sampling == 1:
    input_tensor = tf.keras.layers.Input(shape=input_shape)
    x = input_tensor
  else:
    input_tensor = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.UpSampling2D(
        size=up_sampling, interpolation="bilinear")(input_tensor)
    input_shape = (
        input_shape[0] * up_sampling,
        input_shape[1] * up_sampling,
        input_shape[2])
  base_model = tf.keras.applications.ResNet50(
      weights=(None if no_pretrained_weights else 'imagenet'),
      input_shape=input_shape,
      input_tensor=x,
      include_top=False,
      pooling='avg')
  x = base_model.output
  x = tf.keras.layers.Dense(classes, activation='softmax', name='fc10')(x)
  return tf.keras.Model(inputs=input_tensor, outputs=x)


def train_resnet50(_):
  """Train Resnet50 on CIFAR from the scratch."""

  # Enable/Disable eager based on flags. It's enabled by default.
  utils.init_eager_execution(FLAGS.no_eager)

  # Initialize distribution strategy.
  # TODO(xunkai55): make it work when FLAGS.dist_strat == True.
  # If FLAGS.dist_strat == False, a fake placeholder will be used.
  strategy = utils.get_distribution_strategy(
    FLAGS.num_gpus, no_distribution_strategy=not FLAGS.dist_strat)

  with strategy.scope():
    ds = dataset.Cifar10Dataset()
    train_dataset, test_dataset = ds.to_dataset(FLAGS.batch_size)
    model = get_cifar_model(input_shape=ds.input_shape,
                            classes=ds.num_classes,
                            no_pretrained_weights=FLAGS.no_pretrained_weights,
                            up_sampling=4)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=tf.keras.backend.variable(1e-3), momentum=0.9)
    if FLAGS.no_pretrained_weights:
      lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
          lambda x: 1e-2 if x < 50 else 1e-3 if x < 70 else 1e-4,
          verbose=1)
    else:
      lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
          lambda x: 1e-3 if x < 10 else 1e-4 if x < 30 else 1e-5,
          verbose=1)

    # Add L1L2 regularization to avoid overfitting
    utils.add_global_regularization(model, l2=0.0001)

    checkpoint = utils.prepare_model_saving("resnet50")
    callbacks = [lr_scheduler, checkpoint]

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])

    # Train and evaluate the model
    history = model.fit(
        train_dataset,
        epochs=FLAGS.train_epochs,
        callbacks=callbacks,
        validation_data=test_dataset,
        steps_per_epoch=int(np.ceil(ds.num_train_examples / FLAGS.batch_size)),
        validation_steps=int(np.ceil(ds.num_test_examples / FLAGS.batch_size))
    )

  # Clear the session explicitly to avoid session delete error
  tf.keras.backend.clear_session()


def main(_):
  train_resnet50(FLAGS)


if __name__ == "__main__":
  absl.logging.set_verbosity(absl.logging.INFO)
  utils.define_flags()
  FLAGS = flags.FLAGS
  absl_app.run(main)
