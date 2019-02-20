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
import tensorflow.keras.backend as K
# pylint: enable=g-bad-import-order

from official.keras_application_models.v2 import dataset
from official.keras_application_models.v2 import utils


FLAGS = flags.FLAGS


def get_cifar_model(input_shape,
                    classes,
                    no_pretrained_weights=False,
                    up_sampling=1):
  # Prepare the preprocessing layer, if necessary.
  if up_sampling == 1:
    input_tensor = tf.keras.layers.Input(shape=input_shape)
    x = input_tensor
  else:
    # The ResNet50 starts with 2 consequtive 2-stride convolution, which is not
    # suitable for small images in CIFAR dataset. Applying upsampling here could
    # neutralize the effect.
    input_tensor = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.UpSampling2D(
        size=up_sampling, interpolation="bilinear")(input_tensor)
    input_shape = (
        input_shape[0] * up_sampling,
        input_shape[1] * up_sampling,
        input_shape[2])

  # Load the Keras model.
  base_model = tf.keras.applications.ResNet50(
      weights=(None if no_pretrained_weights else 'imagenet'),
      input_shape=input_shape,
      input_tensor=x,
      include_top=False,
      pooling='avg')

  # Add FC layers for classification.
  x = base_model.output
  x = tf.keras.layers.Dense(classes, activation='softmax', name='fc10')(x)
  return tf.keras.Model(inputs=input_tensor, outputs=x)


def run(_):
  """Train Resnet50 on CIFAR from the scratch."""

  # Enable/Disable eager based on flags. It's enabled by default.
  utils.init_eager_execution(FLAGS.no_eager)

  # Initialize distribution strategy.
  strategy = utils.get_distribution_strategy(
    FLAGS.num_gpus, no_distribution_strategy=not FLAGS.dist_strat)

  with strategy.scope():
    # Prepare dataset.
    ds = dataset.Cifar10Dataset(
        batch_size=FLAGS.batch_size,
        data_augmentation=False)
    train_dataset, test_dataset = ds.train_dataset, ds.test_dataset

    # Prepare model. Setting up_sampling=4 to get a better performance
    model = get_cifar_model(input_shape=ds.image_shape,
                            classes=ds.num_classes,
                            no_pretrained_weights=FLAGS.no_pretrained_weights,
                            up_sampling=4)

    # Prepare optimizers, regularization and lr schedulers.
    # TODO(xunkai): Dist Strat seems not supporting LR Scheduler callback yet.
    # Refactor this part, although it doesn't really affect fine tuning.
    if FLAGS.no_pretrained_weights:
      lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
          lambda x: 1e-2 if x < 50 else 1e-3 if x < 70 else 1e-4,
          verbose=1)
    else:
      lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
          lambda x: 1e-3 if x < 10 else 1e-4 if x < 30 else 1e-5,
          verbose=1)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=K.variable(1e-3), momentum=0.9)
    utils.add_global_regularization(model, l2=0.0001)

    # Prepare model saving checkpoint.
    checkpoint = utils.prepare_model_saving("resnet50")

    # Compile, train and evaluate the model
    callbacks = [lr_scheduler, checkpoint]
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    history = model.fit(
        train_dataset,
        epochs=FLAGS.train_epochs,
        callbacks=callbacks,
        validation_data=test_dataset,
        steps_per_epoch=int(np.ceil(ds.num_train_examples / FLAGS.batch_size)),
        validation_steps=int(np.ceil(ds.num_test_examples / FLAGS.batch_size)),
    )

  # Clear the session explicitly to avoid session delete error
  tf.keras.backend.clear_session()
  return history


def main(_):
  return run(FLAGS)


if __name__ == "__main__":
  absl.logging.set_verbosity(absl.logging.INFO)
  utils.define_flags()
  absl_app.run(main)
