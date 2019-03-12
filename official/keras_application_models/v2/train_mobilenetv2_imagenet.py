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


def train_mobilenetv2(_):
  """Train MobileNetV2 on ImageNet from the scratch."""

  # Initialize distribution strategy.
  strategy = utils.get_distribution_strategy(
    FLAGS.num_gpus, no_distribution_strategy=not FLAGS.dist_strat)

  global_batch_size = FLAGS.batch_size * FLAGS.num_gpus

  with strategy.scope():
    dataset_builder = dataset.ImageNetDataset()
    image_shape = (224, 224)
    train_ds, test_ds = dataset_builder.to_tf_dataset(
        global_batch_size, image_shape)

    model = tf.keras.applications.MobileNetV2(
        weights=(None if FLAGS.no_pretrained_weights else "imagenet"),
        input_shape=image_shape + (3,),
        include_top=True,
        classes=dataset_builder.num_classes)

    initial_lr = 0.045 * FLAGS.num_gpus
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=tf.keras.backend.variable(initial_lr), momentum=0.9)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda x, lr: lr * 0.316 if x > 0 and x % 15 == 0 else lr,
        verbose=1)

    # Add L1L2 regularization to avoid overfitting
    if FLAGS.no_pretrained_weights:
      decay = 0.00004 * FLAGS.num_gpus
      for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type == "Conv2D":
          layer.kerner_regularizer = tf.keras.regularizers.l2(decay)

    checkpoint = utils.prepare_model_saving("mobilenetv2")
    callbacks = [lr_scheduler, checkpoint]

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy", "top_k_categorical_accuracy"])

    # Train and evaluate the model
    history = model.fit(
        train_ds,
        epochs=FLAGS.train_epochs,
        callbacks=callbacks,
        steps_per_epoch=int(
            np.ceil(dataset_builder.num_train / global_batch_size)),
        validation_data=test_ds,
        validation_steps=int(
            np.ceil(dataset_builder.num_test / global_batch_size)),
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
