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

from official.keras_application_models.v2 import datasets
from official.keras_application_models.v2 import utils


def prepare_dataset_builder():
  # Split this function out of `run` for easy testing with different datasets.
  return datasets.ImageNetDatasetBuilder()


def run(dataset_builder, flags_obj):
  """Train MobileNetV2 on ImageNet from the scratch.

  Args:
    dataset_builder: Object which helps to build datasets and contains meta
      info as well. Required members:
        to_dataset(
            batch_size: int,
            image_shape: (int, int),
            take_train_num: int) -> (tf.data.Dataset, tf.data.Dataset)
        num_classes: int
        num_train: int
        num_test: int
  """

  # Initialize distribution strategy.
  strategy = utils.get_distribution_strategy(
    flags_obj.num_gpus, no_distribution_strategy=not flags_obj.dist_strat)

  # MirroredStrategy will divide batches per GPU.
  global_batch_size = flags_obj.batch_size * flags_obj.num_gpus

  with strategy.scope():
    image_shape = (224, 224)
    train_ds, test_ds = dataset_builder.to_dataset(
        global_batch_size, image_shape,
        take_train_num=flags_obj.limit_train_num)

    model = tf.keras.applications.MobileNetV2(
        weights=(None if flags_obj.no_pretrained_weights else "imagenet"),
        input_shape=image_shape + (3,),
        include_top=True,
        classes=dataset_builder.num_classes)

    if flags_obj.no_pretrained_weights:
      initial_lr = 0.045 * flags_obj.num_gpus
      lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
          lambda x, lr: lr * 0.94 if x > 0 else lr,
          verbose=1)
    else:
      initial_lr = 0.001 * flags_obj.num_gpus
      lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
          lambda x, lr: lr * 0.316 if x > 0 and x % 10 == 0 else lr,
          verbose=1)

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=tf.keras.backend.variable(initial_lr), momentum=0.9)

    # To train it from scratch, we need L2 regularization to avoid overfitting.
    if flags_obj.no_pretrained_weights:
      decay = 0.00004 * flags_obj.num_gpus
      for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type == "Conv2D":
          layer.kerner_regularizer = tf.keras.regularizers.l2(decay)

    callbacks = [lr_scheduler]
    if flags_obj.enable_model_saving:
      checkpoint = utils.prepare_model_saving("mobilenetv2")
      callbacks.append(checkpoint)

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["acc", "top_k_categorical_accuracy"])

    # Train and evaluate the model
    history = model.fit(
        train_ds,
        epochs=flags_obj.train_epochs,
        callbacks=callbacks,
        steps_per_epoch=int(
            np.ceil(dataset_builder.num_train / global_batch_size)),
        validation_data=test_ds,
        validation_steps=int(
            np.ceil(dataset_builder.num_test / global_batch_size)),
    )

  # Clear the session explicitly to avoid session delete error
  tf.keras.backend.clear_session()
  return {
      # Optmizer.iterations is a MirroredVariable for distributed training.
      "iters": optimizer.iterations.read_value(),
      "history": history.history,
  }


def main(_):
  dataset_builder = prepare_dataset_builder()
  run(dataset_builder, flags.FLAGS)


if __name__ == "__main__":
  absl.logging.set_verbosity(absl.logging.INFO)
  utils.define_flags()
  absl_app.run(main)

