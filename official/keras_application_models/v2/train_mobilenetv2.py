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


def prepare_dataset_builder(dataset, num_dataset_private_threads):
  if dataset.startswith("imagenet"):
    return datasets.ImageNetDatasetBuilder(num_dataset_private_threads)
  elif dataset.startswith("cifar10") or dataset.startswith("cfiar-10"):
    return datasets.Cifar10DatasetBuilder(num_dataset_private_threads)
  else:
    raise InvalidArgumentError("Only ImageNet and CIFAR-10 are supported yet.")


def create_model_for_train(image_shape, num_classes):
  model = tf.keras.applications.MobileNetV2(
      weights=None,
      input_shape=image_shape + (3,),
      include_top=False,
      pooling="avg",
      classes=num_classes)
  # Keras Application model v2 doesn't include dropout. Follow the SLIM settings
  # with dropout and 1-D conv here.
  x = model.output
  # The last layer output tensor shape in MobileNetV2 is (None, ?)
  x = tf.keras.layers.Reshape((1, 1, x.shape[1]), name='reshape_1')(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Conv2D(num_classes, (1, 1),
                             padding="same",
                             name="conv_preds")(x)
  x = tf.keras.layers.Activation("softmax", name="act_softmax")(x)
  x = tf.keras.layers.Reshape((num_classes,), name='reshape_2')(x)
  final_model = tf.keras.models.Model(
      model.inputs, x, name="%s_train" % model.name)
  final_model.summary()
  return final_model


def create_model_for_finetuning(image_shape, num_classes, freeze=True):
  model = tf.keras.applications.MobileNetV2(
      weights="imagenet",
      input_shape=image_shape + (3,),
      include_top=False,
      pooling="avg",
      classes=num_classes)
  if freeze:
    for layer in model.layers:
      layer.trainable = False
  x = model.output
  x = tf.keras.layers.Dense(num_classes, "softmax")(x)
  final_model = tf.keras.models.Model(
      model.inputs, x, name="%s_finetune" % model.name)
  final_model.summary()
  return final_model


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
        label_smoothing=flags_obj.label_smoothing,
        take_train_num=flags_obj.limit_train_num)
    num_classes = dataset_builder.num_classes

    callbacks = []
    if flags_obj.no_pretrained_weights:  # Train
      model = create_model_for_train(image_shape, num_classes)
      if flags_obj.initial_lr > 0:
        initial_lr = flags_obj.initial_lr * flags_obj.num_gpus
      else:
        initial_lr = 0.045 * flags_obj.num_gpus
      lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
          lambda x, lr: lr * 0.98 if x > 0 else lr,
          verbose=1)
    else:  # Finetune
      model = create_model_for_finetuning(image_shape, num_classes)
      if flags_obj.initial_lr > 0:
        initial_lr = flags_obj.initial_lr * flags_obj.num_gpus
      else:
        initial_lr = 0.01 * flags_obj.num_gpus
      lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
          lambda x, lr: lr * 0.98 if x > 0 else lr,
          verbose=1)

    if flags_obj.checkpoint_path:
      model.load_weights(flags_obj.checkpoint_path)

    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=tf.keras.backend.variable(initial_lr))
    callbacks.append(lr_scheduler)

    # To train it from scratch, we need L2 regularization to avoid overfitting.
    if flags_obj.no_pretrained_weights:
      decay = 0.00004 * flags_obj.num_gpus
      for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type == "Conv2D":
          layer.kerner_regularizer = tf.keras.regularizers.l2(decay)

    run_name = "mobilenetv2_%s" % flags_obj.run_name
    # Configure TensorBoard
    monitor = tf.keras.callbacks.TensorBoard(
        log_dir='./logs/%s' % run_name, histogram_freq=0,
        batch_size=global_batch_size, write_graph=True, write_grads=False,
        write_images=False, embeddings_freq=0, embeddings_layer_names=None,
        embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    callbacks.append(monitor)

    if flags_obj.enable_model_saving:
      checkpoint = utils.prepare_model_saving(run_name)
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
  dataset_builder = prepare_dataset_builder(
      flags.FLAGS.dataset, flags.FLAGS.num_dataset_private_threads)
  run(dataset_builder, flags.FLAGS)


if __name__ == "__main__":
  absl.logging.set_verbosity(absl.logging.INFO)
  utils.define_flags()
  absl_app.run(main)

