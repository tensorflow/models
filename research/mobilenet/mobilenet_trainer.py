# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Runs an Image Classification task for MobileNet."""

import os
from typing import Mapping, Text, Any, Type, List

import logging
import tensorflow as tf

from research.mobilenet.dataset_loader import load_tfds, pipeline
from research.mobilenet.dataset_config import DatasetConfig
from research.mobilenet.dataset_config import ImageNetTEConfig
from research.mobilenet.dataset_config import ImageNetConfig
from research.mobilenet.dataset_config import MNISTConfig
from research.mobilenet.mobilenet_config import MobileNetV1Config
from research.mobilenet.mobilenet_config import MobileNetV2Config
from research.mobilenet.mobilenet_config import MobileNetV3Config
from research.mobilenet.mobilenet_v1_model import mobilenet_v1

from official.modeling.hyperparams.base_config import Config


def _get_model_config() -> Mapping[Text, Type[Config]]:
  return {
    'mobilenet_v1': MobileNetV1Config,
    'mobilenet_v2': MobileNetV2Config,
    'mobilenet_v3': MobileNetV3Config,
  }


def _get_model_builder() -> Mapping[Text, Any]:
  return {
    'mobilenet_v1': mobilenet_v1
  }


def _get_dataset_config() -> Mapping[Text, Type[DatasetConfig]]:
  return {
    'imagenet2012': ImageNetConfig,
    'imagenette': ImageNetTEConfig,
    'mnist': MNISTConfig
  }


def _get_metrics(one_hot: bool) -> Mapping[Text, Any]:
  """Get a dict of available metrics to track."""
  if one_hot:
    return {
      # (name, metric_fn)
      'acc': tf.keras.metrics.CategoricalAccuracy(),
      'accuracy': tf.keras.metrics.CategoricalAccuracy(),
      'top_1': tf.keras.metrics.CategoricalAccuracy(),
      'top_5': tf.keras.metrics.TopKCategoricalAccuracy(
        k=5, name='top_5_accuracy'),
    }
  else:
    return {
      # (name, metric_fn)
      'acc': tf.keras.metrics.SparseCategoricalAccuracy(),
      'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
      'top_1': tf.keras.metrics.SparseCategoricalAccuracy(),
      'top_5': tf.keras.metrics.SparseTopKCategoricalAccuracy(
        k=5, name='top_5_accuracy'),
    }


def _get_loss(one_hot: bool) -> Mapping[Text, Any]:
  """Get a dict of available metrics to track."""
  if one_hot:
    return {
      # (name, loss)
      'cross_entropy': tf.keras.losses.CategoricalCrossentropy(),
    }
  else:
    return {
      # (name, loss)
      'cross_entropy': tf.keras.losses.SparseCategoricalCrossentropy(),
    }


def _get_callback(model_dir: Text) -> List[tf.keras.callbacks.Callback]:
  check_point = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(model_dir, 'model.ckpt-{epoch:04d}'),
    verbose=1
  )
  tensorboard = tf.keras.callbacks.TensorBoard(log_dir=model_dir)
  return [check_point, tensorboard]


def get_args():
  """Parse command line arguments"""
  pass


def get_dataset(config: Type[DatasetConfig]) -> tf.data.Dataset:
  """Build dataset for training, evaluation and test"""
  raw_dataset = load_tfds(
    dataset_name=config.name,
    data_dir=config.data_dir,
    download=config.download,
    split=config.split
  )

  dataset = pipeline(
    dataset=raw_dataset,
    config=config
  )

  return dataset


def build_model(model_name: Text,
                dataset_config: Type[DatasetConfig],
                model_config: Type[Config]
                ) -> tf.keras.models.Model:
  """Build mobilenet model given configuration"""

  model_build_function = _get_model_builder().get(model_name)
  if model_build_function:
    image_size = dataset_config.image_size
    channels = dataset_config.num_channels
    model_config.num_classes = dataset_config.num_classes
    return model_build_function(input_shape=(image_size, image_size, channels),
                                config=model_config)
  else:
    raise ValueError('The model {} is not supported.'.format(model_name))


def train_and_eval():
  """Runs the train and eval path using compile/fit."""

  d_config = _get_dataset_config().get('imagenette')
  m_config = _get_model_config().get('mobilenet_v1')

  d_config.batch_size = 2

  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    keras_model = build_model(
      model_name='mobilenet_v1',
      dataset_config=d_config,
      model_config=m_config
    )

    optimizer = tf.keras.optimizers.Adam()
    keras_model.compile(
      optimizer=optimizer,
      loss=[_get_loss(one_hot=d_config.one_hot)['cross_entropy']],
      metrics=[_get_metrics(one_hot=d_config.one_hot)['acc']],
    )

    global_batch_size = d_config.batch_size * strategy.num_replicas_in_sync
    d_config.batch_size = global_batch_size
    train_dataset = get_dataset(d_config)
    d_config.split = 'validation'
    eval_dataset = get_dataset(d_config)

  steps_per_epoch = d_config.num_examples // d_config.batch_size

  # Callbacks
  callbacks_to_use = _get_callback(model_dir='./tmp')
  # Train model
  keras_model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=eval_dataset,
    validation_steps=1000,
    verbose=1,
    callbacks=callbacks_to_use
  )


def main():
  pass


if __name__ == '__main__':
  logging.basicConfig(
    format='%(asctime)-15s:%(levelname)s:%(module)s:%(message)s',
    level=logging.INFO)

  train_and_eval()
