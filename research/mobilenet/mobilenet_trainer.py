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
import logging
from typing import Mapping, Text, Any, Type, List, Dict

from absl import app
from absl import flags
import tensorflow as tf

from research.mobilenet import dataset_loader
from research.mobilenet import mobilenet_v1_model
from research.mobilenet import mobilenet_v2_model
from research.mobilenet.configs import defaults
from research.mobilenet.configs import archs
from research.mobilenet.configs import dataset as dataset_config

from official.modeling.hyperparams import base_config
from official.vision.image_classification.configs import base_configs
from official.vision.image_classification import optimizer_factory


def _get_model_config() -> Mapping[Text, Type[base_config.Config]]:
  return {
    'mobilenet_v1': archs.MobileNetV1Config,
    'mobilenet_v2': archs.MobileNetV2Config,
    'mobilenet_v3': archs.MobileNetV3Config,
  }


def _get_model_builder() -> Mapping[Text, Any]:
  return {
    'mobilenet_v1': mobilenet_v1_model.mobilenet_v1,
    'mobilenet_v2': mobilenet_v2_model.mobilenet_v2
  }


def _get_dataset_config() -> Mapping[Text, Type[dataset_config.DatasetConfig]]:
  return {
    'imagenet2012': dataset_config.ImageNetConfig,
    'imagenette': dataset_config.ImageNetteConfig,
    'mnist': dataset_config.MNISTConfig
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
    save_best_only=True,
    filepath=os.path.join(model_dir, 'model.ckpt-{epoch:04d}'),
    verbose=1
  )
  tensorboard = tf.keras.callbacks.TensorBoard(log_dir=model_dir)
  return [check_point, tensorboard]


def _get_optimizer(
    batch_size: int,
    steps_per_epoch: int,
    lr_name: Text = defaults.LR_NAME_DEFAULT,
    optimizer_name: Text = defaults.OP_NAME_DEFAULT,
    lr_params: Dict[Text, Any] = defaults.LR_CONFIG_DEFAULT,
    optimizer_params: Dict[Text, Any] = defaults.OP_CONFIG_DEFAULT,
) -> tf.keras.optimizers.Optimizer:
  learning_rate_config = base_configs.LearningRateConfig(
    name=lr_name,
    **lr_params)
  optimizer_config = base_configs.OptimizerConfig(
    name=optimizer_name,
    **optimizer_params)
  learning_rate = optimizer_factory.build_learning_rate(
    params=learning_rate_config,
    batch_size=batch_size,
    train_steps=steps_per_epoch)
  optimizer = optimizer_factory.build_optimizer(
    optimizer_name=optimizer_config.name,
    base_learning_rate=learning_rate,
    params=optimizer_config.as_dict())
  return optimizer


def _resume_from_checkpoint(model: tf.keras.Model,
                            model_dir: str,
                            train_steps: int) -> int:
  """Resumes from the latest checkpoint, if possible.

  Loads the model weights and optimizer settings from a checkpoint.
  This function should be used in case of preemption recovery.

  Args:
    model: The model whose weights should be restored.
    model_dir: The directory where model weights were saved.
    train_steps: The number of steps to train.

  Returns:
    The epoch of the latest checkpoint, or 0 if not restoring.

  """
  logging.info('Load from checkpoint is enabled.')
  latest_checkpoint = tf.train.latest_checkpoint(model_dir)
  logging.info('latest_checkpoint: %s', latest_checkpoint)
  if not latest_checkpoint:
    logging.info('No checkpoint detected.')
    return 0

  logging.info('Checkpoint file %s found and restoring from '
               'checkpoint', latest_checkpoint)
  model.load_weights(latest_checkpoint)
  initial_epoch = model.optimizer.iterations // train_steps
  logging.info('Completed loading from checkpoint.')
  logging.info('Resuming from epoch %d', initial_epoch)
  return int(initial_epoch)


def get_flags():
  """Initialize the data extraction parameters.

  Define the arguments with the default values and parses the arguments
  passed to the main program.

  Args:
      args_parser: (argparse.ArgumentParser)
  """
  flags.DEFINE_string(
    'model_name',
    help='MobileNet version name: mobilenet_v1, mobilenet_v2',
    default='mobilenet_v1'
  )
  flags.DEFINE_string(
    'dataset_name',
    help='Dataset name from TDFS to train on: imagenette, imagenet2012',
    default='imagenette'
  )
  flags.DEFINE_string(
    'optimizer_name',
    help='Name of optimizer.',
    default='rmsprop'
  )
  flags.DEFINE_string(
    'learning_scheduler_name',
    help='Name of learning rate scheduler.',
    default='exponential'
  )
  flags.DEFINE_float(
    'learning_rate',
    help='Base learning rate.',
    default=0.008
  )
  flags.DEFINE_integer(
    'batch_size',
    help='Training batch size.',
    default=2 # for testing purpose
  )
  flags.DEFINE_integer(
    'epochs',
    help='Number of epochs.',
    default=5
  )
  flags.DEFINE_string(
    'model_dir',
    help='Working directory.',
    default='./tmp'
  )
  flags.DEFINE_bool(
    'resume_checkpoint',
    help='Whether resume training from previous checkpoint.',
    default=False
  )


def get_dataset(config: Type[dataset_config.DatasetConfig]) -> tf.data.Dataset:
  """Build dataset for training, evaluation and test"""
  raw_dataset = dataset_loader.load_tfds(
    dataset_name=config.name,
    data_dir=config.data_dir,
    download=config.download,
    split=config.split
  )

  dataset = dataset_loader.pipeline(
    dataset=raw_dataset,
    config=config
  )

  return dataset


def build_model(model_name: Text,
                dataset_config: Type[dataset_config.DatasetConfig],
                model_config: Type[base_config.Config]
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


def train_and_eval(params: flags.FlagValues) -> tf.keras.callbacks.History:
  """Runs the train and eval path using compile/fit."""
  d_config = _get_dataset_config().get(params.dataset_name)
  d_config.batch_size = params.batch_size
  m_config = _get_model_config().get(params.model_name)

  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    # build the dataset
    global_batch_size = d_config.batch_size * strategy.num_replicas_in_sync
    d_config.batch_size = global_batch_size
    train_dataset = get_dataset(d_config)
    d_config.split = 'validation'
    eval_dataset = get_dataset(d_config)

    # compute number iterations per epoch
    steps_per_epoch = d_config.num_examples // d_config.batch_size

    # build the model
    keras_model = build_model(
      model_name=params.model_name,
      dataset_config=d_config,
      model_config=m_config
    )

    learning_rate = params.learning_rate
    learning_params = defaults.LR_CONFIG_DEFAULT
    learning_params.update({'initial_lr': learning_rate})

    # build the optimizer
    optimizer = _get_optimizer(
      batch_size=global_batch_size,
      steps_per_epoch=steps_per_epoch,
      lr_name=params.learning_scheduler_name,
      optimizer_name=params.optimizer_name,
      lr_params=learning_params
    )

    # compile model
    keras_model.compile(
      optimizer=optimizer,
      loss=[_get_loss(one_hot=d_config.one_hot)['cross_entropy']],
      metrics=[_get_metrics(one_hot=d_config.one_hot)['acc']],
    )

    initial_epoch = 0
    if params.resume_checkpoint:
      initial_epoch = _resume_from_checkpoint(model=keras_model,
                                              model_dir=params.model_dir,
                                              train_steps=steps_per_epoch)

    # Callbacks
    callbacks_to_use = _get_callback(model_dir=params.model_dir)

  # Train model
  history = keras_model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=params.epochs,
    validation_data=eval_dataset,
    validation_steps=1000,
    initial_epoch=initial_epoch,
    verbose=1,
    callbacks=callbacks_to_use
  )

  return history


def main(_):
  history = train_and_eval(flags.FLAGS)
  if history:
    logging.info('Run history:\n%s', history)


if __name__ == '__main__':
  logging.basicConfig(
    format='%(asctime)-15s:%(levelname)s:%(module)s:%(message)s',
    level=logging.INFO)
  get_flags()
  app.run(main)
