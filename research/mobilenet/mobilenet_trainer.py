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
import tensorflow_addons as tfa

from research.mobilenet import dataset_loader
from research.mobilenet import mobilenet_v1
from research.mobilenet import mobilenet_v2
from research.mobilenet import mobilenet_v3
from research.mobilenet.configs import defaults
from research.mobilenet.configs import archs
from research.mobilenet.configs import dataset

from official.vision.image_classification.configs import base_configs
from official.vision.image_classification import optimizer_factory
from official.vision.image_classification import dataset_factory


def _get_model_config() -> Mapping[Text, Type[archs.MobileNetConfig]]:
  return {
    'mobilenet_v1': archs.MobileNetV1Config,
    'mobilenet_v2': archs.MobileNetV2Config,
    'mobilenet_v3_small': archs.MobileNetV3SmallConfig,
    'mobilenet_v3_large': archs.MobileNetV3LargeConfig
  }


def _get_model_builder() -> Mapping[Text, Any]:
  return {
    'mobilenet_v1': mobilenet_v1.mobilenet_v1,
    'mobilenet_v2': mobilenet_v2.mobilenet_v2,
    'mobilenet_v3_small': mobilenet_v3.mobilenet_v3_small,
    'mobilenet_v3_large': mobilenet_v3.mobilenet_v3_large
  }


def _get_dataset_config() -> Mapping[Text, Type[dataset_factory.DatasetConfig]]:
  return {
    'imagenet2012': dataset.ImageNetConfig,
    'imagenette': dataset.ImageNetteConfig,
    'mnist': dataset.MNISTConfig
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


def _get_callback(model_dir: Text) -> List[tf.keras.callbacks.Callback]:
  """Create callbacks for Keras model training."""
  check_point = tf.keras.callbacks.ModelCheckpoint(
    save_best_only=True,
    filepath=os.path.join(model_dir, 'model.ckpt-{epoch:04d}'),
    verbose=1)
  tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=model_dir, update_freq=100)
  return [check_point, tensorboard]


def _get_optimizer(
    batch_size: int,
    steps_per_epoch: int,
    lr_name: Text = defaults.LR_NAME_DEFAULT,
    optimizer_name: Text = defaults.OP_NAME_DEFAULT,
    lr_params: Dict[Text, Any] = defaults.LR_CONFIG_DEFAULT,
    optimizer_params: Dict[Text, Any] = defaults.OP_CONFIG_DEFAULT,
) -> tf.keras.optimizers.Optimizer:
  """Construct optimizer for model training.

  Args:
    batch_size: batch size
    steps_per_epoch: number of steps per epoch
    lr_name: learn rate scheduler name, e.g., exponential
    optimizer_name: optimizer name, e.g., adam, rmsprop
    lr_params: parameters for initiating learning rate scheduler object
    optimizer_params: parameters for initiating optimizer object

  Returns:
    Return a tf.keras.optimizers.Optimizer object.
  """
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

  """
  flags.DEFINE_string(
    'model_name',
    help='MobileNet version name: mobilenet_v1, mobilenet_v2, '
         'mobilenet_v3_small and mobilenet_v3_large',
    default='mobilenet_v1'
  )
  flags.DEFINE_string(
    'dataset_name',
    help='Dataset name from TDFS to train on: imagenette, imagenet2012',
    default='imagenette'
  )
  flags.DEFINE_string(
    'model_dir',
    help='Working directory.',
    default='./tmp'
  )
  flags.DEFINE_string(
    'data_dir',
    help='Directory for training data.',
    default=None
  )
  flags.DEFINE_bool(
    'resume_checkpoint',
    help='Whether resume training from previous checkpoint.',
    default=False
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
  # for hyperparameter tuning
  flags.DEFINE_float(
    'op_momentum',
    help='Optimizer momentum.',
    default=0.9
  )
  flags.DEFINE_float(
    'op_decay_rate',
    help='Optimizer discounting factor for gradient.',
    default=0.9
  )
  flags.DEFINE_float(
    'lr',
    help='Base learning rate.',
    default=0.008
  )
  flags.DEFINE_float(
    'lr_decay_rate',
    help='Magnitude of learning rate decay.',
    default=0.97
  )
  flags.DEFINE_float(
    'lr_decay_epochs',
    help='Frequency of learning rate decay.',
    default=2.4
  )
  flags.DEFINE_float(
    'label_smoothing',
    help='The amount of label smoothing.',
    default=0.0,
  )
  flags.DEFINE_float(
    'ma_decay_rate',
    help='Exponential moving average decay rate.',
    default=None
  )
  flags.DEFINE_float(
    'dropout_rate',
    help='Dropout rate.',
    default=0.2
  )
  flags.DEFINE_float(
    'std_weight_decay',
    help='Standard weight decay.',
    default=0.00004
  )
  flags.DEFINE_float(
    'truncated_normal_stddev',
    help='The standard deviation of the truncated normal weight initializer.',
    default=0.09
  )
  flags.DEFINE_float(
    'batch_norm_decay',
    help='Batch norm decay.',
    default=0.9997
  )
  flags.DEFINE_integer(
    'batch_size',
    help='Training batch size.',
    default=4  # for testing purpose
  )
  flags.DEFINE_integer(
    'epochs',
    help='Number of epochs.',
    default=5
  )


def get_dataset(config: dataset_factory.DatasetConfig,
                slim_preprocess: bool = False) -> tf.data.Dataset:
  """Build dataset for training, evaluation and test"""
  logging.info("Dataset Config: ")
  logging.info(config)
  if config.builder == 'tfds':
    raw_dataset = dataset_loader.load_tfds(
      dataset_name=config.name,
      data_dir=config.data_dir,
      download=config.download,
      split=config.split
    )
  elif config.builder == 'records':
    raw_dataset = dataset_loader.load_tfrecords(
      data_dir=config.data_dir,
      split=config.split,
      file_shuffle_buffer_size=config.file_shuffle_buffer_size
    )
  else:
    raise ValueError('Only support tfds and tfrecords builder.')

  processed_dataset = dataset_loader.pipeline(
    dataset=raw_dataset,
    config=config,
    slim_preprocess=slim_preprocess
  )

  return processed_dataset


def build_model(model_name: Text,
                dataset_config: dataset_factory.DatasetConfig,
                model_config: archs.MobileNetConfig
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
  logging.info('Run training for {} with {}'.format(params.model_name,
                                                    params.dataset_name))
  logging.info('The CLI params are: {}'.format(params.flag_values_dict()))
  d_config = _get_dataset_config().get(params.dataset_name)()
  m_config = _get_model_config().get(params.model_name)()

  logging.info('Training dataset configuration:', d_config)
  logging.info('Training model configuration:', m_config)

  # override the model params with CLI params
  m_config.num_classes = d_config.num_classes
  m_config.dropout_keep_prob = 1 - params.dropout_rate
  m_config.weight_decay = params.std_weight_decay
  m_config.stddev = params.truncated_normal_stddev
  m_config.batch_norm_decay = params.batch_norm_decay

  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    # override the dataset params with CLI params
    if params.data_dir:
      d_config.data_dir = params.data_dir
    global_batch_size = params.batch_size * strategy.num_replicas_in_sync

    # override the dataset params with CLI params
    # for distributed training, update batch size
    d_config.batch_size = global_batch_size
    # determine whether one_hot is used based on label_smoothing
    d_config.one_hot = params.label_smoothing and params.label_smoothing > 0

    # build train dataset
    train_dataset = get_dataset(d_config)
    # build validation dataset
    d_config.split = 'validation'
    eval_dataset = get_dataset(d_config)

    # compute number iterations per epoch
    steps_per_epoch = d_config.num_examples // d_config.batch_size
    eval_steps = d_config.num_eval_examples // d_config.batch_size

    # build the model
    keras_model = build_model(
      model_name=params.model_name,
      dataset_config=d_config,
      model_config=m_config
    )

    # build the optimizer
    learning_params = defaults.LR_CONFIG_DEFAULT
    learning_params.update({'initial_lr': params.lr,
                            'decay_epochs': params.lr_decay_epochs,
                            'decay_rate': params.lr_decay_rate})
    optimizer_params = defaults.OP_CONFIG_DEFAULT
    optimizer_params.update({'decay': params.op_decay_rate,
                             'momentum': params.op_momentum})
    optimizer = _get_optimizer(
      batch_size=global_batch_size,
      steps_per_epoch=steps_per_epoch,
      lr_name=params.learning_scheduler_name,
      optimizer_name=params.optimizer_name,
      lr_params=learning_params,
      optimizer_params=optimizer_params
    )

    logging.info('Exponential decay rate:{}'.format(params.ma_decay_rate))
    if params.ma_decay_rate:
      optimizer = tfa.optimizers.MovingAverage(
        optimizer=optimizer,
        average_decay=params.ma_decay_rate)

    # compile model
    if d_config.one_hot:
      loss_obj = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=params.label_smoothing)
    else:
      loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()

    keras_model.compile(
      optimizer=optimizer,
      loss=loss_obj,
      metrics=[_get_metrics(one_hot=d_config.one_hot)['acc']],
    )

    logging.info(keras_model.summary())

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
    validation_steps=eval_steps,
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
