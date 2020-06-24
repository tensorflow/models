# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
r"""Training script for UNet-3D."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import functools
import os

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from official.modeling.hyperparams import params_dict
from official.utils import hyperparams_flags
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.vision.segmentation import unet_config
from official.vision.segmentation import unet_data
from official.vision.segmentation import unet_metrics
from official.vision.segmentation import unet_model as unet_model_lib


def define_unet3d_flags():
  """Defines flags for training 3D Unet."""
  hyperparams_flags.initialize_common_flags()

  flags.DEFINE_enum(
      'distribution_strategy', 'tpu', ['tpu', 'mirrored'],
      'Distribution Strategy type to use for training. `tpu` uses TPUStrategy '
      'for running on TPUs, `mirrored` uses GPUs with single host.')
  flags.DEFINE_integer(
      'steps_per_loop', 50,
      'Number of steps to execute in a loop for performance optimization.')
  flags.DEFINE_integer('checkpoint_interval', 100,
                       'Minimum step interval between two checkpoints.')
  flags.DEFINE_integer('epochs', 10, 'Number of epochs to run training.')
  flags.DEFINE_string(
      'gcp_project',
      default=None,
      help='Project name for the Cloud TPU-enabled project. If not specified, we '
      'will attempt to automatically detect the GCE project from metadata.')
  flags.DEFINE_string(
      'eval_checkpoint_dir',
      default=None,
      help='Directory for reading checkpoint file when `mode` == `eval`.')
  flags.DEFINE_multi_integer(
      'input_partition_dims', [1],
      'A list that describes the partition dims for all the tensors.')
  flags.DEFINE_string(
      'mode', 'train', 'Mode to run: train or eval or train_and_eval '
      '(default: train)')
  flags.DEFINE_string('training_file_pattern', None,
                      'Location of the train data.')
  flags.DEFINE_string('eval_file_pattern', None, 'Location of ther eval data')
  flags.DEFINE_float('lr_init_value', 0.0001, 'Initial learning rate.')
  flags.DEFINE_float('lr_decay_rate', 0.9, 'Learning rate decay rate.')
  flags.DEFINE_integer('lr_decay_steps', 100, 'Learning rate decay steps.')


def save_params(params):
  """Save parameters to config files if model_dir is defined."""
  model_dir = params.model_dir
  assert model_dir is not None
  if not tf.io.gfile.exists(model_dir):
    tf.io.gfile.makedirs(model_dir)
  file_name = os.path.join(model_dir, 'params.yaml')
  params_dict.save_params_dict_to_yaml(params, file_name)


def extract_params(flags_obj):
  """Extract configuration parameters for training and evaluation."""
  params = params_dict.ParamsDict(unet_config.UNET_CONFIG,
                                  unet_config.UNET_RESTRICTIONS)

  params = params_dict.override_params_dict(
      params, flags_obj.config_file, is_strict=False)

  if flags_obj.training_file_pattern:
    params.override({'training_file_pattern': flags_obj.training_file_pattern},
                    is_strict=True)
  if flags_obj.eval_file_pattern:
    params.override({'eval_file_pattern': flags_obj.eval_file_pattern},
                    is_strict=True)

  train_epoch_steps = params.train_item_count // params.train_batch_size
  eval_epoch_steps = params.eval_item_count // params.eval_batch_size

  params.override(
      {
          'model_dir': flags_obj.model_dir,
          'eval_checkpoint_dir': flags_obj.eval_checkpoint_dir,
          'mode': flags_obj.mode,
          'distribution_strategy': flags_obj.distribution_strategy,
          'tpu': flags_obj.tpu,
          'num_gpus': flags_obj.num_gpus,
          'init_learning_rate': flags_obj.lr_init_value,
          'lr_decay_rate': flags_obj.lr_decay_rate,
          'lr_decay_steps': train_epoch_steps,
          'train_epoch_steps': train_epoch_steps,
          'eval_epoch_steps': eval_epoch_steps,
          'steps_per_loop': flags_obj.steps_per_loop,
          'epochs': flags_obj.epochs,
          'checkpoint_interval': flags_obj.checkpoint_interval,
      },
      is_strict=False)

  params.validate()
  params.lock()
  return params


def unet3d_callbacks(params, checkpoint_manager=None):
  """Custom callbacks during training."""
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=params.model_dir)

  if checkpoint_manager:
    checkpoint_callback = keras_utils.SimpleCheckpoint(checkpoint_manager)
    return [tensorboard_callback, checkpoint_callback]
  else:
    return [tensorboard_callback]


def get_computation_shape_for_model_parallelism(input_partition_dims):
  """Return computation shape to be used for TPUStrategy spatial partition."""
  num_logical_devices = np.prod(input_partition_dims)
  if num_logical_devices == 1:
    return [1, 1, 1, 1]
  if num_logical_devices == 2:
    return [1, 1, 1, 2]
  if num_logical_devices == 4:
    return [1, 2, 1, 2]
  if num_logical_devices == 8:
    return [2, 2, 1, 2]
  if num_logical_devices == 16:
    return [4, 2, 1, 2]

  raise ValueError('Unsupported number of spatial partition configuration.')


def create_distribution_strategy(params):
  """Creates distribution strategy to use for computation."""

  if params.input_partition_dims is not None:
    if params.distribution_strategy != 'tpu':
      raise ValueError('Spatial partitioning is only supported '
                       'for TPUStrategy.')

    # When `input_partition_dims` is specified create custom TPUStrategy
    # instance with computation shape for model parallelism.
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=params.tpu)
    if params.tpu not in ('', 'local'):
      tf.config.experimental_connect_to_cluster(resolver)

    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    num_replicas = resolver.get_tpu_system_metadata().num_cores // np.prod(
        params.input_partition_dims)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(
        topology,
        num_replicas=num_replicas,
        computation_shape=get_computation_shape_for_model_parallelism(
            params.input_partition_dims))
    return tf.distribute.experimental.TPUStrategy(
        resolver, device_assignment=device_assignment)

  return distribution_utils.get_distribution_strategy(
      distribution_strategy=params.distribution_strategy,
      tpu_address=params.tpu,
      num_gpus=params.num_gpus)


def get_train_dataset(params, ctx=None):
  """Returns training dataset."""
  return unet_data.LiverInput(
      params.training_file_pattern, params, is_training=True)(
          ctx)


def get_eval_dataset(params, ctx=None):
  """Returns evaluation dataset."""
  return unet_data.LiverInput(
      params.training_file_pattern, params, is_training=False)(
          ctx)


def expand_1d(data):
  """Expands 1-dimensional `Tensor`s into 2-dimensional `Tensor`s."""

  def _expand_single_1d_tensor(t):
    if (isinstance(t, tf.Tensor) and isinstance(t.shape, tf.TensorShape) and
        t.shape.rank == 1):
      return tf.expand_dims(t, axis=-1)
    return t

  return tf.nest.map_structure(_expand_single_1d_tensor, data)


def train_step(train_fn, input_partition_dims, data):
  """The logic for one training step with spatial partitioning."""
  # Keras expects rank 2 inputs. As so, expand single rank inputs.
  data = expand_1d(data)
  x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

  if input_partition_dims:
    strategy = tf.distribute.get_strategy()
    x = strategy.experimental_split_to_logical_devices(x, input_partition_dims)
    y = strategy.experimental_split_to_logical_devices(y, input_partition_dims)

  partitioned_data = tf.keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
  return train_fn(partitioned_data)


def test_step(test_fn, input_partition_dims, data):
  """The logic for one testing step with spatial partitioning."""
  # Keras expects rank 2 inputs. As so, expand single rank inputs.
  data = expand_1d(data)
  x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

  if input_partition_dims:
    strategy = tf.distribute.get_strategy()
    x = strategy.experimental_split_to_logical_devices(x, input_partition_dims)
    y = strategy.experimental_split_to_logical_devices(y, input_partition_dims)

  partitioned_data = tf.keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
  return test_fn(partitioned_data)


def train(params, strategy, unet_model, train_input_fn, eval_input_fn):
  """Trains 3D Unet model."""
  assert tf.distribute.has_strategy()

  # Override Keras Model's train_step() and test_step() function so
  # that inputs are spatially partitioned.
  # Note that is `predict()` API is used, then `predict_step()` should also
  # be overriden.
  unet_model.train_step = functools.partial(train_step, unet_model.train_step,
                                            params.input_partition_dims)
  unet_model.test_step = functools.partial(test_step, unet_model.test_step,
                                           params.input_partition_dims)

  optimizer = unet_model_lib.create_optimizer(params.init_learning_rate, params)
  loss_fn = unet_metrics.get_loss_fn(params.mode, params)
  unet_model.compile(
      loss=loss_fn,
      optimizer=optimizer,
      metrics=[unet_metrics.metric_accuracy],
      experimental_steps_per_execution=params.steps_per_loop)

  train_ds = strategy.experimental_distribute_datasets_from_function(
      train_input_fn)
  eval_ds = strategy.experimental_distribute_datasets_from_function(
      eval_input_fn)

  checkpoint = tf.train.Checkpoint(model=unet_model)

  train_epoch_steps = params.train_item_count // params.train_batch_size
  eval_epoch_steps = params.eval_item_count // params.eval_batch_size

  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint,
      directory=params.model_dir,
      max_to_keep=10,
      step_counter=unet_model.optimizer.iterations,
      checkpoint_interval=params.checkpoint_interval)
  checkpoint_manager.restore_or_initialize()

  train_result = unet_model.fit(
      x=train_ds,
      epochs=params.epochs,
      steps_per_epoch=train_epoch_steps,
      validation_data=eval_ds,
      validation_steps=eval_epoch_steps,
      callbacks=unet3d_callbacks(params, checkpoint_manager))
  return train_result


def evaluate(params, strategy, unet_model, input_fn):
  """Reads from checkpoint and evaluate 3D Unet model."""
  assert tf.distribute.has_strategy()

  unet_model.compile(
      metrics=[unet_metrics.metric_accuracy],
      experimental_steps_per_execution=params.steps_per_loop)

  # Override test_step() function so that inputs are spatially partitioned.
  unet_model.test_step = functools.partial(test_step, unet_model.test_step,
                                           params.input_partition_dims)

  # Load checkpoint for evaluation.
  checkpoint = tf.train.Checkpoint(model=unet_model)
  checkpoint_path = tf.train.latest_checkpoint(params.eval_checkpoint_dir)
  status = checkpoint.restore(checkpoint_path)
  status.assert_existing_objects_matched()

  eval_ds = strategy.experimental_distribute_datasets_from_function(input_fn)
  eval_epoch_steps = params.eval_item_count // params.eval_batch_size

  eval_result = unet_model.evaluate(
      x=eval_ds, steps=eval_epoch_steps, callbacks=unet3d_callbacks(params))
  return eval_result


def main(_):
  params = extract_params(flags.FLAGS)
  assert params.mode in {'train', 'eval'}, 'only support train and eval'
  save_params(params)

  input_dtype = params.dtype
  if input_dtype == 'float16' or input_dtype == 'bfloat16':
    policy = tf.keras.mixed_precision.experimental.Policy(
        'mixed_bfloat16' if input_dtype == 'bfloat16' else 'mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  strategy = create_distribution_strategy(params)
  with strategy.scope():
    unet_model = unet_model_lib.build_unet_model(params)

    if params.mode == 'train':
      train(params, strategy, unet_model,
            functools.partial(get_train_dataset, params),
            functools.partial(get_eval_dataset, params))

    elif params.mode == 'eval':
      evaluate(params, strategy, unet_model,
               functools.partial(get_eval_dataset, params))

    else:
      raise Exception('Only `train` mode and `eval` mode are supported.')


if __name__ == '__main__':
  define_unet3d_flags()
  app.run(main)
