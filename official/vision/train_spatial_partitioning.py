# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""TensorFlow Model Garden Vision training driver with spatial partitioning."""
from typing import Sequence

from absl import app
from absl import flags
import gin
import numpy as np
import tensorflow as tf, tf_keras

from official.common import distribute_utils
from official.common import flags as tfm_flags
from official.core import task_factory
from official.core import train_lib
from official.core import train_utils
from official.modeling import performance
from official.vision import registry_imports  # pylint: disable=unused-import


FLAGS = flags.FLAGS


def get_computation_shape_for_model_parallelism(
    input_partition_dims: Sequence[int]) -> Sequence[int]:
  """Returns computation shape to be used for TPUStrategy spatial partition.

  Args:
    input_partition_dims: The number of partitions along each dimension.

  Returns:
    A list of integers specifying the computation shape.

  Raises:
    ValueError: If the number of logical devices is not supported.
  """
  num_logical_devices = np.prod(input_partition_dims)
  if num_logical_devices == 1:
    return [1, 1, 1, 1]
  elif num_logical_devices == 2:
    return [1, 1, 1, 2]
  elif num_logical_devices == 4:
    return [1, 2, 1, 2]
  elif num_logical_devices == 8:
    return [2, 2, 1, 2]
  elif num_logical_devices == 16:
    return [4, 2, 1, 2]
  else:
    raise ValueError(
        'The number of logical devices %d is not supported. Supported numbers '
        'are 1, 2, 4, 8, 16' % num_logical_devices)


def create_distribution_strategy(distribution_strategy,
                                 tpu_address,
                                 input_partition_dims=None,
                                 num_gpus=None):
  """Creates distribution strategy to use for computation."""

  if input_partition_dims is not None:
    if distribution_strategy != 'tpu':
      raise ValueError('Spatial partitioning is only supported '
                       'for TPUStrategy.')

    # When `input_partition_dims` is specified create custom TPUStrategy
    # instance with computation shape for model parallelism.
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=tpu_address)
    if tpu_address not in ('', 'local'):
      tf.config.experimental_connect_to_cluster(resolver)

    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    num_replicas = resolver.get_tpu_system_metadata().num_cores // np.prod(
        input_partition_dims)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(
        topology,
        num_replicas=num_replicas,
        computation_shape=input_partition_dims)
    return tf.distribute.TPUStrategy(
        resolver, experimental_device_assignment=device_assignment)

  return distribute_utils.get_distribution_strategy(
      distribution_strategy=distribution_strategy,
      tpu_address=tpu_address,
      num_gpus=num_gpus)


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  params = train_utils.parse_configuration(FLAGS)
  model_dir = FLAGS.model_dir
  if 'train' in FLAGS.mode:
    # Pure eval modes do not output yaml files. Otherwise continuous eval job
    # may race against the train job for writing the same file.
    train_utils.serialize_config(params, model_dir)

  # Sets mixed_precision policy. Using 'mixed_float16' or 'mixed_bfloat16'
  # can have significant impact on model speeds by utilizing float16 in case of
  # GPUs, and bfloat16 in the case of TPUs. loss_scale takes effect only when
  # dtype is float16
  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype)

  input_partition_dims = None
  if FLAGS.mode == 'train_and_eval':
    if np.prod(params.task.train_input_partition_dims) != np.prod(
        params.task.eval_input_partition_dims):
      raise ValueError('Train and eval input partition dims can not be'
                       'partitioned on the same node')
    else:
      input_partition_dims = get_computation_shape_for_model_parallelism(
          params.task.train_input_partition_dims)
  elif FLAGS.mode == 'train':
    if params.task.train_input_partition_dims:
      input_partition_dims = get_computation_shape_for_model_parallelism(
          params.task.train_input_partition_dims)
  elif FLAGS.mode == 'eval' or FLAGS.mode == 'continuous_eval':
    if params.task.eval_input_partition_dims:
      input_partition_dims = get_computation_shape_for_model_parallelism(
          params.task.eval_input_partition_dims)

  distribution_strategy = create_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      num_gpus=params.runtime.num_gpus,
      input_partition_dims=input_partition_dims,
      tpu_address=params.runtime.tpu)
  with distribution_strategy.scope():
    task = task_factory.get_task(params.task, logging_dir=model_dir)

  train_lib.run_experiment(
      distribution_strategy=distribution_strategy,
      task=task,
      mode=FLAGS.mode,
      params=params,
      model_dir=model_dir)

if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(main)
