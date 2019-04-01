# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Helper functions for running models in a distributed setting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random
import string
import tensorflow as tf

_COLLECTIVE_COMMUNICATION_OPTIONS = {
    None: tf.distribute.experimental.CollectiveCommunication.AUTO,
    "ring": tf.distribute.experimental.CollectiveCommunication.RING,
    "nccl": tf.distribute.experimental.CollectiveCommunication.NCCL
}


def get_distribution_strategy(distribution_strategy="default",
                              num_gpus=0,
                              num_workers=1,
                              all_reduce_alg=None):
  """Return a DistributionStrategy for running the model.

  Args:
    distribution_strategy: a string specify which distribution strategy to use.
      Accepted values are 'off', 'default', 'one_device', 'mirrored',
      'parameter_server', 'multi_worker_mirrored', case insensitive. 'off' means
      not to use Distribution Strategy; 'default' means to choose from
      `MirroredStrategy`, `MultiWorkerMirroredStrategy`, or `OneDeviceStrategy`
      according to the number of GPUs and number of workers.
    num_gpus: Number of GPUs to run this model.
    num_workers: Number of workers to run this model.
    all_reduce_alg: Optional. Specify which algorithm to use when performing
      all-reduce. See tf.contrib.distribute.AllReduceCrossDeviceOps for
      available algorithms when used with `mirrored`, and
      tf.distribute.experimental.CollectiveCommunication when used with
      `multi_worker_mirrored`. If None, DistributionStrategy will choose based
      on device topology.

  Returns:
    tf.distribute.DistibutionStrategy object.
  Raises:
    ValueError: if `distribution_strategy` is 'off' or 'one_device' and
      `num_gpus` is larger than 1; or `num_gpus` is negative.
  """
  if num_gpus < 0:
    raise ValueError("`num_gpus` can not be negative.")

  distribution_strategy = distribution_strategy.lower()
  if distribution_strategy == "off":
    if num_gpus > 1 or num_workers > 1:
      raise ValueError(
          "When {} GPUs and  {} workers are specified, distribution_strategy "
          "flag cannot be set to 'off'.".format(num_gpus, num_workers))
    return None

  if distribution_strategy == "multi_worker_mirrored" or num_workers > 1:
    if all_reduce_alg not in _COLLECTIVE_COMMUNICATION_OPTIONS:
      raise ValueError(
          "When used with `multi_worker_mirrored`, valid values for "
          "all_reduce_alg are [`ring`, `nccl`].  Supplied value: {}".format(
              all_reduce_alg))
    return tf.distribute.experimental.MultiWorkerMirroredStrategy(
        communication=_COLLECTIVE_COMMUNICATION_OPTIONS[all_reduce_alg])

  if (distribution_strategy == "one_device" or
      (distribution_strategy == "default" and num_gpus <= 1)):
    if num_gpus == 0:
      return tf.distribute.OneDeviceStrategy("device:CPU:0")
    else:
      if num_gpus > 1:
        raise ValueError("`OneDeviceStrategy` can not be used for more than "
                         "one device.")
      return tf.distribute.OneDeviceStrategy("device:GPU:0")

  if distribution_strategy in ("mirrored", "default"):
    if num_gpus == 0:
      assert distribution_strategy == "mirrored"
      devices = ["device:CPU:0"]
    else:
      devices = ["device:GPU:%d" % i for i in range(num_gpus)]
    if all_reduce_alg:
      return tf.distribute.MirroredStrategy(
          devices=devices,
          cross_device_ops=tf.contrib.distribute.AllReduceCrossDeviceOps(
              all_reduce_alg, num_packs=2))
    else:
      return tf.distribute.MirroredStrategy(devices=devices)

  if distribution_strategy == "parameter_server":
    return tf.distribute.experimental.ParameterServerStrategy()

  raise ValueError(
      "Unrecognized Distribution Strategy: %r" % distribution_strategy)


def per_device_batch_size(batch_size, num_gpus):
  """For multi-gpu, batch-size must be a multiple of the number of GPUs.


  Note that distribution strategy handles this automatically when used with
  Keras. For using with Estimator, we need to get per GPU batch.

  Args:
    batch_size: Global batch size to be divided among devices. This should be
      equal to num_gpus times the single-GPU batch_size for multi-gpu training.
    num_gpus: How many GPUs are used with DistributionStrategies.

  Returns:
    Batch size per device.

  Raises:
    ValueError: if batch_size is not divisible by number of devices
  """
  if num_gpus <= 1:
    return batch_size

  remainder = batch_size % num_gpus
  if remainder:
    err = ('When running with multiple GPUs, batch size '
           'must be a multiple of the number of available GPUs. Found {} '
           'GPUs with a batch size of {}; try --batch_size={} instead.'
          ).format(num_gpus, batch_size, batch_size - remainder)
    raise ValueError(err)
  return int(batch_size / num_gpus)


# The `SyntheticDataset` is a temporary solution for generating synthetic data
# directly on devices. It is only useful for Keras with Distribution
# Strategies. We will have better support in `tf.data` or Distribution Strategy
# later.
class SyntheticDataset(object):
  """A dataset that generates synthetic data on each device."""

  def __init__(self, dataset, split_by=1):
    self._input_data = {}
    # dataset.take(1) doesn't have GPU kernel.
    with tf.device('device:CPU:0'):
      tensor = tf.data.experimental.get_single_element(dataset.take(1))
    flat_tensor = tf.nest.flatten(tensor)
    variable_data = []
    self._initializers = []
    for t in flat_tensor:
      rebatched_t = tf.split(t, num_or_size_splits=split_by, axis=0)[0]
      assert rebatched_t.shape.is_fully_defined(), rebatched_t.shape
      v = tf.compat.v1.get_local_variable(self.random_name(),
                                          initializer=rebatched_t)
      variable_data.append(v)
      self._initializers.append(v.initializer)
    self._input_data = tf.nest.pack_sequence_as(tensor, variable_data)

  def get_next(self):
    return self._input_data

  def initialize(self):
    if tf.executing_eagerly():
      return tf.no_op()
    else:
      return self._initializers

  def random_name(self, size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def _monkey_patch_dataset_method(strategy):
  """Monkey-patch `strategy`'s `make_dataset_iterator` method."""
  def make_dataset_iterator(self, dataset):
    tf.compat.v1.logging.info('Using pure synthetic data.')
    with self.scope():
      if self.extended._global_batch_size:  # pylint: disable=protected-access
        return SyntheticDataset(dataset, self.num_replicas_in_sync)
      else:
        return SyntheticDataset(dataset)

  strategy.org_make_dataset_iterator = strategy.make_dataset_iterator
  strategy.make_dataset_iterator = make_dataset_iterator


def _undo_monkey_patch_dataset_method(strategy):
  if hasattr(strategy, 'org_make_dataset_iterator'):
    strategy.make_dataset_iterator = strategy.org_make_dataset_iterator


def set_up_synthetic_data():
  _monkey_patch_dataset_method(tf.distribute.OneDeviceStrategy)
  _monkey_patch_dataset_method(tf.distribute.MirroredStrategy)
  # TODO(tobyboyd): Remove when contrib.distribute is all in core.
  if hasattr(tf, 'contrib'):
    _monkey_patch_dataset_method(tf.contrib.distribute.MirroredStrategy)
    _monkey_patch_dataset_method(tf.contrib.distribute.OneDeviceStrategy)
  else:
    print('Contrib missing: Skip monkey patch tf.contrib.distribute.*')


def undo_set_up_synthetic_data():
  _undo_monkey_patch_dataset_method(tf.distribute.OneDeviceStrategy)
  _undo_monkey_patch_dataset_method(tf.distribute.MirroredStrategy)
  # TODO(tobyboyd): Remove when contrib.distribute is all in core.
  if hasattr(tf, 'contrib'):
    _undo_monkey_patch_dataset_method(tf.contrib.distribute.MirroredStrategy)
    _undo_monkey_patch_dataset_method(tf.contrib.distribute.OneDeviceStrategy)
  else:
    print('Contrib missing: Skip remove monkey patch tf.contrib.distribute.*')


def configure_cluster(worker_hosts=None, task_index=-1):
  """Set multi-worker cluster spec in TF_CONFIG environment variable.

  Args:
    worker_hosts: comma-separated list of worker ip:port pairs.

  Returns:
    Number of workers in the cluster.
  """
  tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
  if tf_config:
    num_workers = len(tf_config['cluster']['worker'])
    if tf_config['cluster'].get('chief', None):
      num_workers += 1
  elif worker_hosts:
    workers = worker_hosts.split(',')
    num_workers = len(workers)
    if num_workers > 1 and task_index < 0:
      raise ValueError('Must specify task_index when number of workers > 1')
    task_index = 0 if num_workers == 1 else task_index
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': workers
        },
        'task': {'type': 'worker', 'index': task_index}
    })
  else:
    num_workers = 1
  return num_workers
