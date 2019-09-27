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

from official.utils.misc import tpu_lib


def _collective_communication(all_reduce_alg):
  """Return a CollectiveCommunication based on all_reduce_alg.

  Args:
    all_reduce_alg: a string specifying which collective communication to pick,
      or None.

  Returns:
    tf.distribute.experimental.CollectiveCommunication object

  Raises:
    ValueError: if `all_reduce_alg` not in [None, 'ring', 'nccl']
  """
  collective_communication_options = {
      None: tf.distribute.experimental.CollectiveCommunication.AUTO,
      "ring": tf.distribute.experimental.CollectiveCommunication.RING,
      "nccl": tf.distribute.experimental.CollectiveCommunication.NCCL
  }
  if all_reduce_alg not in collective_communication_options:
    raise ValueError(
        "When used with `multi_worker_mirrored`, valid values for "
        "all_reduce_alg are ['ring', 'nccl'].  Supplied value: {}".format(
            all_reduce_alg))
  return collective_communication_options[all_reduce_alg]


def _mirrored_cross_device_ops(all_reduce_alg, num_packs):
  """Return a CrossDeviceOps based on all_reduce_alg and num_packs.

  Args:
    all_reduce_alg: a string specifying which cross device op to pick, or None.
    num_packs: an integer specifying number of packs for the cross device op.

  Returns:
    tf.distribute.CrossDeviceOps object or None.

  Raises:
    ValueError: if `all_reduce_alg` not in [None, 'nccl', 'hierarchical_copy'].
  """
  if all_reduce_alg is None:
    return None
  mirrored_all_reduce_options = {
      "nccl": tf.distribute.NcclAllReduce,
      "hierarchical_copy": tf.distribute.HierarchicalCopyAllReduce
  }
  if all_reduce_alg not in mirrored_all_reduce_options:
    raise ValueError(
        "When used with `mirrored`, valid values for all_reduce_alg are "
        "['nccl', 'hierarchical_copy'].  Supplied value: {}".format(
            all_reduce_alg))
  cross_device_ops_class = mirrored_all_reduce_options[all_reduce_alg]
  return cross_device_ops_class(num_packs=num_packs)


def get_distribution_strategy(distribution_strategy="default",
                              num_gpus=0,
                              num_workers=1,
                              all_reduce_alg=None,
                              num_packs=1,
                              tpu_address=None):
  """Return a DistributionStrategy for running the model.

  Args:
    distribution_strategy: a string specifying which distribution strategy to
      use. Accepted values are 'off', 'default', 'one_device', 'mirrored',
      'parameter_server', 'multi_worker_mirrored', and 'tpu' -- case insensitive.
      'off' means not to use Distribution Strategy; 'default' means to choose from
      `MirroredStrategy`, `MultiWorkerMirroredStrategy`, or `OneDeviceStrategy`
      according to the number of GPUs and number of workers. 'tpu' means to use
      TPUStrategy using `tpu_address`.
    num_gpus: Number of GPUs to run this model.
    num_workers: Number of workers to run this model.
    all_reduce_alg: Optional. Specifies which algorithm to use when performing
      all-reduce. For `MirroredStrategy`, valid values are "nccl" and
      "hierarchical_copy". For `MultiWorkerMirroredStrategy`, valid values are
      "ring" and "nccl".  If None, DistributionStrategy will choose based on
      device topology.
    num_packs: Optional.  Sets the `num_packs` in `tf.distribute.NcclAllReduce`
      or `tf.distribute.HierarchicalCopyAllReduce` for `MirroredStrategy`.
    tpu_address: Optional. String that represents TPU to connect to. Must not
      be None if `distribution_strategy` is set to `tpu`.
  Returns:
    tf.distribute.DistibutionStrategy object.
  Raises:
    ValueError: if `distribution_strategy` is 'off' or 'one_device' and
      `num_gpus` is larger than 1; or `num_gpus` is negative or if
      `distribution_strategy` is `tpu` but `tpu_address` is not specified.
  """
  if num_gpus < 0:
    raise ValueError("`num_gpus` can not be negative.")

  distribution_strategy = distribution_strategy.lower()
  if distribution_strategy == "off":
    if num_gpus > 1:
      raise ValueError(
          "When {} GPUs and  {} workers are specified, distribution_strategy "
          "flag cannot be set to 'off'.".format(num_gpus, num_workers))
    return None

  if distribution_strategy == "tpu":
    # When tpu_address is an empty string, we communicate with local TPUs.
    cluster_resolver = tpu_lib.tpu_initialize(tpu_address)
    return tf.distribute.experimental.TPUStrategy(cluster_resolver)

  if distribution_strategy == "multi_worker_mirrored":
    return tf.distribute.experimental.MultiWorkerMirroredStrategy(
        communication=_collective_communication(all_reduce_alg))

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
    return tf.distribute.MirroredStrategy(
        devices=devices,
        cross_device_ops=_mirrored_cross_device_ops(all_reduce_alg, num_packs))

  if distribution_strategy == "parameter_server":
    return tf.distribute.experimental.ParameterServerStrategy()

  raise ValueError(
      "Unrecognized Distribution Strategy: %r" % distribution_strategy)


def per_replica_batch_size(batch_size, num_gpus):
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
    # dataset.take(1) doesn't have GPU kernel.
    with tf.device('device:CPU:0'):
      tensor = tf.data.experimental.get_single_element(dataset.take(1))
    flat_tensor = tf.nest.flatten(tensor)
    variable_data = []
    initializers = []
    for t in flat_tensor:
      rebatched_t = tf.split(t, num_or_size_splits=split_by, axis=0)[0]
      assert rebatched_t.shape.is_fully_defined(), rebatched_t.shape
      v = tf.compat.v1.get_local_variable(self._random_name(),
                                          initializer=rebatched_t)
      variable_data.append(v)
      initializers.append(v.initializer)
    input_data = tf.nest.pack_sequence_as(tensor, variable_data)
    self._iterator = SyntheticIterator(input_data, initializers)

  def _random_name(self, size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

  def __iter__(self):
    return self._iterator

  def make_one_shot_iterator(self):
    return self._iterator

  def make_initializable_iterator(self):
    return self._iterator


class SyntheticIterator(object):
  """A dataset that generates synthetic data on each device."""

  def __init__(self, input_data, initializers):
    self._input_data = input_data
    self._initializers = initializers

  def get_next(self):
    return self._input_data

  def next(self):
    return self.__next__()

  def __next__(self):
    try:
      return self.get_next()
    except tf.errors.OutOfRangeError:
      raise StopIteration

  def initialize(self):
    if tf.executing_eagerly():
      return tf.no_op()
    else:
      return self._initializers


def _monkey_patch_dataset_method(strategy):
  """Monkey-patch `strategy`'s `make_dataset_iterator` method."""
  def make_dataset(self, dataset):
    tf.compat.v1.logging.info('Using pure synthetic data.')
    with self.scope():
      if self.extended._global_batch_size:  # pylint: disable=protected-access
        return SyntheticDataset(dataset, self.num_replicas_in_sync)
      else:
        return SyntheticDataset(dataset)

  def make_iterator(self, dataset):
    dist_dataset = make_dataset(self, dataset)
    return iter(dist_dataset)

  strategy.orig_make_dataset_iterator = strategy.make_dataset_iterator
  strategy.make_dataset_iterator = make_iterator
  strategy.orig_distribute_dataset = strategy.experimental_distribute_dataset
  strategy.experimental_distribute_dataset = make_dataset


def _undo_monkey_patch_dataset_method(strategy):
  if hasattr(strategy, 'orig_make_dataset_iterator'):
    strategy.make_dataset_iterator = strategy.orig_make_dataset_iterator
  if hasattr(strategy, 'orig_distribute_dataset'):
    strategy.make_dataset_iterator = strategy.orig_distribute_dataset


def set_up_synthetic_data():
  _monkey_patch_dataset_method(tf.distribute.OneDeviceStrategy)
  _monkey_patch_dataset_method(tf.distribute.MirroredStrategy)
  _monkey_patch_dataset_method(
      tf.distribute.experimental.MultiWorkerMirroredStrategy)
  # TODO(tobyboyd): Remove when contrib.distribute is all in core.
  if hasattr(tf, 'contrib'):
    _monkey_patch_dataset_method(tf.contrib.distribute.MirroredStrategy)
    _monkey_patch_dataset_method(tf.contrib.distribute.OneDeviceStrategy)
    _monkey_patch_dataset_method(
        tf.contrib.distribute.CollectiveAllReduceStrategy)
  else:
    print('Contrib missing: Skip monkey patch tf.contrib.distribute.*')


def undo_set_up_synthetic_data():
  _undo_monkey_patch_dataset_method(tf.distribute.OneDeviceStrategy)
  _undo_monkey_patch_dataset_method(tf.distribute.MirroredStrategy)
  _undo_monkey_patch_dataset_method(
      tf.distribute.experimental.MultiWorkerMirroredStrategy)
  # TODO(tobyboyd): Remove when contrib.distribute is all in core.
  if hasattr(tf, 'contrib'):
    _undo_monkey_patch_dataset_method(tf.contrib.distribute.MirroredStrategy)
    _undo_monkey_patch_dataset_method(tf.contrib.distribute.OneDeviceStrategy)
    _undo_monkey_patch_dataset_method(
        tf.contrib.distribute.CollectiveAllReduceStrategy)
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
    num_workers = (len(tf_config['cluster'].get('chief', [])) +
                   len(tf_config['cluster'].get('worker', [])))
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


def get_strategy_scope(strategy):
  if strategy:
    strategy_scope = strategy.scope()
  else:
    strategy_scope = DummyContextManager()

  return strategy_scope


class DummyContextManager(object):

  def __enter__(self):
    pass

  def __exit__(self, *args):
    pass
