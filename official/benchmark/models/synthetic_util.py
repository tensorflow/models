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
"""Helper functions to generate data directly on devices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import string

from absl import logging
import tensorflow as tf


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
    logging.info('Using pure synthetic data.')
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


def undo_set_up_synthetic_data():
  _undo_monkey_patch_dataset_method(tf.distribute.OneDeviceStrategy)
  _undo_monkey_patch_dataset_method(tf.distribute.MirroredStrategy)
  _undo_monkey_patch_dataset_method(
      tf.distribute.experimental.MultiWorkerMirroredStrategy)
