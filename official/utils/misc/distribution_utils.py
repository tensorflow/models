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

import tensorflow as tf


def get_distribution_strategy(num_gpus,
                              all_reduce_alg=None,
                              turn_off_distribution_strategy=False):
  """Return a DistributionStrategy for running the model.

  Args:
    num_gpus: Number of GPUs to run this model.
    all_reduce_alg: Specify which algorithm to use when performing all-reduce.
      See tf.contrib.distribute.AllReduceCrossDeviceOps for available
      algorithms. If None, DistributionStrategy will choose based on device
      topology.
    turn_off_distribution_strategy: when set to True, do not use any
      distribution strategy. Note that when it is True, and num_gpus is
      larger than 1, it will raise a ValueError.

  Returns:
    tf.contrib.distribute.DistibutionStrategy object.
  Raises:
    ValueError: if turn_off_distribution_strategy is True and num_gpus is
    larger than 1
  """
  if num_gpus == 0:
    if turn_off_distribution_strategy:
      return None
    else:
      return tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")
  elif num_gpus == 1:
    if turn_off_distribution_strategy:
      return None
    else:
      return tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")
  elif turn_off_distribution_strategy:
    raise ValueError("When {} GPUs are specified, "
                     "turn_off_distribution_strategy flag cannot be set to"
                     "True.".format(num_gpus))
  else:  # num_gpus > 1 and not turn_off_distribution_strategy
    devices = ["device:GPU:%d" % i for i in range(num_gpus)]
    if all_reduce_alg:
      return tf.distribute.MirroredStrategy(
          devices=devices,
          cross_device_ops=tf.contrib.distribute.AllReduceCrossDeviceOps(
              all_reduce_alg, num_packs=2))
    else:
      return tf.distribute.MirroredStrategy(devices=devices)


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
    err = ("When running with multiple GPUs, batch size "
           "must be a multiple of the number of available GPUs. Found {} "
           "GPUs with a batch size of {}; try --batch_size={} instead."
          ).format(num_gpus, batch_size, batch_size - remainder)
    raise ValueError(err)
  return int(batch_size / num_gpus)
