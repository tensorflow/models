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
""" Tests for distribution util functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.misc import distribution_utils


class GetDistributionStrategyTest(tf.test.TestCase):
  """Tests for get_distribution_strategy."""
  def test_one_device_strategy_cpu(self):
    ds = distribution_utils.get_distribution_strategy(num_gpus=0)
    self.assertEquals(ds.num_replicas_in_sync, 1)
    self.assertEquals(len(ds.extended.worker_devices), 1)
    self.assertIn('CPU', ds.extended.worker_devices[0])

  def test_one_device_strategy_gpu(self):
    ds = distribution_utils.get_distribution_strategy(num_gpus=1)
    self.assertEquals(ds.num_replicas_in_sync, 1)
    self.assertEquals(len(ds.extended.worker_devices), 1)
    self.assertIn('GPU', ds.extended.worker_devices[0])

  def test_mirrored_strategy(self):
    ds = distribution_utils.get_distribution_strategy(num_gpus=5)
    self.assertEquals(ds.num_replicas_in_sync, 5)
    self.assertEquals(len(ds.extended.worker_devices), 5)
    for device in ds.extended.worker_devices:
      self.assertIn('GPU', device)


class PerDeviceBatchSizeTest(tf.test.TestCase):
  """Tests for per_device_batch_size."""

  def test_batch_size(self):
    self.assertEquals(
        distribution_utils.per_device_batch_size(147, num_gpus=0), 147)
    self.assertEquals(
        distribution_utils.per_device_batch_size(147, num_gpus=1), 147)
    self.assertEquals(
        distribution_utils.per_device_batch_size(147, num_gpus=7), 21)

  def test_batch_size_with_remainder(self):
    with self.assertRaises(ValueError):
        distribution_utils.per_device_batch_size(147, num_gpus=5)


if __name__ == "__main__":
  tf.test.main()
