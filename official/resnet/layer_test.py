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
"""Test that the definitions of ResNet layers haven't changed.

These tests will fail if either:
  a)  The graph of a resnet layer changes and the change is significant enough
      that it can no longer load existing checkpoints.
  b)  The numerical results produced by the layer change.

A warning will be issued if the graph changes, but the checkpoint still loads.

In the event that a layer change is intended, or the TensorFlow implementation
of a layer changes (and thus changes the graph), regenerate using the command:

  $ python3 layer_test.py -regen
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

import tensorflow as tf   # pylint: disable=g-bad-import-order
from official.resnet import resnet_model
from official.utils.misc import keras_utils
from official.utils.testing import reference_data


DATA_FORMAT = "channels_last"  # CPU instructions often preclude channels_first
BATCH_SIZE = 32
BLOCK_TESTS = [
    dict(bottleneck=True, projection=True, resnet_version=1, width=8,
         channels=4),
    dict(bottleneck=True, projection=True, resnet_version=2, width=8,
         channels=4),
    dict(bottleneck=True, projection=False, resnet_version=1, width=8,
         channels=4),
    dict(bottleneck=True, projection=False, resnet_version=2, width=8,
         channels=4),
    dict(bottleneck=False, projection=True, resnet_version=1, width=8,
         channels=4),
    dict(bottleneck=False, projection=True, resnet_version=2, width=8,
         channels=4),
    dict(bottleneck=False, projection=False, resnet_version=1, width=8,
         channels=4),
    dict(bottleneck=False, projection=False, resnet_version=2, width=8,
         channels=4),
]


class BaseTest(reference_data.BaseTest):
  """Tests for core ResNet layers."""

  def setUp(self):
    super(BaseTest, self).setUp()
    if keras_utils.is_v2_0:
      tf.compat.v1.disable_eager_execution()

  @property
  def test_name(self):
    return "resnet"

  def _batch_norm_ops(self, test=False):
    name = "batch_norm"

    g = tf.Graph()
    with g.as_default():
      tf.compat.v1.set_random_seed(self.name_to_seed(name))
      input_tensor = tf.compat.v1.get_variable(
          "input_tensor", dtype=tf.float32,
          initializer=tf.random.uniform((32, 16, 16, 3), maxval=1)
      )
      layer = resnet_model.batch_norm(
          inputs=input_tensor, data_format=DATA_FORMAT, training=True)

    self._save_or_test_ops(
        name=name, graph=g, ops_to_eval=[input_tensor, layer], test=test,
        correctness_function=self.default_correctness_function
    )

  def make_projection(self, filters_out, strides, data_format):
    """1D convolution with stride projector.

    Args:
      filters_out: Number of filters in the projection.
      strides: Stride length for convolution.
      data_format: channels_first or channels_last

    Returns:
      A CNN projector function with kernel_size 1.
    """
    def projection_shortcut(inputs):
      return resnet_model.conv2d_fixed_padding(
          inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
          data_format=data_format)
    return projection_shortcut

  def _resnet_block_ops(self, test, batch_size, bottleneck, projection,
                        resnet_version, width, channels):
    """Test whether resnet block construction has changed.

    Args:
      test: Whether or not to run as a test case.
      batch_size: Number of points in the fake image. This is needed due to
        batch normalization.
      bottleneck: Whether or not to use bottleneck layers.
      projection: Whether or not to project the input.
      resnet_version: Which version of ResNet to test.
      width: The width of the fake image.
      channels: The number of channels in the fake image.
    """

    name = "batch-size-{}_{}{}_version-{}_width-{}_channels-{}".format(
        batch_size,
        "bottleneck" if bottleneck else "building",
        "_projection" if projection else "",
        resnet_version,
        width,
        channels
    )

    if resnet_version == 1:
      block_fn = resnet_model._building_block_v1
      if bottleneck:
        block_fn = resnet_model._bottleneck_block_v1
    else:
      block_fn = resnet_model._building_block_v2
      if bottleneck:
        block_fn = resnet_model._bottleneck_block_v2

    g = tf.Graph()
    with g.as_default():
      tf.compat.v1.set_random_seed(self.name_to_seed(name))
      strides = 1
      channels_out = channels
      projection_shortcut = None
      if projection:
        strides = 2
        channels_out *= strides
        projection_shortcut = self.make_projection(
            filters_out=channels_out, strides=strides, data_format=DATA_FORMAT)

      filters = channels_out
      if bottleneck:
        filters = channels_out // 4

      input_tensor = tf.compat.v1.get_variable(
          "input_tensor", dtype=tf.float32,
          initializer=tf.random.uniform((batch_size, width, width, channels),
                                        maxval=1)
      )

      layer = block_fn(inputs=input_tensor, filters=filters, training=True,
                       projection_shortcut=projection_shortcut, strides=strides,
                       data_format=DATA_FORMAT)

    self._save_or_test_ops(
        name=name, graph=g, ops_to_eval=[input_tensor, layer], test=test,
        correctness_function=self.default_correctness_function
    )

  @unittest.skipIf(tf.test.is_built_with_cuda(), "Results only match CPU.")
  def test_batch_norm(self):
    """Tests batch norm layer correctness.

    Test fails on a GTX 1080 with the last value being significantly different:
    7.629395e-05 (expected) -> -4.159546e-02 (actual). The tests passes on CPU
    on TF 1.0 and TF 2.0.
    """
    self._batch_norm_ops(test=True)

  def test_block_0(self):
    self._resnet_block_ops(test=True, batch_size=BATCH_SIZE, **BLOCK_TESTS[0])

  @unittest.skipIf(tf.test.is_built_with_cuda(), "Results only match CPU.")
  def test_block_1(self):
    """Test bottleneck=True, projection=False, resnet_version=1.

    Test fails on a GTX 1080 but would pass with tolerances moved from
    1e-06 to 1e-05. Being TF 1.0 and this was not setup as a GPU test originally
    it makes sense to disable it on GPU vs. research.
    """
    self._resnet_block_ops(test=True, batch_size=BATCH_SIZE, **BLOCK_TESTS[1])

  @unittest.skipIf(tf.test.is_built_with_cuda(), "Results only match CPU.")
  def test_block_2(self):
    """Test bottleneck=True, projection=True, resnet_version=2, width=8.

    Test fails on a GTX 1080 but would pass with tolerances moved from
    1e-06 to 1e-05. Being TF 1.0 and this was not setup as a GPU test originally
    it makes sense to disable it on GPU.
    """
    self._resnet_block_ops(test=True, batch_size=BATCH_SIZE, **BLOCK_TESTS[2])

  def test_block_3(self):
    self._resnet_block_ops(test=True, batch_size=BATCH_SIZE, **BLOCK_TESTS[3])

  def test_block_4(self):
    self._resnet_block_ops(test=True, batch_size=BATCH_SIZE, **BLOCK_TESTS[4])

  def test_block_5(self):
    self._resnet_block_ops(test=True, batch_size=BATCH_SIZE, **BLOCK_TESTS[5])

  def test_block_6(self):
    self._resnet_block_ops(test=True, batch_size=BATCH_SIZE, **BLOCK_TESTS[6])

  def test_block_7(self):
    self._resnet_block_ops(test=True, batch_size=BATCH_SIZE, **BLOCK_TESTS[7])

  def regenerate(self):
    """Create reference data files for ResNet layer tests."""
    self._batch_norm_ops(test=False)
    for block_params in BLOCK_TESTS:
      self._resnet_block_ops(test=False, batch_size=BATCH_SIZE, **block_params)


if __name__ == "__main__":
  reference_data.main(argv=sys.argv, test_class=BaseTest)
