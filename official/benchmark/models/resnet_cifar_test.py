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
"""Test the keras ResNet model with Cifar data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.platform import googletest
from official.benchmark.models import cifar_preprocessing
from official.benchmark.models import resnet_cifar_main
from official.utils.testing import integration


class KerasCifarTest(googletest.TestCase):
  """Unit tests for Keras ResNet with Cifar."""

  _extra_flags = [
      "-batch_size", "4",
      "-train_steps", "1",
      "-use_synthetic_data", "true"
  ]
  _tempdir = None

  def get_temp_dir(self):
    if not self._tempdir:
      self._tempdir = tempfile.mkdtemp(dir=googletest.GetTempDir())
    return self._tempdir

  @classmethod
  def setUpClass(cls):  # pylint: disable=invalid-name
    super(KerasCifarTest, cls).setUpClass()
    resnet_cifar_main.define_cifar_flags()

  def setUp(self):
    super(KerasCifarTest, self).setUp()
    cifar_preprocessing.NUM_IMAGES["validation"] = 4

  def tearDown(self):
    super(KerasCifarTest, self).tearDown()
    tf.io.gfile.rmtree(self.get_temp_dir())

  def test_end_to_end_no_dist_strat(self):
    """Test Keras model with 1 GPU, no distribution strategy."""

    extra_flags = [
        "-distribution_strategy", "off",
        "-model_dir", "keras_cifar_no_dist_strat",
        "-data_format", "channels_last",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
        main=resnet_cifar_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

  def test_end_to_end_graph_no_dist_strat(self):
    """Test Keras model in legacy graph mode with 1 GPU, no dist strat."""
    extra_flags = [
        "-enable_eager", "false",
        "-distribution_strategy", "off",
        "-model_dir", "keras_cifar_graph_no_dist_strat",
        "-data_format", "channels_last",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
        main=resnet_cifar_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

  def test_end_to_end_1_gpu(self):
    """Test Keras model with 1 GPU."""

    if context.num_gpus() < 1:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(1, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "1",
        "-distribution_strategy", "mirrored",
        "-model_dir", "keras_cifar_1_gpu",
        "-data_format", "channels_last",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
        main=resnet_cifar_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

  def test_end_to_end_graph_1_gpu(self):
    """Test Keras model in legacy graph mode with 1 GPU."""
    if context.num_gpus() < 1:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(1, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "1",
        "-noenable_eager",
        "-distribution_strategy", "mirrored",
        "-model_dir", "keras_cifar_graph_1_gpu",
        "-data_format", "channels_last",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
        main=resnet_cifar_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

  def test_end_to_end_2_gpu(self):
    """Test Keras model with 2 GPUs."""

    if context.num_gpus() < 2:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(2, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "2",
        "-distribution_strategy", "mirrored",
        "-model_dir", "keras_cifar_2_gpu",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
        main=resnet_cifar_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

  def test_end_to_end_graph_2_gpu(self):
    """Test Keras model in legacy graph mode with 2 GPUs."""
    if context.num_gpus() < 2:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(2, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "2",
        "-enable_eager", "false",
        "-distribution_strategy", "mirrored",
        "-model_dir", "keras_cifar_graph_2_gpu",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
        main=resnet_cifar_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )


if __name__ == "__main__":
  googletest.main()
