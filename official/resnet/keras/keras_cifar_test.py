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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tempfile import mkdtemp

import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.resnet import cifar10_main
from official.resnet.keras import keras_cifar_main
from official.resnet.keras import keras_common
from official.resnet.keras import keras_test_base
from official.utils.testing import integration
from tensorflow.python.platform import googletest

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class KerasCifarTest(keras_test_base.KerasTestBase)
  """Unit tests for Keras ResNet with Cifar."""

  def test_cifar10_end_to_end_1_gpu_no_dist_strat(self):
    """Test Keras model with 1 GPU, no distribution strategy."""
    extra_flags = [
        "-num_gpus", "1",
        "-enable_eager", "true",
        "-distribution_strategy", "off",
        "-model_dir", "cifar_1_gpu_no_dist_strat",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_cifar_main.main,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_cifar10_end_to_end_graph_1_gpu_no_dist_strat(self):
    """Test Keras model in legacy graph mode with 1 GPU, no dist strat."""
    extra_flags = [
        "-num_gpus", "1",
        "-enable_eager", "false",
        "-distribution_strategy", "off",
        "-model_dir", "cifar_graph_1_gpu_no_dist_strat",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_cifar_main.main,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_cifar10_end_to_end_1_gpu(self):
    """Test Keras model with 1 GPU."""
    extra_flags = [
        "-num_gpus", "1",
        "-enable_eager", "true",
        "-distribution_strategy", "default",
        "-model_dir", "cifar_1_gpu",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_cifar_main.main,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_cifar10_end_to_end_xla_1_gpu(self):
    """Test Keras model with XLA and 1 GPU."""
    extra_flags = [
        "-num_gpus", "1",
        "-enable_eager", "true",
        "-enable_xla", "true",
        "-distribution_strategy", "default",
        "-model_dir", "cifar_xla_1_gpu",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_cifar_main.main,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_cifar10_end_to_end_1_gpu_fp16(self):
    """Test Keras model with 1 GPU and fp16."""
    extra_flags = [
        "-num_gpus", "1",
        "-enable_eager", "true",
        "-distribution_strategy", "default",
        "-model_dir", "cifar_1_gpu_fp16",
        "-dtype", "fp16",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_cifar_main.main,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_cifar10_end_to_end_xla_1_gpu_fp16(self):
    """Test Keras model with XLA, 1 GPU and fp16."""
    extra_flags = [
        "-num_gpus", "1",
        "-enable_eager", "true",
        "-enable_xla", "true",
        "-distribution_strategy", "default",
        "-model_dir", "cifar_xla_1_gpu_fp16",
        "-dtype", "fp16",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_cifar_main.main,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_cifar10_end_to_end_graph_1_gpu(self):
    """Test Keras model in legacy graph mode with 1 GPU."""
    extra_flags = [
        "-num_gpus", "1",
        "-enable_eager", "false",
        "-distribution_strategy", "default",
        "-model_dir", "cifar_graph_1_gpu",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_cifar_main.main,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_cifar10_end_to_end_graph_xla_1_gpu(self):
    """Test Keras model in legacy graph mode with XLA and 1 GPU."""
    extra_flags = [
        "-num_gpus", "1",
        "-enable_eager", "false",
        "-enable_xla", "true",
        "-distribution_strategy", "default",
        "-model_dir", "cifar_graph_xla_1_gpu",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_cifar_main.main,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_cifar10_end_to_end_2_gpu(self):
    """Test Keras model with 2 GPUs."""
    extra_flags = [
        "-num_gpus", "2",
        "-enable_eager", "true",
        "-distribution_strategy", "default",
        "-model_dir", "cifar_2_gpu",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_cifar_main.main,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_cifar10_end_to_end_2_gpu_tweaked(self):
    """Test Keras model with manual config tuning and 2 GPUs."""
    extra_flags = [
        "-num_gpus", "2",
        "-enable_eager", "true",
        "-distribution_strategy", "default",
        "-model_dir", "cifar_2_gpu_tweaked",
        "-datasets_num_private_threads", "14",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_cifar_main.main,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_cifar10_end_to_end_xla_2_gpu(self):
    """Test Keras model with XLA and 2 GPUs."""
    extra_flags = [
        "-num_gpus", "2",
        "-enable_eager", "true",
        "-enable_xla", "true",
        "-distribution_strategy", "default",
        "-model_dir", "cifar_xla_2_gpu",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_cifar_main.main,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_cifar10_end_to_end_2_gpu_fp16(self):
    """Test Keras model with 2 GPUs and fp16."""
    extra_flags = [
        "-num_gpus", "2",
        "-dtype", "fp16",
        "-enable_eager", "true",
        "-distribution_strategy", "default",
        "-model_dir", "cifar_2_gpu_fp16",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_cifar_main.main,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_cifar10_end_to_end_2_gpu_fp16_tweaked(self):
    """Test Keras model with 2 GPUs and fp16."""
    extra_flags = [
        "-num_gpus", "2",
        "-dtype", "fp16",
        "-enable_eager", "true",
        "-distribution_strategy", "default",
        "-model_dir", "cifar_2_gpu_fp16",
        "-tf_gpu_thread_mode", "gpu_private",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_cifar_main.main,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_cifar10_end_to_end_xla_2_gpu_fp16(self):
    """Test Keras model with XLA, 2 GPUs and fp16."""
    extra_flags = [
        "-num_gpus", "2",
        "-dtype", "fp16",
        "-enable_eager", "true",
        "-enable_xla", "true",
        "-distribution_strategy", "default",
        "-model_dir", "cifar_xla_2_gpu_fp16",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_cifar_main.main,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_cifar10_end_to_end_xla_2_gpu_fp16_tweaked(self):
    """Test Keras model with manual config tuning, XLA, 2 GPUs and fp16."""
    extra_flags = [
        "-num_gpus", "2",
        "-dtype", "fp16",
        "-enable_eager", "true",
        "-enable_xla", "true",
        "-distribution_strategy", "default",
        "-model_dir", "cifar_xla_2_gpu_fp16_tweaked",
        "-tf_gpu_thread_mode", "gpu_private",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_cifar_main.main,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_cifar10_end_to_end_xla_2_gpu_fp16_tensorboard_tweaked(self):
    """Test to track Tensorboard performance overhead."""
    extra_flags = [
        "-num_gpus", "2",
        "-dtype", "fp16",
        "-enable_eager", "true",
        "-enable_xla", "true",
        "-distribution_strategy", "default",
        "-model_dir", "cifar_xla_2_gpu_fp16_tensorboard_tweaked",
        "-tf_gpu_thread_mode", "gpu_private",
        "-enable_tensorboard", "true",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_cifar_main.main,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_cifar10_end_to_end_graph_2_gpu(self):
    """Test Keras model in legacy graph mode with 2 GPUs."""
    extra_flags = [
        "-num_gpus", "2",
        "-enable_eager", "false",
        "-distribution_strategy", "default",
        "-model_dir", "cifar_graph_2_gpu",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_cifar_main.main,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_cifar10_end_to_end_graph_xla_2_gpu(self):
    """Test Keras model in legacy graph mode with XLA and 2 GPUs."""
    extra_flags = [
        "-num_gpus", "2",
        "-enable_eager", "false",
        "-enable_xla", "true",
        "-distribution_strategy", "default",
        "-model_dir", "cifar_graph_xla_2_gpu",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_cifar_main.main,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_cifar10_end_to_end_graph_xla_2_gpu_fp16_tweaked(self):
    """Test Keras model in legacy graph mode with manual config tuning, XLA,
       2 GPUs and fp16.
    """
    extra_flags = [
        "-num_gpus", "2",
        "-dtype", "fp16",
        "-enable_eager", "false",
        "-enable_xla", "true",
        "-distribution_strategy", "default",
        "-model_dir", "cifar_graph_xla_2_gpu_fp16_tweaked",
        "-tf_gpu_thread_mode", "gpu_private",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_cifar_main.main,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

if __name__ == "__main__":
  googletest.main()
