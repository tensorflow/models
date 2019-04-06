# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Test the keras ResNet model with ImageNet data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tempfile import mkdtemp
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.resnet import imagenet_main
from official.resnet.keras import keras_common
from official.resnet.keras import keras_imagenet_main
from official.utils.testing import integration
from tensorflow.python.eager import context
from tensorflow.python.platform import googletest


class KerasImagenetTest(googletest.TestCase):
  """Unit tests for Keras ResNet with ImageNet."""

  _num_validation_images = None
  _extra_flags = [
          '-batch_size', '4',
          '-train_steps', '1',
          '-use_synthetic_data', 'true'
  ]
  _tempdir = None

  def get_temp_dir(self):
    if not self._tempdir:
      self._tempdir = mkdtemp(dir=googletest.GetTempDir())
    return self._tempdir

  @classmethod
  def setUpClass(cls):  # pylint: disable=invalid-name
    super(KerasImagenetTest, cls).setUpClass()
    imagenet_main.define_imagenet_flags()
    keras_common.define_keras_flags()

  def setUp(self):
    super(KerasImagenetTest, self).setUp()
    self._num_validation_images = imagenet_main.NUM_IMAGES['validation']
    imagenet_main.NUM_IMAGES['validation'] = 4

  def tearDown(self):
    super(KerasImagenetTest, self).tearDown()
    tf.io.gfile.rmtree(self.get_temp_dir())
    imagenet_main.NUM_IMAGES['validation'] = self._num_validation_images

  def test_end_to_end_1_gpu_no_dist_strat(self):
    """Test Keras model with 1 GPU, no distribution strategy."""
    if context.num_gpus() < 1:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(1, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "1",
        "-enable_eager", "true",
        "-distribution_strategy", "off",
        "-model_dir", "keras_imagenet_1_gpu_no_dist_strat",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_imagenet_main.run,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_end_to_end_graph_1_gpu_no_dist_strat(self):
    """Test Keras model in legacy graph mode with 1 GPU, no dist strat."""
    if context.num_gpus() < 1:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(1, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "1",
        "-enable_eager", "false",
        "-distribution_strategy", "off",
        "-model_dir", "keras_imagenet_graph_1_gpu_no_dist_strat",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_imagenet_main.run,
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
        "-enable_eager", "true",
        "-distribution_strategy", "default",
        "-model_dir", "keras_imagenet_1_gpu",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_imagenet_main.run,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_end_to_end_xla_1_gpu(self):
    """Test Keras model with XLA and 1 GPU."""
    if context.num_gpus() < 1:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(1, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "1",
        "-enable_eager", "true",
        "-enable_xla", "true",
        "-distribution_strategy", "default",
        "-model_dir", "keras_imagenet_xla_1_gpu",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_imagenet_main.run,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_end_to_end_1_gpu_fp16(self):
    """Test Keras model with 1 GPU and fp16."""
    if context.num_gpus() < 1:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(1, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "1",
        "-enable_eager", "true",
        "-distribution_strategy", "default",
        "-model_dir", "keras_imagenet_1_gpu_fp16",
        "-dtype", "fp16",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_imagenet_main.run,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_end_to_end_xla_1_gpu_fp16(self):
    """Test Keras model with XLA, 1 GPU and fp16."""
    if context.num_gpus() < 1:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(1, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "1",
        "-enable_eager", "true",
        "-enable_xla", "true",
        "-distribution_strategy", "default",
        "-model_dir", "keras_imagenet_xla_1_gpu_fp16",
        "-dtype", "fp16",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_imagenet_main.run,
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
        "-enable_eager", "false",
        "-distribution_strategy", "default",
        "-model_dir", "keras_imagenet_graph_1_gpu",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_imagenet_main.run,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_end_to_end_graph_xla_1_gpu(self):
    """Test Keras model in legacy graph mode with XLA and 1 GPU."""
    if context.num_gpus() < 1:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(1, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "1",
        "-enable_eager", "false",
        "-enable_xla", "true",
        "-distribution_strategy", "default",
        "-model_dir", "keras_imagenet_graph_xla_1_gpu",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_imagenet_main.run,
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
        "-enable_eager", "true",
        "-distribution_strategy", "default",
        "-model_dir", "keras_imagenet_2_gpu",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_imagenet_main.run,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_end_to_end_2_gpu_tweaked(self):
    """Test Keras model with manual config tuning and 2 GPUs."""
    if context.num_gpus() < 2:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(2, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "2",
        "-enable_eager", "true",
        "-distribution_strategy", "default",
        "-model_dir", "keras_imagenet_2_gpu_tweaked",
        "-datasets_num_private_threads", "14",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_imagenet_main.run,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_end_to_end_xla_2_gpu(self):
    """Test Keras model with XLA and 2 GPUs."""
    if context.num_gpus() < 2:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(2, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "2",
        "-enable_eager", "true",
        "-enable_xla", "true",
        "-distribution_strategy", "default",
        "-model_dir", "keras_imagenet_xla_2_gpu",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_imagenet_main.run,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_end_to_end_2_gpu_fp16(self):
    """Test Keras model with 2 GPUs and fp16."""
    if context.num_gpus() < 2:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(2, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "2",
        "-dtype", "fp16",
        "-enable_eager", "true",
        "-distribution_strategy", "default",
        "-model_dir", "keras_imagenet_2_gpu_fp16",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_imagenet_main.run,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_end_to_end_2_gpu_fp16_tweaked(self):
    """Test Keras model with 2 GPUs and fp16."""
    if context.num_gpus() < 2:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(2, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "2",
        "-dtype", "fp16",
        "-enable_eager", "true",
        "-distribution_strategy", "default",
        "-model_dir", "keras_imagenet_2_gpu_fp16",
        "-tf_gpu_thread_mode", "gpu_private",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_imagenet_main.run,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_end_to_end_xla_2_gpu_fp16(self):
    """Test Keras model with XLA, 2 GPUs and fp16."""
    if context.num_gpus() < 2:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(2, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "2",
        "-dtype", "fp16",
        "-enable_eager", "true",
        "-enable_xla", "true",
        "-distribution_strategy", "default",
        "-model_dir", "keras_imagenet_xla_2_gpu_fp16",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_imagenet_main.run,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_end_to_end_xla_2_gpu_fp16_tweaked(self):
    """Test Keras model with manual config tuning, XLA, 2 GPUs and fp16."""
    if context.num_gpus() < 2:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(2, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "2",
        "-dtype", "fp16",
        "-enable_eager", "true",
        "-enable_xla", "true",
        "-distribution_strategy", "default",
        "-model_dir", "keras_imagenet_xla_2_gpu_fp16_tweaked",
        "-tf_gpu_thread_mode", "gpu_private",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_imagenet_main.run,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_end_to_end_xla_2_gpu_fp16_tensorboard_tweaked(self):
    """Test to track Tensorboard performance overhead."""
    if context.num_gpus() < 2:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(2, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "2",
        "-dtype", "fp16",
        "-enable_eager", "true",
        "-enable_xla", "true",
        "-distribution_strategy", "default",
        "-model_dir", "keras_imagenet_xla_2_gpu_fp16_tensorboard_tweaked",
        "-tf_gpu_thread_mode", "gpu_private",
        "-enable_tensorboard", "true",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_imagenet_main.run,
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
        "-distribution_strategy", "default",
        "-model_dir", "keras_imagenet_graph_2_gpu",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_imagenet_main.run,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_end_to_end_graph_xla_2_gpu(self):
    """Test Keras model in legacy graph mode with XLA and 2 GPUs."""
    if context.num_gpus() < 2:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(2, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "2",
        "-enable_eager", "false",
        "-enable_xla", "true",
        "-distribution_strategy", "default",
        "-model_dir", "keras_imagenet_graph_xla_2_gpu",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_imagenet_main.run,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )

  def test_end_to_end_graph_xla_2_gpu_fp16_tweaked(self):
    """Test Keras model in legacy graph mode with manual config tuning, XLA,
       2 GPUs and fp16.
    """
    if context.num_gpus() < 2:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(2, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "2",
        "-dtype", "fp16",
        "-enable_eager", "false",
        "-enable_xla", "true",
        "-distribution_strategy", "default",
        "-model_dir", "keras_imagenet_graph_xla_2_gpu_fp16_tweaked",
        "-tf_gpu_thread_mode", "gpu_private",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
      main=keras_imagenet_main.run,
      tmp_root=self.get_temp_dir(),
      extra_flags=extra_flags
    )


if __name__ == '__main__':
  googletest.main()
