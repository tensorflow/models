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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.eager import context
from official.benchmark.models import resnet_imagenet_main
from official.utils.testing import integration
from official.vision.image_classification.resnet import imagenet_preprocessing


@parameterized.parameters(
    "resnet",
    # "resnet_polynomial_decay",  b/151854314
    "mobilenet",
    # "mobilenet_polynomial_decay"  b/151854314
)
class KerasImagenetTest(tf.test.TestCase):
  """Unit tests for Keras Models with ImageNet."""
  _default_flags_dict = [
      "-batch_size", "4",
      "-train_steps", "1",
      "-use_synthetic_data", "true",
      "-data_format", "channels_last",
  ]
  _extra_flags_dict = {
      "resnet": [
          "-model", "resnet50_v1.5",
          "-optimizer", "resnet50_default",
      ],
      "resnet_polynomial_decay": [
          "-model", "resnet50_v1.5",
          "-optimizer", "resnet50_default",
          "-pruning_method", "polynomial_decay",
      ],
      "mobilenet": [
          "-model", "mobilenet",
          "-optimizer", "mobilenet_default",
      ],
      "mobilenet_polynomial_decay": [
          "-model", "mobilenet",
          "-optimizer", "mobilenet_default",
          "-pruning_method", "polynomial_decay",
      ],
  }
  _tempdir = None

  @classmethod
  def setUpClass(cls):  # pylint: disable=invalid-name
    super(KerasImagenetTest, cls).setUpClass()
    resnet_imagenet_main.define_imagenet_keras_flags()

  def setUp(self):
    super(KerasImagenetTest, self).setUp()
    imagenet_preprocessing.NUM_IMAGES["validation"] = 4
    self.policy = \
        tf.keras.mixed_precision.experimental.global_policy()

  def tearDown(self):
    super(KerasImagenetTest, self).tearDown()
    tf.io.gfile.rmtree(self.get_temp_dir())
    tf.keras.mixed_precision.experimental.set_policy(self.policy)

  def get_extra_flags_dict(self, flags_key):
    return self._extra_flags_dict[flags_key] + self._default_flags_dict

  def test_end_to_end_no_dist_strat(self, flags_key):
    """Test Keras model with 1 GPU, no distribution strategy."""

    extra_flags = [
        "-distribution_strategy", "off",
    ]
    extra_flags = extra_flags + self.get_extra_flags_dict(flags_key)

    integration.run_synthetic(
        main=resnet_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

  def test_end_to_end_graph_no_dist_strat(self, flags_key):
    """Test Keras model in legacy graph mode with 1 GPU, no dist strat."""
    extra_flags = [
        "-enable_eager", "false",
        "-distribution_strategy", "off",
    ]
    extra_flags = extra_flags + self.get_extra_flags_dict(flags_key)

    integration.run_synthetic(
        main=resnet_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

  def test_end_to_end_1_gpu(self, flags_key):
    """Test Keras model with 1 GPU."""

    if context.num_gpus() < 1:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(1, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "1",
        "-distribution_strategy", "mirrored",
        "-enable_checkpoint_and_export", "1",
    ]
    extra_flags = extra_flags + self.get_extra_flags_dict(flags_key)

    integration.run_synthetic(
        main=resnet_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

  def test_end_to_end_1_gpu_fp16(self, flags_key):
    """Test Keras model with 1 GPU and fp16."""

    if context.num_gpus() < 1:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available"
          .format(1, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "1",
        "-dtype", "fp16",
        "-distribution_strategy", "mirrored",
    ]
    extra_flags = extra_flags + self.get_extra_flags_dict(flags_key)

    if "polynomial_decay" in extra_flags:
      self.skipTest("Pruning with fp16 is not currently supported.")

    integration.run_synthetic(
        main=resnet_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

  def test_end_to_end_2_gpu(self, flags_key):
    """Test Keras model with 2 GPUs."""

    if context.num_gpus() < 2:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(2, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "2",
        "-distribution_strategy", "mirrored",
    ]
    extra_flags = extra_flags + self.get_extra_flags_dict(flags_key)

    integration.run_synthetic(
        main=resnet_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

  def test_end_to_end_xla_2_gpu(self, flags_key):
    """Test Keras model with XLA and 2 GPUs."""

    if context.num_gpus() < 2:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(2, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "2",
        "-enable_xla", "true",
        "-distribution_strategy", "mirrored",
    ]
    extra_flags = extra_flags + self.get_extra_flags_dict(flags_key)

    integration.run_synthetic(
        main=resnet_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

  def test_end_to_end_2_gpu_fp16(self, flags_key):
    """Test Keras model with 2 GPUs and fp16."""

    if context.num_gpus() < 2:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(2, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "2",
        "-dtype", "fp16",
        "-distribution_strategy", "mirrored",
    ]
    extra_flags = extra_flags + self.get_extra_flags_dict(flags_key)

    if "polynomial_decay" in extra_flags:
      self.skipTest("Pruning with fp16 is not currently supported.")

    integration.run_synthetic(
        main=resnet_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

  def test_end_to_end_xla_2_gpu_fp16(self, flags_key):
    """Test Keras model with XLA, 2 GPUs and fp16."""
    if context.num_gpus() < 2:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(2, context.num_gpus()))

    extra_flags = [
        "-num_gpus", "2",
        "-dtype", "fp16",
        "-enable_xla", "true",
        "-distribution_strategy", "mirrored",
    ]
    extra_flags = extra_flags + self.get_extra_flags_dict(flags_key)

    if "polynomial_decay" in extra_flags:
      self.skipTest("Pruning with fp16 is not currently supported.")

    integration.run_synthetic(
        main=resnet_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )


if __name__ == "__main__":
  tf.test.main()
