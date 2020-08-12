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
"""Test the keras ResNet model with ImageNet data on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf
from official.benchmark.models import resnet_imagenet_main
from official.utils.testing import integration
from official.vision.image_classification.resnet import imagenet_preprocessing


class KerasImagenetTest(tf.test.TestCase, parameterized.TestCase):
  """Unit tests for Keras Models with ImageNet."""

  _extra_flags_dict = {
      "resnet": [
          "-batch_size",
          "4",
          "-train_steps",
          "1",
          "-use_synthetic_data",
          "true"
          "-model",
          "resnet50_v1.5",
          "-optimizer",
          "resnet50_default",
      ],
      "resnet_polynomial_decay": [
          "-batch_size",
          "4",
          "-train_steps",
          "1",
          "-use_synthetic_data",
          "true",
          "-model",
          "resnet50_v1.5",
          "-optimizer",
          "resnet50_default",
          "-pruning_method",
          "polynomial_decay",
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

  @parameterized.parameters([
      "resnet",
      # "resnet_polynomial_decay"  b/151854314
  ])
  def test_end_to_end_tpu(self, flags_key):
    """Test Keras model with TPU distribution strategy."""

    extra_flags = [
        "-distribution_strategy",
        "tpu",
        "-data_format",
        "channels_last",
        "-enable_checkpoint_and_export",
        "1",
    ]
    extra_flags = extra_flags + self._extra_flags_dict[flags_key]

    integration.run_synthetic(
        main=resnet_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags)

  @parameterized.parameters(["resnet"])
  def test_end_to_end_tpu_bf16(self, flags_key):
    """Test Keras model with TPU and bfloat16 activation."""

    extra_flags = [
        "-distribution_strategy",
        "tpu",
        "-data_format",
        "channels_last",
        "-dtype",
        "bf16",
    ]
    extra_flags = extra_flags + self._extra_flags_dict[flags_key]

    integration.run_synthetic(
        main=resnet_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags)


if __name__ == "__main__":
  tf.test.main()
