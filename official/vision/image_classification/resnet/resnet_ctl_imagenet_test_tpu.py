# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Test the ResNet model with ImageNet data using CTL."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from official.utils.testing import integration
from official.vision.image_classification.resnet import common
from official.vision.image_classification.resnet import imagenet_preprocessing
from official.vision.image_classification.resnet import resnet_ctl_imagenet_main


class CtlImagenetTest(tf.test.TestCase):
  """Unit tests for Keras ResNet with ImageNet using CTL."""

  _extra_flags = [
      '-batch_size', '4',
      '-train_steps', '4',
      '-use_synthetic_data', 'true',
      '-distribution_strategy', 'tpu',
      '-data_format', 'channels_last',
      '-use_tf_function', 'true',
      '-single_l2_loss_op', 'true'
  ]
  _tempdir = None

  @classmethod
  def setUpClass(cls):  # pylint: disable=invalid-name
    super(CtlImagenetTest, cls).setUpClass()
    common.define_keras_flags()

  def setUp(self):
    super(CtlImagenetTest, self).setUp()
    imagenet_preprocessing.NUM_IMAGES['validation'] = 4
    self.policy = \
        tf.compat.v2.keras.mixed_precision.experimental.global_policy()

  def tearDown(self):
    super(CtlImagenetTest, self).tearDown()
    tf.io.gfile.rmtree(self.get_temp_dir())
    tf.compat.v2.keras.mixed_precision.experimental.set_policy(self.policy)

  def test_end_to_end_tpu(self):
    """Test Keras model with TPU distribution strategy."""

    model_dir = os.path.join(self.get_temp_dir(), 'ctl_imagenet_tpu_dist_strat')
    extra_flags = ['-model_dir', model_dir]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
        main=resnet_ctl_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

  def test_end_to_end_tpu_bf16(self):
    """Test Keras model with TPU and bfloat16 activation."""

    model_dir = os.path.join(self.get_temp_dir(),
                             'ctl_imagenet_tpu_dist_strat_bf16')
    extra_flags = [
        '-model_dir', model_dir,
        '-dtype', 'bf16',
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
        main=resnet_ctl_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

  def test_end_to_end_tpu_with_mlir(self):
    """Test Keras model with TPU distribution strategy and MLIR bridge."""

    tf.config.experimental.enable_mlir_bridge()
    model_dir = os.path.join(self.get_temp_dir(),
                             'ctl_imagenet_tpu_dist_strat_mlir')
    extra_flags = ['-model_dir', model_dir]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
        main=resnet_ctl_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

  def test_end_to_end_tpu_bf16_with_mlir(self):
    """Test Keras model with TPU and bfloat16 activation and MLIR bridge."""

    tf.config.experimental.enable_mlir_bridge()
    model_dir = os.path.join(self.get_temp_dir(),
                             'ctl_imagenet_tpu_dist_strat_bf16_mlir')
    extra_flags = [
        '-model_dir', model_dir,
        '-dtype', 'bf16',
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
        main=resnet_ctl_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

if __name__ == '__main__':
  tf.test.main()
