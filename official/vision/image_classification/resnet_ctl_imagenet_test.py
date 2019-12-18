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

import tempfile

import tensorflow.compat.v2 as tf

from tensorflow.python.eager import context
from official.utils.testing import integration
from official.vision.image_classification import common
from official.vision.image_classification import imagenet_preprocessing
from official.vision.image_classification import resnet_ctl_imagenet_main


class CtlImagenetTest(tf.test.TestCase):
  """Unit tests for Keras ResNet with ImageNet using CTL."""

  _extra_flags = [
      '-batch_size', '4',
      '-train_steps', '4',
      '-use_synthetic_data', 'true'
  ]
  _tempdir = None

  def get_temp_dir(self):
    if not self._tempdir:
      self._tempdir = tempfile.mkdtemp(
          dir=super(CtlImagenetTest, self).get_temp_dir())
    return self._tempdir

  @classmethod
  def setUpClass(cls):
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

  def test_end_to_end_no_dist_strat(self):
    """Test Keras model with 1 GPU, no distribution strategy."""

    extra_flags = [
        '-distribution_strategy', 'off',
        '-model_dir', 'ctl_imagenet_no_dist_strat',
        '-data_format', 'channels_last',
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
        main=resnet_ctl_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

  def test_end_to_end_2_gpu(self):
    """Test Keras model with 2 GPUs."""
    num_gpus = '2'
    if context.num_gpus() < 2:
      num_gpus = '0'

    extra_flags = [
        '-num_gpus', num_gpus,
        '-distribution_strategy', 'mirrored',
        '-model_dir', 'ctl_imagenet_2_gpu',
        '-data_format', 'channels_last',
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
        main=resnet_ctl_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

if __name__ == '__main__':
  assert tf.version.VERSION.startswith('2.')
  tf.test.main()
