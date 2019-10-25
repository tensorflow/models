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

from tempfile import mkdtemp
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.platform import googletest
from official.resnet.ctl import ctl_common
from official.resnet.ctl import ctl_imagenet_main
from official.vision.image_classification import imagenet_preprocessing
from official.vision.image_classification import common
from official.utils.misc import keras_utils
from official.utils.testing import integration


class CtlImagenetTest(googletest.TestCase):
  """Unit tests for Keras ResNet with ImageNet using CTL."""

  _extra_flags = [
      '-batch_size', '4',
      '-train_steps', '4',
      '-use_synthetic_data', 'true'
  ]
  _tempdir = None

  def get_temp_dir(self):
    if not self._tempdir:
      self._tempdir = mkdtemp(dir=googletest.GetTempDir())
    return self._tempdir

  @classmethod
  def setUpClass(cls):  # pylint: disable=invalid-name
    super(CtlImagenetTest, cls).setUpClass()
    common.define_keras_flags()
    ctl_common.define_ctl_flags()

  def setUp(self):
    super(CtlImagenetTest, self).setUp()
    if not keras_utils.is_v2_0():
      tf.compat.v1.enable_v2_behavior()
    imagenet_preprocessing.NUM_IMAGES['validation'] = 4

  def tearDown(self):
    super(CtlImagenetTest, self).tearDown()
    tf.io.gfile.rmtree(self.get_temp_dir())

  def test_end_to_end_no_dist_strat(self):
    """Test Keras model with 1 GPU, no distribution strategy."""

    extra_flags = [
        '-distribution_strategy', 'off',
        '-model_dir', 'ctl_imagenet_no_dist_strat',
        '-data_format', 'channels_last',
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
        main=ctl_imagenet_main.run,
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
        '-distribution_strategy', 'default',
        '-model_dir', 'ctl_imagenet_2_gpu',
        '-data_format', 'channels_last',
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
        main=ctl_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

if __name__ == '__main__':
  googletest.main()
