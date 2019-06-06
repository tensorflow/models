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
"""Test the ResNet model with ImageNet data using CTL."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tempfile import mkdtemp
import tensorflow as tf

from official.resnet import imagenet_main
from official.resnet.ctl import ctl_imagenet_main
from official.resnet.ctl import ctl_common
from official.utils.testing import integration
# pylint: disable=ungrouped-imports
from tensorflow.python.eager import context
from tensorflow.python.platform import googletest


class CtlImagenetTest(googletest.TestCase):
  """Unit tests for Keras ResNet with ImageNet using CTL."""

  _extra_flags = [
      '-batch_size', '4',
      '-train_steps', '1',
      '-use_synthetic_data', 'true',
      'enable_eager', 'true'
  ]
  _tempdir = None

  def get_temp_dir(self):
    if not self._tempdir:
      self._tempdir = mkdtemp(dir=googletest.GetTempDir())
    return self._tempdir

  @classmethod
  def setUpClass(cls):  # pylint: disable=invalid-name
    super(CtlImagenetTest, cls).setUpClass()
    imagenet_main.define_imagenet_flags()
    ctl_common.define_ctl_flags()

  def setUp(self):
    super(CtlImagenetTest, self).setUp()
    imagenet_main.NUM_IMAGES['validation'] = 4

  def tearDown(self):
    super(CtlImagenetTest, self).tearDown()
    tf.io.gfile.rmtree(self.get_temp_dir())

  def test_end_to_end_1_gpu(self):
    """Test Keras model with 1 GPU."""
    if context.num_gpus() < 1:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(1, context.num_gpus()))
    tf.enable_v2_behavior()
    extra_flags = [
        "-num_gpus", "1",
        "-distribution_strategy", "default",
        "-model_dir", "ctl_imagenet_1_gpu",
        "-data_format", "channels_last",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
        main=ctl_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

  def test_end_to_end_2_gpu(self):
    """Test Keras model with 2 GPUs."""
    if context.num_gpus() < 2:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(2, context.num_gpus()))
    tf.enable_v2_behavior()
    extra_flags = [
        "-num_gpus", "2",
        "-distribution_strategy", "default",
        "-model_dir", "ctl_imagenet_2_gpu",
    ]
    extra_flags = extra_flags + self._extra_flags

    integration.run_synthetic(
        main=ctl_imagenet_main.run,
        tmp_root=self.get_temp_dir(),
        extra_flags=extra_flags
    )

if __name__ == '__main__':
  googletest.main()
