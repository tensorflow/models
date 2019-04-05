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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.resnet import cifar10_main
from official.resnet.keras import keras_common
from tensorflow.python.platform import googletest

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class KerasTestBase(googletest.TestCase):
  """Base class for unit tests."""

  _num_validation_images = None
  _extra_flags = [
          '-batch_size', '4',
          '-train_steps', '1',
          '-use_synthetic_data', 'true'
  ]
  _tempdir = None

  @classmethod
  def setUpClass(cls):  # pylint: disable=invalid-name
    super(KerasCifarTest, cls).setUpClass()
    cifar10_main.define_cifar_flags()
    keras_common.define_keras_flags()

  def setUp(self):
    super(KerasCifarTest, self).setUp()
    self._num_validation_images = cifar10_main.NUM_IMAGES['validation']
    cifar10_main.NUM_IMAGES['validation'] = 4

  def tearDown(self):
    super(KerasCifarTest, self).tearDown()
    tf.io.gfile.rmtree(self.get_temp_dir())
    cifar10_main.NUM_IMAGES['validation'] = self._num_validation_images

  def get_temp_dir(self):
    if not self._tempdir:
      self._tempdir = mkdtemp(dir=googletest.GetTempDir())
    return self._tempdir

