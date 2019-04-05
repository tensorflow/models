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

from official.resnet import imagenet_main
from official.resnet.keras import keras_common
from official.resnet.keras import keras_imagenet_main
from official.resnet.keras import keras_test_base
from official.utils.testing import integration
from tensorflow.python.platform import googletest

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class KerasImagenetTest(keras_test_base.KerasTestBase)
  """Unit tests for Keras ResNet with ImageNet."""

  def test_imagenet_end_to_end_keras_synthetic_v1(self):
    integration.run_synthetic(
        main=keras_imagenet_main.main, tmp_root=self.get_temp_dir(),
        extra_flags=['-resnet_version', '1', '-batch_size', '4',
                     '-train_steps', '1']
    )


if __name__ == '__main__':
  googletest.main()
