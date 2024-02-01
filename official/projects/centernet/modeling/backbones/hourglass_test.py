# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for hourglass module."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.projects.centernet.common import registry_imports  # pylint: disable=unused-import
from official.projects.centernet.configs import backbones
from official.projects.centernet.modeling.backbones import hourglass
from official.vision.configs import common


class HourglassTest(tf.test.TestCase, parameterized.TestCase):

  def test_hourglass(self):
    backbone = hourglass.build_hourglass(
        input_specs=tf_keras.layers.InputSpec(shape=[None, 512, 512, 3]),
        backbone_config=backbones.Backbone(type='hourglass'),
        norm_activation_config=common.NormActivation(use_sync_bn=True)
    )
    inputs = np.zeros((2, 512, 512, 3), dtype=np.float32)
    outputs = backbone(inputs)
    self.assertEqual(outputs['2_0'].shape, (2, 128, 128, 256))
    self.assertEqual(outputs['2'].shape, (2, 128, 128, 256))


if __name__ == '__main__':
  tf.test.main()
