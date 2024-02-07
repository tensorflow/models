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

"""Tests for backbones_3d."""

import tensorflow as tf, tf_keras

from official.projects.const_cl.configs import backbones_3d


class Backbones3DTest(tf.test.TestCase):

  def test_conv3dy_config(self):
    config = backbones_3d.Backbone3D(
        type='resnet_3dy',
        resnet_3d=backbones_3d.ResNet3DY50())
    config.validate()


if __name__ == '__main__':
  tf.test.main()
