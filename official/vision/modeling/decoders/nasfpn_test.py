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

"""Tests for NAS-FPN."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.vision.modeling.backbones import resnet
from official.vision.modeling.decoders import nasfpn


class NASFPNTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (256, 3, 7, False),
      (256, 3, 7, True),
  )
  def test_network_creation(self, input_size, min_level, max_level,
                            use_separable_conv):
    """Test creation of NAS-FPN."""
    tf_keras.backend.set_image_data_format('channels_last')

    inputs = tf_keras.Input(shape=(input_size, input_size, 3), batch_size=1)

    num_filters = 256
    backbone = resnet.ResNet(model_id=50)
    network = nasfpn.NASFPN(
        input_specs=backbone.output_specs,
        min_level=min_level,
        max_level=max_level,
        num_filters=num_filters,
        use_separable_conv=use_separable_conv)

    endpoints = backbone(inputs)
    feats = network(endpoints)

    for level in range(min_level, max_level + 1):
      self.assertIn(str(level), feats)
      self.assertAllEqual(
          [1, input_size // 2**level, input_size // 2**level, num_filters],
          feats[str(level)].shape.as_list())


if __name__ == '__main__':
  tf.test.main()
