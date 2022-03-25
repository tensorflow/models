# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for VIT."""

from absl.testing import parameterized
import tensorflow as tf

from official.projects.vit.modeling import vit


class VisionTransformerTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (224, 85798656),
      (256, 85844736),
  )
  def test_network_creation(self, input_size, params_count):
    """Test creation of VisionTransformer family models."""
    tf.keras.backend.set_image_data_format('channels_last')
    input_specs = tf.keras.layers.InputSpec(
        shape=[2, input_size, input_size, 3])
    network = vit.VisionTransformer(input_specs=input_specs)

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    _ = network(inputs)
    self.assertEqual(network.count_params(), params_count)


if __name__ == '__main__':
  tf.test.main()
