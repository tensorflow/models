# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for aspp."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.modeling.backbones import basnet_en
from official.vision.beta.modeling.decoders import basnet_de


class BASNet_De_Test(parameterized.TestCase, tf.test.TestCase):
  """
  @parameterized.parameters(
      (3, [6, 12, 18, 24], 128),
      (3, [6, 12, 18], 128),
      (3, [6, 12], 256),
      (4, [6, 12, 18, 24], 128),
      (4, [6, 12, 18], 128),
      (4, [6, 12], 256),
  )
  """
  def test_network_creation(self):
    """Test creation of BASNet Decoder."""

    input_size = 224
    tf.keras.backend.set_image_data_format('channels_last')

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)

    backbone = basnet_en.BASNet_En()

    network = basnet_de.BASNet_De(
        input_specs=backbone.output_specs  
    )

    endpoints = backbone(inputs)
    sups = network(endpoints)

    self.assertIn(str(6), sups)
    self.assertAllEqual(
        [1, input_size, input_size, 1],
        sups[str(6)].shape.as_list())

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        input_specs=basnet_en.BASNet_En().output_specs,
        use_separable_conv=False,
        activation='relu',
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    network = basnet_de.BASNet_De(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = basnet_de.BASNet_De.from_config(network.get_config())

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == '__main__':
  tf.test.main()

"""
input_size = 224
tf.keras.backend.set_image_data_format('channels_last')

inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)

backbone = basnet_en.BASNet_En()

print(backbone.output_specs)

network = basnet_de.BASNet_De(
      input_specs=backbone.output_specs  
  )

endpoints = backbone(inputs)
print('endpoints')
print(endpoints)
sup = network(endpoints)
print('sup')
print(sup)
"""
