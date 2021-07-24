# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for decoder factory functions."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.distribute import combinations
from official.vision.beta import configs
from official.vision.beta.configs import decoders as decoders_cfg
from official.vision.beta.modeling import decoders
from official.vision.beta.modeling.decoders import factory


class FactoryTest(tf.test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          num_filters=[128, 256], use_separable_conv=[True, False]))
  def test_fpn_decoder_creation(self, num_filters, use_separable_conv):
    """Test creation of FPN decoder."""
    min_level = 3
    max_level = 7
    input_specs = {}
    for level in range(min_level, max_level):
      input_specs[str(level)] = tf.TensorShape(
          [1, 128 // (2**level), 128 // (2**level), 3])

    network = decoders.FPN(
        input_specs=input_specs,
        num_filters=num_filters,
        use_separable_conv=use_separable_conv,
        use_sync_bn=True)

    model_config = configs.retinanet.RetinaNet()
    model_config.min_level = min_level
    model_config.max_level = max_level
    model_config.num_classes = 10
    model_config.input_size = [None, None, 3]
    model_config.decoder = decoders_cfg.Decoder(
        type='fpn',
        fpn=decoders_cfg.FPN(
            num_filters=num_filters, use_separable_conv=use_separable_conv))

    factory_network = factory.build_decoder(
        input_specs=input_specs, model_config=model_config)

    network_config = network.get_config()
    factory_network_config = factory_network.get_config()

    self.assertEqual(network_config, factory_network_config)

  @combinations.generate(
      combinations.combine(
          num_filters=[128, 256],
          num_repeats=[3, 5],
          use_separable_conv=[True, False]))
  def test_nasfpn_decoder_creation(self, num_filters, num_repeats,
                                   use_separable_conv):
    """Test creation of NASFPN decoder."""
    min_level = 3
    max_level = 7
    input_specs = {}
    for level in range(min_level, max_level):
      input_specs[str(level)] = tf.TensorShape(
          [1, 128 // (2**level), 128 // (2**level), 3])

    network = decoders.NASFPN(
        input_specs=input_specs,
        num_filters=num_filters,
        num_repeats=num_repeats,
        use_separable_conv=use_separable_conv,
        use_sync_bn=True)

    model_config = configs.retinanet.RetinaNet()
    model_config.min_level = min_level
    model_config.max_level = max_level
    model_config.num_classes = 10
    model_config.input_size = [None, None, 3]
    model_config.decoder = decoders_cfg.Decoder(
        type='nasfpn',
        nasfpn=decoders_cfg.NASFPN(
            num_filters=num_filters,
            num_repeats=num_repeats,
            use_separable_conv=use_separable_conv))

    factory_network = factory.build_decoder(
        input_specs=input_specs, model_config=model_config)

    network_config = network.get_config()
    factory_network_config = factory_network.get_config()

    self.assertEqual(network_config, factory_network_config)

  @combinations.generate(
      combinations.combine(
          level=[3, 4],
          dilation_rates=[[6, 12, 18], [6, 12]],
          num_filters=[128, 256]))
  def test_aspp_decoder_creation(self, level, dilation_rates, num_filters):
    """Test creation of ASPP decoder."""
    input_specs = {'1': tf.TensorShape([1, 128, 128, 3])}

    network = decoders.ASPP(
        level=level,
        dilation_rates=dilation_rates,
        num_filters=num_filters,
        use_sync_bn=True)

    model_config = configs.semantic_segmentation.SemanticSegmentationModel()
    model_config.num_classes = 10
    model_config.input_size = [None, None, 3]
    model_config.decoder = decoders_cfg.Decoder(
        type='aspp',
        aspp=decoders_cfg.ASPP(
            level=level, dilation_rates=dilation_rates,
            num_filters=num_filters))

    factory_network = factory.build_decoder(
        input_specs=input_specs, model_config=model_config)

    network_config = network.get_config()
    factory_network_config = factory_network.get_config()

    self.assertEqual(network_config, factory_network_config)

  def test_identity_decoder_creation(self):
    """Test creation of identity decoder."""
    model_config = configs.retinanet.RetinaNet()
    model_config.num_classes = 2
    model_config.input_size = [None, None, 3]

    model_config.decoder = decoders_cfg.Decoder(
        type='identity', identity=decoders_cfg.Identity())

    factory_network = factory.build_decoder(
        input_specs=None, model_config=model_config)

    self.assertIsNone(factory_network)


if __name__ == '__main__':
  tf.test.main()
