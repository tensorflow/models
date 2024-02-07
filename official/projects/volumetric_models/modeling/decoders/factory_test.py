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

"""Tests for factory functions."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from tensorflow.python.distribute import combinations
from official.projects.volumetric_models.configs import decoders as decoders_cfg
from official.projects.volumetric_models.configs import semantic_segmentation_3d as semantic_segmentation_3d_exp
from official.projects.volumetric_models.modeling import decoders
from official.projects.volumetric_models.modeling.decoders import factory


class FactoryTest(tf.test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(model_id=[2, 3],))
  def test_unet_3d_decoder_creation(self, model_id):
    """Test creation of UNet 3D decoder."""
    # Create test input for decoders based on input model_id.
    input_specs = {}
    for level in range(model_id):
      input_specs[str(level + 1)] = tf.TensorShape(
          [1, 128 // (2**level), 128 // (2**level), 128 // (2**level), 1])

    network = decoders.UNet3DDecoder(
        model_id=model_id,
        input_specs=input_specs,
        use_sync_bn=True,
        use_batch_normalization=True,
        use_deconvolution=True)

    model_config = semantic_segmentation_3d_exp.SemanticSegmentationModel3D()
    model_config.num_classes = 2
    model_config.num_channels = 1
    model_config.input_size = [None, None, None]
    model_config.decoder = decoders_cfg.Decoder(
        type='unet_3d_decoder',
        unet_3d_decoder=decoders_cfg.UNet3DDecoder(model_id=model_id))

    factory_network = factory.build_decoder(
        input_specs=input_specs, model_config=model_config)

    network_config = network.get_config()
    factory_network_config = factory_network.get_config()
    print(network_config)
    print(factory_network_config)

    self.assertEqual(network_config, factory_network_config)

  def test_identity_creation(self):
    """Test creation of identity decoder."""
    model_config = semantic_segmentation_3d_exp.SemanticSegmentationModel3D()
    model_config.num_classes = 2
    model_config.num_channels = 3
    model_config.input_size = [None, None, None]

    model_config.decoder = decoders_cfg.Decoder(
        type='identity', identity=decoders_cfg.Identity())

    factory_network = factory.build_decoder(
        input_specs=None, model_config=model_config)

    self.assertIsNone(factory_network)


if __name__ == '__main__':
  tf.test.main()
