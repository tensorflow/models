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

# Lint as: python3
"""Tests for Panoptic Deeplab network."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow.python.distribute import combinations

from official.vision.beta.modeling import backbones
from official.vision.beta.modeling.decoders import aspp
from official.vision.beta.projects.panoptic_maskrcnn.modeling.heads import panoptic_deeplab_heads
from official.vision.beta.projects.panoptic_maskrcnn.modeling import panoptic_deeplab_model

class PanopticDeeplabNetworkTest(parameterized.TestCase, tf.test.TestCase):

  @combinations.generate(
      combinations.combine(
          level=[2, 3, 4],
          input_size=[256, 512],
          low_level=[(4, 3), (3, 2)],
          shared_decoder=[True, False],
          training=[True, False]))
  def test_panoptic_deeplab_network_creation(
      self, input_size, level, low_level, shared_decoder, training):
    """Test for creation of a panoptic deep lab network."""
    num_classes = 10
    inputs = np.random.rand(2, input_size, input_size, 3)
    tf.keras.backend.set_image_data_format('channels_last')
    backbone = backbones.ResNet(model_id=50)

    semantic_decoder = aspp.ASPP(
        level=level, dilation_rates=[6, 12, 18])

    if shared_decoder:
      instance_decoder = semantic_decoder
    else:
      instance_decoder = aspp.ASPP(
          level=level, dilation_rates=[6, 12, 18])

    semantic_head = panoptic_deeplab_heads.SemanticHead(
        num_classes,                                    
        level=level,
        low_level=low_level,
        low_level_num_filters=(64, 32))

    instance_head = panoptic_deeplab_heads.InstanceHead(
        level=level,
        low_level=low_level,
        low_level_num_filters=(64, 32))

    model = panoptic_deeplab_model.PanopticDeeplabModel(
        backbone=backbone,
        semantic_decoder=semantic_decoder,
        instance_decoder=instance_decoder,
        semantic_head=semantic_head,
        instance_head=instance_head)

    outputs = model(inputs, training=training)

      
    self.assertIn('segmentation_outputs', outputs)
    self.assertIn('instance_center_prediction', outputs)
    self.assertIn('instance_center_regression', outputs)

    self.assertAllEqual(
        [2, input_size // (2**low_level[-1]), 
         input_size //(2**low_level[-1]), 
         num_classes],
        outputs['segmentation_outputs'].numpy().shape)
    self.assertAllEqual(
        [2, input_size // (2**low_level[-1]),
         input_size // (2**low_level[-1]),
         1],
        outputs['instance_center_prediction'].numpy().shape)
    self.assertAllEqual(
        [2, input_size // (2**low_level[-1]),
         input_size // (2**low_level[-1]),
         2],
        outputs['instance_center_regression'].numpy().shape)

  @combinations.generate(
      combinations.combine(
          level=[2, 3, 4],
          low_level=[(4, 3), (3, 2)],
          shared_decoder=[True, False]))
  def test_serialize_deserialize(self, level, low_level, shared_decoder):
    """Validate the network can be serialized and deserialized."""
    num_classes = 10
    backbone = backbones.ResNet(model_id=50)

    semantic_decoder = aspp.ASPP(
        level=level, dilation_rates=[6, 12, 18])

    if shared_decoder:
      instance_decoder = semantic_decoder
    else:
      instance_decoder = aspp.ASPP(
          level=level, dilation_rates=[6, 12, 18])

    semantic_head = panoptic_deeplab_heads.SemanticHead(
        num_classes,
        level=level,
        low_level=low_level,
        low_level_num_filters=(64, 32))

    instance_head = panoptic_deeplab_heads.InstanceHead(
        level=level,
        low_level=low_level,
        low_level_num_filters=(64, 32))

    model = panoptic_deeplab_model.PanopticDeeplabModel(
        backbone=backbone,
        semantic_decoder=semantic_decoder,
        instance_decoder=instance_decoder,
        semantic_head=semantic_head,
        instance_head=instance_head)

    config = model.get_config()
    new_model = panoptic_deeplab_model.PanopticDeeplabModel.from_config(config)

    # Validate that the config can be forced to JSON.
    _ = new_model.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(model.get_config(), new_model.get_config())


if __name__ == '__main__':
  tf.test.main()
