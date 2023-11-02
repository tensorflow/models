# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for segmentation network."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.vision.modeling import backbones
from official.vision.modeling import segmentation_model
from official.vision.modeling.decoders import fpn
from official.vision.modeling.heads import segmentation_heads


class SegmentationNetworkTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (128, 2),
      (128, 3),
      (128, 4),
      (256, 2),
      (256, 3),
      (256, 4),
  )
  def test_segmentation_network_creation(
      self, input_size, level):
    """Test for creation of a segmentation network."""
    num_classes = 10
    inputs = np.random.rand(2, input_size, input_size, 3)
    tf_keras.backend.set_image_data_format('channels_last')
    backbone = backbones.ResNet(model_id=50)

    decoder = fpn.FPN(
        input_specs=backbone.output_specs, min_level=2, max_level=7)
    head = segmentation_heads.SegmentationHead(num_classes, level=level)

    model = segmentation_model.SegmentationModel(
        backbone=backbone,
        decoder=decoder,
        head=head,
        mask_scoring_head=None,
    )

    outputs = model(inputs)
    self.assertAllEqual(
        [2, input_size // (2**level), input_size // (2**level), num_classes],
        outputs['logits'].numpy().shape)

  def test_serialize_deserialize(self):
    """Validate the network can be serialized and deserialized."""
    num_classes = 3
    backbone = backbones.ResNet(model_id=50)
    decoder = fpn.FPN(
        input_specs=backbone.output_specs, min_level=3, max_level=7)
    head = segmentation_heads.SegmentationHead(num_classes, level=3)
    model = segmentation_model.SegmentationModel(
        backbone=backbone,
        decoder=decoder,
        head=head
    )

    config = model.get_config()
    new_model = segmentation_model.SegmentationModel.from_config(config)

    # Validate that the config can be forced to JSON.
    _ = new_model.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(model.get_config(), new_model.get_config())


if __name__ == '__main__':
  tf.test.main()
