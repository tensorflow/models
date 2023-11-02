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
from official.projects.volumetric_models.modeling import backbones
from official.projects.volumetric_models.modeling import decoders
from official.projects.volumetric_models.modeling.heads import segmentation_heads_3d
from official.vision.modeling import segmentation_model


class SegmentationNetworkUNet3DTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ([32, 32], 4),
      ([64, 64], 4),
      ([64, 64], 2),
      ([128, 64], 2),
  )
  def test_segmentation_network_unet3d_creation(self, input_size, depth):
    """Test for creation of a segmentation network."""
    num_classes = 2
    inputs = np.random.rand(2, input_size[0], input_size[0], input_size[1], 3)
    tf_keras.backend.set_image_data_format('channels_last')
    backbone = backbones.UNet3D(model_id=depth)

    decoder = decoders.UNet3DDecoder(
        model_id=depth, input_specs=backbone.output_specs)
    head = segmentation_heads_3d.SegmentationHead3D(
        num_classes, level=1, num_convs=0)

    model = segmentation_model.SegmentationModel(
        backbone=backbone, decoder=decoder, head=head)

    outputs = model(inputs)
    self.assertAllEqual(
        [2, input_size[0], input_size[0], input_size[1], num_classes],
        outputs['logits'].numpy().shape)

  def test_serialize_deserialize(self):
    """Validate the network can be serialized and deserialized."""
    num_classes = 3
    backbone = backbones.UNet3D(model_id=4)
    decoder = decoders.UNet3DDecoder(
        model_id=4, input_specs=backbone.output_specs)
    head = segmentation_heads_3d.SegmentationHead3D(
        num_classes, level=1, num_convs=0)
    model = segmentation_model.SegmentationModel(
        backbone=backbone, decoder=decoder, head=head)

    config = model.get_config()
    new_model = segmentation_model.SegmentationModel.from_config(config)

    # Validate that the config can be forced to JSON.
    _ = new_model.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(model.get_config(), new_model.get_config())


if __name__ == '__main__':
  tf.test.main()
