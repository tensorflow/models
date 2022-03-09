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

"""Tests for factory.py."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.vision.beta.projects.panoptic_maskrcnn.configs import panoptic_maskrcnn as panoptic_maskrcnn_cfg
from official.vision.beta.projects.panoptic_maskrcnn.modeling import factory
from official.vision.configs import backbones
from official.vision.configs import decoders
from official.vision.configs import semantic_segmentation


class PanopticMaskRCNNBuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ('resnet', (640, 640), 'dilated_resnet', 'fpn'),
      ('resnet', (640, 640), 'dilated_resnet', 'aspp'),
      ('resnet', (640, 640), None, 'fpn'),
      ('resnet', (640, 640), None, 'aspp'),
      ('resnet', (640, 640), None, None),
      ('resnet', (None, None), 'dilated_resnet', 'fpn'),
      ('resnet', (None, None), 'dilated_resnet', 'aspp'),
      ('resnet', (None, None), None, 'fpn'),
      ('resnet', (None, None), None, 'aspp'),
      ('resnet', (None, None), None, None))
  def test_builder(self, backbone_type, input_size, segmentation_backbone_type,
                   segmentation_decoder_type):
    num_classes = 2
    input_specs = tf.keras.layers.InputSpec(
        shape=[None, input_size[0], input_size[1], 3])
    segmentation_output_stride = 16
    level = int(np.math.log2(segmentation_output_stride))
    segmentation_model = semantic_segmentation.SemanticSegmentationModel(
        num_classes=2,
        backbone=backbones.Backbone(type=segmentation_backbone_type),
        decoder=decoders.Decoder(type=segmentation_decoder_type),
        head=semantic_segmentation.SegmentationHead(level=level))
    model_config = panoptic_maskrcnn_cfg.PanopticMaskRCNN(
        num_classes=num_classes,
        segmentation_model=segmentation_model,
        backbone=backbones.Backbone(type=backbone_type),
        shared_backbone=segmentation_backbone_type is None,
        shared_decoder=segmentation_decoder_type is None)
    l2_regularizer = tf.keras.regularizers.l2(5e-5)
    _ = factory.build_panoptic_maskrcnn(
        input_specs=input_specs,
        model_config=model_config,
        l2_regularizer=l2_regularizer)

if __name__ == '__main__':
  tf.test.main()
