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

"""Tests for Panoptic Deeplab network."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import combinations
from official.vision.beta.projects.panoptic_maskrcnn.modeling import panoptic_deeplab_model
from official.vision.beta.projects.panoptic_maskrcnn.modeling.heads import panoptic_deeplab_heads
from official.vision.beta.projects.panoptic_maskrcnn.modeling.layers import panoptic_deeplab_merge
from official.vision.modeling import backbones
from official.vision.modeling.decoders import aspp


class PanopticDeeplabNetworkTest(parameterized.TestCase, tf.test.TestCase):

  @combinations.generate(
      combinations.combine(
          level=[2, 3, 4],
          input_size=[256, 512],
          low_level=[[4, 3], [3, 2]],
          shared_decoder=[True, False],
          training=[True, False]))
  def test_panoptic_deeplab_network_creation(
      self, input_size, level, low_level, shared_decoder, training):
    """Test for creation of a panoptic deeplab network."""
    batch_size = 2 if training else 1
    num_classes = 10
    inputs = np.random.rand(batch_size, input_size, input_size, 3)

    image_info = tf.convert_to_tensor(
        [[[input_size, input_size], [input_size, input_size], [1, 1], [0, 0]]])
    image_info = tf.tile(image_info, [batch_size, 1, 1])

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

    post_processor = panoptic_deeplab_merge.PostProcessor(
        output_size=[input_size, input_size],
        center_score_threshold=0.1,
        thing_class_ids=[1, 2, 3, 4],
        label_divisor=[256],
        stuff_area_limit=4096,
        ignore_label=0,
        nms_kernel=41,
        keep_k_centers=41,
        rescale_predictions=True)

    model = panoptic_deeplab_model.PanopticDeeplabModel(
        backbone=backbone,
        semantic_decoder=semantic_decoder,
        instance_decoder=instance_decoder,
        semantic_head=semantic_head,
        instance_head=instance_head,
        post_processor=post_processor)

    outputs = model(
        inputs=inputs,
        image_info=image_info,
        training=training)

    if training:
      self.assertIn('segmentation_outputs', outputs)
      self.assertIn('instance_centers_heatmap', outputs)
      self.assertIn('instance_centers_offset', outputs)

      self.assertAllEqual(
          [2, input_size // (2**low_level[-1]),
           input_size //(2**low_level[-1]),
           num_classes],
          outputs['segmentation_outputs'].numpy().shape)
      self.assertAllEqual(
          [2, input_size // (2**low_level[-1]),
           input_size // (2**low_level[-1]),
           1],
          outputs['instance_centers_heatmap'].numpy().shape)
      self.assertAllEqual(
          [2, input_size // (2**low_level[-1]),
           input_size // (2**low_level[-1]),
           2],
          outputs['instance_centers_offset'].numpy().shape)

    else:
      self.assertIn('panoptic_outputs', outputs)
      self.assertIn('category_mask', outputs)
      self.assertIn('instance_mask', outputs)
      self.assertIn('instance_centers', outputs)
      self.assertIn('instance_scores', outputs)
      self.assertIn('segmentation_outputs', outputs)

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

    post_processor = panoptic_deeplab_merge.PostProcessor(
        output_size=[640, 640],
        center_score_threshold=0.1,
        thing_class_ids=[1, 2, 3, 4],
        label_divisor=[256],
        stuff_area_limit=4096,
        ignore_label=0,
        nms_kernel=41,
        keep_k_centers=41,
        rescale_predictions=True)

    model = panoptic_deeplab_model.PanopticDeeplabModel(
        backbone=backbone,
        semantic_decoder=semantic_decoder,
        instance_decoder=instance_decoder,
        semantic_head=semantic_head,
        instance_head=instance_head,
        post_processor=post_processor)

    config = model.get_config()
    new_model = panoptic_deeplab_model.PanopticDeeplabModel.from_config(config)

    # Validate that the config can be forced to JSON.
    _ = new_model.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(model.get_config(), new_model.get_config())


if __name__ == '__main__':
  tf.test.main()
