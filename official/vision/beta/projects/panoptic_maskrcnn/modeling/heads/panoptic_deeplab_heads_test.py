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

"""Tests for panoptic_deeplab_heads.py."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.vision.beta.projects.panoptic_maskrcnn.modeling.heads import panoptic_deeplab_heads


class PanopticDeeplabHeadsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (2, (2,), (48,)),
      (3, (2,), (48,)),
      (2, (2,), (48,)),
      (2, (2,), (48,)),
      (3, (2,), (48,)),
      (3, (2,), (48,)),
      (4, (4, 3), (64, 32)),
      (4, (3, 2), (64, 32)))
  def test_forward(self, level, low_level, low_level_num_filters):
    backbone_features = {
        '3': np.random.rand(2, 128, 128, 16),
        '4': np.random.rand(2, 64, 64, 16),
        '5': np.random.rand(2, 32, 32, 16),
    }
    decoder_features = {
        '3': np.random.rand(2, 128, 128, 64),
        '4': np.random.rand(2, 64, 64, 64),
        '5': np.random.rand(2, 32, 32, 64),
        '6': np.random.rand(2, 16, 16, 64),
    }

    backbone_features['2'] = np.random.rand(2, 256, 256, 16)
    decoder_features['2'] = np.random.rand(2, 256, 256, 64)
    num_classes = 10
    semantic_head = panoptic_deeplab_heads.SemanticHead(
        num_classes=num_classes,
        level=level,
        low_level=low_level,
        low_level_num_filters=low_level_num_filters)

    instance_head = panoptic_deeplab_heads.InstanceHead(
        level=level,
        low_level=low_level,
        low_level_num_filters=low_level_num_filters)

    semantic_outputs = semantic_head((backbone_features, decoder_features))
    instance_outputs = instance_head((backbone_features, decoder_features))

    if str(level) in decoder_features:
      h, w = decoder_features[str(low_level[-1])].shape[1:3]
      self.assertAllEqual(
          semantic_outputs.numpy().shape,
          [2, h, w, num_classes])
      self.assertAllEqual(
          instance_outputs['instance_centers_heatmap'].numpy().shape,
          [2, h, w, 1])
      self.assertAllEqual(
          instance_outputs['instance_centers_offset'].numpy().shape,
          [2, h, w, 2])

  def test_serialize_deserialize(self):
    semantic_head = panoptic_deeplab_heads.SemanticHead(num_classes=2, level=3)
    instance_head = panoptic_deeplab_heads.InstanceHead(level=3)

    semantic_head_config = semantic_head.get_config()
    instance_head_config = instance_head.get_config()

    new_semantic_head = panoptic_deeplab_heads.SemanticHead.from_config(
        semantic_head_config)
    new_instance_head = panoptic_deeplab_heads.InstanceHead.from_config(
        instance_head_config)

    self.assertAllEqual(semantic_head.get_config(),
                        new_semantic_head.get_config())
    self.assertAllEqual(instance_head.get_config(),
                        new_instance_head.get_config())


if __name__ == '__main__':
  tf.test.main()
