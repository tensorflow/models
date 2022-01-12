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
"""Tests for segmentation_heads.py."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.vision.beta.projects.panoptic_maskrcnn.modeling.heads import instance_center_head


class InstanceCenterHeadTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (2, 'pyramid_fusion', None, None, 2, 48),
      (3, 'pyramid_fusion', None, None, 2, 48),
      (2, 'panoptic_fpn_fusion', 2, 5, 2, 48),
      (2, 'panoptic_fpn_fusion', 2, 6, 2, 48),
      (3, 'panoptic_fpn_fusion', 3, 5, 2, 48),
      (3, 'panoptic_fpn_fusion', 3, 6, 2, 48),
      (4, 'panoptic_deeplab_fusion', None, None, (4, 3), (64, 32)),
      (4, 'panoptic_deeplab_fusion', None, None, (3, 2), (64, 32)))
  def test_forward(self, level, feature_fusion,
                   decoder_min_level, decoder_max_level,
                   low_level, low_level_num_filters):
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

    if 'panoptic' in feature_fusion:
      backbone_features['2'] = np.random.rand(2, 256, 256, 16)
      decoder_features['2'] = np.random.rand(2, 256, 256, 64)

    head = instance_center_head.InstanceCenterHead(
        level=level,
        low_level=low_level,
        low_level_num_filters=low_level_num_filters,
        feature_fusion=feature_fusion,
        decoder_min_level=decoder_min_level,
        decoder_max_level=decoder_max_level,
        num_decoder_filters=64)

    outputs = head((backbone_features, decoder_features))

    if str(level) in decoder_features:
      if feature_fusion == 'panoptic_deeplab_fusion':
        h, w = decoder_features[str(low_level[-1])].shape[1:3]
      else:
        h, w = decoder_features[str(level)].shape[1:3]
      self.assertAllEqual(
          outputs['instance_center_prediction'].numpy().shape,
          [2, h, w, 1])
      self.assertAllEqual(
          outputs['instance_center_regression'].numpy().shape,
          [2, h, w, 2])


  def test_serialize_deserialize(self):
    head = instance_center_head.InstanceCenterHead(level=3)
    config = head.get_config()
    new_head = instance_center_head.InstanceCenterHead.from_config(config)
    self.assertAllEqual(head.get_config(), new_head.get_config())

if __name__ == '__main__':
  tf.test.main()
