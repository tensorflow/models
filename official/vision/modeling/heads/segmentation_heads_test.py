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

"""Tests for segmentation_heads.py."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.vision.modeling.heads import segmentation_heads


class SegmentationHeadTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (2, 'pyramid_fusion', None, None),
      (3, 'pyramid_fusion', None, None),
      (2, 'panoptic_fpn_fusion', 2, 5),
      (2, 'panoptic_fpn_fusion', 2, 6),
      (3, 'panoptic_fpn_fusion', 3, 5),
      (3, 'panoptic_fpn_fusion', 3, 6),
      (3, 'deeplabv3plus', 3, 6),
      (3, 'deeplabv3plus_sum_to_merge', 3, 6))
  def test_forward(self, level, feature_fusion,
                   decoder_min_level, decoder_max_level):
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

    if feature_fusion == 'panoptic_fpn_fusion':
      backbone_features['2'] = np.random.rand(2, 256, 256, 16)
      decoder_features['2'] = np.random.rand(2, 256, 256, 64)

    head = segmentation_heads.SegmentationHead(
        num_classes=10,
        level=level,
        low_level=decoder_min_level,
        low_level_num_filters=64,
        feature_fusion=feature_fusion,
        decoder_min_level=decoder_min_level,
        decoder_max_level=decoder_max_level,
        num_decoder_filters=64)

    logits = head((backbone_features, decoder_features))

    if str(level) in decoder_features:
      self.assertAllEqual(logits.numpy().shape, [
          2, decoder_features[str(level)].shape[1],
          decoder_features[str(level)].shape[2], 10
      ])

  def test_serialize_deserialize(self):
    head = segmentation_heads.SegmentationHead(num_classes=10, level=3)
    config = head.get_config()
    new_head = segmentation_heads.SegmentationHead.from_config(config)
    self.assertAllEqual(head.get_config(), new_head.get_config())


class MaskScoringHeadTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (1, 1, 64, [4, 4]),
      (2, 1, 64, [4, 4]),
      (3, 1, 64, [4, 4]),
      (1, 2, 32, [8, 8]),
      (2, 2, 32, [8, 8]),
      (3, 2, 32, [8, 8]),)
  def test_forward(self, num_convs, num_fcs,
                   num_filters, fc_input_size):
    features = np.random.rand(2, 64, 64, 16)

    head = segmentation_heads.MaskScoring(
        num_classes=2,
        num_convs=num_convs,
        num_filters=num_filters,
        fc_dims=128,
        num_fcs=num_fcs,
        fc_input_size=fc_input_size)

    scores = head(features)
    self.assertAllEqual(scores.numpy().shape, [2, 2])

  def test_serialize_deserialize(self):
    head = segmentation_heads.MaskScoring(
        num_classes=2, fc_input_size=[4, 4], fc_dims=128)
    config = head.get_config()
    new_head = segmentation_heads.MaskScoring.from_config(config)
    self.assertAllEqual(head.get_config(), new_head.get_config())

if __name__ == '__main__':
  tf.test.main()
