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

# Lint as: python3
"""Tests for dense_prediction_heads.py."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.projects.qat.vision.modeling.heads import dense_prediction_heads


class RetinaNetHeadQuantizedTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (False, False, False),
      (False, True, False),
      (True, False, True),
      (True, True, True),
  )
  def test_forward(self, use_separable_conv, use_sync_bn, has_att_heads):
    if has_att_heads:
      attribute_heads = [dict(name='depth', type='regression', size=1)]
    else:
      attribute_heads = None

    retinanet_head = dense_prediction_heads.RetinaNetHeadQuantized(
        min_level=3,
        max_level=4,
        num_classes=3,
        num_anchors_per_location=3,
        num_convs=2,
        num_filters=256,
        attribute_heads=attribute_heads,
        use_separable_conv=use_separable_conv,
        activation='relu',
        use_sync_bn=use_sync_bn,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    features = {
        '3': np.random.rand(2, 128, 128, 16),
        '4': np.random.rand(2, 64, 64, 16),
    }
    scores, boxes, attributes = retinanet_head(features)
    self.assertAllEqual(scores['3'].numpy().shape, [2, 128, 128, 9])
    self.assertAllEqual(scores['4'].numpy().shape, [2, 64, 64, 9])
    self.assertAllEqual(boxes['3'].numpy().shape, [2, 128, 128, 12])
    self.assertAllEqual(boxes['4'].numpy().shape, [2, 64, 64, 12])
    if has_att_heads:
      for att in attributes.values():
        self.assertAllEqual(att['3'].numpy().shape, [2, 128, 128, 3])
        self.assertAllEqual(att['4'].numpy().shape, [2, 64, 64, 3])

  def test_serialize_deserialize(self):
    retinanet_head = dense_prediction_heads.RetinaNetHeadQuantized(
        min_level=3,
        max_level=7,
        num_classes=3,
        num_anchors_per_location=9,
        num_convs=2,
        num_filters=16,
        attribute_heads=None,
        use_separable_conv=False,
        activation='relu',
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    config = retinanet_head.get_config()
    new_retinanet_head = (
        dense_prediction_heads.RetinaNetHead.from_config(config))
    self.assertAllEqual(
        retinanet_head.get_config(), new_retinanet_head.get_config())

