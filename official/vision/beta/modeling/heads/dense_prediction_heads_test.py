# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for dense_prediction_heads.py."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.vision.beta.modeling.heads import dense_prediction_heads


class RetinaNetHeadTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (False, False),
      (False, True),
      (True, False),
      (True, True),
  )
  def test_forward(self, use_separable_conv, use_sync_bn):
    retinanet_head = dense_prediction_heads.RetinaNetHead(
        min_level=3,
        max_level=4,
        num_classes=3,
        num_anchors_per_location=3,
        num_convs=2,
        num_filters=256,
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
    scores, boxes = retinanet_head(features)
    self.assertAllEqual(scores['3'].numpy().shape, [2, 128, 128, 9])
    self.assertAllEqual(scores['4'].numpy().shape, [2, 64, 64, 9])
    self.assertAllEqual(boxes['3'].numpy().shape, [2, 128, 128, 12])
    self.assertAllEqual(boxes['4'].numpy().shape, [2, 64, 64, 12])

  def test_serialize_deserialize(self):
    retinanet_head = dense_prediction_heads.RetinaNetHead(
        min_level=3,
        max_level=7,
        num_classes=3,
        num_anchors_per_location=9,
        num_convs=2,
        num_filters=16,
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


class RpnHeadTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (False, False),
      (False, True),
      (True, False),
      (True, True),
  )
  def test_forward(self, use_separable_conv, use_sync_bn):
    rpn_head = dense_prediction_heads.RPNHead(
        min_level=3,
        max_level=4,
        num_anchors_per_location=3,
        num_convs=2,
        num_filters=256,
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
    scores, boxes = rpn_head(features)
    self.assertAllEqual(scores['3'].numpy().shape, [2, 128, 128, 3])
    self.assertAllEqual(scores['4'].numpy().shape, [2, 64, 64, 3])
    self.assertAllEqual(boxes['3'].numpy().shape, [2, 128, 128, 12])
    self.assertAllEqual(boxes['4'].numpy().shape, [2, 64, 64, 12])

  def test_serialize_deserialize(self):
    rpn_head = dense_prediction_heads.RPNHead(
        min_level=3,
        max_level=7,
        num_anchors_per_location=9,
        num_convs=2,
        num_filters=16,
        use_separable_conv=False,
        activation='relu',
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    config = rpn_head.get_config()
    new_rpn_head = dense_prediction_heads.RPNHead.from_config(config)
    self.assertAllEqual(rpn_head.get_config(), new_rpn_head.get_config())


if __name__ == '__main__':
  tf.test.main()
