# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for dense_prediction_heads.py."""

import unittest

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from tensorflow.python.distribute import combinations
from official.vision.modeling.heads import dense_prediction_heads


def get_attribute_heads(att_head_type):
  if att_head_type == 'regression_head':
    return [
        dict(
            name='depth',
            type='regression',
            size=1,
            prediction_tower_name='',
            num_convs=1,
            num_filters=128,
        )
    ]
  elif att_head_type == 'classification_head':
    return [
        dict(
            name='depth',
            type='classification',
            size=1,
            prediction_tower_name='')
    ]
  elif att_head_type == 'shared_prediction_tower_attribute_heads':
    return [
        dict(
            name='attr_1', type='regression', size=1, prediction_tower_name=''
        ),
        dict(
            name='attr_2',
            type='classification',
            size=1,
            prediction_tower_name='tower_1',
        ),
        dict(
            name='attr_3',
            type='regression',
            size=1,
            prediction_tower_name='tower_1',
        ),
    ]
  else:
    raise ValueError('Undefined attribute type.')


class RetinaNetHeadTest(parameterized.TestCase, tf.test.TestCase):

  @combinations.generate(
      combinations.combine(
          use_separable_conv=[True, False],
          use_sync_bn=[True, False],
          share_level_convs=[True, False],
      )
  )
  def test_forward_without_attribute_head(
      self, use_separable_conv, use_sync_bn, share_level_convs
  ):
    retinanet_head = dense_prediction_heads.RetinaNetHead(
        min_level=3,
        max_level=4,
        num_classes=3,
        num_anchors_per_location=3,
        num_convs=2,
        num_filters=256,
        attribute_heads=None,
        use_separable_conv=use_separable_conv,
        activation='relu',
        use_sync_bn=use_sync_bn,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
        share_level_convs=share_level_convs,
    )
    features = {
        '3': np.random.rand(2, 128, 128, 16),
        '4': np.random.rand(2, 64, 64, 16),
    }
    scores, boxes, _ = retinanet_head(features)
    self.assertAllEqual(scores['3'].numpy().shape, [2, 128, 128, 9])
    self.assertAllEqual(scores['4'].numpy().shape, [2, 64, 64, 9])
    self.assertAllEqual(boxes['3'].numpy().shape, [2, 128, 128, 12])
    self.assertAllEqual(boxes['4'].numpy().shape, [2, 64, 64, 12])

  @parameterized.parameters(
      (False, 'regression_head', False),
      (True, 'classification_head', True),
      (True, 'shared_prediction_tower_attribute_heads', False),
  )
  def test_forward_with_attribute_head(
      self,
      use_sync_bn,
      att_head_type,
      share_classification_heads,
  ):
    retinanet_head = dense_prediction_heads.RetinaNetHead(
        min_level=3,
        max_level=4,
        num_classes=3,
        num_anchors_per_location=3,
        num_convs=2,
        num_filters=256,
        attribute_heads=get_attribute_heads(att_head_type),
        share_classification_heads=share_classification_heads,
        use_separable_conv=True,
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
    for att in attributes.values():
      self.assertAllEqual(att['3'].numpy().shape, [2, 128, 128, 3])
      self.assertAllEqual(att['4'].numpy().shape, [2, 64, 64, 3])
    if att_head_type == 'regression_head':
      self.assertLen(retinanet_head._att_convs['depth'], 1)
      self.assertEqual(retinanet_head._att_convs['depth'][0].filters, 128)

  @unittest.expectedFailure
  def test_forward_shared_prediction_tower_with_share_classification_heads(
      self):
    share_classification_heads = True
    attribute_heads = get_attribute_heads(
        'shared_prediction_tower_attribute_heads')

    retinanet_head = dense_prediction_heads.RetinaNetHead(
        min_level=3,
        max_level=4,
        num_classes=3,
        num_anchors_per_location=3,
        num_convs=2,
        num_filters=256,
        attribute_heads=attribute_heads,
        share_classification_heads=share_classification_heads,
        use_separable_conv=True,
        activation='relu',
        use_sync_bn=True,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    features = {
        '3': np.random.rand(2, 128, 128, 16),
        '4': np.random.rand(2, 64, 64, 16),
    }
    retinanet_head(features)

  def test_forward_with_num_anchors_per_location_by_level(self):
    bs = 2
    retinanet_head = dense_prediction_heads.RetinaNetHead(
        min_level=3,
        max_level=4,
        num_classes=7,
        num_anchors_per_location={'3': 2, '4': 5},
        num_convs=0,
        num_filters=123,
        attribute_heads=None,
        share_level_convs=False,
    )
    features = {
        '3': np.random.rand(bs, 32, 32, 11),
        '4': np.random.rand(bs, 16, 16, 13),
    }
    scores, boxes, _ = retinanet_head(features)
    self.assertAllEqual(scores['3'].numpy().shape, [bs, 32, 32, 2 * 7])
    self.assertAllEqual(boxes['3'].numpy().shape, [bs, 32, 32, 2 * 4])
    self.assertAllEqual(scores['4'].numpy().shape, [bs, 16, 16, 5 * 7])
    self.assertAllEqual(boxes['4'].numpy().shape, [bs, 16, 16, 5 * 4])

  def test_serialize_deserialize(self):
    retinanet_head = dense_prediction_heads.RetinaNetHead(
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
