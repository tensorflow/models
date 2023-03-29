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

"""Tests for instance_heads.py."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.vision.modeling.heads import instance_heads


class DetectionHeadTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (0, 0, False, False),
      (0, 1, False, False),
      (1, 0, False, False),
      (1, 1, False, False),
  )
  def test_forward(self, num_convs, num_fcs, use_separable_conv, use_sync_bn):
    detection_head = instance_heads.DetectionHead(
        num_classes=3,
        num_convs=num_convs,
        num_filters=16,
        use_separable_conv=use_separable_conv,
        num_fcs=num_fcs,
        fc_dims=4,
        activation='relu',
        use_sync_bn=use_sync_bn,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    roi_features = np.random.rand(2, 10, 128, 128, 16)
    scores, boxes = detection_head(roi_features)
    self.assertAllEqual(scores.numpy().shape, [2, 10, 3])
    self.assertAllEqual(boxes.numpy().shape, [2, 10, 12])

  def test_serialize_deserialize(self):
    detection_head = instance_heads.DetectionHead(
        num_classes=91,
        num_convs=0,
        num_filters=256,
        use_separable_conv=False,
        num_fcs=2,
        fc_dims=1024,
        activation='relu',
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    config = detection_head.get_config()
    new_detection_head = instance_heads.DetectionHead.from_config(config)
    self.assertAllEqual(
        detection_head.get_config(), new_detection_head.get_config())


class MaskHeadTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (1, 1, False),
      (1, 2, False),
      (2, 1, False),
      (2, 2, False),
  )
  def test_forward(self, upsample_factor, num_convs, use_sync_bn):
    mask_head = instance_heads.MaskHead(
        num_classes=3,
        upsample_factor=upsample_factor,
        num_convs=num_convs,
        num_filters=16,
        use_separable_conv=False,
        activation='relu',
        use_sync_bn=use_sync_bn,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    roi_features = np.random.rand(2, 10, 14, 14, 16)
    roi_classes = np.zeros((2, 10))
    masks = mask_head([roi_features, roi_classes])
    self.assertAllEqual(
        masks.numpy().shape,
        [2, 10, 14 * upsample_factor, 14 * upsample_factor])

  def test_serialize_deserialize(self):
    mask_head = instance_heads.MaskHead(
        num_classes=3,
        upsample_factor=2,
        num_convs=1,
        num_filters=256,
        use_separable_conv=False,
        activation='relu',
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    config = mask_head.get_config()
    new_mask_head = instance_heads.MaskHead.from_config(config)
    self.assertAllEqual(
        mask_head.get_config(), new_mask_head.get_config())

  def test_forward_class_agnostic(self):
    mask_head = instance_heads.MaskHead(
        num_classes=3,
        class_agnostic=True
    )
    roi_features = np.random.rand(2, 10, 14, 14, 16)
    roi_classes = np.zeros((2, 10))
    masks = mask_head([roi_features, roi_classes])
    self.assertAllEqual(masks.numpy().shape, [2, 10, 28, 28])


if __name__ == '__main__':
  tf.test.main()
