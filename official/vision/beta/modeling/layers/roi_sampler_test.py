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
"""Tests for roi_sampler.py."""

# Import libraries
import numpy as np
import tensorflow as tf

from official.vision.beta.modeling.layers import roi_sampler


class ROISamplerTest(tf.test.TestCase):

  def test_roi_sampler(self):
    boxes_np = np.array(
        [[[0, 0, 5, 5], [2.5, 2.5, 7.5, 7.5],
          [5, 5, 10, 10], [7.5, 7.5, 12.5, 12.5]]])
    boxes = tf.constant(boxes_np, dtype=tf.float32)

    gt_boxes_np = np.array(
        [[[10, 10, 15, 15], [2.5, 2.5, 7.5, 7.5],
          [-1, -1, -1, -1]]])
    gt_boxes = tf.constant(gt_boxes_np, dtype=tf.float32)
    gt_classes_np = np.array([[2, 10, -1]])
    gt_classes = tf.constant(gt_classes_np, dtype=tf.int32)

    generator = roi_sampler.ROISampler(
        mix_gt_boxes=True,
        num_sampled_rois=2,
        foreground_fraction=0.5,
        foreground_iou_threshold=0.5,
        background_iou_high_threshold=0.5,
        background_iou_low_threshold=0.0)

    # Runs on TPU.
    strategy = tf.distribute.experimental.TPUStrategy()
    with strategy.scope():
      _ = generator(boxes, gt_boxes, gt_classes)

    # Runs on CPU.
    _ = generator(boxes, gt_boxes, gt_classes)

  def test_serialize_deserialize(self):
    kwargs = dict(
        mix_gt_boxes=True,
        num_sampled_rois=512,
        foreground_fraction=0.25,
        foreground_iou_threshold=0.5,
        background_iou_high_threshold=0.5,
        background_iou_low_threshold=0.5,
    )
    generator = roi_sampler.ROISampler(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(generator.get_config(), expected_config)

    new_generator = roi_sampler.ROISampler.from_config(
        generator.get_config())

    self.assertAllEqual(generator.get_config(), new_generator.get_config())


if __name__ == '__main__':
  tf.test.main()
