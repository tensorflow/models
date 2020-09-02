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
"""Tests for mask_sampler.py."""

# Import libraries
import numpy as np
import tensorflow as tf

from official.vision.beta.modeling.layers import mask_sampler


class SampleAndCropForegroundMasksTest(tf.test.TestCase):

  def test_sample_and_crop_foreground_masks(self):
    candidate_rois_np = np.array(
        [[[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1],
          [2, 2, 4, 4], [1, 1, 5, 5]]])
    candidate_rois = tf.constant(candidate_rois_np, dtype=tf.float32)

    candidate_gt_boxes_np = np.array(
        [[[0, 0, 0.6, 0.6], [0, 0, 0, 0],
          [1, 1, 3, 3], [1, 1, 3, 3]]])
    candidate_gt_boxes = tf.constant(candidate_gt_boxes_np, dtype=tf.float32)

    candidate_gt_classes_np = np.array([[4, 0, 0, 2]])
    candidate_gt_classes = tf.constant(
        candidate_gt_classes_np, dtype=tf.float32)

    candidate_gt_indices_np = np.array([[10, -1, -1, 20]])
    candidate_gt_indices = tf.constant(
        candidate_gt_indices_np, dtype=tf.int32)

    gt_masks_np = np.random.rand(1, 100, 32, 32)
    gt_masks = tf.constant(gt_masks_np, dtype=tf.float32)

    num_mask_samples_per_image = 2
    mask_target_size = 28

    # Runs on TPU.
    strategy = tf.distribute.experimental.TPUStrategy()
    with strategy.scope():
      foreground_rois, foreground_classes, cropped_foreground_masks = (
          mask_sampler._sample_and_crop_foreground_masks(
              candidate_rois, candidate_gt_boxes, candidate_gt_classes,
              candidate_gt_indices, gt_masks, num_mask_samples_per_image,
              mask_target_size))
    foreground_rois_tpu = foreground_rois.numpy()
    foreground_classes_tpu = foreground_classes.numpy()
    cropped_foreground_masks_tpu = cropped_foreground_masks.numpy()

    foreground_rois, foreground_classes, cropped_foreground_masks = (
        mask_sampler._sample_and_crop_foreground_masks(
            candidate_rois, candidate_gt_boxes, candidate_gt_classes,
            candidate_gt_indices, gt_masks, num_mask_samples_per_image,
            mask_target_size))
    foreground_rois_cpu = foreground_rois.numpy()
    foreground_classes_cpu = foreground_classes.numpy()
    cropped_foreground_masks_cpu = cropped_foreground_masks.numpy()

    # consistency.
    self.assertAllEqual(foreground_rois_tpu.shape, foreground_rois_cpu.shape)
    self.assertAllEqual(
        foreground_classes_tpu.shape, foreground_classes_cpu.shape)
    self.assertAllEqual(
        cropped_foreground_masks_tpu.shape, cropped_foreground_masks_cpu.shape)

    # correctnesss.
    self.assertAllEqual(foreground_rois_tpu.shape, [1, 2, 4])
    self.assertAllEqual(foreground_classes_tpu.shape, [1, 2])
    self.assertAllEqual(cropped_foreground_masks_tpu.shape, [1, 2, 28, 28])


class MaskSamplerTest(tf.test.TestCase):

  def test_mask_sampler(self):
    candidate_rois_np = np.array(
        [[[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1],
          [2, 2, 4, 4], [1, 1, 5, 5]]])
    candidate_rois = tf.constant(candidate_rois_np, dtype=tf.float32)

    candidate_gt_boxes_np = np.array(
        [[[0, 0, 0.6, 0.6], [0, 0, 0, 0],
          [1, 1, 3, 3], [1, 1, 3, 3]]])
    candidate_gt_boxes = tf.constant(candidate_gt_boxes_np, dtype=tf.float32)

    candidate_gt_classes_np = np.array([[4, 0, 0, 2]])
    candidate_gt_classes = tf.constant(
        candidate_gt_classes_np, dtype=tf.float32)

    candidate_gt_indices_np = np.array([[10, -1, -1, 20]])
    candidate_gt_indices = tf.constant(
        candidate_gt_indices_np, dtype=tf.int32)

    gt_masks_np = np.random.rand(1, 100, 32, 32)
    gt_masks = tf.constant(gt_masks_np, dtype=tf.float32)

    sampler = mask_sampler.MaskSampler(28, 2)

    foreground_rois, foreground_classes, cropped_foreground_masks = sampler(
        candidate_rois, candidate_gt_boxes, candidate_gt_classes,
        candidate_gt_indices, gt_masks)

    # correctnesss.
    self.assertAllEqual(foreground_rois.numpy().shape, [1, 2, 4])
    self.assertAllEqual(foreground_classes.numpy().shape, [1, 2])
    self.assertAllEqual(cropped_foreground_masks.numpy().shape, [1, 2, 28, 28])

  def test_serialize_deserialize(self):
    kwargs = dict(
        mask_target_size=7,
        num_sampled_masks=10,
    )
    sampler = mask_sampler.MaskSampler(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(sampler.get_config(), expected_config)

    new_sampler = mask_sampler.MaskSampler.from_config(
        sampler.get_config())

    self.assertAllEqual(sampler.get_config(), new_sampler.get_config())


if __name__ == '__main__':
  tf.test.main()
