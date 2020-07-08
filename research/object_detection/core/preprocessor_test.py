# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for object_detection.core.preprocessor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from absl.testing import parameterized
import numpy as np
import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

from object_detection.core import preprocessor
from object_detection.core import preprocessor_cache
from object_detection.core import standard_fields as fields
from object_detection.utils import test_case
from object_detection.utils import tf_version

if six.PY2:
  import mock  # pylint: disable=g-import-not-at-top
else:
  mock = unittest.mock  # pylint: disable=g-import-not-at-top


class PreprocessorTest(test_case.TestCase, parameterized.TestCase):

  def createColorfulTestImage(self):
    ch255 = tf.fill([1, 100, 200, 1], tf.constant(255, dtype=tf.uint8))
    ch128 = tf.fill([1, 100, 200, 1], tf.constant(128, dtype=tf.uint8))
    ch0 = tf.fill([1, 100, 200, 1], tf.constant(0, dtype=tf.uint8))
    imr = tf.concat([ch255, ch0, ch0], 3)
    img = tf.concat([ch255, ch255, ch0], 3)
    imb = tf.concat([ch255, ch0, ch255], 3)
    imw = tf.concat([ch128, ch128, ch128], 3)
    imu = tf.concat([imr, img], 2)
    imd = tf.concat([imb, imw], 2)
    im = tf.concat([imu, imd], 1)
    return im

  def createTestImages(self):
    images_r = tf.constant([[[128, 128, 128, 128], [0, 0, 128, 128],
                             [0, 128, 128, 128], [192, 192, 128, 128]]],
                           dtype=tf.uint8)
    images_r = tf.expand_dims(images_r, 3)
    images_g = tf.constant([[[0, 0, 128, 128], [0, 0, 128, 128],
                             [0, 128, 192, 192], [192, 192, 128, 192]]],
                           dtype=tf.uint8)
    images_g = tf.expand_dims(images_g, 3)
    images_b = tf.constant([[[128, 128, 192, 0], [0, 0, 128, 192],
                             [0, 128, 128, 0], [192, 192, 192, 128]]],
                           dtype=tf.uint8)
    images_b = tf.expand_dims(images_b, 3)
    images = tf.concat([images_r, images_g, images_b], 3)
    return images

  def createEmptyTestBoxes(self):
    boxes = tf.constant([[]], dtype=tf.float32)
    return boxes

  def createTestBoxes(self):
    boxes = tf.constant(
        [[0.0, 0.25, 0.75, 1.0], [0.25, 0.5, 0.75, 1.0]], dtype=tf.float32)
    return boxes

  def createTestGroundtruthWeights(self):
    return tf.constant([1.0, 0.5], dtype=tf.float32)

  def createTestMasks(self):
    mask = np.array([
        [[255.0, 0.0, 0.0],
         [255.0, 0.0, 0.0],
         [255.0, 0.0, 0.0]],
        [[255.0, 255.0, 0.0],
         [255.0, 255.0, 0.0],
         [255.0, 255.0, 0.0]]])
    return tf.constant(mask, dtype=tf.float32)

  def createTestKeypoints(self):
    keypoints_np = np.array([
        [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
        [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]],
    ])
    keypoints = tf.constant(keypoints_np, dtype=tf.float32)
    keypoint_visibilities = tf.constant(
        [
            [True, True, False],
            [False, True, True]
        ])
    return keypoints, keypoint_visibilities

  def createTestKeypointsInsideCrop(self):
    keypoints = np.array([
        [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]],
        [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]],
    ])
    return tf.constant(keypoints, dtype=tf.float32)

  def createTestKeypointsOutsideCrop(self):
    keypoints = np.array([
        [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
        [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
    ])
    return tf.constant(keypoints, dtype=tf.float32)

  def createTestDensePose(self):
    dp_num_points = tf.constant([1, 3], dtype=tf.int32)
    dp_part_ids = tf.constant(
        [[4, 0, 0],
         [1, 0, 5]], dtype=tf.int32)
    dp_surface_coords = tf.constant(
        [
            # Instance 0.
            [[0.1, 0.2, 0.6, 0.7],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0]],
            # Instance 1.
            [[0.8, 0.9, 0.2, 0.4],
             [0.1, 0.3, 0.2, 0.8],
             [0.6, 1.0, 0.3, 0.4]],
        ], dtype=tf.float32)
    return dp_num_points, dp_part_ids, dp_surface_coords

  def createKeypointFlipPermutation(self):
    return [0, 2, 1]

  def createKeypointRotPermutation(self):
    return [0, 2, 1]

  def createTestLabels(self):
    labels = tf.constant([1, 2], dtype=tf.int32)
    return labels

  def createTestLabelsLong(self):
    labels = tf.constant([1, 2, 4], dtype=tf.int32)
    return labels

  def createTestBoxesOutOfImage(self):
    boxes = tf.constant(
        [[-0.1, 0.25, 0.75, 1], [0.25, 0.5, 0.75, 1.1]], dtype=tf.float32)
    return boxes

  def createTestMultiClassScores(self):
    return tf.constant([[1.0, 0.0], [0.5, 0.5]], dtype=tf.float32)

  def expectedImagesAfterNormalization(self):
    images_r = tf.constant([[[0, 0, 0, 0], [-1, -1, 0, 0],
                             [-1, 0, 0, 0], [0.5, 0.5, 0, 0]]],
                           dtype=tf.float32)
    images_r = tf.expand_dims(images_r, 3)
    images_g = tf.constant([[[-1, -1, 0, 0], [-1, -1, 0, 0],
                             [-1, 0, 0.5, 0.5], [0.5, 0.5, 0, 0.5]]],
                           dtype=tf.float32)
    images_g = tf.expand_dims(images_g, 3)
    images_b = tf.constant([[[0, 0, 0.5, -1], [-1, -1, 0, 0.5],
                             [-1, 0, 0, -1], [0.5, 0.5, 0.5, 0]]],
                           dtype=tf.float32)
    images_b = tf.expand_dims(images_b, 3)
    images = tf.concat([images_r, images_g, images_b], 3)
    return images

  def expectedMaxImageAfterColorScale(self):
    images_r = tf.constant([[[0.1, 0.1, 0.1, 0.1], [-0.9, -0.9, 0.1, 0.1],
                             [-0.9, 0.1, 0.1, 0.1], [0.6, 0.6, 0.1, 0.1]]],
                           dtype=tf.float32)
    images_r = tf.expand_dims(images_r, 3)
    images_g = tf.constant([[[-0.9, -0.9, 0.1, 0.1], [-0.9, -0.9, 0.1, 0.1],
                             [-0.9, 0.1, 0.6, 0.6], [0.6, 0.6, 0.1, 0.6]]],
                           dtype=tf.float32)
    images_g = tf.expand_dims(images_g, 3)
    images_b = tf.constant([[[0.1, 0.1, 0.6, -0.9], [-0.9, -0.9, 0.1, 0.6],
                             [-0.9, 0.1, 0.1, -0.9], [0.6, 0.6, 0.6, 0.1]]],
                           dtype=tf.float32)
    images_b = tf.expand_dims(images_b, 3)
    images = tf.concat([images_r, images_g, images_b], 3)
    return images

  def expectedMinImageAfterColorScale(self):
    images_r = tf.constant([[[-0.1, -0.1, -0.1, -0.1], [-1, -1, -0.1, -0.1],
                             [-1, -0.1, -0.1, -0.1], [0.4, 0.4, -0.1, -0.1]]],
                           dtype=tf.float32)
    images_r = tf.expand_dims(images_r, 3)
    images_g = tf.constant([[[-1, -1, -0.1, -0.1], [-1, -1, -0.1, -0.1],
                             [-1, -0.1, 0.4, 0.4], [0.4, 0.4, -0.1, 0.4]]],
                           dtype=tf.float32)
    images_g = tf.expand_dims(images_g, 3)
    images_b = tf.constant([[[-0.1, -0.1, 0.4, -1], [-1, -1, -0.1, 0.4],
                             [-1, -0.1, -0.1, -1], [0.4, 0.4, 0.4, -0.1]]],
                           dtype=tf.float32)
    images_b = tf.expand_dims(images_b, 3)
    images = tf.concat([images_r, images_g, images_b], 3)
    return images

  def expectedImagesAfterLeftRightFlip(self):
    images_r = tf.constant([[[0, 0, 0, 0], [0, 0, -1, -1],
                             [0, 0, 0, -1], [0, 0, 0.5, 0.5]]],
                           dtype=tf.float32)
    images_r = tf.expand_dims(images_r, 3)
    images_g = tf.constant([[[0, 0, -1, -1], [0, 0, -1, -1],
                             [0.5, 0.5, 0, -1], [0.5, 0, 0.5, 0.5]]],
                           dtype=tf.float32)
    images_g = tf.expand_dims(images_g, 3)
    images_b = tf.constant([[[-1, 0.5, 0, 0], [0.5, 0, -1, -1],
                             [-1, 0, 0, -1], [0, 0.5, 0.5, 0.5]]],
                           dtype=tf.float32)
    images_b = tf.expand_dims(images_b, 3)
    images = tf.concat([images_r, images_g, images_b], 3)
    return images

  def expectedImagesAfterUpDownFlip(self):
    images_r = tf.constant([[[0.5, 0.5, 0, 0], [-1, 0, 0, 0],
                             [-1, -1, 0, 0], [0, 0, 0, 0]]],
                           dtype=tf.float32)
    images_r = tf.expand_dims(images_r, 3)
    images_g = tf.constant([[[0.5, 0.5, 0, 0.5], [-1, 0, 0.5, 0.5],
                             [-1, -1, 0, 0], [-1, -1, 0, 0]]],
                           dtype=tf.float32)
    images_g = tf.expand_dims(images_g, 3)
    images_b = tf.constant([[[0.5, 0.5, 0.5, 0], [-1, 0, 0, -1],
                             [-1, -1, 0, 0.5], [0, 0, 0.5, -1]]],
                           dtype=tf.float32)
    images_b = tf.expand_dims(images_b, 3)
    images = tf.concat([images_r, images_g, images_b], 3)
    return images

  def expectedImagesAfterRot90(self):
    images_r = tf.constant([[[0, 0, 0, 0], [0, 0, 0, 0],
                             [0, -1, 0, 0.5], [0, -1, -1, 0.5]]],
                           dtype=tf.float32)
    images_r = tf.expand_dims(images_r, 3)
    images_g = tf.constant([[[0, 0, 0.5, 0.5], [0, 0, 0.5, 0],
                             [-1, -1, 0, 0.5], [-1, -1, -1, 0.5]]],
                           dtype=tf.float32)
    images_g = tf.expand_dims(images_g, 3)
    images_b = tf.constant([[[-1, 0.5, -1, 0], [0.5, 0, 0, 0.5],
                             [0, -1, 0, 0.5], [0, -1, -1, 0.5]]],
                           dtype=tf.float32)
    images_b = tf.expand_dims(images_b, 3)
    images = tf.concat([images_r, images_g, images_b], 3)
    return images

  def expectedBoxesAfterLeftRightFlip(self):
    boxes = tf.constant([[0.0, 0.0, 0.75, 0.75], [0.25, 0.0, 0.75, 0.5]],
                        dtype=tf.float32)
    return boxes

  def expectedBoxesAfterUpDownFlip(self):
    boxes = tf.constant([[0.25, 0.25, 1.0, 1.0], [0.25, 0.5, 0.75, 1.0]],
                        dtype=tf.float32)
    return boxes

  def expectedBoxesAfterRot90(self):
    boxes = tf.constant(
        [[0.0, 0.0, 0.75, 0.75], [0.0, 0.25, 0.5, 0.75]], dtype=tf.float32)
    return boxes

  def expectedMasksAfterLeftRightFlip(self):
    mask = np.array([
        [[0.0, 0.0, 255.0],
         [0.0, 0.0, 255.0],
         [0.0, 0.0, 255.0]],
        [[0.0, 255.0, 255.0],
         [0.0, 255.0, 255.0],
         [0.0, 255.0, 255.0]]])
    return tf.constant(mask, dtype=tf.float32)

  def expectedMasksAfterUpDownFlip(self):
    mask = np.array([
        [[255.0, 0.0, 0.0],
         [255.0, 0.0, 0.0],
         [255.0, 0.0, 0.0]],
        [[255.0, 255.0, 0.0],
         [255.0, 255.0, 0.0],
         [255.0, 255.0, 0.0]]])
    return tf.constant(mask, dtype=tf.float32)

  def expectedMasksAfterRot90(self):
    mask = np.array([
        [[0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [255.0, 255.0, 255.0]],
        [[0.0, 0.0, 0.0],
         [255.0, 255.0, 255.0],
         [255.0, 255.0, 255.0]]])
    return tf.constant(mask, dtype=tf.float32)

  def expectedLabelScoresAfterThresholding(self):
    return tf.constant([1.0], dtype=tf.float32)

  def expectedBoxesAfterThresholding(self):
    return tf.constant([[0.0, 0.25, 0.75, 1.0]], dtype=tf.float32)

  def expectedLabelsAfterThresholding(self):
    return tf.constant([1], dtype=tf.float32)

  def expectedMultiClassScoresAfterThresholding(self):
    return tf.constant([[1.0, 0.0]], dtype=tf.float32)

  def expectedMasksAfterThresholding(self):
    mask = np.array([
        [[255.0, 0.0, 0.0],
         [255.0, 0.0, 0.0],
         [255.0, 0.0, 0.0]]])
    return tf.constant(mask, dtype=tf.float32)

  def expectedKeypointsAfterThresholding(self):
    keypoints = np.array([
        [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
    ])
    return tf.constant(keypoints, dtype=tf.float32)

  def expectedLabelScoresAfterThresholdingWithMissingScore(self):
    return tf.constant([np.nan], dtype=tf.float32)

  def expectedBoxesAfterThresholdingWithMissingScore(self):
    return tf.constant([[0.25, 0.5, 0.75, 1]], dtype=tf.float32)

  def expectedLabelsAfterThresholdingWithMissingScore(self):
    return tf.constant([2], dtype=tf.float32)

  def expectedLabelScoresAfterDropping(self):
    return tf.constant([0.5], dtype=tf.float32)

  def expectedBoxesAfterDropping(self):
    return tf.constant([[0.25, 0.5, 0.75, 1.0]], dtype=tf.float32)

  def expectedLabelsAfterDropping(self):
    return tf.constant([2], dtype=tf.float32)

  def expectedMultiClassScoresAfterDropping(self):
    return tf.constant([[0.5, 0.5]], dtype=tf.float32)

  def expectedMasksAfterDropping(self):
    masks = np.array([[[255.0, 255.0, 0.0], [255.0, 255.0, 0.0],
                       [255.0, 255.0, 0.0]]])
    return tf.constant(masks, dtype=tf.float32)

  def expectedKeypointsAfterDropping(self):
    keypoints = np.array([[[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]]])
    return tf.constant(keypoints, dtype=tf.float32)

  def expectedLabelsAfterRemapping(self):
    return tf.constant([3, 3, 4], dtype=tf.float32)

  def testRgbToGrayscale(self):
    def graph_fn():
      images = self.createTestImages()
      grayscale_images = preprocessor._rgb_to_grayscale(images)
      expected_images = tf.image.rgb_to_grayscale(images)
      return grayscale_images, expected_images
    (grayscale_images, expected_images) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(expected_images, grayscale_images)

  def testNormalizeImage(self):
    def graph_fn():
      preprocess_options = [(preprocessor.normalize_image, {
          'original_minval': 0,
          'original_maxval': 256,
          'target_minval': -1,
          'target_maxval': 1
      })]
      images = self.createTestImages()
      tensor_dict = {fields.InputDataFields.image: images}
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocess_options)
      images = tensor_dict[fields.InputDataFields.image]
      images_expected = self.expectedImagesAfterNormalization()
      return images, images_expected
    images_, images_expected_ = self.execute_cpu(graph_fn, [])
    images_shape_ = images_.shape
    images_expected_shape_ = images_expected_.shape
    expected_shape = [1, 4, 4, 3]
    self.assertAllEqual(images_expected_shape_, images_shape_)
    self.assertAllEqual(images_shape_, expected_shape)
    self.assertAllClose(images_, images_expected_)

  def testRetainBoxesAboveThreshold(self):
    def graph_fn():
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      (retained_boxes, retained_labels,
       retained_weights) = preprocessor.retain_boxes_above_threshold(
           boxes, labels, weights, threshold=0.6)
      return [
          retained_boxes, retained_labels, retained_weights,
          self.expectedBoxesAfterThresholding(),
          self.expectedLabelsAfterThresholding(),
          self.expectedLabelScoresAfterThresholding()
      ]

    (retained_boxes_, retained_labels_, retained_weights_,
     expected_retained_boxes_, expected_retained_labels_,
     expected_retained_weights_) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(
        retained_boxes_, expected_retained_boxes_)
    self.assertAllClose(
        retained_labels_, expected_retained_labels_)
    self.assertAllClose(
        retained_weights_, expected_retained_weights_)

  def testRetainBoxesAboveThresholdWithMultiClassScores(self):
    def graph_fn():
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      multiclass_scores = self.createTestMultiClassScores()
      (_, _, _,
       retained_multiclass_scores) = preprocessor.retain_boxes_above_threshold(
           boxes,
           labels,
           weights,
           multiclass_scores=multiclass_scores,
           threshold=0.6)
      return [
          retained_multiclass_scores,
          self.expectedMultiClassScoresAfterThresholding()
      ]

    (retained_multiclass_scores_,
     expected_retained_multiclass_scores_) = self.execute(graph_fn, [])
    self.assertAllClose(retained_multiclass_scores_,
                        expected_retained_multiclass_scores_)

  def testRetainBoxesAboveThresholdWithMasks(self):
    def graph_fn():
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      masks = self.createTestMasks()
      _, _, _, retained_masks = preprocessor.retain_boxes_above_threshold(
          boxes, labels, weights, masks, threshold=0.6)
      return [
          retained_masks, self.expectedMasksAfterThresholding()]
    retained_masks_, expected_retained_masks_ = self.execute_cpu(graph_fn, [])

    self.assertAllClose(
        retained_masks_, expected_retained_masks_)

  def testRetainBoxesAboveThresholdWithKeypoints(self):
    def graph_fn():
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      keypoints, _ = self.createTestKeypoints()
      (_, _, _, retained_keypoints) = preprocessor.retain_boxes_above_threshold(
          boxes, labels, weights, keypoints=keypoints, threshold=0.6)
      return [retained_keypoints, self.expectedKeypointsAfterThresholding()]

    (retained_keypoints_,
     expected_retained_keypoints_) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(retained_keypoints_, expected_retained_keypoints_)

  def testDropLabelProbabilistically(self):
    def graph_fn():
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      (retained_boxes, retained_labels,
       retained_weights) = preprocessor.drop_label_probabilistically(
           boxes, labels, weights, dropped_label=1, drop_probability=1.0)
      return [
          retained_boxes, retained_labels, retained_weights,
          self.expectedBoxesAfterDropping(),
          self.expectedLabelsAfterDropping(),
          self.expectedLabelScoresAfterDropping()
      ]

    (retained_boxes_, retained_labels_, retained_weights_,
     expected_retained_boxes_, expected_retained_labels_,
     expected_retained_weights_) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(retained_boxes_, expected_retained_boxes_)
    self.assertAllClose(retained_labels_, expected_retained_labels_)
    self.assertAllClose(retained_weights_, expected_retained_weights_)

  def testDropLabelProbabilisticallyWithMultiClassScores(self):
    def graph_fn():
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      multiclass_scores = self.createTestMultiClassScores()
      (_, _, _,
       retained_multiclass_scores) = preprocessor.drop_label_probabilistically(
           boxes,
           labels,
           weights,
           multiclass_scores=multiclass_scores,
           dropped_label=1,
           drop_probability=1.0)
      return [retained_multiclass_scores,
              self.expectedMultiClassScoresAfterDropping()]
    (retained_multiclass_scores_,
     expected_retained_multiclass_scores_) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(retained_multiclass_scores_,
                        expected_retained_multiclass_scores_)

  def testDropLabelProbabilisticallyWithMasks(self):
    def graph_fn():
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      masks = self.createTestMasks()
      (_, _, _, retained_masks) = preprocessor.drop_label_probabilistically(
          boxes,
          labels,
          weights,
          masks=masks,
          dropped_label=1,
          drop_probability=1.0)
      return [retained_masks, self.expectedMasksAfterDropping()]
    (retained_masks_, expected_retained_masks_) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(retained_masks_, expected_retained_masks_)

  def testDropLabelProbabilisticallyWithKeypoints(self):
    def graph_fn():
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      keypoints, _ = self.createTestKeypoints()
      (_, _, _, retained_keypoints) = preprocessor.drop_label_probabilistically(
          boxes,
          labels,
          weights,
          keypoints=keypoints,
          dropped_label=1,
          drop_probability=1.0)
      return [retained_keypoints, self.expectedKeypointsAfterDropping()]

    (retained_keypoints_,
     expected_retained_keypoints_) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(retained_keypoints_, expected_retained_keypoints_)

  def testRemapLabels(self):
    def graph_fn():
      labels = self.createTestLabelsLong()
      remapped_labels = preprocessor.remap_labels(labels, [1, 2], 3)
      return [remapped_labels, self.expectedLabelsAfterRemapping()]

    (remapped_labels_, expected_remapped_labels_) = self.execute_cpu(graph_fn,
                                                                     [])
    self.assertAllClose(remapped_labels_, expected_remapped_labels_)

  def testFlipBoxesLeftRight(self):
    def graph_fn():
      boxes = self.createTestBoxes()
      flipped_boxes = preprocessor._flip_boxes_left_right(boxes)
      expected_boxes = self.expectedBoxesAfterLeftRightFlip()
      return flipped_boxes, expected_boxes
    flipped_boxes, expected_boxes = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(flipped_boxes.flatten(), expected_boxes.flatten())

  def testFlipBoxesUpDown(self):
    def graph_fn():
      boxes = self.createTestBoxes()
      flipped_boxes = preprocessor._flip_boxes_up_down(boxes)
      expected_boxes = self.expectedBoxesAfterUpDownFlip()
      return flipped_boxes, expected_boxes
    flipped_boxes, expected_boxes = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(flipped_boxes.flatten(), expected_boxes.flatten())

  def testRot90Boxes(self):
    def graph_fn():
      boxes = self.createTestBoxes()
      rotated_boxes = preprocessor._rot90_boxes(boxes)
      expected_boxes = self.expectedBoxesAfterRot90()
      return rotated_boxes, expected_boxes
    rotated_boxes, expected_boxes = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(rotated_boxes.flatten(), expected_boxes.flatten())

  def testFlipMasksLeftRight(self):
    def graph_fn():
      test_mask = self.createTestMasks()
      flipped_mask = preprocessor._flip_masks_left_right(test_mask)
      expected_mask = self.expectedMasksAfterLeftRightFlip()
      return flipped_mask, expected_mask
    flipped_mask, expected_mask = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(flipped_mask.flatten(), expected_mask.flatten())

  def testFlipMasksUpDown(self):
    def graph_fn():
      test_mask = self.createTestMasks()
      flipped_mask = preprocessor._flip_masks_up_down(test_mask)
      expected_mask = self.expectedMasksAfterUpDownFlip()
      return  flipped_mask, expected_mask
    flipped_mask, expected_mask = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(flipped_mask.flatten(), expected_mask.flatten())

  def testRot90Masks(self):
    def graph_fn():
      test_mask = self.createTestMasks()
      rotated_mask = preprocessor._rot90_masks(test_mask)
      expected_mask = self.expectedMasksAfterRot90()
      return [rotated_mask, expected_mask]
    rotated_mask, expected_mask = self.execute(graph_fn, [])
    self.assertAllEqual(rotated_mask.flatten(), expected_mask.flatten())

  def _testPreprocessorCache(self,
                             preprocess_options,
                             test_boxes=False,
                             test_masks=False,
                             test_keypoints=False):
    if self.is_tf2(): return
    def graph_fn():
      cache = preprocessor_cache.PreprocessorCache()
      images = self.createTestImages()
      boxes = self.createTestBoxes()
      weights = self.createTestGroundtruthWeights()
      classes = self.createTestLabels()
      masks = self.createTestMasks()
      keypoints, _ = self.createTestKeypoints()
      preprocessor_arg_map = preprocessor.get_default_func_arg_map(
          include_instance_masks=test_masks, include_keypoints=test_keypoints)
      out = []
      for _ in range(2):
        tensor_dict = {
            fields.InputDataFields.image: images,
            fields.InputDataFields.groundtruth_weights: weights
        }
        if test_boxes:
          tensor_dict[fields.InputDataFields.groundtruth_boxes] = boxes
          tensor_dict[fields.InputDataFields.groundtruth_classes] = classes
        if test_masks:
          tensor_dict[fields.InputDataFields.groundtruth_instance_masks] = masks
        if test_keypoints:
          tensor_dict[fields.InputDataFields.groundtruth_keypoints] = keypoints
        out.append(
            preprocessor.preprocess(tensor_dict, preprocess_options,
                                    preprocessor_arg_map, cache))
      return out

    out1, out2 = self.execute_cpu_tf1(graph_fn, [])
    for (_, v1), (_, v2) in zip(out1.items(), out2.items()):
      self.assertAllClose(v1, v2)

  def testRandomHorizontalFlip(self):
    def graph_fn():
      preprocess_options = [(preprocessor.random_horizontal_flip, {})]
      images = self.expectedImagesAfterNormalization()
      boxes = self.createTestBoxes()
      tensor_dict = {fields.InputDataFields.image: images,
                     fields.InputDataFields.groundtruth_boxes: boxes}
      images_expected1 = self.expectedImagesAfterLeftRightFlip()
      boxes_expected1 = self.expectedBoxesAfterLeftRightFlip()
      images_expected2 = images
      boxes_expected2 = boxes
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocess_options)
      images = tensor_dict[fields.InputDataFields.image]
      boxes = tensor_dict[fields.InputDataFields.groundtruth_boxes]

      boxes_diff1 = tf.squared_difference(boxes, boxes_expected1)
      boxes_diff2 = tf.squared_difference(boxes, boxes_expected2)
      boxes_diff = tf.multiply(boxes_diff1, boxes_diff2)
      boxes_diff_expected = tf.zeros_like(boxes_diff)

      images_diff1 = tf.squared_difference(images, images_expected1)
      images_diff2 = tf.squared_difference(images, images_expected2)
      images_diff = tf.multiply(images_diff1, images_diff2)
      images_diff_expected = tf.zeros_like(images_diff)
      return [images_diff, images_diff_expected, boxes_diff,
              boxes_diff_expected]
    (images_diff_, images_diff_expected_, boxes_diff_,
     boxes_diff_expected_) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(boxes_diff_, boxes_diff_expected_)
    self.assertAllClose(images_diff_, images_diff_expected_)

  def testRandomHorizontalFlipWithEmptyBoxes(self):
    def graph_fn():
      preprocess_options = [(preprocessor.random_horizontal_flip, {})]
      images = self.expectedImagesAfterNormalization()
      boxes = self.createEmptyTestBoxes()
      tensor_dict = {fields.InputDataFields.image: images,
                     fields.InputDataFields.groundtruth_boxes: boxes}
      images_expected1 = self.expectedImagesAfterLeftRightFlip()
      boxes_expected = self.createEmptyTestBoxes()
      images_expected2 = images
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocess_options)
      images = tensor_dict[fields.InputDataFields.image]
      boxes = tensor_dict[fields.InputDataFields.groundtruth_boxes]

      images_diff1 = tf.squared_difference(images, images_expected1)
      images_diff2 = tf.squared_difference(images, images_expected2)
      images_diff = tf.multiply(images_diff1, images_diff2)
      images_diff_expected = tf.zeros_like(images_diff)
      return [images_diff, images_diff_expected, boxes, boxes_expected]
    (images_diff_, images_diff_expected_, boxes_,
     boxes_expected_) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(boxes_, boxes_expected_)
    self.assertAllClose(images_diff_, images_diff_expected_)

  def testRandomHorizontalFlipWithCache(self):
    keypoint_flip_permutation = self.createKeypointFlipPermutation()
    preprocess_options = [
        (preprocessor.random_horizontal_flip,
         {'keypoint_flip_permutation': keypoint_flip_permutation})]
    self._testPreprocessorCache(preprocess_options,
                                test_boxes=True,
                                test_masks=True,
                                test_keypoints=True)


  def testRandomVerticalFlip(self):

    def graph_fn():
      preprocess_options = [(preprocessor.random_vertical_flip, {})]
      images = self.expectedImagesAfterNormalization()
      boxes = self.createTestBoxes()
      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes
      }
      images_expected1 = self.expectedImagesAfterUpDownFlip()
      boxes_expected1 = self.expectedBoxesAfterUpDownFlip()
      images_expected2 = images
      boxes_expected2 = boxes
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocess_options)
      images = tensor_dict[fields.InputDataFields.image]
      boxes = tensor_dict[fields.InputDataFields.groundtruth_boxes]

      boxes_diff1 = tf.squared_difference(boxes, boxes_expected1)
      boxes_diff2 = tf.squared_difference(boxes, boxes_expected2)
      boxes_diff = tf.multiply(boxes_diff1, boxes_diff2)
      boxes_diff_expected = tf.zeros_like(boxes_diff)

      images_diff1 = tf.squared_difference(images, images_expected1)
      images_diff2 = tf.squared_difference(images, images_expected2)
      images_diff = tf.multiply(images_diff1, images_diff2)
      images_diff_expected = tf.zeros_like(images_diff)
      return [
          images_diff, images_diff_expected, boxes_diff, boxes_diff_expected
      ]

    (images_diff_, images_diff_expected_, boxes_diff_,
     boxes_diff_expected_) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(boxes_diff_, boxes_diff_expected_)
    self.assertAllClose(images_diff_, images_diff_expected_)

  def testRandomVerticalFlipWithEmptyBoxes(self):

    def graph_fn():
      preprocess_options = [(preprocessor.random_vertical_flip, {})]
      images = self.expectedImagesAfterNormalization()
      boxes = self.createEmptyTestBoxes()
      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes
      }
      images_expected1 = self.expectedImagesAfterUpDownFlip()
      boxes_expected = self.createEmptyTestBoxes()
      images_expected2 = images
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocess_options)
      images = tensor_dict[fields.InputDataFields.image]
      boxes = tensor_dict[fields.InputDataFields.groundtruth_boxes]

      images_diff1 = tf.squared_difference(images, images_expected1)
      images_diff2 = tf.squared_difference(images, images_expected2)
      images_diff = tf.multiply(images_diff1, images_diff2)
      images_diff_expected = tf.zeros_like(images_diff)
      return [images_diff, images_diff_expected, boxes, boxes_expected]

    (images_diff_, images_diff_expected_, boxes_,
     boxes_expected_) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(boxes_, boxes_expected_)
    self.assertAllClose(images_diff_, images_diff_expected_)

  def testRandomVerticalFlipWithCache(self):
    keypoint_flip_permutation = self.createKeypointFlipPermutation()
    preprocess_options = [
        (preprocessor.random_vertical_flip,
         {'keypoint_flip_permutation': keypoint_flip_permutation})]
    self._testPreprocessorCache(preprocess_options,
                                test_boxes=True,
                                test_masks=True,
                                test_keypoints=True)

  def testRunRandomVerticalFlipWithMaskAndKeypoints(self):
    preprocess_options = [(preprocessor.random_vertical_flip, {})]
    image_height = 3
    image_width = 3
    images = tf.random_uniform([1, image_height, image_width, 3])
    boxes = self.createTestBoxes()
    masks = self.createTestMasks()
    keypoints, _ = self.createTestKeypoints()
    keypoint_flip_permutation = self.createKeypointFlipPermutation()
    tensor_dict = {
        fields.InputDataFields.image: images,
        fields.InputDataFields.groundtruth_boxes: boxes,
        fields.InputDataFields.groundtruth_instance_masks: masks,
        fields.InputDataFields.groundtruth_keypoints: keypoints
    }
    preprocess_options = [
        (preprocessor.random_vertical_flip,
         {'keypoint_flip_permutation': keypoint_flip_permutation})]
    preprocessor_arg_map = preprocessor.get_default_func_arg_map(
        include_instance_masks=True, include_keypoints=True)
    tensor_dict = preprocessor.preprocess(
        tensor_dict, preprocess_options, func_arg_map=preprocessor_arg_map)
    boxes = tensor_dict[fields.InputDataFields.groundtruth_boxes]
    masks = tensor_dict[fields.InputDataFields.groundtruth_instance_masks]
    keypoints = tensor_dict[fields.InputDataFields.groundtruth_keypoints]
    self.assertIsNotNone(boxes)
    self.assertIsNotNone(masks)
    self.assertIsNotNone(keypoints)

  def testRandomRotation90(self):

    def graph_fn():
      preprocess_options = [(preprocessor.random_rotation90, {})]
      images = self.expectedImagesAfterNormalization()
      boxes = self.createTestBoxes()
      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes
      }
      images_expected1 = self.expectedImagesAfterRot90()
      boxes_expected1 = self.expectedBoxesAfterRot90()
      images_expected2 = images
      boxes_expected2 = boxes
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocess_options)
      images = tensor_dict[fields.InputDataFields.image]
      boxes = tensor_dict[fields.InputDataFields.groundtruth_boxes]

      boxes_diff1 = tf.squared_difference(boxes, boxes_expected1)
      boxes_diff2 = tf.squared_difference(boxes, boxes_expected2)
      boxes_diff = tf.multiply(boxes_diff1, boxes_diff2)
      boxes_diff_expected = tf.zeros_like(boxes_diff)

      images_diff1 = tf.squared_difference(images, images_expected1)
      images_diff2 = tf.squared_difference(images, images_expected2)
      images_diff = tf.multiply(images_diff1, images_diff2)
      images_diff_expected = tf.zeros_like(images_diff)
      return [
          images_diff, images_diff_expected, boxes_diff, boxes_diff_expected
      ]

    (images_diff_, images_diff_expected_, boxes_diff_,
     boxes_diff_expected_) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(boxes_diff_, boxes_diff_expected_)
    self.assertAllClose(images_diff_, images_diff_expected_)

  def testRandomRotation90WithEmptyBoxes(self):

    def graph_fn():
      preprocess_options = [(preprocessor.random_rotation90, {})]
      images = self.expectedImagesAfterNormalization()
      boxes = self.createEmptyTestBoxes()
      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes
      }
      images_expected1 = self.expectedImagesAfterRot90()
      boxes_expected = self.createEmptyTestBoxes()
      images_expected2 = images
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocess_options)
      images = tensor_dict[fields.InputDataFields.image]
      boxes = tensor_dict[fields.InputDataFields.groundtruth_boxes]

      images_diff1 = tf.squared_difference(images, images_expected1)
      images_diff2 = tf.squared_difference(images, images_expected2)
      images_diff = tf.multiply(images_diff1, images_diff2)
      images_diff_expected = tf.zeros_like(images_diff)
      return [images_diff, images_diff_expected, boxes, boxes_expected]

    (images_diff_, images_diff_expected_, boxes_,
     boxes_expected_) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(boxes_, boxes_expected_)
    self.assertAllClose(images_diff_, images_diff_expected_)

  def testRandomRotation90WithCache(self):
    preprocess_options = [(preprocessor.random_rotation90, {})]
    self._testPreprocessorCache(preprocess_options,
                                test_boxes=True,
                                test_masks=True,
                                test_keypoints=True)

  def testRunRandomRotation90WithMaskAndKeypoints(self):
    image_height = 3
    image_width = 3
    images = tf.random_uniform([1, image_height, image_width, 3])
    boxes = self.createTestBoxes()
    masks = self.createTestMasks()
    keypoints, _ = self.createTestKeypoints()
    keypoint_rot_permutation = self.createKeypointRotPermutation()
    tensor_dict = {
        fields.InputDataFields.image: images,
        fields.InputDataFields.groundtruth_boxes: boxes,
        fields.InputDataFields.groundtruth_instance_masks: masks,
        fields.InputDataFields.groundtruth_keypoints: keypoints
    }
    preprocess_options = [(preprocessor.random_rotation90, {
        'keypoint_rot_permutation': keypoint_rot_permutation
    })]
    preprocessor_arg_map = preprocessor.get_default_func_arg_map(
        include_instance_masks=True, include_keypoints=True)
    tensor_dict = preprocessor.preprocess(
        tensor_dict, preprocess_options, func_arg_map=preprocessor_arg_map)
    boxes = tensor_dict[fields.InputDataFields.groundtruth_boxes]
    masks = tensor_dict[fields.InputDataFields.groundtruth_instance_masks]
    keypoints = tensor_dict[fields.InputDataFields.groundtruth_keypoints]
    self.assertIsNotNone(boxes)
    self.assertIsNotNone(masks)
    self.assertIsNotNone(keypoints)

  def testRandomPixelValueScale(self):

    def graph_fn():
      preprocessing_options = []
      preprocessing_options.append((preprocessor.normalize_image, {
          'original_minval': 0,
          'original_maxval': 255,
          'target_minval': 0,
          'target_maxval': 1
      }))
      preprocessing_options.append((preprocessor.random_pixel_value_scale, {}))
      images = self.createTestImages()
      tensor_dict = {fields.InputDataFields.image: images}
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocessing_options)
      images_min = tf.cast(images, dtype=tf.float32) * 0.9 / 255.0
      images_max = tf.cast(images, dtype=tf.float32) * 1.1 / 255.0
      images = tensor_dict[fields.InputDataFields.image]
      values_greater = tf.greater_equal(images, images_min)
      values_less = tf.less_equal(images, images_max)
      values_true = tf.fill([1, 4, 4, 3], True)
      return [values_greater, values_less, values_true]

    (values_greater_, values_less_,
     values_true_) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(values_greater_, values_true_)
    self.assertAllClose(values_less_, values_true_)

  def testRandomPixelValueScaleWithCache(self):
    preprocess_options = []
    preprocess_options.append((preprocessor.normalize_image, {
        'original_minval': 0,
        'original_maxval': 255,
        'target_minval': 0,
        'target_maxval': 1
    }))
    preprocess_options.append((preprocessor.random_pixel_value_scale, {}))
    self._testPreprocessorCache(preprocess_options,
                                test_boxes=True,
                                test_masks=False,
                                test_keypoints=False)

  def testRandomImageScale(self):

    def graph_fn():
      preprocess_options = [(preprocessor.random_image_scale, {})]
      images_original = self.createTestImages()
      tensor_dict = {fields.InputDataFields.image: images_original}
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocess_options)
      images_scaled = tensor_dict[fields.InputDataFields.image]
      images_original_shape = tf.shape(images_original)
      images_scaled_shape = tf.shape(images_scaled)
      return [images_original_shape, images_scaled_shape]

    (images_original_shape_,
     images_scaled_shape_) = self.execute_cpu(graph_fn, [])
    self.assertLessEqual(images_original_shape_[1] * 0.5,
                         images_scaled_shape_[1])
    self.assertGreaterEqual(images_original_shape_[1] * 2.0,
                            images_scaled_shape_[1])
    self.assertLessEqual(images_original_shape_[2] * 0.5,
                         images_scaled_shape_[2])
    self.assertGreaterEqual(images_original_shape_[2] * 2.0,
                            images_scaled_shape_[2])

  def testRandomImageScaleWithCache(self):
    preprocess_options = [(preprocessor.random_image_scale, {})]
    self._testPreprocessorCache(preprocess_options,
                                test_boxes=False,
                                test_masks=False,
                                test_keypoints=False)

  def testRandomRGBtoGray(self):

    def graph_fn():
      preprocess_options = [(preprocessor.random_rgb_to_gray, {})]
      images_original = self.createTestImages()
      tensor_dict = {fields.InputDataFields.image: images_original}
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocess_options)
      images_gray = tensor_dict[fields.InputDataFields.image]
      images_gray_r, images_gray_g, images_gray_b = tf.split(
          value=images_gray, num_or_size_splits=3, axis=3)
      images_r, images_g, images_b = tf.split(
          value=images_original, num_or_size_splits=3, axis=3)
      images_r_diff1 = tf.squared_difference(
          tf.cast(images_r, dtype=tf.float32),
          tf.cast(images_gray_r, dtype=tf.float32))
      images_r_diff2 = tf.squared_difference(
          tf.cast(images_gray_r, dtype=tf.float32),
          tf.cast(images_gray_g, dtype=tf.float32))
      images_r_diff = tf.multiply(images_r_diff1, images_r_diff2)
      images_g_diff1 = tf.squared_difference(
          tf.cast(images_g, dtype=tf.float32),
          tf.cast(images_gray_g, dtype=tf.float32))
      images_g_diff2 = tf.squared_difference(
          tf.cast(images_gray_g, dtype=tf.float32),
          tf.cast(images_gray_b, dtype=tf.float32))
      images_g_diff = tf.multiply(images_g_diff1, images_g_diff2)
      images_b_diff1 = tf.squared_difference(
          tf.cast(images_b, dtype=tf.float32),
          tf.cast(images_gray_b, dtype=tf.float32))
      images_b_diff2 = tf.squared_difference(
          tf.cast(images_gray_b, dtype=tf.float32),
          tf.cast(images_gray_r, dtype=tf.float32))
      images_b_diff = tf.multiply(images_b_diff1, images_b_diff2)
      image_zero1 = tf.constant(0, dtype=tf.float32, shape=[1, 4, 4, 1])
      return [images_r_diff, images_g_diff, images_b_diff, image_zero1]

    (images_r_diff_, images_g_diff_, images_b_diff_,
     image_zero1_) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(images_r_diff_, image_zero1_)
    self.assertAllClose(images_g_diff_, image_zero1_)
    self.assertAllClose(images_b_diff_, image_zero1_)

  def testRandomRGBtoGrayWithCache(self):
    preprocess_options = [(
        preprocessor.random_rgb_to_gray, {'probability': 0.5})]
    self._testPreprocessorCache(preprocess_options,
                                test_boxes=False,
                                test_masks=False,
                                test_keypoints=False)

  def testRandomAdjustBrightness(self):

    def graph_fn():
      preprocessing_options = []
      preprocessing_options.append((preprocessor.normalize_image, {
          'original_minval': 0,
          'original_maxval': 255,
          'target_minval': 0,
          'target_maxval': 1
      }))
      preprocessing_options.append((preprocessor.random_adjust_brightness, {}))
      images_original = self.createTestImages()
      tensor_dict = {fields.InputDataFields.image: images_original}
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocessing_options)
      images_bright = tensor_dict[fields.InputDataFields.image]
      image_original_shape = tf.shape(images_original)
      image_bright_shape = tf.shape(images_bright)
      return [image_original_shape, image_bright_shape]

    (image_original_shape_,
     image_bright_shape_) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(image_original_shape_, image_bright_shape_)

  def testRandomAdjustBrightnessWithCache(self):
    preprocess_options = []
    preprocess_options.append((preprocessor.normalize_image, {
        'original_minval': 0,
        'original_maxval': 255,
        'target_minval': 0,
        'target_maxval': 1
    }))
    preprocess_options.append((preprocessor.random_adjust_brightness, {}))
    self._testPreprocessorCache(preprocess_options,
                                test_boxes=False,
                                test_masks=False,
                                test_keypoints=False)

  def testRandomAdjustContrast(self):

    def graph_fn():
      preprocessing_options = []
      preprocessing_options.append((preprocessor.normalize_image, {
          'original_minval': 0,
          'original_maxval': 255,
          'target_minval': 0,
          'target_maxval': 1
      }))
      preprocessing_options.append((preprocessor.random_adjust_contrast, {}))
      images_original = self.createTestImages()
      tensor_dict = {fields.InputDataFields.image: images_original}
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocessing_options)
      images_contrast = tensor_dict[fields.InputDataFields.image]
      image_original_shape = tf.shape(images_original)
      image_contrast_shape = tf.shape(images_contrast)
      return [image_original_shape, image_contrast_shape]

    (image_original_shape_,
     image_contrast_shape_) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(image_original_shape_, image_contrast_shape_)

  def testRandomAdjustContrastWithCache(self):
    preprocess_options = []
    preprocess_options.append((preprocessor.normalize_image, {
        'original_minval': 0,
        'original_maxval': 255,
        'target_minval': 0,
        'target_maxval': 1
    }))
    preprocess_options.append((preprocessor.random_adjust_contrast, {}))
    self._testPreprocessorCache(preprocess_options,
                                test_boxes=False,
                                test_masks=False,
                                test_keypoints=False)

  def testRandomAdjustHue(self):

    def graph_fn():
      preprocessing_options = []
      preprocessing_options.append((preprocessor.normalize_image, {
          'original_minval': 0,
          'original_maxval': 255,
          'target_minval': 0,
          'target_maxval': 1
      }))
      preprocessing_options.append((preprocessor.random_adjust_hue, {}))
      images_original = self.createTestImages()
      tensor_dict = {fields.InputDataFields.image: images_original}
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocessing_options)
      images_hue = tensor_dict[fields.InputDataFields.image]
      image_original_shape = tf.shape(images_original)
      image_hue_shape = tf.shape(images_hue)
      return [image_original_shape, image_hue_shape]

    (image_original_shape_, image_hue_shape_) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(image_original_shape_, image_hue_shape_)

  def testRandomAdjustHueWithCache(self):
    preprocess_options = []
    preprocess_options.append((preprocessor.normalize_image, {
        'original_minval': 0,
        'original_maxval': 255,
        'target_minval': 0,
        'target_maxval': 1
    }))
    preprocess_options.append((preprocessor.random_adjust_hue, {}))
    self._testPreprocessorCache(preprocess_options,
                                test_boxes=False,
                                test_masks=False,
                                test_keypoints=False)

  def testRandomDistortColor(self):

    def graph_fn():
      preprocessing_options = []
      preprocessing_options.append((preprocessor.normalize_image, {
          'original_minval': 0,
          'original_maxval': 255,
          'target_minval': 0,
          'target_maxval': 1
      }))
      preprocessing_options.append((preprocessor.random_distort_color, {}))
      images_original = self.createTestImages()
      images_original_shape = tf.shape(images_original)
      tensor_dict = {fields.InputDataFields.image: images_original}
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocessing_options)
      images_distorted_color = tensor_dict[fields.InputDataFields.image]
      images_distorted_color_shape = tf.shape(images_distorted_color)
      return [images_original_shape, images_distorted_color_shape]

    (images_original_shape_,
     images_distorted_color_shape_) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(images_original_shape_, images_distorted_color_shape_)

  def testRandomDistortColorWithCache(self):
    preprocess_options = []
    preprocess_options.append((preprocessor.normalize_image, {
        'original_minval': 0,
        'original_maxval': 255,
        'target_minval': 0,
        'target_maxval': 1
    }))
    preprocess_options.append((preprocessor.random_distort_color, {}))
    self._testPreprocessorCache(preprocess_options,
                                test_boxes=False,
                                test_masks=False,
                                test_keypoints=False)

  def testRandomJitterBoxes(self):

    def graph_fn():
      preprocessing_options = []
      preprocessing_options.append((preprocessor.random_jitter_boxes, {}))
      boxes = self.createTestBoxes()
      boxes_shape = tf.shape(boxes)
      tensor_dict = {fields.InputDataFields.groundtruth_boxes: boxes}
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocessing_options)
      distorted_boxes = tensor_dict[fields.InputDataFields.groundtruth_boxes]
      distorted_boxes_shape = tf.shape(distorted_boxes)
      return [boxes_shape, distorted_boxes_shape]

    (boxes_shape_, distorted_boxes_shape_) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(boxes_shape_, distorted_boxes_shape_)

  def testRandomCropImage(self):

    def graph_fn():
      preprocessing_options = []
      preprocessing_options.append((preprocessor.normalize_image, {
          'original_minval': 0,
          'original_maxval': 255,
          'target_minval': 0,
          'target_maxval': 1
      }))
      preprocessing_options.append((preprocessor.random_crop_image, {}))
      images = self.createTestImages()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
      }
      distorted_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                      preprocessing_options)
      distorted_images = distorted_tensor_dict[fields.InputDataFields.image]
      distorted_boxes = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]
      boxes_rank = tf.rank(boxes)
      distorted_boxes_rank = tf.rank(distorted_boxes)
      images_rank = tf.rank(images)
      distorted_images_rank = tf.rank(distorted_images)
      return [
          boxes_rank, distorted_boxes_rank, images_rank, distorted_images_rank
      ]

    (boxes_rank_, distorted_boxes_rank_, images_rank_,
     distorted_images_rank_) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(boxes_rank_, distorted_boxes_rank_)
    self.assertAllEqual(images_rank_, distorted_images_rank_)

  def testRandomCropImageWithCache(self):
    preprocess_options = [(preprocessor.random_rgb_to_gray,
                           {'probability': 0.5}),
                          (preprocessor.normalize_image, {
                              'original_minval': 0,
                              'original_maxval': 255,
                              'target_minval': 0,
                              'target_maxval': 1,
                          }),
                          (preprocessor.random_crop_image, {})]
    self._testPreprocessorCache(preprocess_options,
                                test_boxes=True,
                                test_masks=False,
                                test_keypoints=False)

  def testRandomCropImageGrayscale(self):

    def graph_fn():
      preprocessing_options = [(preprocessor.rgb_to_gray, {}),
                               (preprocessor.normalize_image, {
                                   'original_minval': 0,
                                   'original_maxval': 255,
                                   'target_minval': 0,
                                   'target_maxval': 1,
                               }), (preprocessor.random_crop_image, {})]
      images = self.createTestImages()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
      }
      distorted_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                      preprocessing_options)
      distorted_images = distorted_tensor_dict[fields.InputDataFields.image]
      distorted_boxes = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]
      boxes_rank = tf.rank(boxes)
      distorted_boxes_rank = tf.rank(distorted_boxes)
      images_rank = tf.rank(images)
      distorted_images_rank = tf.rank(distorted_images)
      return [
          boxes_rank, distorted_boxes_rank, images_rank, distorted_images_rank
      ]

    (boxes_rank_, distorted_boxes_rank_, images_rank_,
     distorted_images_rank_) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(boxes_rank_, distorted_boxes_rank_)
    self.assertAllEqual(images_rank_, distorted_images_rank_)

  def testRandomCropImageWithBoxOutOfImage(self):

    def graph_fn():
      preprocessing_options = []
      preprocessing_options.append((preprocessor.normalize_image, {
          'original_minval': 0,
          'original_maxval': 255,
          'target_minval': 0,
          'target_maxval': 1
      }))
      preprocessing_options.append((preprocessor.random_crop_image, {}))
      images = self.createTestImages()
      boxes = self.createTestBoxesOutOfImage()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
      }
      distorted_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                      preprocessing_options)
      distorted_images = distorted_tensor_dict[fields.InputDataFields.image]
      distorted_boxes = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]
      boxes_rank = tf.rank(boxes)
      distorted_boxes_rank = tf.rank(distorted_boxes)
      images_rank = tf.rank(images)
      distorted_images_rank = tf.rank(distorted_images)
      return [
          boxes_rank, distorted_boxes_rank, images_rank, distorted_images_rank
      ]

    (boxes_rank_, distorted_boxes_rank_, images_rank_,
     distorted_images_rank_) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(boxes_rank_, distorted_boxes_rank_)
    self.assertAllEqual(images_rank_, distorted_images_rank_)

  def testRandomCropImageWithRandomCoefOne(self):

    def graph_fn():
      preprocessing_options = [(preprocessor.normalize_image, {
          'original_minval': 0,
          'original_maxval': 255,
          'target_minval': 0,
          'target_maxval': 1
      })]

      images = self.createTestImages()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights
      }
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocessing_options)
      images = tensor_dict[fields.InputDataFields.image]

      preprocessing_options = [(preprocessor.random_crop_image, {
          'random_coef': 1.0
      })]
      distorted_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                      preprocessing_options)

      distorted_images = distorted_tensor_dict[fields.InputDataFields.image]
      distorted_boxes = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]
      distorted_labels = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_classes]
      distorted_weights = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_weights]
      boxes_shape = tf.shape(boxes)
      distorted_boxes_shape = tf.shape(distorted_boxes)
      images_shape = tf.shape(images)
      distorted_images_shape = tf.shape(distorted_images)
      return [
          boxes_shape, distorted_boxes_shape, images_shape,
          distorted_images_shape, images, distorted_images, boxes,
          distorted_boxes, labels, distorted_labels, weights, distorted_weights
      ]

    (boxes_shape_, distorted_boxes_shape_, images_shape_,
     distorted_images_shape_, images_, distorted_images_, boxes_,
     distorted_boxes_, labels_, distorted_labels_, weights_,
     distorted_weights_) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(boxes_shape_, distorted_boxes_shape_)
    self.assertAllEqual(images_shape_, distorted_images_shape_)
    self.assertAllClose(images_, distorted_images_)
    self.assertAllClose(boxes_, distorted_boxes_)
    self.assertAllEqual(labels_, distorted_labels_)
    self.assertAllEqual(weights_, distorted_weights_)

  def testRandomCropWithMockSampleDistortedBoundingBox(self):
    def graph_fn():
      preprocessing_options = [(preprocessor.normalize_image, {
          'original_minval': 0,
          'original_maxval': 255,
          'target_minval': 0,
          'target_maxval': 1
      })]

      images = self.createColorfulTestImage()
      boxes = tf.constant([[0.1, 0.1, 0.8, 0.3], [0.2, 0.4, 0.75, 0.75],
                           [0.3, 0.1, 0.4, 0.7]],
                          dtype=tf.float32)
      labels = tf.constant([1, 7, 11], dtype=tf.int32)
      weights = tf.constant([1.0, 0.5, 0.6], dtype=tf.float32)

      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
      }
      tensor_dict = preprocessor.preprocess(tensor_dict,
                                            preprocessing_options)
      images = tensor_dict[fields.InputDataFields.image]

      preprocessing_options = [(preprocessor.random_crop_image, {})]

      with mock.patch.object(tf.image, 'sample_distorted_bounding_box'
                            ) as mock_sample_distorted_bounding_box:
        mock_sample_distorted_bounding_box.return_value = (tf.constant(
            [6, 143, 0], dtype=tf.int32), tf.constant(
                [190, 237, -1], dtype=tf.int32), tf.constant(
                    [[[0.03, 0.3575, 0.98, 0.95]]], dtype=tf.float32))
        distorted_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                        preprocessing_options)

      distorted_boxes = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]
      distorted_labels = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_classes]
      distorted_weights = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_weights]
      expected_boxes = tf.constant(
          [[0.178947, 0.07173, 0.75789469, 0.66244733],
           [0.28421, 0.0, 0.38947365, 0.57805908]],
          dtype=tf.float32)
      expected_labels = tf.constant([7, 11], dtype=tf.int32)
      expected_weights = tf.constant([0.5, 0.6], dtype=tf.float32)
      return [
          distorted_boxes, distorted_labels, distorted_weights,
          expected_boxes, expected_labels, expected_weights
      ]

    (distorted_boxes_, distorted_labels_, distorted_weights_, expected_boxes_,
     expected_labels_, expected_weights_) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(distorted_boxes_, expected_boxes_)
    self.assertAllEqual(distorted_labels_, expected_labels_)
    self.assertAllEqual(distorted_weights_, expected_weights_)

  def testRandomCropWithoutClipBoxes(self):

    def graph_fn():
      preprocessing_options = [(preprocessor.normalize_image, {
          'original_minval': 0,
          'original_maxval': 255,
          'target_minval': 0,
          'target_maxval': 1
      })]

      images = self.createColorfulTestImage()
      boxes = tf.constant([[0.1, 0.1, 0.8, 0.3],
                           [0.2, 0.4, 0.75, 0.75],
                           [0.3, 0.1, 0.4, 0.7]], dtype=tf.float32)
      keypoints = tf.constant([
          [[0.1, 0.1], [0.8, 0.3]],
          [[0.2, 0.4], [0.75, 0.75]],
          [[0.3, 0.1], [0.4, 0.7]],
      ], dtype=tf.float32)
      labels = tf.constant([1, 7, 11], dtype=tf.int32)
      weights = tf.constant([1.0, 0.5, 0.6], dtype=tf.float32)

      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_keypoints: keypoints,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
      }
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocessing_options)

      preprocessing_options = [(preprocessor.random_crop_image, {
          'clip_boxes': False,
      })]
      preprocessor_arg_map = preprocessor.get_default_func_arg_map(
          include_keypoints=True)
      with mock.patch.object(tf.image, 'sample_distorted_bounding_box'
                            ) as mock_sample_distorted_bounding_box:
        mock_sample_distorted_bounding_box.return_value = (tf.constant(
            [6, 143, 0], dtype=tf.int32), tf.constant(
                [190, 237, -1], dtype=tf.int32), tf.constant(
                    [[[0.03, 0.3575, 0.98, 0.95]]], dtype=tf.float32))
        distorted_tensor_dict = preprocessor.preprocess(
            tensor_dict, preprocessing_options,
            func_arg_map=preprocessor_arg_map)
      distorted_boxes = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]
      distorted_keypoints = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_keypoints]
      distorted_labels = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_classes]
      distorted_weights = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_weights]
      expected_boxes = tf.constant(
          [[0.178947, 0.07173, 0.75789469, 0.66244733],
           [0.28421, -0.434599, 0.38947365, 0.57805908]],
          dtype=tf.float32)
      expected_keypoints = tf.constant(
          [[[0.178947, 0.07173], [0.75789469, 0.66244733]],
           [[0.28421, -0.434599], [0.38947365, 0.57805908]]],
          dtype=tf.float32)
      expected_labels = tf.constant([7, 11], dtype=tf.int32)
      expected_weights = tf.constant([0.5, 0.6], dtype=tf.float32)
      return [distorted_boxes, distorted_keypoints, distorted_labels,
              distorted_weights, expected_boxes, expected_keypoints,
              expected_labels, expected_weights]

    (distorted_boxes_, distorted_keypoints_, distorted_labels_,
     distorted_weights_, expected_boxes_, expected_keypoints_, expected_labels_,
     expected_weights_) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(distorted_boxes_, expected_boxes_)
    self.assertAllClose(distorted_keypoints_, expected_keypoints_)
    self.assertAllEqual(distorted_labels_, expected_labels_)
    self.assertAllEqual(distorted_weights_, expected_weights_)

  def testRandomCropImageWithMultiClassScores(self):
    def graph_fn():
      preprocessing_options = []
      preprocessing_options.append((preprocessor.normalize_image, {
          'original_minval': 0,
          'original_maxval': 255,
          'target_minval': 0,
          'target_maxval': 1
      }))
      preprocessing_options.append((preprocessor.random_crop_image, {}))
      images = self.createTestImages()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      multiclass_scores = self.createTestMultiClassScores()

      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
          fields.InputDataFields.multiclass_scores: multiclass_scores
      }
      distorted_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                      preprocessing_options)
      distorted_images = distorted_tensor_dict[fields.InputDataFields.image]
      distorted_boxes = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]
      distorted_multiclass_scores = distorted_tensor_dict[
          fields.InputDataFields.multiclass_scores]
      boxes_rank = tf.rank(boxes)
      distorted_boxes_rank = tf.rank(distorted_boxes)
      images_rank = tf.rank(images)
      distorted_images_rank = tf.rank(distorted_images)
      multiclass_scores_rank = tf.rank(multiclass_scores)
      distorted_multiclass_scores_rank = tf.rank(distorted_multiclass_scores)
      return [
          boxes_rank, distorted_boxes, distorted_boxes_rank, images_rank,
          distorted_images_rank, multiclass_scores_rank,
          distorted_multiclass_scores_rank, distorted_multiclass_scores
      ]

    (boxes_rank_, distorted_boxes_, distorted_boxes_rank_, images_rank_,
     distorted_images_rank_, multiclass_scores_rank_,
     distorted_multiclass_scores_rank_,
     distorted_multiclass_scores_) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(boxes_rank_, distorted_boxes_rank_)
    self.assertAllEqual(images_rank_, distorted_images_rank_)
    self.assertAllEqual(multiclass_scores_rank_,
                        distorted_multiclass_scores_rank_)
    self.assertAllEqual(distorted_boxes_.shape[0],
                        distorted_multiclass_scores_.shape[0])

  def testStrictRandomCropImageWithGroundtruthWeights(self):
    def graph_fn():
      image = self.createColorfulTestImage()[0]
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      with mock.patch.object(
          tf.image,
          'sample_distorted_bounding_box'
      ) as mock_sample_distorted_bounding_box:
        mock_sample_distorted_bounding_box.return_value = (
            tf.constant([6, 143, 0], dtype=tf.int32),
            tf.constant([190, 237, -1], dtype=tf.int32),
            tf.constant([[[0.03, 0.3575, 0.98, 0.95]]], dtype=tf.float32))
        new_image, new_boxes, new_labels, new_groundtruth_weights = (
            preprocessor._strict_random_crop_image(
                image, boxes, labels, weights))
        return [new_image, new_boxes, new_labels, new_groundtruth_weights]
    (new_image, new_boxes, _,
     new_groundtruth_weights) = self.execute_cpu(graph_fn, [])
    expected_boxes = np.array(
        [[0.0, 0.0, 0.75789469, 1.0],
         [0.23157893, 0.24050637, 0.75789469, 1.0]], dtype=np.float32)
    self.assertAllEqual(new_image.shape, [190, 237, 3])
    self.assertAllEqual(new_groundtruth_weights, [1.0, 0.5])
    self.assertAllClose(
        new_boxes.flatten(), expected_boxes.flatten())

  def testStrictRandomCropImageWithMasks(self):
    def graph_fn():
      image = self.createColorfulTestImage()[0]
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      masks = tf.random_uniform([2, 200, 400], dtype=tf.float32)
      with mock.patch.object(
          tf.image,
          'sample_distorted_bounding_box'
      ) as mock_sample_distorted_bounding_box:
        mock_sample_distorted_bounding_box.return_value = (
            tf.constant([6, 143, 0], dtype=tf.int32),
            tf.constant([190, 237, -1], dtype=tf.int32),
            tf.constant([[[0.03, 0.3575, 0.98, 0.95]]], dtype=tf.float32))
        new_image, new_boxes, new_labels, new_weights, new_masks = (
            preprocessor._strict_random_crop_image(
                image, boxes, labels, weights, masks=masks))
        return [new_image, new_boxes, new_labels, new_weights, new_masks]
    (new_image, new_boxes, _, _,
     new_masks) = self.execute_cpu(graph_fn, [])
    expected_boxes = np.array(
        [[0.0, 0.0, 0.75789469, 1.0],
         [0.23157893, 0.24050637, 0.75789469, 1.0]], dtype=np.float32)
    self.assertAllEqual(new_image.shape, [190, 237, 3])
    self.assertAllEqual(new_masks.shape, [2, 190, 237])
    self.assertAllClose(
        new_boxes.flatten(), expected_boxes.flatten())

  def testStrictRandomCropImageWithKeypoints(self):
    def graph_fn():
      image = self.createColorfulTestImage()[0]
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      keypoints, keypoint_visibilities = self.createTestKeypoints()
      with mock.patch.object(
          tf.image,
          'sample_distorted_bounding_box'
      ) as mock_sample_distorted_bounding_box:
        mock_sample_distorted_bounding_box.return_value = (
            tf.constant([6, 143, 0], dtype=tf.int32),
            tf.constant([190, 237, -1], dtype=tf.int32),
            tf.constant([[[0.03, 0.3575, 0.98, 0.95]]], dtype=tf.float32))
        (new_image, new_boxes, new_labels, new_weights, new_keypoints,
         new_keypoint_visibilities) = preprocessor._strict_random_crop_image(
             image, boxes, labels, weights, keypoints=keypoints,
             keypoint_visibilities=keypoint_visibilities)
        return [new_image, new_boxes, new_labels, new_weights, new_keypoints,
                new_keypoint_visibilities]
    (new_image, new_boxes, _, _, new_keypoints,
     new_keypoint_visibilities) = self.execute_cpu(graph_fn, [])
    expected_boxes = np.array([
        [0.0, 0.0, 0.75789469, 1.0],
        [0.23157893, 0.24050637, 0.75789469, 1.0],], dtype=np.float32)
    expected_keypoints = np.array([
        [[np.nan, np.nan],
         [np.nan, np.nan],
         [np.nan, np.nan]],
        [[0.38947368, 0.07173],
         [0.49473682, 0.24050637],
         [0.60000002, 0.40928277]]
    ], dtype=np.float32)
    expected_keypoint_visibilities = [
        [False, False, False],
        [False, True, True]
    ]
    self.assertAllEqual(new_image.shape, [190, 237, 3])
    self.assertAllClose(
        new_boxes, expected_boxes)
    self.assertAllClose(
        new_keypoints, expected_keypoints)
    self.assertAllEqual(
        new_keypoint_visibilities, expected_keypoint_visibilities)

  def testRunRandomCropImageWithMasks(self):
    def graph_fn():
      image = self.createColorfulTestImage()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      masks = tf.random_uniform([2, 200, 400], dtype=tf.float32)

      tensor_dict = {
          fields.InputDataFields.image: image,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
          fields.InputDataFields.groundtruth_instance_masks: masks,
      }

      preprocessor_arg_map = preprocessor.get_default_func_arg_map(
          include_instance_masks=True)

      preprocessing_options = [(preprocessor.random_crop_image, {})]

      with mock.patch.object(
          tf.image,
          'sample_distorted_bounding_box'
      ) as mock_sample_distorted_bounding_box:
        mock_sample_distorted_bounding_box.return_value = (
            tf.constant([6, 143, 0], dtype=tf.int32),
            tf.constant([190, 237, -1], dtype=tf.int32),
            tf.constant([[[0.03, 0.3575, 0.98, 0.95]]], dtype=tf.float32))
        distorted_tensor_dict = preprocessor.preprocess(
            tensor_dict,
            preprocessing_options,
            func_arg_map=preprocessor_arg_map)
        distorted_image = distorted_tensor_dict[fields.InputDataFields.image]
        distorted_boxes = distorted_tensor_dict[
            fields.InputDataFields.groundtruth_boxes]
        distorted_labels = distorted_tensor_dict[
            fields.InputDataFields.groundtruth_classes]
        distorted_masks = distorted_tensor_dict[
            fields.InputDataFields.groundtruth_instance_masks]
        return [distorted_image, distorted_boxes, distorted_labels,
                distorted_masks]
    (distorted_image_, distorted_boxes_, distorted_labels_,
     distorted_masks_) = self.execute_cpu(graph_fn, [])
    expected_boxes = np.array([
        [0.0, 0.0, 0.75789469, 1.0],
        [0.23157893, 0.24050637, 0.75789469, 1.0],
    ], dtype=np.float32)
    self.assertAllEqual(distorted_image_.shape, [1, 190, 237, 3])
    self.assertAllEqual(distorted_masks_.shape, [2, 190, 237])
    self.assertAllEqual(distorted_labels_, [1, 2])
    self.assertAllClose(
        distorted_boxes_.flatten(), expected_boxes.flatten())

  def testRunRandomCropImageWithKeypointsInsideCrop(self):
    def graph_fn():
      image = self.createColorfulTestImage()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      keypoints = self.createTestKeypointsInsideCrop()

      tensor_dict = {
          fields.InputDataFields.image: image,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_keypoints: keypoints,
          fields.InputDataFields.groundtruth_weights: weights
      }

      preprocessor_arg_map = preprocessor.get_default_func_arg_map(
          include_keypoints=True)

      preprocessing_options = [(preprocessor.random_crop_image, {})]

      with mock.patch.object(
          tf.image,
          'sample_distorted_bounding_box'
      ) as mock_sample_distorted_bounding_box:
        mock_sample_distorted_bounding_box.return_value = (
            tf.constant([6, 143, 0], dtype=tf.int32),
            tf.constant([190, 237, -1], dtype=tf.int32),
            tf.constant([[[0.03, 0.3575, 0.98, 0.95]]], dtype=tf.float32))
        distorted_tensor_dict = preprocessor.preprocess(
            tensor_dict,
            preprocessing_options,
            func_arg_map=preprocessor_arg_map)
        distorted_image = distorted_tensor_dict[fields.InputDataFields.image]
        distorted_boxes = distorted_tensor_dict[
            fields.InputDataFields.groundtruth_boxes]
        distorted_labels = distorted_tensor_dict[
            fields.InputDataFields.groundtruth_classes]
        distorted_keypoints = distorted_tensor_dict[
            fields.InputDataFields.groundtruth_keypoints]
        return [distorted_image, distorted_boxes, distorted_labels,
                distorted_keypoints]
    (distorted_image_, distorted_boxes_, distorted_labels_,
     distorted_keypoints_) = self.execute_cpu(graph_fn, [])
    expected_boxes = np.array([
        [0.0, 0.0, 0.75789469, 1.0],
        [0.23157893, 0.24050637, 0.75789469, 1.0],
    ], dtype=np.float32)
    expected_keypoints = np.array([
        [[0.38947368, 0.07173],
         [0.49473682, 0.24050637],
         [0.60000002, 0.40928277]],
        [[0.38947368, 0.07173],
         [0.49473682, 0.24050637],
         [0.60000002, 0.40928277]]
    ])
    self.assertAllEqual(distorted_image_.shape, [1, 190, 237, 3])
    self.assertAllEqual(distorted_labels_, [1, 2])
    self.assertAllClose(
        distorted_boxes_.flatten(), expected_boxes.flatten())
    self.assertAllClose(
        distorted_keypoints_.flatten(), expected_keypoints.flatten())

  def testRunRandomCropImageWithKeypointsOutsideCrop(self):
    def graph_fn():
      image = self.createColorfulTestImage()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      keypoints = self.createTestKeypointsOutsideCrop()

      tensor_dict = {
          fields.InputDataFields.image: image,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
          fields.InputDataFields.groundtruth_keypoints: keypoints
      }

      preprocessor_arg_map = preprocessor.get_default_func_arg_map(
          include_keypoints=True)

      preprocessing_options = [(preprocessor.random_crop_image, {})]

      with mock.patch.object(
          tf.image,
          'sample_distorted_bounding_box'
      ) as mock_sample_distorted_bounding_box:
        mock_sample_distorted_bounding_box.return_value = (
            tf.constant([6, 143, 0], dtype=tf.int32),
            tf.constant([190, 237, -1], dtype=tf.int32),
            tf.constant([[[0.03, 0.3575, 0.98, 0.95]]], dtype=tf.float32))
        distorted_tensor_dict = preprocessor.preprocess(
            tensor_dict,
            preprocessing_options,
            func_arg_map=preprocessor_arg_map)
        distorted_image = distorted_tensor_dict[fields.InputDataFields.image]
        distorted_boxes = distorted_tensor_dict[
            fields.InputDataFields.groundtruth_boxes]
        distorted_labels = distorted_tensor_dict[
            fields.InputDataFields.groundtruth_classes]
        distorted_keypoints = distorted_tensor_dict[
            fields.InputDataFields.groundtruth_keypoints]
        return [distorted_image, distorted_boxes, distorted_labels,
                distorted_keypoints]
      (distorted_image_, distorted_boxes_, distorted_labels_,
       distorted_keypoints_) = self.execute_cpu(graph_fn, [])

      expected_boxes = np.array([
          [0.0, 0.0, 0.75789469, 1.0],
          [0.23157893, 0.24050637, 0.75789469, 1.0],
      ], dtype=np.float32)
      expected_keypoints = np.array([
          [[np.nan, np.nan],
           [np.nan, np.nan],
           [np.nan, np.nan]],
          [[np.nan, np.nan],
           [np.nan, np.nan],
           [np.nan, np.nan]],
      ])
      self.assertAllEqual(distorted_image_.shape, [1, 190, 237, 3])
      self.assertAllEqual(distorted_labels_, [1, 2])
      self.assertAllClose(
          distorted_boxes_.flatten(), expected_boxes.flatten())
      self.assertAllClose(
          distorted_keypoints_.flatten(), expected_keypoints.flatten())

  def testRunRandomCropImageWithDensePose(self):
    def graph_fn():
      image = self.createColorfulTestImage()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      dp_num_points, dp_part_ids, dp_surface_coords = self.createTestDensePose()

      tensor_dict = {
          fields.InputDataFields.image: image,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
          fields.InputDataFields.groundtruth_dp_num_points: dp_num_points,
          fields.InputDataFields.groundtruth_dp_part_ids: dp_part_ids,
          fields.InputDataFields.groundtruth_dp_surface_coords:
              dp_surface_coords
      }

      preprocessor_arg_map = preprocessor.get_default_func_arg_map(
          include_dense_pose=True)

      preprocessing_options = [(preprocessor.random_crop_image, {})]

      with mock.patch.object(
          tf.image,
          'sample_distorted_bounding_box'
      ) as mock_sample_distorted_bounding_box:
        mock_sample_distorted_bounding_box.return_value = (
            tf.constant([6, 40, 0], dtype=tf.int32),
            tf.constant([134, 340, -1], dtype=tf.int32),
            tf.constant([[[0.03, 0.1, 0.7, 0.95]]], dtype=tf.float32))
        distorted_tensor_dict = preprocessor.preprocess(
            tensor_dict,
            preprocessing_options,
            func_arg_map=preprocessor_arg_map)
        distorted_image = distorted_tensor_dict[fields.InputDataFields.image]
        distorted_dp_num_points = distorted_tensor_dict[
            fields.InputDataFields.groundtruth_dp_num_points]
        distorted_dp_part_ids = distorted_tensor_dict[
            fields.InputDataFields.groundtruth_dp_part_ids]
        distorted_dp_surface_coords = distorted_tensor_dict[
            fields.InputDataFields.groundtruth_dp_surface_coords]
        return [distorted_image, distorted_dp_num_points, distorted_dp_part_ids,
                distorted_dp_surface_coords]
    (distorted_image_, distorted_dp_num_points_, distorted_dp_part_ids_,
     distorted_dp_surface_coords_) = self.execute_cpu(graph_fn, [])
    expected_dp_num_points = np.array([1, 1])
    expected_dp_part_ids = np.array([[4], [0]])
    expected_dp_surface_coords = np.array([
        [[0.10447761, 0.1176470, 0.6, 0.7]],
        [[0.10447761, 0.2352941, 0.2, 0.8]],
    ])
    self.assertAllEqual(distorted_image_.shape, [1, 134, 340, 3])
    self.assertAllEqual(distorted_dp_num_points_, expected_dp_num_points)
    self.assertAllEqual(distorted_dp_part_ids_, expected_dp_part_ids)
    self.assertAllClose(distorted_dp_surface_coords_,
                        expected_dp_surface_coords)

  def testRunRetainBoxesAboveThreshold(self):
    def graph_fn():
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()

      tensor_dict = {
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
      }

      preprocessing_options = [
          (preprocessor.retain_boxes_above_threshold, {'threshold': 0.6})
      ]
      preprocessor_arg_map = preprocessor.get_default_func_arg_map()
      retained_tensor_dict = preprocessor.preprocess(
          tensor_dict, preprocessing_options, func_arg_map=preprocessor_arg_map)
      retained_boxes = retained_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]
      retained_labels = retained_tensor_dict[
          fields.InputDataFields.groundtruth_classes]
      retained_weights = retained_tensor_dict[
          fields.InputDataFields.groundtruth_weights]
      return [retained_boxes, retained_labels, retained_weights,
              self.expectedBoxesAfterThresholding(),
              self.expectedLabelsAfterThresholding(),
              self.expectedLabelScoresAfterThresholding()]

    (retained_boxes_, retained_labels_, retained_weights_,
     expected_retained_boxes_, expected_retained_labels_,
     expected_retained_weights_) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(retained_boxes_, expected_retained_boxes_)
    self.assertAllClose(retained_labels_, expected_retained_labels_)
    self.assertAllClose(
        retained_weights_, expected_retained_weights_)

  def testRunRetainBoxesAboveThresholdWithMasks(self):
    def graph_fn():
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      masks = self.createTestMasks()

      tensor_dict = {
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
          fields.InputDataFields.groundtruth_instance_masks: masks
      }

      preprocessor_arg_map = preprocessor.get_default_func_arg_map(
          include_label_weights=True,
          include_instance_masks=True)

      preprocessing_options = [
          (preprocessor.retain_boxes_above_threshold, {'threshold': 0.6})
      ]

      retained_tensor_dict = preprocessor.preprocess(
          tensor_dict, preprocessing_options, func_arg_map=preprocessor_arg_map)
      retained_masks = retained_tensor_dict[
          fields.InputDataFields.groundtruth_instance_masks]
      return [retained_masks, self.expectedMasksAfterThresholding()]
    (retained_masks_, expected_masks_) = self.execute(graph_fn, [])
    self.assertAllClose(retained_masks_, expected_masks_)

  def testRunRetainBoxesAboveThresholdWithKeypoints(self):
    def graph_fn():
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      keypoints, _ = self.createTestKeypoints()

      tensor_dict = {
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
          fields.InputDataFields.groundtruth_keypoints: keypoints
      }

      preprocessor_arg_map = preprocessor.get_default_func_arg_map(
          include_keypoints=True)

      preprocessing_options = [
          (preprocessor.retain_boxes_above_threshold, {'threshold': 0.6})
      ]

      retained_tensor_dict = preprocessor.preprocess(
          tensor_dict, preprocessing_options, func_arg_map=preprocessor_arg_map)
      retained_keypoints = retained_tensor_dict[
          fields.InputDataFields.groundtruth_keypoints]
      return [retained_keypoints, self.expectedKeypointsAfterThresholding()]
    (retained_keypoints_, expected_keypoints_) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(retained_keypoints_, expected_keypoints_)

  def testRandomCropToAspectRatioWithCache(self):
    preprocess_options = [(preprocessor.random_crop_to_aspect_ratio, {})]
    self._testPreprocessorCache(preprocess_options,
                                test_boxes=True,
                                test_masks=False,
                                test_keypoints=False)

  def testRunRandomCropToAspectRatioWithMasks(self):
    def graph_fn():
      image = self.createColorfulTestImage()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      masks = tf.random_uniform([2, 200, 400], dtype=tf.float32)

      tensor_dict = {
          fields.InputDataFields.image: image,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
          fields.InputDataFields.groundtruth_instance_masks: masks
      }

      preprocessor_arg_map = preprocessor.get_default_func_arg_map(
          include_instance_masks=True)

      preprocessing_options = [(preprocessor.random_crop_to_aspect_ratio, {})]

      with mock.patch.object(preprocessor,
                             '_random_integer') as mock_random_integer:
        mock_random_integer.return_value = tf.constant(0, dtype=tf.int32)
        distorted_tensor_dict = preprocessor.preprocess(
            tensor_dict,
            preprocessing_options,
            func_arg_map=preprocessor_arg_map)
        distorted_image = distorted_tensor_dict[fields.InputDataFields.image]
        distorted_boxes = distorted_tensor_dict[
            fields.InputDataFields.groundtruth_boxes]
        distorted_labels = distorted_tensor_dict[
            fields.InputDataFields.groundtruth_classes]
        distorted_masks = distorted_tensor_dict[
            fields.InputDataFields.groundtruth_instance_masks]
        return [
            distorted_image, distorted_boxes, distorted_labels, distorted_masks
        ]

    (distorted_image_, distorted_boxes_, distorted_labels_,
     distorted_masks_) = self.execute_cpu(graph_fn, [])
    expected_boxes = np.array([0.0, 0.5, 0.75, 1.0], dtype=np.float32)
    self.assertAllEqual(distorted_image_.shape, [1, 200, 200, 3])
    self.assertAllEqual(distorted_labels_, [1])
    self.assertAllClose(distorted_boxes_.flatten(),
                        expected_boxes.flatten())
    self.assertAllEqual(distorted_masks_.shape, [1, 200, 200])

  def testRunRandomCropToAspectRatioWithKeypoints(self):
    def graph_fn():
      image = self.createColorfulTestImage()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      keypoints, _ = self.createTestKeypoints()

      tensor_dict = {
          fields.InputDataFields.image: image,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
          fields.InputDataFields.groundtruth_keypoints: keypoints
      }

      preprocessor_arg_map = preprocessor.get_default_func_arg_map(
          include_keypoints=True)

      preprocessing_options = [(preprocessor.random_crop_to_aspect_ratio, {})]

      with mock.patch.object(preprocessor,
                             '_random_integer') as mock_random_integer:
        mock_random_integer.return_value = tf.constant(0, dtype=tf.int32)
        distorted_tensor_dict = preprocessor.preprocess(
            tensor_dict,
            preprocessing_options,
            func_arg_map=preprocessor_arg_map)
        distorted_image = distorted_tensor_dict[fields.InputDataFields.image]
        distorted_boxes = distorted_tensor_dict[
            fields.InputDataFields.groundtruth_boxes]
        distorted_labels = distorted_tensor_dict[
            fields.InputDataFields.groundtruth_classes]
        distorted_keypoints = distorted_tensor_dict[
            fields.InputDataFields.groundtruth_keypoints]
        return [distorted_image, distorted_boxes, distorted_labels,
                distorted_keypoints]
    (distorted_image_, distorted_boxes_, distorted_labels_,
     distorted_keypoints_) = self.execute_cpu(graph_fn, [])
    expected_boxes = np.array([0.0, 0.5, 0.75, 1.0], dtype=np.float32)
    expected_keypoints = np.array(
        [[0.1, 0.2], [0.2, 0.4], [0.3, 0.6]], dtype=np.float32)
    self.assertAllEqual(distorted_image_.shape, [1, 200, 200, 3])
    self.assertAllEqual(distorted_labels_, [1])
    self.assertAllClose(distorted_boxes_.flatten(),
                        expected_boxes.flatten())
    self.assertAllClose(distorted_keypoints_.flatten(),
                        expected_keypoints.flatten())

  def testRandomPadToAspectRatioWithCache(self):
    preprocess_options = [(preprocessor.random_pad_to_aspect_ratio, {})]
    self._testPreprocessorCache(preprocess_options,
                                test_boxes=True,
                                test_masks=True,
                                test_keypoints=True)

  def testRunRandomPadToAspectRatioWithMinMaxPaddedSizeRatios(self):
    def graph_fn():
      image = self.createColorfulTestImage()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()

      tensor_dict = {
          fields.InputDataFields.image: image,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels
      }

      preprocessor_arg_map = preprocessor.get_default_func_arg_map()
      preprocessing_options = [(preprocessor.random_pad_to_aspect_ratio,
                                {'min_padded_size_ratio': (4.0, 4.0),
                                 'max_padded_size_ratio': (4.0, 4.0)})]

      distorted_tensor_dict = preprocessor.preprocess(
          tensor_dict, preprocessing_options, func_arg_map=preprocessor_arg_map)
      distorted_image = distorted_tensor_dict[fields.InputDataFields.image]
      distorted_boxes = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]
      distorted_labels = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_classes]
      return [distorted_image, distorted_boxes, distorted_labels]

    distorted_image_, distorted_boxes_, distorted_labels_ = self.execute_cpu(
        graph_fn, [])
    expected_boxes = np.array(
        [[0.0, 0.125, 0.1875, 0.5], [0.0625, 0.25, 0.1875, 0.5]],
        dtype=np.float32)
    self.assertAllEqual(distorted_image_.shape, [1, 800, 800, 3])
    self.assertAllEqual(distorted_labels_, [1, 2])
    self.assertAllClose(distorted_boxes_.flatten(),
                        expected_boxes.flatten())

  def testRunRandomPadToAspectRatioWithMasks(self):
    def graph_fn():
      image = self.createColorfulTestImage()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      masks = tf.random_uniform([2, 200, 400], dtype=tf.float32)

      tensor_dict = {
          fields.InputDataFields.image: image,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_instance_masks: masks
      }

      preprocessor_arg_map = preprocessor.get_default_func_arg_map(
          include_instance_masks=True)

      preprocessing_options = [(preprocessor.random_pad_to_aspect_ratio, {})]

      distorted_tensor_dict = preprocessor.preprocess(
          tensor_dict, preprocessing_options, func_arg_map=preprocessor_arg_map)
      distorted_image = distorted_tensor_dict[fields.InputDataFields.image]
      distorted_boxes = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]
      distorted_labels = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_classes]
      distorted_masks = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_instance_masks]
      return [
          distorted_image, distorted_boxes, distorted_labels, distorted_masks
      ]

    (distorted_image_, distorted_boxes_, distorted_labels_,
     distorted_masks_) = self.execute_cpu(graph_fn, [])
    expected_boxes = np.array(
        [[0.0, 0.25, 0.375, 1.0], [0.125, 0.5, 0.375, 1.0]], dtype=np.float32)
    self.assertAllEqual(distorted_image_.shape, [1, 400, 400, 3])
    self.assertAllEqual(distorted_labels_, [1, 2])
    self.assertAllClose(distorted_boxes_.flatten(),
                        expected_boxes.flatten())
    self.assertAllEqual(distorted_masks_.shape, [2, 400, 400])

  def testRunRandomPadToAspectRatioWithKeypoints(self):
    def graph_fn():
      image = self.createColorfulTestImage()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      keypoints, _ = self.createTestKeypoints()

      tensor_dict = {
          fields.InputDataFields.image: image,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_keypoints: keypoints
      }

      preprocessor_arg_map = preprocessor.get_default_func_arg_map(
          include_keypoints=True)

      preprocessing_options = [(preprocessor.random_pad_to_aspect_ratio, {})]

      distorted_tensor_dict = preprocessor.preprocess(
          tensor_dict, preprocessing_options, func_arg_map=preprocessor_arg_map)
      distorted_image = distorted_tensor_dict[fields.InputDataFields.image]
      distorted_boxes = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]
      distorted_labels = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_classes]
      distorted_keypoints = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_keypoints]
      return [
          distorted_image, distorted_boxes, distorted_labels,
          distorted_keypoints
      ]

    (distorted_image_, distorted_boxes_, distorted_labels_,
     distorted_keypoints_) = self.execute_cpu(graph_fn, [])
    expected_boxes = np.array(
        [[0.0, 0.25, 0.375, 1.0], [0.125, 0.5, 0.375, 1.0]], dtype=np.float32)
    expected_keypoints = np.array([
        [[0.05, 0.1], [0.1, 0.2], [0.15, 0.3]],
        [[0.2, 0.4], [0.25, 0.5], [0.3, 0.6]],
    ], dtype=np.float32)
    self.assertAllEqual(distorted_image_.shape, [1, 400, 400, 3])
    self.assertAllEqual(distorted_labels_, [1, 2])
    self.assertAllClose(distorted_boxes_.flatten(),
                        expected_boxes.flatten())
    self.assertAllClose(distorted_keypoints_.flatten(),
                        expected_keypoints.flatten())

  def testRandomPadImageWithCache(self):
    preprocess_options = [(preprocessor.normalize_image, {
        'original_minval': 0,
        'original_maxval': 255,
        'target_minval': 0,
        'target_maxval': 1,}), (preprocessor.random_pad_image, {})]
    self._testPreprocessorCache(preprocess_options,
                                test_boxes=True,
                                test_masks=True,
                                test_keypoints=True)

  def testRandomPadImage(self):
    def graph_fn():
      preprocessing_options = [(preprocessor.normalize_image, {
          'original_minval': 0,
          'original_maxval': 255,
          'target_minval': 0,
          'target_maxval': 1
      })]

      images = self.createTestImages()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
      }
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocessing_options)
      images = tensor_dict[fields.InputDataFields.image]

      preprocessing_options = [(preprocessor.random_pad_image, {})]
      padded_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                   preprocessing_options)

      padded_images = padded_tensor_dict[fields.InputDataFields.image]
      padded_boxes = padded_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]
      boxes_shape = tf.shape(boxes)
      padded_boxes_shape = tf.shape(padded_boxes)
      images_shape = tf.shape(images)
      padded_images_shape = tf.shape(padded_images)
      return [boxes_shape, padded_boxes_shape, images_shape,
              padded_images_shape, boxes, padded_boxes]
    (boxes_shape_, padded_boxes_shape_, images_shape_,
     padded_images_shape_, boxes_, padded_boxes_) = self.execute_cpu(graph_fn,
                                                                     [])
    self.assertAllEqual(boxes_shape_, padded_boxes_shape_)
    self.assertTrue((images_shape_[1] >= padded_images_shape_[1] * 0.5).all)
    self.assertTrue((images_shape_[2] >= padded_images_shape_[2] * 0.5).all)
    self.assertTrue((images_shape_[1] <= padded_images_shape_[1]).all)
    self.assertTrue((images_shape_[2] <= padded_images_shape_[2]).all)
    self.assertTrue(np.all((boxes_[:, 2] - boxes_[:, 0]) >= (
        padded_boxes_[:, 2] - padded_boxes_[:, 0])))
    self.assertTrue(np.all((boxes_[:, 3] - boxes_[:, 1]) >= (
        padded_boxes_[:, 3] - padded_boxes_[:, 1])))

  @parameterized.parameters(
      {'include_dense_pose': False},
  )
  def testRandomPadImageWithKeypointsAndMasks(self, include_dense_pose):
    def graph_fn():
      preprocessing_options = [(preprocessor.normalize_image, {
          'original_minval': 0,
          'original_maxval': 255,
          'target_minval': 0,
          'target_maxval': 1
      })]

      images = self.createTestImages()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      masks = self.createTestMasks()
      keypoints, _ = self.createTestKeypoints()
      _, _, dp_surface_coords = self.createTestDensePose()
      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_instance_masks: masks,
          fields.InputDataFields.groundtruth_keypoints: keypoints,
          fields.InputDataFields.groundtruth_dp_surface_coords:
              dp_surface_coords
      }
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocessing_options)
      images = tensor_dict[fields.InputDataFields.image]

      preprocessing_options = [(preprocessor.random_pad_image, {})]
      func_arg_map = preprocessor.get_default_func_arg_map(
          include_instance_masks=True,
          include_keypoints=True,
          include_keypoint_visibilities=True,
          include_dense_pose=include_dense_pose)
      padded_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                   preprocessing_options,
                                                   func_arg_map=func_arg_map)

      padded_images = padded_tensor_dict[fields.InputDataFields.image]
      padded_boxes = padded_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]
      padded_masks = padded_tensor_dict[
          fields.InputDataFields.groundtruth_instance_masks]
      padded_keypoints = padded_tensor_dict[
          fields.InputDataFields.groundtruth_keypoints]
      boxes_shape = tf.shape(boxes)
      padded_boxes_shape = tf.shape(padded_boxes)
      padded_masks_shape = tf.shape(padded_masks)
      keypoints_shape = tf.shape(keypoints)
      padded_keypoints_shape = tf.shape(padded_keypoints)
      images_shape = tf.shape(images)
      padded_images_shape = tf.shape(padded_images)
      outputs = [boxes_shape, padded_boxes_shape, padded_masks_shape,
                 keypoints_shape, padded_keypoints_shape, images_shape,
                 padded_images_shape, boxes, padded_boxes, keypoints,
                 padded_keypoints]
      if include_dense_pose:
        padded_dp_surface_coords = padded_tensor_dict[
            fields.InputDataFields.groundtruth_dp_surface_coords]
        outputs.extend([dp_surface_coords, padded_dp_surface_coords])
      return outputs

    outputs = self.execute_cpu(graph_fn, [])
    boxes_shape_ = outputs[0]
    padded_boxes_shape_ = outputs[1]
    padded_masks_shape_ = outputs[2]
    keypoints_shape_ = outputs[3]
    padded_keypoints_shape_ = outputs[4]
    images_shape_ = outputs[5]
    padded_images_shape_ = outputs[6]
    boxes_ = outputs[7]
    padded_boxes_ = outputs[8]
    keypoints_ = outputs[9]
    padded_keypoints_ = outputs[10]

    self.assertAllEqual(boxes_shape_, padded_boxes_shape_)
    self.assertAllEqual(keypoints_shape_, padded_keypoints_shape_)
    self.assertTrue((images_shape_[1] >= padded_images_shape_[1] * 0.5).all)
    self.assertTrue((images_shape_[2] >= padded_images_shape_[2] * 0.5).all)
    self.assertTrue((images_shape_[1] <= padded_images_shape_[1]).all)
    self.assertTrue((images_shape_[2] <= padded_images_shape_[2]).all)
    self.assertAllEqual(padded_masks_shape_[1:3], padded_images_shape_[1:3])
    self.assertTrue(np.all((boxes_[:, 2] - boxes_[:, 0]) >= (
        padded_boxes_[:, 2] - padded_boxes_[:, 0])))
    self.assertTrue(np.all((boxes_[:, 3] - boxes_[:, 1]) >= (
        padded_boxes_[:, 3] - padded_boxes_[:, 1])))
    self.assertTrue(np.all((keypoints_[1, :, 0] - keypoints_[0, :, 0]) >= (
        padded_keypoints_[1, :, 0] - padded_keypoints_[0, :, 0])))
    self.assertTrue(np.all((keypoints_[1, :, 1] - keypoints_[0, :, 1]) >= (
        padded_keypoints_[1, :, 1] - padded_keypoints_[0, :, 1])))
    if include_dense_pose:
      dp_surface_coords = outputs[11]
      padded_dp_surface_coords = outputs[12]
      self.assertAllClose(padded_dp_surface_coords[:, :, 2:],
                          dp_surface_coords[:, :, 2:])

  def testRandomAbsolutePadImage(self):
    height_padding = 10
    width_padding = 20
    def graph_fn():
      images = self.createTestImages()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      tensor_dict = {
          fields.InputDataFields.image: tf.cast(images, dtype=tf.float32),
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
      }
      preprocessing_options = [(preprocessor.random_absolute_pad_image, {
          'max_height_padding': height_padding,
          'max_width_padding': width_padding})]
      padded_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                   preprocessing_options)
      original_shape = tf.shape(images)
      final_shape = tf.shape(padded_tensor_dict[fields.InputDataFields.image])
      return original_shape, final_shape
    for _ in range(100):
      original_shape, output_shape = self.execute_cpu(graph_fn, [])
      _, height, width, _ = original_shape
      self.assertGreaterEqual(output_shape[1], height)
      self.assertLess(output_shape[1], height + height_padding)
      self.assertGreaterEqual(output_shape[2], width)
      self.assertLess(output_shape[2], width + width_padding)

  def testRandomAbsolutePadImageWithKeypoints(self):
    height_padding = 10
    width_padding = 20
    def graph_fn():
      images = self.createTestImages()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      keypoints, _ = self.createTestKeypoints()
      tensor_dict = {
          fields.InputDataFields.image: tf.cast(images, dtype=tf.float32),
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_keypoints: keypoints,
      }

      preprocessing_options = [(preprocessor.random_absolute_pad_image, {
          'max_height_padding': height_padding,
          'max_width_padding': width_padding
      })]
      padded_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                   preprocessing_options)
      original_shape = tf.shape(images)
      final_shape = tf.shape(padded_tensor_dict[fields.InputDataFields.image])
      padded_keypoints = padded_tensor_dict[
          fields.InputDataFields.groundtruth_keypoints]
      return (original_shape, final_shape, padded_keypoints)
    for _ in range(100):
      original_shape, output_shape, padded_keypoints_ = self.execute_cpu(
          graph_fn, [])
      _, height, width, _ = original_shape
      self.assertGreaterEqual(output_shape[1], height)
      self.assertLess(output_shape[1], height + height_padding)
      self.assertGreaterEqual(output_shape[2], width)
      self.assertLess(output_shape[2], width + width_padding)
      # Verify the keypoints are populated. The correctness of the keypoint
      # coordinates are already tested in random_pad_image function.
      self.assertEqual(padded_keypoints_.shape, (2, 3, 2))

  def testRandomCropPadImageWithCache(self):
    preprocess_options = [(preprocessor.normalize_image, {
        'original_minval': 0,
        'original_maxval': 255,
        'target_minval': 0,
        'target_maxval': 1,}), (preprocessor.random_crop_pad_image, {})]
    self._testPreprocessorCache(preprocess_options,
                                test_boxes=True,
                                test_masks=True,
                                test_keypoints=True)

  def testRandomCropPadImageWithRandomCoefOne(self):
    def graph_fn():
      preprocessing_options = [(preprocessor.normalize_image, {
          'original_minval': 0,
          'original_maxval': 255,
          'target_minval': 0,
          'target_maxval': 1
      })]

      images = self.createTestImages()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
      }
      tensor_dict = preprocessor.preprocess(tensor_dict, preprocessing_options)
      images = tensor_dict[fields.InputDataFields.image]

      preprocessing_options = [(preprocessor.random_crop_pad_image, {
          'random_coef': 1.0
      })]
      padded_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                   preprocessing_options)

      padded_images = padded_tensor_dict[fields.InputDataFields.image]
      padded_boxes = padded_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]
      boxes_shape = tf.shape(boxes)
      padded_boxes_shape = tf.shape(padded_boxes)
      images_shape = tf.shape(images)
      padded_images_shape = tf.shape(padded_images)
      return [boxes_shape, padded_boxes_shape, images_shape,
              padded_images_shape, boxes, padded_boxes]
    (boxes_shape_, padded_boxes_shape_, images_shape_,
     padded_images_shape_, boxes_, padded_boxes_) = self.execute_cpu(graph_fn,
                                                                     [])
    self.assertAllEqual(boxes_shape_, padded_boxes_shape_)
    self.assertTrue((images_shape_[1] >= padded_images_shape_[1] * 0.5).all)
    self.assertTrue((images_shape_[2] >= padded_images_shape_[2] * 0.5).all)
    self.assertTrue((images_shape_[1] <= padded_images_shape_[1]).all)
    self.assertTrue((images_shape_[2] <= padded_images_shape_[2]).all)
    self.assertTrue(np.all((boxes_[:, 2] - boxes_[:, 0]) >= (
        padded_boxes_[:, 2] - padded_boxes_[:, 0])))
    self.assertTrue(np.all((boxes_[:, 3] - boxes_[:, 1]) >= (
        padded_boxes_[:, 3] - padded_boxes_[:, 1])))

  def testRandomCropToAspectRatio(self):
    def graph_fn():
      images = self.createTestImages()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
      }
      tensor_dict = preprocessor.preprocess(tensor_dict, [])
      images = tensor_dict[fields.InputDataFields.image]

      preprocessing_options = [(preprocessor.random_crop_to_aspect_ratio, {
          'aspect_ratio': 2.0
      })]
      cropped_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                    preprocessing_options)

      cropped_images = cropped_tensor_dict[fields.InputDataFields.image]
      cropped_boxes = cropped_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]
      boxes_shape = tf.shape(boxes)
      cropped_boxes_shape = tf.shape(cropped_boxes)
      images_shape = tf.shape(images)
      cropped_images_shape = tf.shape(cropped_images)
      return [
          boxes_shape, cropped_boxes_shape, images_shape, cropped_images_shape
      ]

    (boxes_shape_, cropped_boxes_shape_, images_shape_,
     cropped_images_shape_) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(boxes_shape_, cropped_boxes_shape_)
    self.assertEqual(images_shape_[1], cropped_images_shape_[1] * 2)
    self.assertEqual(images_shape_[2], cropped_images_shape_[2])

  def testRandomPadToAspectRatio(self):
    def graph_fn():
      images = self.createTestImages()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
      }
      tensor_dict = preprocessor.preprocess(tensor_dict, [])
      images = tensor_dict[fields.InputDataFields.image]

      preprocessing_options = [(preprocessor.random_pad_to_aspect_ratio, {
          'aspect_ratio': 2.0
      })]
      padded_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                   preprocessing_options)

      padded_images = padded_tensor_dict[fields.InputDataFields.image]
      padded_boxes = padded_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]
      boxes_shape = tf.shape(boxes)
      padded_boxes_shape = tf.shape(padded_boxes)
      images_shape = tf.shape(images)
      padded_images_shape = tf.shape(padded_images)
      return [
          boxes_shape, padded_boxes_shape, images_shape, padded_images_shape
      ]

    (boxes_shape_, padded_boxes_shape_, images_shape_,
     padded_images_shape_) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(boxes_shape_, padded_boxes_shape_)
    self.assertEqual(images_shape_[1], padded_images_shape_[1])
    self.assertEqual(2 * images_shape_[2], padded_images_shape_[2])

  def testRandomBlackPatchesWithCache(self):
    preprocess_options = []
    preprocess_options.append((preprocessor.normalize_image, {
        'original_minval': 0,
        'original_maxval': 255,
        'target_minval': 0,
        'target_maxval': 1
    }))
    preprocess_options.append((preprocessor.random_black_patches, {
        'size_to_image_ratio': 0.5
    }))
    self._testPreprocessorCache(preprocess_options,
                                test_boxes=True,
                                test_masks=True,
                                test_keypoints=True)

  def testRandomBlackPatches(self):
    def graph_fn():
      preprocessing_options = []
      preprocessing_options.append((preprocessor.normalize_image, {
          'original_minval': 0,
          'original_maxval': 255,
          'target_minval': 0,
          'target_maxval': 1
      }))
      preprocessing_options.append((preprocessor.random_black_patches, {
          'size_to_image_ratio': 0.5
      }))
      images = self.createTestImages()
      tensor_dict = {fields.InputDataFields.image: images}
      blacked_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                    preprocessing_options)
      blacked_images = blacked_tensor_dict[fields.InputDataFields.image]
      images_shape = tf.shape(images)
      blacked_images_shape = tf.shape(blacked_images)
      return [images_shape, blacked_images_shape]
    (images_shape_, blacked_images_shape_) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(images_shape_, blacked_images_shape_)

  def testRandomJpegQuality(self):
    def graph_fn():
      preprocessing_options = [(preprocessor.random_jpeg_quality, {
          'min_jpeg_quality': 0,
          'max_jpeg_quality': 100
      })]
      images = self.createTestImages()
      tensor_dict = {fields.InputDataFields.image: images}
      processed_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                      preprocessing_options)
      encoded_images = processed_tensor_dict[fields.InputDataFields.image]
      images_shape = tf.shape(images)
      encoded_images_shape = tf.shape(encoded_images)
      return [images_shape, encoded_images_shape]
    images_shape_out, encoded_images_shape_out = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(images_shape_out, encoded_images_shape_out)

  def testRandomJpegQualityKeepsStaticChannelShape(self):
    # Set at least three weeks past the forward compatibility horizon for
    # tf 1.14 of 2019/11/01.
    # https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/python/compat/compat.py#L30
    if not tf.compat.forward_compatible(year=2019, month=12, day=1):
      self.skipTest('Skipping test for future functionality.')
    preprocessing_options = [(preprocessor.random_jpeg_quality, {
        'min_jpeg_quality': 0,
        'max_jpeg_quality': 100
    })]
    images = self.createTestImages()
    tensor_dict = {fields.InputDataFields.image: images}
    processed_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                    preprocessing_options)
    encoded_images = processed_tensor_dict[fields.InputDataFields.image]
    images_static_channels = images.shape[-1]
    encoded_images_static_channels = encoded_images.shape[-1]
    self.assertEqual(images_static_channels, encoded_images_static_channels)

  def testRandomJpegQualityWithCache(self):
    preprocessing_options = [(preprocessor.random_jpeg_quality, {
        'min_jpeg_quality': 0,
        'max_jpeg_quality': 100
    })]
    self._testPreprocessorCache(preprocessing_options)

  def testRandomJpegQualityWithRandomCoefOne(self):
    def graph_fn():
      preprocessing_options = [(preprocessor.random_jpeg_quality, {
          'random_coef': 1.0
      })]
      images = self.createTestImages()
      tensor_dict = {fields.InputDataFields.image: images}
      processed_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                      preprocessing_options)
      encoded_images = processed_tensor_dict[fields.InputDataFields.image]
      images_shape = tf.shape(images)
      encoded_images_shape = tf.shape(encoded_images)
      return [images, encoded_images, images_shape, encoded_images_shape]

    (images_out, encoded_images_out, images_shape_out,
     encoded_images_shape_out) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(images_shape_out, encoded_images_shape_out)
    self.assertAllEqual(images_out, encoded_images_out)

  def testRandomDownscaleToTargetPixels(self):
    def graph_fn():
      preprocessing_options = [(preprocessor.random_downscale_to_target_pixels,
                                {
                                    'min_target_pixels': 100,
                                    'max_target_pixels': 101
                                })]
      images = tf.random_uniform([1, 25, 100, 3])
      tensor_dict = {fields.InputDataFields.image: images}
      processed_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                      preprocessing_options)
      downscaled_images = processed_tensor_dict[fields.InputDataFields.image]
      downscaled_shape = tf.shape(downscaled_images)
      return downscaled_shape
    expected_shape = [1, 5, 20, 3]
    downscaled_shape_out = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(downscaled_shape_out, expected_shape)

  def testRandomDownscaleToTargetPixelsWithMasks(self):
    def graph_fn():
      preprocessing_options = [(preprocessor.random_downscale_to_target_pixels,
                                {
                                    'min_target_pixels': 100,
                                    'max_target_pixels': 101
                                })]
      images = tf.random_uniform([1, 25, 100, 3])
      masks = tf.random_uniform([10, 25, 100])
      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_instance_masks: masks
      }
      preprocessor_arg_map = preprocessor.get_default_func_arg_map(
          include_instance_masks=True)
      processed_tensor_dict = preprocessor.preprocess(
          tensor_dict, preprocessing_options, func_arg_map=preprocessor_arg_map)
      downscaled_images = processed_tensor_dict[fields.InputDataFields.image]
      downscaled_masks = processed_tensor_dict[
          fields.InputDataFields.groundtruth_instance_masks]
      downscaled_images_shape = tf.shape(downscaled_images)
      downscaled_masks_shape = tf.shape(downscaled_masks)
      return [downscaled_images_shape, downscaled_masks_shape]
    expected_images_shape = [1, 5, 20, 3]
    expected_masks_shape = [10, 5, 20]
    (downscaled_images_shape_out,
     downscaled_masks_shape_out) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(downscaled_images_shape_out, expected_images_shape)
    self.assertAllEqual(downscaled_masks_shape_out, expected_masks_shape)

  @parameterized.parameters(
      {'test_masks': False},
      {'test_masks': True}
  )
  def testRandomDownscaleToTargetPixelsWithCache(self, test_masks):
    preprocessing_options = [(preprocessor.random_downscale_to_target_pixels, {
        'min_target_pixels': 100,
        'max_target_pixels': 999
    })]
    self._testPreprocessorCache(preprocessing_options, test_masks=test_masks)

  def testRandomDownscaleToTargetPixelsWithRandomCoefOne(self):
    def graph_fn():
      preprocessing_options = [(preprocessor.random_downscale_to_target_pixels,
                                {
                                    'random_coef': 1.0,
                                    'min_target_pixels': 10,
                                    'max_target_pixels': 20,
                                })]
      images = tf.random_uniform([1, 25, 100, 3])
      tensor_dict = {fields.InputDataFields.image: images}
      processed_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                      preprocessing_options)
      downscaled_images = processed_tensor_dict[fields.InputDataFields.image]
      images_shape = tf.shape(images)
      downscaled_images_shape = tf.shape(downscaled_images)
      return [images, downscaled_images, images_shape, downscaled_images_shape]
    (images_out, downscaled_images_out, images_shape_out,
     downscaled_images_shape_out) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(images_shape_out, downscaled_images_shape_out)
    self.assertAllEqual(images_out, downscaled_images_out)

  def testRandomDownscaleToTargetPixelsIgnoresSmallImages(self):
    def graph_fn():
      preprocessing_options = [(preprocessor.random_downscale_to_target_pixels,
                                {
                                    'min_target_pixels': 1000,
                                    'max_target_pixels': 1001
                                })]
      images = tf.random_uniform([1, 10, 10, 3])
      tensor_dict = {fields.InputDataFields.image: images}
      processed_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                      preprocessing_options)
      downscaled_images = processed_tensor_dict[fields.InputDataFields.image]
      images_shape = tf.shape(images)
      downscaled_images_shape = tf.shape(downscaled_images)
      return [images, downscaled_images, images_shape, downscaled_images_shape]
    (images_out, downscaled_images_out, images_shape_out,
     downscaled_images_shape_out) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(images_shape_out, downscaled_images_shape_out)
    self.assertAllEqual(images_out, downscaled_images_out)

  def testRandomPatchGaussianShape(self):
    preprocessing_options = [(preprocessor.random_patch_gaussian, {
        'min_patch_size': 1,
        'max_patch_size': 200,
        'min_gaussian_stddev': 0.0,
        'max_gaussian_stddev': 2.0
    })]
    images = self.createTestImages()
    tensor_dict = {fields.InputDataFields.image: images}
    processed_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                    preprocessing_options)
    patched_images = processed_tensor_dict[fields.InputDataFields.image]
    images_shape = tf.shape(images)
    patched_images_shape = tf.shape(patched_images)
    self.assertAllEqual(images_shape, patched_images_shape)

  def testRandomPatchGaussianClippedToLowerBound(self):
    def graph_fn():
      preprocessing_options = [(preprocessor.random_patch_gaussian, {
          'min_patch_size': 20,
          'max_patch_size': 40,
          'min_gaussian_stddev': 50,
          'max_gaussian_stddev': 100
      })]
      images = tf.zeros([1, 5, 4, 3])
      tensor_dict = {fields.InputDataFields.image: images}
      processed_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                      preprocessing_options)
      patched_images = processed_tensor_dict[fields.InputDataFields.image]
      return patched_images
    patched_images = self.execute_cpu(graph_fn, [])
    self.assertAllGreaterEqual(patched_images, 0.0)

  def testRandomPatchGaussianClippedToUpperBound(self):
    def graph_fn():
      preprocessing_options = [(preprocessor.random_patch_gaussian, {
          'min_patch_size': 20,
          'max_patch_size': 40,
          'min_gaussian_stddev': 50,
          'max_gaussian_stddev': 100
      })]
      images = tf.constant(255.0, shape=[1, 5, 4, 3])
      tensor_dict = {fields.InputDataFields.image: images}
      processed_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                      preprocessing_options)
      patched_images = processed_tensor_dict[fields.InputDataFields.image]
      return patched_images
    patched_images = self.execute_cpu(graph_fn, [])
    self.assertAllLessEqual(patched_images, 255.0)

  def testRandomPatchGaussianWithCache(self):
    preprocessing_options = [(preprocessor.random_patch_gaussian, {
        'min_patch_size': 1,
        'max_patch_size': 200,
        'min_gaussian_stddev': 0.0,
        'max_gaussian_stddev': 2.0
    })]
    self._testPreprocessorCache(preprocessing_options)

  def testRandomPatchGaussianWithRandomCoefOne(self):
    def graph_fn():
      preprocessing_options = [(preprocessor.random_patch_gaussian, {
          'random_coef': 1.0
      })]
      images = self.createTestImages()
      tensor_dict = {fields.InputDataFields.image: images}
      processed_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                      preprocessing_options)
      patched_images = processed_tensor_dict[fields.InputDataFields.image]
      images_shape = tf.shape(images)
      patched_images_shape = tf.shape(patched_images)
      return patched_images_shape, patched_images, images_shape, images
    (patched_images_shape, patched_images, images_shape,
     images) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(images_shape, patched_images_shape)
    self.assertAllEqual(images, patched_images)

  @unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
  def testAutoAugmentImage(self):
    def graph_fn():
      preprocessing_options = []
      preprocessing_options.append((preprocessor.autoaugment_image, {
          'policy_name': 'v1'
      }))
      images = self.createTestImages()
      boxes = self.createTestBoxes()
      tensor_dict = {fields.InputDataFields.image: images,
                     fields.InputDataFields.groundtruth_boxes: boxes}
      autoaugment_tensor_dict = preprocessor.preprocess(
          tensor_dict, preprocessing_options)
      augmented_images = autoaugment_tensor_dict[fields.InputDataFields.image]
      augmented_boxes = autoaugment_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]
      images_shape = tf.shape(images)
      boxes_shape = tf.shape(boxes)
      augmented_images_shape = tf.shape(augmented_images)
      augmented_boxes_shape = tf.shape(augmented_boxes)
      return [images_shape, boxes_shape, augmented_images_shape,
              augmented_boxes_shape]
    (images_shape_, boxes_shape_, augmented_images_shape_,
     augmented_boxes_shape_) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(images_shape_, augmented_images_shape_)
    self.assertAllEqual(boxes_shape_, augmented_boxes_shape_)

  def testRandomResizeMethodWithCache(self):
    preprocess_options = []
    preprocess_options.append((preprocessor.normalize_image, {
        'original_minval': 0,
        'original_maxval': 255,
        'target_minval': 0,
        'target_maxval': 1
    }))
    preprocess_options.append((preprocessor.random_resize_method, {
        'target_size': (75, 150)
    }))
    self._testPreprocessorCache(preprocess_options,
                                test_boxes=True,
                                test_masks=True,
                                test_keypoints=True)

  def testRandomResizeMethod(self):
    def graph_fn():
      preprocessing_options = []
      preprocessing_options.append((preprocessor.normalize_image, {
          'original_minval': 0,
          'original_maxval': 255,
          'target_minval': 0,
          'target_maxval': 1
      }))
      preprocessing_options.append((preprocessor.random_resize_method, {
          'target_size': (75, 150)
      }))
      images = self.createTestImages()
      tensor_dict = {fields.InputDataFields.image: images}
      resized_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                    preprocessing_options)
      resized_images = resized_tensor_dict[fields.InputDataFields.image]
      resized_images_shape = tf.shape(resized_images)
      expected_images_shape = tf.constant([1, 75, 150, 3], dtype=tf.int32)
      return [expected_images_shape, resized_images_shape]
    (expected_images_shape_, resized_images_shape_) = self.execute_cpu(graph_fn,
                                                                       [])
    self.assertAllEqual(expected_images_shape_,
                        resized_images_shape_)

  def testResizeImageWithMasks(self):
    """Tests image resizing, checking output sizes."""
    in_image_shape_list = [[60, 40, 3], [15, 30, 3]]
    in_masks_shape_list = [[15, 60, 40], [10, 15, 30]]
    height = 50
    width = 100
    expected_image_shape_list = [[50, 100, 3], [50, 100, 3]]
    expected_masks_shape_list = [[15, 50, 100], [10, 50, 100]]
    def graph_fn(in_image_shape, in_masks_shape):
      in_image = tf.random_uniform(in_image_shape)
      in_masks = tf.random_uniform(in_masks_shape)
      out_image, out_masks, _ = preprocessor.resize_image(
          in_image, in_masks, new_height=height, new_width=width)
      out_image_shape = tf.shape(out_image)
      out_masks_shape = tf.shape(out_masks)
      return out_image_shape, out_masks_shape
    for (in_image_shape, expected_image_shape, in_masks_shape,
         expected_mask_shape) in zip(in_image_shape_list,
                                     expected_image_shape_list,
                                     in_masks_shape_list,
                                     expected_masks_shape_list):
      (out_image_shape,
       out_masks_shape) = self.execute_cpu(graph_fn, [
           np.array(in_image_shape, np.int32),
           np.array(in_masks_shape, np.int32)
       ])
      self.assertAllEqual(out_image_shape, expected_image_shape)
      self.assertAllEqual(out_masks_shape, expected_mask_shape)

  def testResizeImageWithMasksTensorInputHeightAndWidth(self):
    """Tests image resizing, checking output sizes."""
    in_image_shape_list = [[60, 40, 3], [15, 30, 3]]
    in_masks_shape_list = [[15, 60, 40], [10, 15, 30]]
    expected_image_shape_list = [[50, 100, 3], [50, 100, 3]]
    expected_masks_shape_list = [[15, 50, 100], [10, 50, 100]]
    def graph_fn(in_image_shape, in_masks_shape):
      height = tf.constant(50, dtype=tf.int32)
      width = tf.constant(100, dtype=tf.int32)
      in_image = tf.random_uniform(in_image_shape)
      in_masks = tf.random_uniform(in_masks_shape)
      out_image, out_masks, _ = preprocessor.resize_image(
          in_image, in_masks, new_height=height, new_width=width)
      out_image_shape = tf.shape(out_image)
      out_masks_shape = tf.shape(out_masks)
      return out_image_shape, out_masks_shape
    for (in_image_shape, expected_image_shape, in_masks_shape,
         expected_mask_shape) in zip(in_image_shape_list,
                                     expected_image_shape_list,
                                     in_masks_shape_list,
                                     expected_masks_shape_list):
      (out_image_shape,
       out_masks_shape) = self.execute_cpu(graph_fn, [
           np.array(in_image_shape, np.int32),
           np.array(in_masks_shape, np.int32)
       ])
      self.assertAllEqual(out_image_shape, expected_image_shape)
      self.assertAllEqual(out_masks_shape, expected_mask_shape)

  def testResizeImageWithNoInstanceMask(self):
    """Tests image resizing, checking output sizes."""
    in_image_shape_list = [[60, 40, 3], [15, 30, 3]]
    in_masks_shape_list = [[0, 60, 40], [0, 15, 30]]
    height = 50
    width = 100
    expected_image_shape_list = [[50, 100, 3], [50, 100, 3]]
    expected_masks_shape_list = [[0, 50, 100], [0, 50, 100]]
    def graph_fn(in_image_shape, in_masks_shape):
      in_image = tf.random_uniform(in_image_shape)
      in_masks = tf.random_uniform(in_masks_shape)
      out_image, out_masks, _ = preprocessor.resize_image(
          in_image, in_masks, new_height=height, new_width=width)
      out_image_shape = tf.shape(out_image)
      out_masks_shape = tf.shape(out_masks)
      return out_image_shape, out_masks_shape
    for (in_image_shape, expected_image_shape, in_masks_shape,
         expected_mask_shape) in zip(in_image_shape_list,
                                     expected_image_shape_list,
                                     in_masks_shape_list,
                                     expected_masks_shape_list):
      (out_image_shape,
       out_masks_shape) = self.execute_cpu(graph_fn, [
           np.array(in_image_shape, np.int32),
           np.array(in_masks_shape, np.int32)
       ])
      self.assertAllEqual(out_image_shape, expected_image_shape)
      self.assertAllEqual(out_masks_shape, expected_mask_shape)

  def testResizeToRangePreservesStaticSpatialShape(self):
    """Tests image resizing, checking output sizes."""
    in_shape_list = [[60, 40, 3], [15, 30, 3], [15, 50, 3]]
    min_dim = 50
    max_dim = 100
    expected_shape_list = [[75, 50, 3], [50, 100, 3], [30, 100, 3]]

    for in_shape, expected_shape in zip(in_shape_list, expected_shape_list):
      in_image = tf.random_uniform(in_shape)
      out_image, _ = preprocessor.resize_to_range(
          in_image, min_dimension=min_dim, max_dimension=max_dim)
      self.assertAllEqual(out_image.get_shape().as_list(), expected_shape)

  def testResizeToRangeWithDynamicSpatialShape(self):
    """Tests image resizing, checking output sizes."""
    in_shape_list = [[60, 40, 3], [15, 30, 3], [15, 50, 3]]
    min_dim = 50
    max_dim = 100
    expected_shape_list = [[75, 50, 3], [50, 100, 3], [30, 100, 3]]
    def graph_fn(in_image_shape):
      in_image = tf.random_uniform(in_image_shape)
      out_image, _ = preprocessor.resize_to_range(
          in_image, min_dimension=min_dim, max_dimension=max_dim)
      out_image_shape = tf.shape(out_image)
      return out_image_shape
    for in_shape, expected_shape in zip(in_shape_list, expected_shape_list):
      out_image_shape = self.execute_cpu(graph_fn, [np.array(in_shape,
                                                             np.int32)])
      self.assertAllEqual(out_image_shape, expected_shape)

  def testResizeToRangeWithPadToMaxDimensionReturnsCorrectShapes(self):
    in_shape_list = [[60, 40, 3], [15, 30, 3], [15, 50, 3]]
    min_dim = 50
    max_dim = 100
    expected_shape_list = [[100, 100, 3], [100, 100, 3], [100, 100, 3]]
    def graph_fn(in_image):
      out_image, _ = preprocessor.resize_to_range(
          in_image,
          min_dimension=min_dim,
          max_dimension=max_dim,
          pad_to_max_dimension=True)
      return tf.shape(out_image)
    for in_shape, expected_shape in zip(in_shape_list, expected_shape_list):
      out_image_shape = self.execute_cpu(
          graph_fn, [np.random.rand(*in_shape).astype('f')])
      self.assertAllEqual(out_image_shape, expected_shape)

  def testResizeToRangeWithPadToMaxDimensionReturnsCorrectTensor(self):
    in_image_np = np.array([[[0, 1, 2]]], np.float32)
    ex_image_np = np.array(
        [[[0, 1, 2], [123.68, 116.779, 103.939]],
         [[123.68, 116.779, 103.939], [123.68, 116.779, 103.939]]], np.float32)
    min_dim = 1
    max_dim = 2
    def graph_fn(in_image):
      out_image, _ = preprocessor.resize_to_range(
          in_image,
          min_dimension=min_dim,
          max_dimension=max_dim,
          pad_to_max_dimension=True,
          per_channel_pad_value=(123.68, 116.779, 103.939))
      return out_image
    out_image_np = self.execute_cpu(graph_fn, [in_image_np])
    self.assertAllClose(ex_image_np, out_image_np)

  def testResizeToRangeWithMasksPreservesStaticSpatialShape(self):
    """Tests image resizing, checking output sizes."""
    in_image_shape_list = [[60, 40, 3], [15, 30, 3]]
    in_masks_shape_list = [[15, 60, 40], [10, 15, 30]]
    min_dim = 50
    max_dim = 100
    expected_image_shape_list = [[75, 50, 3], [50, 100, 3]]
    expected_masks_shape_list = [[15, 75, 50], [10, 50, 100]]

    for (in_image_shape, expected_image_shape, in_masks_shape,
         expected_mask_shape) in zip(in_image_shape_list,
                                     expected_image_shape_list,
                                     in_masks_shape_list,
                                     expected_masks_shape_list):
      in_image = tf.random_uniform(in_image_shape)
      in_masks = tf.random_uniform(in_masks_shape)
      out_image, out_masks, _ = preprocessor.resize_to_range(
          in_image, in_masks, min_dimension=min_dim, max_dimension=max_dim)
      self.assertAllEqual(out_masks.get_shape().as_list(), expected_mask_shape)
      self.assertAllEqual(out_image.get_shape().as_list(), expected_image_shape)

  def testResizeToRangeWithMasksAndPadToMaxDimension(self):
    """Tests image resizing, checking output sizes."""
    in_image_shape_list = [[60, 40, 3], [15, 30, 3]]
    in_masks_shape_list = [[15, 60, 40], [10, 15, 30]]
    min_dim = 50
    max_dim = 100
    expected_image_shape_list = [[100, 100, 3], [100, 100, 3]]
    expected_masks_shape_list = [[15, 100, 100], [10, 100, 100]]
    def graph_fn(in_image, in_masks):
      out_image, out_masks, _ = preprocessor.resize_to_range(
          in_image, in_masks, min_dimension=min_dim,
          max_dimension=max_dim, pad_to_max_dimension=True)
      out_image_shape = tf.shape(out_image)
      out_masks_shape = tf.shape(out_masks)
      return [out_image_shape, out_masks_shape]
    for (in_image_shape, expected_image_shape, in_masks_shape,
         expected_mask_shape) in zip(in_image_shape_list,
                                     expected_image_shape_list,
                                     in_masks_shape_list,
                                     expected_masks_shape_list):
      out_image_shape, out_masks_shape = self.execute_cpu(
          graph_fn, [
              np.random.rand(*in_image_shape).astype('f'),
              np.random.rand(*in_masks_shape).astype('f'),
          ])
      self.assertAllEqual(out_image_shape, expected_image_shape)
      self.assertAllEqual(out_masks_shape, expected_mask_shape)

  def testResizeToRangeWithMasksAndDynamicSpatialShape(self):
    """Tests image resizing, checking output sizes."""
    in_image_shape_list = [[60, 40, 3], [15, 30, 3]]
    in_masks_shape_list = [[15, 60, 40], [10, 15, 30]]
    min_dim = 50
    max_dim = 100
    expected_image_shape_list = [[75, 50, 3], [50, 100, 3]]
    expected_masks_shape_list = [[15, 75, 50], [10, 50, 100]]
    def graph_fn(in_image, in_masks):
      out_image, out_masks, _ = preprocessor.resize_to_range(
          in_image, in_masks, min_dimension=min_dim, max_dimension=max_dim)
      out_image_shape = tf.shape(out_image)
      out_masks_shape = tf.shape(out_masks)
      return [out_image_shape, out_masks_shape]
    for (in_image_shape, expected_image_shape, in_masks_shape,
         expected_mask_shape) in zip(in_image_shape_list,
                                     expected_image_shape_list,
                                     in_masks_shape_list,
                                     expected_masks_shape_list):
      out_image_shape, out_masks_shape = self.execute_cpu(
          graph_fn, [
              np.random.rand(*in_image_shape).astype('f'),
              np.random.rand(*in_masks_shape).astype('f'),
          ])
      self.assertAllEqual(out_image_shape, expected_image_shape)
      self.assertAllEqual(out_masks_shape, expected_mask_shape)

  def testResizeToRangeWithInstanceMasksTensorOfSizeZero(self):
    """Tests image resizing, checking output sizes."""
    in_image_shape_list = [[60, 40, 3], [15, 30, 3]]
    in_masks_shape_list = [[0, 60, 40], [0, 15, 30]]
    min_dim = 50
    max_dim = 100
    expected_image_shape_list = [[75, 50, 3], [50, 100, 3]]
    expected_masks_shape_list = [[0, 75, 50], [0, 50, 100]]
    def graph_fn(in_image, in_masks):
      out_image, out_masks, _ = preprocessor.resize_to_range(
          in_image, in_masks, min_dimension=min_dim, max_dimension=max_dim)
      out_image_shape = tf.shape(out_image)
      out_masks_shape = tf.shape(out_masks)
      return [out_image_shape, out_masks_shape]
    for (in_image_shape, expected_image_shape, in_masks_shape,
         expected_mask_shape) in zip(in_image_shape_list,
                                     expected_image_shape_list,
                                     in_masks_shape_list,
                                     expected_masks_shape_list):
      out_image_shape, out_masks_shape = self.execute_cpu(
          graph_fn, [
              np.random.rand(*in_image_shape).astype('f'),
              np.random.rand(*in_masks_shape).astype('f'),
          ])
      self.assertAllEqual(out_image_shape, expected_image_shape)
      self.assertAllEqual(out_masks_shape, expected_mask_shape)

  def testResizeToRange4DImageTensor(self):
    image = tf.random_uniform([1, 200, 300, 3])
    with self.assertRaises(ValueError):
      preprocessor.resize_to_range(image, 500, 600)

  def testResizeToRangeSameMinMax(self):
    """Tests image resizing, checking output sizes."""
    in_shape_list = [[312, 312, 3], [299, 299, 3]]
    min_dim = 320
    max_dim = 320
    expected_shape_list = [[320, 320, 3], [320, 320, 3]]
    def graph_fn(in_shape):
      in_image = tf.random_uniform(in_shape)
      out_image, _ = preprocessor.resize_to_range(
          in_image, min_dimension=min_dim, max_dimension=max_dim)
      out_image_shape = tf.shape(out_image)
      return out_image_shape
    for in_shape, expected_shape in zip(in_shape_list, expected_shape_list):
      out_image_shape = self.execute_cpu(graph_fn, [np.array(in_shape,
                                                             np.int32)])
      self.assertAllEqual(out_image_shape, expected_shape)

  def testResizeToMaxDimensionTensorShapes(self):
    """Tests both cases where image should and shouldn't be resized."""
    in_image_shape_list = [[100, 50, 3], [15, 30, 3]]
    in_masks_shape_list = [[15, 100, 50], [10, 15, 30]]
    max_dim = 50
    expected_image_shape_list = [[50, 25, 3], [15, 30, 3]]
    expected_masks_shape_list = [[15, 50, 25], [10, 15, 30]]
    def graph_fn(in_image_shape, in_masks_shape):
      in_image = tf.random_uniform(in_image_shape)
      in_masks = tf.random_uniform(in_masks_shape)
      out_image, out_masks, _ = preprocessor.resize_to_max_dimension(
          in_image, in_masks, max_dimension=max_dim)
      out_image_shape = tf.shape(out_image)
      out_masks_shape = tf.shape(out_masks)
      return [out_image_shape, out_masks_shape]
    for (in_image_shape, expected_image_shape, in_masks_shape,
         expected_mask_shape) in zip(in_image_shape_list,
                                     expected_image_shape_list,
                                     in_masks_shape_list,
                                     expected_masks_shape_list):
      out_image_shape, out_masks_shape = self.execute_cpu(
          graph_fn, [
              np.array(in_image_shape, np.int32),
              np.array(in_masks_shape, np.int32)
          ])
      self.assertAllEqual(out_image_shape, expected_image_shape)
      self.assertAllEqual(out_masks_shape, expected_mask_shape)

  def testResizeToMaxDimensionWithInstanceMasksTensorOfSizeZero(self):
    """Tests both cases where image should and shouldn't be resized."""
    in_image_shape_list = [[100, 50, 3], [15, 30, 3]]
    in_masks_shape_list = [[0, 100, 50], [0, 15, 30]]
    max_dim = 50
    expected_image_shape_list = [[50, 25, 3], [15, 30, 3]]
    expected_masks_shape_list = [[0, 50, 25], [0, 15, 30]]

    def graph_fn(in_image_shape, in_masks_shape):
      in_image = tf.random_uniform(in_image_shape)
      in_masks = tf.random_uniform(in_masks_shape)
      out_image, out_masks, _ = preprocessor.resize_to_max_dimension(
          in_image, in_masks, max_dimension=max_dim)
      out_image_shape = tf.shape(out_image)
      out_masks_shape = tf.shape(out_masks)
      return [out_image_shape, out_masks_shape]

    for (in_image_shape, expected_image_shape, in_masks_shape,
         expected_mask_shape) in zip(in_image_shape_list,
                                     expected_image_shape_list,
                                     in_masks_shape_list,
                                     expected_masks_shape_list):
      out_image_shape, out_masks_shape = self.execute_cpu(
          graph_fn, [
              np.array(in_image_shape, np.int32),
              np.array(in_masks_shape, np.int32)
          ])
      self.assertAllEqual(out_image_shape, expected_image_shape)
      self.assertAllEqual(out_masks_shape, expected_mask_shape)

  def testResizeToMaxDimensionRaisesErrorOn4DImage(self):
    image = tf.random_uniform([1, 200, 300, 3])
    with self.assertRaises(ValueError):
      preprocessor.resize_to_max_dimension(image, 500)

  def testResizeToMinDimensionTensorShapes(self):
    in_image_shape_list = [[60, 55, 3], [15, 30, 3]]
    in_masks_shape_list = [[15, 60, 55], [10, 15, 30]]
    min_dim = 50
    expected_image_shape_list = [[60, 55, 3], [50, 100, 3]]
    expected_masks_shape_list = [[15, 60, 55], [10, 50, 100]]
    def graph_fn(in_image_shape, in_masks_shape):
      in_image = tf.random_uniform(in_image_shape)
      in_masks = tf.random_uniform(in_masks_shape)
      out_image, out_masks, _ = preprocessor.resize_to_min_dimension(
          in_image, in_masks, min_dimension=min_dim)
      out_image_shape = tf.shape(out_image)
      out_masks_shape = tf.shape(out_masks)
      return [out_image_shape, out_masks_shape]
    for (in_image_shape, expected_image_shape, in_masks_shape,
         expected_mask_shape) in zip(in_image_shape_list,
                                     expected_image_shape_list,
                                     in_masks_shape_list,
                                     expected_masks_shape_list):
      out_image_shape, out_masks_shape = self.execute_cpu(
          graph_fn, [
              np.array(in_image_shape, np.int32),
              np.array(in_masks_shape, np.int32)
          ])
      self.assertAllEqual(out_image_shape, expected_image_shape)
      self.assertAllEqual(out_masks_shape, expected_mask_shape)

  def testResizeToMinDimensionWithInstanceMasksTensorOfSizeZero(self):
    """Tests image resizing, checking output sizes."""
    in_image_shape_list = [[60, 40, 3], [15, 30, 3]]
    in_masks_shape_list = [[0, 60, 40], [0, 15, 30]]
    min_dim = 50
    expected_image_shape_list = [[75, 50, 3], [50, 100, 3]]
    expected_masks_shape_list = [[0, 75, 50], [0, 50, 100]]
    def graph_fn(in_image_shape, in_masks_shape):
      in_image = tf.random_uniform(in_image_shape)
      in_masks = tf.random_uniform(in_masks_shape)
      out_image, out_masks, _ = preprocessor.resize_to_min_dimension(
          in_image, in_masks, min_dimension=min_dim)
      out_image_shape = tf.shape(out_image)
      out_masks_shape = tf.shape(out_masks)
      return [out_image_shape, out_masks_shape]
    for (in_image_shape, expected_image_shape, in_masks_shape,
         expected_mask_shape) in zip(in_image_shape_list,
                                     expected_image_shape_list,
                                     in_masks_shape_list,
                                     expected_masks_shape_list):
      out_image_shape, out_masks_shape = self.execute_cpu(
          graph_fn, [
              np.array(in_image_shape, np.int32),
              np.array(in_masks_shape, np.int32)
          ])
      self.assertAllEqual(out_image_shape, expected_image_shape)
      self.assertAllEqual(out_masks_shape, expected_mask_shape)

  def testResizeToMinDimensionRaisesErrorOn4DImage(self):
    image = tf.random_uniform([1, 200, 300, 3])
    with self.assertRaises(ValueError):
      preprocessor.resize_to_min_dimension(image, 500)

  def testResizePadToMultipleNoMasks(self):
    """Tests resizing when padding to multiple without masks."""
    def graph_fn():
      image = tf.ones((200, 100, 3), dtype=tf.float32)
      out_image, out_shape = preprocessor.resize_pad_to_multiple(
          image, multiple=32)
      return out_image, out_shape

    out_image, out_shape = self.execute_cpu(graph_fn, [])
    self.assertAllClose(out_image.sum(), 200 * 100 * 3)
    self.assertAllEqual(out_shape, (200, 100, 3))
    self.assertAllEqual(out_image.shape, (224, 128, 3))

  def testResizePadToMultipleWithMasks(self):
    """Tests resizing when padding to multiple with masks."""
    def graph_fn():
      image = tf.ones((200, 100, 3), dtype=tf.float32)
      masks = tf.ones((10, 200, 100), dtype=tf.float32)

      _, out_masks, out_shape = preprocessor.resize_pad_to_multiple(
          image, multiple=32, masks=masks)
      return [out_masks, out_shape]

    out_masks, out_shape = self.execute_cpu(graph_fn, [])
    self.assertAllClose(out_masks.sum(), 200 * 100 * 10)
    self.assertAllEqual(out_shape, (200, 100, 3))
    self.assertAllEqual(out_masks.shape, (10, 224, 128))

  def testResizePadToMultipleEmptyMasks(self):
    """Tests resizing when padding to multiple with an empty mask."""
    def graph_fn():
      image = tf.ones((200, 100, 3), dtype=tf.float32)
      masks = tf.ones((0, 200, 100), dtype=tf.float32)
      _, out_masks, out_shape = preprocessor.resize_pad_to_multiple(
          image, multiple=32, masks=masks)
      return [out_masks, out_shape]
    out_masks, out_shape = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(out_shape, (200, 100, 3))
    self.assertAllEqual(out_masks.shape, (0, 224, 128))

  def testScaleBoxesToPixelCoordinates(self):
    """Tests box scaling, checking scaled values."""
    def graph_fn():
      in_shape = [60, 40, 3]
      in_boxes = [[0.1, 0.2, 0.4, 0.6],
                  [0.5, 0.3, 0.9, 0.7]]
      in_image = tf.random_uniform(in_shape)
      in_boxes = tf.constant(in_boxes)
      _, out_boxes = preprocessor.scale_boxes_to_pixel_coordinates(
          in_image, boxes=in_boxes)
      return out_boxes
    expected_boxes = [[6., 8., 24., 24.],
                      [30., 12., 54., 28.]]
    out_boxes = self.execute_cpu(graph_fn, [])
    self.assertAllClose(out_boxes, expected_boxes)

  def testScaleBoxesToPixelCoordinatesWithKeypoints(self):
    """Tests box and keypoint scaling, checking scaled values."""
    def graph_fn():
      in_shape = [60, 40, 3]
      in_boxes = self.createTestBoxes()
      in_keypoints, _ = self.createTestKeypoints()
      in_image = tf.random_uniform(in_shape)
      (_, out_boxes,
       out_keypoints) = preprocessor.scale_boxes_to_pixel_coordinates(
           in_image, boxes=in_boxes, keypoints=in_keypoints)
      return out_boxes, out_keypoints
    expected_boxes = [[0., 10., 45., 40.],
                      [15., 20., 45., 40.]]
    expected_keypoints = [
        [[6., 4.], [12., 8.], [18., 12.]],
        [[24., 16.], [30., 20.], [36., 24.]],
    ]
    out_boxes_, out_keypoints_ = self.execute_cpu(graph_fn, [])
    self.assertAllClose(out_boxes_, expected_boxes)
    self.assertAllClose(out_keypoints_, expected_keypoints)

  def testSubtractChannelMean(self):
    """Tests whether channel means have been subtracted."""
    def graph_fn():
      image = tf.zeros((240, 320, 3))
      means = [1, 2, 3]
      actual = preprocessor.subtract_channel_mean(image, means=means)
      return actual
    actual = self.execute_cpu(graph_fn, [])
    self.assertTrue((actual[:, :, 0], -1))
    self.assertTrue((actual[:, :, 1], -2))
    self.assertTrue((actual[:, :, 2], -3))

  def testOneHotEncoding(self):
    """Tests one hot encoding of multiclass labels."""
    def graph_fn():
      labels = tf.constant([1, 4, 2], dtype=tf.int32)
      one_hot = preprocessor.one_hot_encoding(labels, num_classes=5)
      return one_hot
    one_hot = self.execute_cpu(graph_fn, [])
    self.assertAllEqual([0, 1, 1, 0, 1], one_hot)

  def testRandomSelfConcatImageVertically(self):

    def graph_fn():
      images = self.createTestImages()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      confidences = weights
      scores = self.createTestMultiClassScores()

      tensor_dict = {
          fields.InputDataFields.image: tf.cast(images, dtype=tf.float32),
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
          fields.InputDataFields.groundtruth_confidences: confidences,
          fields.InputDataFields.multiclass_scores: scores,
      }

      preprocessing_options = [(preprocessor.random_self_concat_image, {
          'concat_vertical_probability': 1.0,
          'concat_horizontal_probability': 0.0,
      })]
      func_arg_map = preprocessor.get_default_func_arg_map(
          True, True, True)
      output_tensor_dict = preprocessor.preprocess(
          tensor_dict, preprocessing_options, func_arg_map=func_arg_map)

      original_shape = tf.shape(images)[1:3]
      final_shape = tf.shape(output_tensor_dict[fields.InputDataFields.image])[
          1:3]
      return [
          original_shape,
          boxes,
          labels,
          confidences,
          scores,
          final_shape,
          output_tensor_dict[fields.InputDataFields.groundtruth_boxes],
          output_tensor_dict[fields.InputDataFields.groundtruth_classes],
          output_tensor_dict[fields.InputDataFields.groundtruth_confidences],
          output_tensor_dict[fields.InputDataFields.multiclass_scores],
      ]
    (original_shape, boxes, labels, confidences, scores, final_shape, new_boxes,
     new_labels, new_confidences, new_scores) = self.execute(graph_fn, [])
    self.assertAllEqual(final_shape, original_shape * np.array([2, 1]))
    self.assertAllEqual(2 * boxes.size, new_boxes.size)
    self.assertAllEqual(2 * labels.size, new_labels.size)
    self.assertAllEqual(2 * confidences.size, new_confidences.size)
    self.assertAllEqual(2 * scores.size, new_scores.size)

  def testRandomSelfConcatImageHorizontally(self):
    def graph_fn():
      images = self.createTestImages()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      confidences = weights
      scores = self.createTestMultiClassScores()

      tensor_dict = {
          fields.InputDataFields.image: tf.cast(images, dtype=tf.float32),
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
          fields.InputDataFields.groundtruth_confidences: confidences,
          fields.InputDataFields.multiclass_scores: scores,
      }

      preprocessing_options = [(preprocessor.random_self_concat_image, {
          'concat_vertical_probability': 0.0,
          'concat_horizontal_probability': 1.0,
      })]
      func_arg_map = preprocessor.get_default_func_arg_map(
          True, True, True)
      output_tensor_dict = preprocessor.preprocess(
          tensor_dict, preprocessing_options, func_arg_map=func_arg_map)

      original_shape = tf.shape(images)[1:3]
      final_shape = tf.shape(output_tensor_dict[fields.InputDataFields.image])[
          1:3]
      return [
          original_shape,
          boxes,
          labels,
          confidences,
          scores,
          final_shape,
          output_tensor_dict[fields.InputDataFields.groundtruth_boxes],
          output_tensor_dict[fields.InputDataFields.groundtruth_classes],
          output_tensor_dict[fields.InputDataFields.groundtruth_confidences],
          output_tensor_dict[fields.InputDataFields.multiclass_scores],
      ]
    (original_shape, boxes, labels, confidences, scores, final_shape, new_boxes,
     new_labels, new_confidences, new_scores) = self.execute(graph_fn, [])
    self.assertAllEqual(final_shape, original_shape * np.array([1, 2]))
    self.assertAllEqual(2 * boxes.size, new_boxes.size)
    self.assertAllEqual(2 * labels.size, new_labels.size)
    self.assertAllEqual(2 * confidences.size, new_confidences.size)
    self.assertAllEqual(2 * scores.size, new_scores.size)

  def testSSDRandomCropWithCache(self):
    preprocess_options = [
        (preprocessor.normalize_image, {
            'original_minval': 0,
            'original_maxval': 255,
            'target_minval': 0,
            'target_maxval': 1
        }),
        (preprocessor.ssd_random_crop, {})]
    self._testPreprocessorCache(preprocess_options,
                                test_boxes=True,
                                test_masks=False,
                                test_keypoints=False)

  def testSSDRandomCrop(self):
    def graph_fn():
      preprocessing_options = [
          (preprocessor.normalize_image, {
              'original_minval': 0,
              'original_maxval': 255,
              'target_minval': 0,
              'target_maxval': 1
          }),
          (preprocessor.ssd_random_crop, {})]
      images = self.createTestImages()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
      }
      distorted_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                      preprocessing_options)
      distorted_images = distorted_tensor_dict[fields.InputDataFields.image]
      distorted_boxes = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]

      images_rank = tf.rank(images)
      distorted_images_rank = tf.rank(distorted_images)
      boxes_rank = tf.rank(boxes)
      distorted_boxes_rank = tf.rank(distorted_boxes)
      return [boxes_rank, distorted_boxes_rank, images_rank,
              distorted_images_rank]
    (boxes_rank_, distorted_boxes_rank_, images_rank_,
     distorted_images_rank_) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(boxes_rank_, distorted_boxes_rank_)
    self.assertAllEqual(images_rank_, distorted_images_rank_)

  def testSSDRandomCropWithMultiClassScores(self):
    def graph_fn():
      preprocessing_options = [(preprocessor.normalize_image, {
          'original_minval': 0,
          'original_maxval': 255,
          'target_minval': 0,
          'target_maxval': 1
      }), (preprocessor.ssd_random_crop, {})]
      images = self.createTestImages()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      multiclass_scores = self.createTestMultiClassScores()

      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.multiclass_scores: multiclass_scores,
          fields.InputDataFields.groundtruth_weights: weights,
      }
      preprocessor_arg_map = preprocessor.get_default_func_arg_map(
          include_multiclass_scores=True)
      distorted_tensor_dict = preprocessor.preprocess(
          tensor_dict, preprocessing_options, func_arg_map=preprocessor_arg_map)
      distorted_images = distorted_tensor_dict[fields.InputDataFields.image]
      distorted_boxes = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]
      distorted_multiclass_scores = distorted_tensor_dict[
          fields.InputDataFields.multiclass_scores]

      images_rank = tf.rank(images)
      distorted_images_rank = tf.rank(distorted_images)
      boxes_rank = tf.rank(boxes)
      distorted_boxes_rank = tf.rank(distorted_boxes)
      multiclass_scores_rank = tf.rank(multiclass_scores)
      distorted_multiclass_scores_rank = tf.rank(distorted_multiclass_scores)
      return [
          boxes_rank, distorted_boxes, distorted_boxes_rank, images_rank,
          distorted_images_rank, multiclass_scores_rank,
          distorted_multiclass_scores, distorted_multiclass_scores_rank
      ]

    (boxes_rank_, distorted_boxes_, distorted_boxes_rank_, images_rank_,
     distorted_images_rank_, multiclass_scores_rank_,
     distorted_multiclass_scores_,
     distorted_multiclass_scores_rank_) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(boxes_rank_, distorted_boxes_rank_)
    self.assertAllEqual(images_rank_, distorted_images_rank_)
    self.assertAllEqual(multiclass_scores_rank_,
                        distorted_multiclass_scores_rank_)
    self.assertAllEqual(distorted_boxes_.shape[0],
                        distorted_multiclass_scores_.shape[0])

  def testSSDRandomCropPad(self):
    def graph_fn():
      images = self.createTestImages()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      preprocessing_options = [
          (preprocessor.normalize_image, {
              'original_minval': 0,
              'original_maxval': 255,
              'target_minval': 0,
              'target_maxval': 1
          }),
          (preprocessor.ssd_random_crop_pad, {})]
      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights,
      }
      distorted_tensor_dict = preprocessor.preprocess(tensor_dict,
                                                      preprocessing_options)
      distorted_images = distorted_tensor_dict[fields.InputDataFields.image]
      distorted_boxes = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]

      images_rank = tf.rank(images)
      distorted_images_rank = tf.rank(distorted_images)
      boxes_rank = tf.rank(boxes)
      distorted_boxes_rank = tf.rank(distorted_boxes)
      return [
          boxes_rank, distorted_boxes_rank, images_rank, distorted_images_rank
      ]
    (boxes_rank_, distorted_boxes_rank_, images_rank_,
     distorted_images_rank_) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(boxes_rank_, distorted_boxes_rank_)
    self.assertAllEqual(images_rank_, distorted_images_rank_)

  def testSSDRandomCropFixedAspectRatioWithCache(self):
    preprocess_options = [
        (preprocessor.normalize_image, {
            'original_minval': 0,
            'original_maxval': 255,
            'target_minval': 0,
            'target_maxval': 1
        }),
        (preprocessor.ssd_random_crop_fixed_aspect_ratio, {})]
    self._testPreprocessorCache(preprocess_options,
                                test_boxes=True,
                                test_masks=False,
                                test_keypoints=False)

  def _testSSDRandomCropFixedAspectRatio(self,
                                         include_multiclass_scores,
                                         include_instance_masks,
                                         include_keypoints):
    def graph_fn():
      images = self.createTestImages()
      boxes = self.createTestBoxes()
      labels = self.createTestLabels()
      weights = self.createTestGroundtruthWeights()
      preprocessing_options = [(preprocessor.normalize_image, {
          'original_minval': 0,
          'original_maxval': 255,
          'target_minval': 0,
          'target_maxval': 1
      }), (preprocessor.ssd_random_crop_fixed_aspect_ratio, {})]
      tensor_dict = {
          fields.InputDataFields.image: images,
          fields.InputDataFields.groundtruth_boxes: boxes,
          fields.InputDataFields.groundtruth_classes: labels,
          fields.InputDataFields.groundtruth_weights: weights
      }
      if include_multiclass_scores:
        multiclass_scores = self.createTestMultiClassScores()
        tensor_dict[fields.InputDataFields.multiclass_scores] = (
            multiclass_scores)
      if include_instance_masks:
        masks = self.createTestMasks()
        tensor_dict[fields.InputDataFields.groundtruth_instance_masks] = masks
      if include_keypoints:
        keypoints, _ = self.createTestKeypoints()
        tensor_dict[fields.InputDataFields.groundtruth_keypoints] = keypoints

      preprocessor_arg_map = preprocessor.get_default_func_arg_map(
          include_multiclass_scores=include_multiclass_scores,
          include_instance_masks=include_instance_masks,
          include_keypoints=include_keypoints)
      distorted_tensor_dict = preprocessor.preprocess(
          tensor_dict, preprocessing_options, func_arg_map=preprocessor_arg_map)
      distorted_images = distorted_tensor_dict[fields.InputDataFields.image]
      distorted_boxes = distorted_tensor_dict[
          fields.InputDataFields.groundtruth_boxes]
      images_rank = tf.rank(images)
      distorted_images_rank = tf.rank(distorted_images)
      boxes_rank = tf.rank(boxes)
      distorted_boxes_rank = tf.rank(distorted_boxes)
      return [boxes_rank, distorted_boxes_rank, images_rank,
              distorted_images_rank]

    (boxes_rank_, distorted_boxes_rank_, images_rank_,
     distorted_images_rank_) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(boxes_rank_, distorted_boxes_rank_)
    self.assertAllEqual(images_rank_, distorted_images_rank_)

  def testSSDRandomCropFixedAspectRatio(self):
    self._testSSDRandomCropFixedAspectRatio(include_multiclass_scores=False,
                                            include_instance_masks=False,
                                            include_keypoints=False)

  def testSSDRandomCropFixedAspectRatioWithMultiClassScores(self):
    self._testSSDRandomCropFixedAspectRatio(include_multiclass_scores=True,
                                            include_instance_masks=False,
                                            include_keypoints=False)

  def testSSDRandomCropFixedAspectRatioWithMasksAndKeypoints(self):
    self._testSSDRandomCropFixedAspectRatio(include_multiclass_scores=False,
                                            include_instance_masks=True,
                                            include_keypoints=True)

  def testSSDRandomCropFixedAspectRatioWithLabelScoresMasksAndKeypoints(self):
    self._testSSDRandomCropFixedAspectRatio(include_multiclass_scores=False,
                                            include_instance_masks=True,
                                            include_keypoints=True)

  def testConvertClassLogitsToSoftmax(self):
    def graph_fn():
      multiclass_scores = tf.constant(
          [[1.0, 0.0], [0.5, 0.5], [1000, 1]], dtype=tf.float32)
      temperature = 2.0

      converted_multiclass_scores = (
          preprocessor.convert_class_logits_to_softmax(
              multiclass_scores=multiclass_scores, temperature=temperature))
      return converted_multiclass_scores
    converted_multiclass_scores_ = self.execute_cpu(graph_fn, [])
    expected_converted_multiclass_scores = [[0.62245935, 0.37754068],
                                            [0.5, 0.5],
                                            [1, 0]]
    self.assertAllClose(converted_multiclass_scores_,
                        expected_converted_multiclass_scores)

  @parameterized.named_parameters(
      ('scale_1', 1.0),
      ('scale_1.5', 1.5),
      ('scale_0.5', 0.5)
  )
  def test_square_crop_by_scale(self, scale):
    def graph_fn():
      image = np.random.randn(256, 256, 1)

      masks = tf.constant(image[:, :, 0].reshape(1, 256, 256))
      image = tf.constant(image)
      keypoints = tf.constant([[[0.25, 0.25], [0.75, 0.75]]])

      boxes = tf.constant([[0.25, .25, .75, .75]])
      labels = tf.constant([[1]])
      label_weights = tf.constant([[1.]])

      (new_image, new_boxes, _, _, new_masks,
       new_keypoints) = preprocessor.random_square_crop_by_scale(
           image,
           boxes,
           labels,
           label_weights,
           masks=masks,
           keypoints=keypoints,
           max_border=256,
           scale_min=scale,
           scale_max=scale)
      return new_image, new_boxes, new_masks, new_keypoints
    image, boxes, masks, keypoints = self.execute_cpu(graph_fn, [])
    ymin, xmin, ymax, xmax = boxes[0]
    self.assertAlmostEqual(ymax - ymin, 0.5 / scale)
    self.assertAlmostEqual(xmax - xmin, 0.5 / scale)

    k1 = keypoints[0, 0]
    k2 = keypoints[0, 1]
    self.assertAlmostEqual(k2[0] - k1[0], 0.5 / scale)
    self.assertAlmostEqual(k2[1] - k1[1], 0.5 / scale)

    size = max(image.shape)
    self.assertAlmostEqual(scale * 256.0, size)

    self.assertAllClose(image[:, :, 0], masks[0, :, :])

  @parameterized.named_parameters(('scale_0_1', 0.1), ('scale_1_0', 1.0),
                                  ('scale_2_0', 2.0))
  def test_random_scale_crop_and_pad_to_square(self, scale):

    def graph_fn():
      image = np.random.randn(512, 256, 1)
      box_centers = [0.25, 0.5, 0.75]
      box_size = 0.1
      box_corners = []
      box_labels = []
      box_label_weights = []
      keypoints = []
      masks = []
      for center_y in box_centers:
        for center_x in box_centers:
          box_corners.append(
              [center_y - box_size / 2.0, center_x - box_size / 2.0,
               center_y + box_size / 2.0, center_x + box_size / 2.0])
          box_labels.append([1])
          box_label_weights.append([1.])
          keypoints.append(
              [[center_y - box_size / 2.0, center_x - box_size / 2.0],
               [center_y + box_size / 2.0, center_x + box_size / 2.0]])
          masks.append(image[:, :, 0].reshape(512, 256))

      image = tf.constant(image)
      boxes = tf.constant(box_corners)
      labels = tf.constant(box_labels)
      label_weights = tf.constant(box_label_weights)
      keypoints = tf.constant(keypoints)
      masks = tf.constant(np.stack(masks))

      (new_image, new_boxes, _, _, new_masks,
       new_keypoints) = preprocessor.random_scale_crop_and_pad_to_square(
           image,
           boxes,
           labels,
           label_weights,
           masks=masks,
           keypoints=keypoints,
           scale_min=scale,
           scale_max=scale,
           output_size=512)
      return new_image, new_boxes, new_masks, new_keypoints

    image, boxes, masks, keypoints = self.execute_cpu(graph_fn, [])

    # Since random_scale_crop_and_pad_to_square may prune and clip boxes,
    # we only need to find one of the boxes that was not clipped and check
    # that it matches the expected dimensions. Note, assertAlmostEqual(a, b)
    # is equivalent to round(a-b, 7) == 0.
    any_box_has_correct_size = False
    effective_scale_y = int(scale * 512) / 512.0
    effective_scale_x = int(scale * 256) / 512.0
    expected_size_y = 0.1 * effective_scale_y
    expected_size_x = 0.1 * effective_scale_x
    for box in boxes:
      ymin, xmin, ymax, xmax = box
      any_box_has_correct_size |= (
          (round(ymin, 7) != 0.0) and (round(xmin, 7) != 0.0) and
          (round(ymax, 7) != 1.0) and (round(xmax, 7) != 1.0) and
          (round((ymax - ymin) - expected_size_y, 7) == 0.0) and
          (round((xmax - xmin) - expected_size_x, 7) == 0.0))
    self.assertTrue(any_box_has_correct_size)

    # Similar to the approach above where we check for at least one box with the
    # expected dimensions, we check for at least one pair of keypoints whose
    # distance matches the expected dimensions.
    any_keypoint_pair_has_correct_dist = False
    for keypoint_pair in keypoints:
      ymin, xmin = keypoint_pair[0]
      ymax, xmax = keypoint_pair[1]
      any_keypoint_pair_has_correct_dist |= (
          (round(ymin, 7) != 0.0) and (round(xmin, 7) != 0.0) and
          (round(ymax, 7) != 1.0) and (round(xmax, 7) != 1.0) and
          (round((ymax - ymin) - expected_size_y, 7) == 0.0) and
          (round((xmax - xmin) - expected_size_x, 7) == 0.0))
    self.assertTrue(any_keypoint_pair_has_correct_dist)

    self.assertAlmostEqual(512.0, image.shape[0])
    self.assertAlmostEqual(512.0, image.shape[1])

    self.assertAllClose(image[:, :, 0],
                        masks[0, :, :])


if __name__ == '__main__':
  tf.test.main()
