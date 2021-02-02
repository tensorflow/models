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

"""Tests for object_detection.core.target_assigner."""
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.box_coders import keypoint_box_coder
from object_detection.box_coders import mean_stddev_box_coder
from object_detection.core import box_list
from object_detection.core import region_similarity_calculator
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner as targetassigner
from object_detection.matchers import argmax_matcher
from object_detection.utils import np_box_ops
from object_detection.utils import test_case
from object_detection.utils import tf_version


class TargetAssignerTest(test_case.TestCase):

  def test_assign_agnostic(self):
    def graph_fn(anchor_means, groundtruth_box_corners):
      similarity_calc = region_similarity_calculator.IouSimilarity()
      matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5,
                                             unmatched_threshold=0.5)
      box_coder = mean_stddev_box_coder.MeanStddevBoxCoder(stddev=0.1)
      target_assigner = targetassigner.TargetAssigner(
          similarity_calc, matcher, box_coder)
      anchors_boxlist = box_list.BoxList(anchor_means)
      groundtruth_boxlist = box_list.BoxList(groundtruth_box_corners)
      result = target_assigner.assign(
          anchors_boxlist, groundtruth_boxlist, unmatched_class_label=None)
      (cls_targets, cls_weights, reg_targets, reg_weights, _) = result
      return (cls_targets, cls_weights, reg_targets, reg_weights)

    anchor_means = np.array([[0.0, 0.0, 0.5, 0.5],
                             [0.5, 0.5, 1.0, 0.8],
                             [0, 0.5, .5, 1.0]], dtype=np.float32)
    groundtruth_box_corners = np.array([[0.0, 0.0, 0.5, 0.5],
                                        [0.5, 0.5, 0.9, 0.9]],
                                       dtype=np.float32)
    exp_cls_targets = [[1], [1], [0]]
    exp_cls_weights = [[1], [1], [1]]
    exp_reg_targets = [[0, 0, 0, 0],
                       [0, 0, -1, 1],
                       [0, 0, 0, 0]]
    exp_reg_weights = [1, 1, 0]

    (cls_targets_out,
     cls_weights_out, reg_targets_out, reg_weights_out) = self.execute(
         graph_fn, [anchor_means, groundtruth_box_corners])
    self.assertAllClose(cls_targets_out, exp_cls_targets)
    self.assertAllClose(cls_weights_out, exp_cls_weights)
    self.assertAllClose(reg_targets_out, exp_reg_targets)
    self.assertAllClose(reg_weights_out, exp_reg_weights)
    self.assertEqual(cls_targets_out.dtype, np.float32)
    self.assertEqual(cls_weights_out.dtype, np.float32)
    self.assertEqual(reg_targets_out.dtype, np.float32)
    self.assertEqual(reg_weights_out.dtype, np.float32)

  def test_assign_class_agnostic_with_ignored_matches(self):
    # Note: test is very similar to above. The third box matched with an IOU
    # of 0.35, which is between the matched and unmatched threshold. This means
    # That like above the expected classification targets are [1, 1, 0].
    # Unlike above, the third target is ignored and therefore expected
    # classification weights are [1, 1, 0].
    def graph_fn(anchor_means, groundtruth_box_corners):
      similarity_calc = region_similarity_calculator.IouSimilarity()
      matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5,
                                             unmatched_threshold=0.3)
      box_coder = mean_stddev_box_coder.MeanStddevBoxCoder(stddev=0.1)
      target_assigner = targetassigner.TargetAssigner(
          similarity_calc, matcher, box_coder)
      anchors_boxlist = box_list.BoxList(anchor_means)
      groundtruth_boxlist = box_list.BoxList(groundtruth_box_corners)
      result = target_assigner.assign(
          anchors_boxlist, groundtruth_boxlist, unmatched_class_label=None)
      (cls_targets, cls_weights, reg_targets, reg_weights, _) = result
      return (cls_targets, cls_weights, reg_targets, reg_weights)

    anchor_means = np.array([[0.0, 0.0, 0.5, 0.5],
                             [0.5, 0.5, 1.0, 0.8],
                             [0.0, 0.5, .9, 1.0]], dtype=np.float32)
    groundtruth_box_corners = np.array([[0.0, 0.0, 0.5, 0.5],
                                        [0.5, 0.5, 0.9, 0.9]], dtype=np.float32)
    exp_cls_targets = [[1], [1], [0]]
    exp_cls_weights = [[1], [1], [0]]
    exp_reg_targets = [[0, 0, 0, 0],
                       [0, 0, -1, 1],
                       [0, 0, 0, 0]]
    exp_reg_weights = [1, 1, 0]
    (cls_targets_out,
     cls_weights_out, reg_targets_out, reg_weights_out) = self.execute(
         graph_fn, [anchor_means, groundtruth_box_corners])
    self.assertAllClose(cls_targets_out, exp_cls_targets)
    self.assertAllClose(cls_weights_out, exp_cls_weights)
    self.assertAllClose(reg_targets_out, exp_reg_targets)
    self.assertAllClose(reg_weights_out, exp_reg_weights)
    self.assertEqual(cls_targets_out.dtype, np.float32)
    self.assertEqual(cls_weights_out.dtype, np.float32)
    self.assertEqual(reg_targets_out.dtype, np.float32)
    self.assertEqual(reg_weights_out.dtype, np.float32)

  def test_assign_agnostic_with_keypoints(self):

    def graph_fn(anchor_means, groundtruth_box_corners,
                 groundtruth_keypoints):
      similarity_calc = region_similarity_calculator.IouSimilarity()
      matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5,
                                             unmatched_threshold=0.5)
      box_coder = keypoint_box_coder.KeypointBoxCoder(
          num_keypoints=6, scale_factors=[10.0, 10.0, 5.0, 5.0])
      target_assigner = targetassigner.TargetAssigner(
          similarity_calc, matcher, box_coder)
      anchors_boxlist = box_list.BoxList(anchor_means)
      groundtruth_boxlist = box_list.BoxList(groundtruth_box_corners)
      groundtruth_boxlist.add_field(fields.BoxListFields.keypoints,
                                    groundtruth_keypoints)
      result = target_assigner.assign(
          anchors_boxlist, groundtruth_boxlist, unmatched_class_label=None)
      (cls_targets, cls_weights, reg_targets, reg_weights, _) = result
      return (cls_targets, cls_weights, reg_targets, reg_weights)

    anchor_means = np.array([[0.0, 0.0, 0.5, 0.5],
                             [0.5, 0.5, 1.0, 1.0],
                             [0.0, 0.5, .9, 1.0]], dtype=np.float32)
    groundtruth_box_corners = np.array([[0.0, 0.0, 0.5, 0.5],
                                        [0.45, 0.45, 0.95, 0.95]],
                                       dtype=np.float32)
    groundtruth_keypoints = np.array(
        [[[0.1, 0.2], [0.1, 0.3], [0.2, 0.2], [0.2, 0.2], [0.1, 0.1], [0.9, 0]],
         [[0, 0.3], [0.2, 0.4], [0.5, 0.6], [0, 0.6], [0.8, 0.2], [0.2, 0.4]]],
        dtype=np.float32)
    exp_cls_targets = [[1], [1], [0]]
    exp_cls_weights = [[1], [1], [1]]
    exp_reg_targets = [[0, 0, 0, 0, -3, -1, -3, 1, -1, -1, -1, -1, -3, -3, 13,
                        -5],
                       [-1, -1, 0, 0, -15, -9, -11, -7, -5, -3, -15, -3, 1, -11,
                        -11, -7],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    exp_reg_weights = [1, 1, 0]
    (cls_targets_out, cls_weights_out, reg_targets_out,
     reg_weights_out) = self.execute(graph_fn, [anchor_means,
                                                groundtruth_box_corners,
                                                groundtruth_keypoints])
    self.assertAllClose(cls_targets_out, exp_cls_targets)
    self.assertAllClose(cls_weights_out, exp_cls_weights)
    self.assertAllClose(reg_targets_out, exp_reg_targets)
    self.assertAllClose(reg_weights_out, exp_reg_weights)
    self.assertEqual(cls_targets_out.dtype, np.float32)
    self.assertEqual(cls_weights_out.dtype, np.float32)
    self.assertEqual(reg_targets_out.dtype, np.float32)
    self.assertEqual(reg_weights_out.dtype, np.float32)

  def test_assign_class_agnostic_with_keypoints_and_ignored_matches(self):
    # Note: test is very similar to above. The third box matched with an IOU
    # of 0.35, which is between the matched and unmatched threshold. This means
    # That like above the expected classification targets are [1, 1, 0].
    # Unlike above, the third target is ignored and therefore expected
    # classification weights are [1, 1, 0].
    def graph_fn(anchor_means, groundtruth_box_corners,
                 groundtruth_keypoints):
      similarity_calc = region_similarity_calculator.IouSimilarity()
      matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5,
                                             unmatched_threshold=0.5)
      box_coder = keypoint_box_coder.KeypointBoxCoder(
          num_keypoints=6, scale_factors=[10.0, 10.0, 5.0, 5.0])
      target_assigner = targetassigner.TargetAssigner(
          similarity_calc, matcher, box_coder)
      anchors_boxlist = box_list.BoxList(anchor_means)
      groundtruth_boxlist = box_list.BoxList(groundtruth_box_corners)
      groundtruth_boxlist.add_field(fields.BoxListFields.keypoints,
                                    groundtruth_keypoints)
      result = target_assigner.assign(
          anchors_boxlist, groundtruth_boxlist, unmatched_class_label=None)
      (cls_targets, cls_weights, reg_targets, reg_weights, _) = result
      return (cls_targets, cls_weights, reg_targets, reg_weights)

    anchor_means = np.array([[0.0, 0.0, 0.5, 0.5],
                             [0.5, 0.5, 1.0, 1.0],
                             [0.0, 0.5, .9, 1.0]], dtype=np.float32)
    groundtruth_box_corners = np.array([[0.0, 0.0, 0.5, 0.5],
                                        [0.45, 0.45, 0.95, 0.95]],
                                       dtype=np.float32)
    groundtruth_keypoints = np.array(
        [[[0.1, 0.2], [0.1, 0.3], [0.2, 0.2], [0.2, 0.2], [0.1, 0.1], [0.9, 0]],
         [[0, 0.3], [0.2, 0.4], [0.5, 0.6], [0, 0.6], [0.8, 0.2], [0.2, 0.4]]],
        dtype=np.float32)
    exp_cls_targets = [[1], [1], [0]]
    exp_cls_weights = [[1], [1], [1]]
    exp_reg_targets = [[0, 0, 0, 0, -3, -1, -3, 1, -1, -1, -1, -1, -3, -3, 13,
                        -5],
                       [-1, -1, 0, 0, -15, -9, -11, -7, -5, -3, -15, -3, 1, -11,
                        -11, -7],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    exp_reg_weights = [1, 1, 0]
    (cls_targets_out, cls_weights_out, reg_targets_out,
     reg_weights_out) = self.execute(graph_fn, [anchor_means,
                                                groundtruth_box_corners,
                                                groundtruth_keypoints])
    self.assertAllClose(cls_targets_out, exp_cls_targets)
    self.assertAllClose(cls_weights_out, exp_cls_weights)
    self.assertAllClose(reg_targets_out, exp_reg_targets)
    self.assertAllClose(reg_weights_out, exp_reg_weights)
    self.assertEqual(cls_targets_out.dtype, np.float32)
    self.assertEqual(cls_weights_out.dtype, np.float32)
    self.assertEqual(reg_targets_out.dtype, np.float32)
    self.assertEqual(reg_weights_out.dtype, np.float32)

  def test_assign_multiclass(self):

    def graph_fn(anchor_means, groundtruth_box_corners, groundtruth_labels):
      similarity_calc = region_similarity_calculator.IouSimilarity()
      matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5,
                                             unmatched_threshold=0.5)
      box_coder = mean_stddev_box_coder.MeanStddevBoxCoder(stddev=0.1)
      unmatched_class_label = tf.constant([1, 0, 0, 0, 0, 0, 0], tf.float32)
      target_assigner = targetassigner.TargetAssigner(
          similarity_calc, matcher, box_coder)

      anchors_boxlist = box_list.BoxList(anchor_means)
      groundtruth_boxlist = box_list.BoxList(groundtruth_box_corners)
      result = target_assigner.assign(
          anchors_boxlist,
          groundtruth_boxlist,
          groundtruth_labels,
          unmatched_class_label=unmatched_class_label)
      (cls_targets, cls_weights, reg_targets, reg_weights, _) = result
      return (cls_targets, cls_weights, reg_targets, reg_weights)

    anchor_means = np.array([[0.0, 0.0, 0.5, 0.5],
                             [0.5, 0.5, 1.0, 0.8],
                             [0, 0.5, .5, 1.0],
                             [.75, 0, 1.0, .25]], dtype=np.float32)
    groundtruth_box_corners = np.array([[0.0, 0.0, 0.5, 0.5],
                                        [0.5, 0.5, 0.9, 0.9],
                                        [.75, 0, .95, .27]], dtype=np.float32)
    groundtruth_labels = np.array([[0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 0],
                                   [0, 0, 0, 1, 0, 0, 0]], dtype=np.float32)

    exp_cls_targets = [[0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0],
                       [1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0]]
    exp_cls_weights = [[1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1]]
    exp_reg_targets = [[0, 0, 0, 0],
                       [0, 0, -1, 1],
                       [0, 0, 0, 0],
                       [0, 0, -.5, .2]]
    exp_reg_weights = [1, 1, 0, 1]

    (cls_targets_out,
     cls_weights_out, reg_targets_out, reg_weights_out) = self.execute(
         graph_fn, [anchor_means, groundtruth_box_corners, groundtruth_labels])
    self.assertAllClose(cls_targets_out, exp_cls_targets)
    self.assertAllClose(cls_weights_out, exp_cls_weights)
    self.assertAllClose(reg_targets_out, exp_reg_targets)
    self.assertAllClose(reg_weights_out, exp_reg_weights)
    self.assertEqual(cls_targets_out.dtype, np.float32)
    self.assertEqual(cls_weights_out.dtype, np.float32)
    self.assertEqual(reg_targets_out.dtype, np.float32)
    self.assertEqual(reg_weights_out.dtype, np.float32)

  def test_assign_multiclass_with_groundtruth_weights(self):

    def graph_fn(anchor_means, groundtruth_box_corners, groundtruth_labels,
                 groundtruth_weights):
      similarity_calc = region_similarity_calculator.IouSimilarity()
      matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5,
                                             unmatched_threshold=0.5)
      box_coder = mean_stddev_box_coder.MeanStddevBoxCoder(stddev=0.1)
      unmatched_class_label = tf.constant([1, 0, 0, 0, 0, 0, 0], tf.float32)
      target_assigner = targetassigner.TargetAssigner(
          similarity_calc, matcher, box_coder)

      anchors_boxlist = box_list.BoxList(anchor_means)
      groundtruth_boxlist = box_list.BoxList(groundtruth_box_corners)
      result = target_assigner.assign(
          anchors_boxlist,
          groundtruth_boxlist,
          groundtruth_labels,
          unmatched_class_label=unmatched_class_label,
          groundtruth_weights=groundtruth_weights)
      (_, cls_weights, _, reg_weights, _) = result
      return (cls_weights, reg_weights)

    anchor_means = np.array([[0.0, 0.0, 0.5, 0.5],
                             [0.5, 0.5, 1.0, 0.8],
                             [0, 0.5, .5, 1.0],
                             [.75, 0, 1.0, .25]], dtype=np.float32)
    groundtruth_box_corners = np.array([[0.0, 0.0, 0.5, 0.5],
                                        [0.5, 0.5, 0.9, 0.9],
                                        [.75, 0, .95, .27]], dtype=np.float32)
    groundtruth_labels = np.array([[0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 0],
                                   [0, 0, 0, 1, 0, 0, 0]], dtype=np.float32)
    groundtruth_weights = np.array([0.3, 0., 0.5], dtype=np.float32)

    # background class gets weight of 1.
    exp_cls_weights = [[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                       [0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
    exp_reg_weights = [0.3, 0., 0., 0.5]  # background class gets weight of 0.

    (cls_weights_out, reg_weights_out) = self.execute(graph_fn, [
        anchor_means, groundtruth_box_corners, groundtruth_labels,
        groundtruth_weights
    ])
    self.assertAllClose(cls_weights_out, exp_cls_weights)
    self.assertAllClose(reg_weights_out, exp_reg_weights)

  def test_assign_multidimensional_class_targets(self):

    def graph_fn(anchor_means, groundtruth_box_corners, groundtruth_labels):
      similarity_calc = region_similarity_calculator.IouSimilarity()
      matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5,
                                             unmatched_threshold=0.5)
      box_coder = mean_stddev_box_coder.MeanStddevBoxCoder(stddev=0.1)

      unmatched_class_label = tf.constant([[0, 0], [0, 0]], tf.float32)
      target_assigner = targetassigner.TargetAssigner(
          similarity_calc, matcher, box_coder)

      anchors_boxlist = box_list.BoxList(anchor_means)
      groundtruth_boxlist = box_list.BoxList(groundtruth_box_corners)
      result = target_assigner.assign(
          anchors_boxlist,
          groundtruth_boxlist,
          groundtruth_labels,
          unmatched_class_label=unmatched_class_label)
      (cls_targets, cls_weights, reg_targets, reg_weights, _) = result
      return (cls_targets, cls_weights, reg_targets, reg_weights)

    anchor_means = np.array([[0.0, 0.0, 0.5, 0.5],
                             [0.5, 0.5, 1.0, 0.8],
                             [0, 0.5, .5, 1.0],
                             [.75, 0, 1.0, .25]], dtype=np.float32)
    groundtruth_box_corners = np.array([[0.0, 0.0, 0.5, 0.5],
                                        [0.5, 0.5, 0.9, 0.9],
                                        [.75, 0, .95, .27]], dtype=np.float32)

    groundtruth_labels = np.array([[[0, 1], [1, 0]],
                                   [[1, 0], [0, 1]],
                                   [[0, 1], [1, .5]]], np.float32)

    exp_cls_targets = [[[0, 1], [1, 0]],
                       [[1, 0], [0, 1]],
                       [[0, 0], [0, 0]],
                       [[0, 1], [1, .5]]]
    exp_cls_weights = [[[1, 1], [1, 1]],
                       [[1, 1], [1, 1]],
                       [[1, 1], [1, 1]],
                       [[1, 1], [1, 1]]]
    exp_reg_targets = [[0, 0, 0, 0],
                       [0, 0, -1, 1],
                       [0, 0, 0, 0],
                       [0, 0, -.5, .2]]
    exp_reg_weights = [1, 1, 0, 1]
    (cls_targets_out,
     cls_weights_out, reg_targets_out, reg_weights_out) = self.execute(
         graph_fn, [anchor_means, groundtruth_box_corners, groundtruth_labels])
    self.assertAllClose(cls_targets_out, exp_cls_targets)
    self.assertAllClose(cls_weights_out, exp_cls_weights)
    self.assertAllClose(reg_targets_out, exp_reg_targets)
    self.assertAllClose(reg_weights_out, exp_reg_weights)
    self.assertEqual(cls_targets_out.dtype, np.float32)
    self.assertEqual(cls_weights_out.dtype, np.float32)
    self.assertEqual(reg_targets_out.dtype, np.float32)
    self.assertEqual(reg_weights_out.dtype, np.float32)

  def test_assign_empty_groundtruth(self):

    def graph_fn(anchor_means, groundtruth_box_corners, groundtruth_labels):
      similarity_calc = region_similarity_calculator.IouSimilarity()
      matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5,
                                             unmatched_threshold=0.5)
      box_coder = mean_stddev_box_coder.MeanStddevBoxCoder(stddev=0.1)
      unmatched_class_label = tf.constant([0, 0, 0], tf.float32)
      anchors_boxlist = box_list.BoxList(anchor_means)
      groundtruth_boxlist = box_list.BoxList(groundtruth_box_corners)
      target_assigner = targetassigner.TargetAssigner(
          similarity_calc, matcher, box_coder)
      result = target_assigner.assign(
          anchors_boxlist,
          groundtruth_boxlist,
          groundtruth_labels,
          unmatched_class_label=unmatched_class_label)
      (cls_targets, cls_weights, reg_targets, reg_weights, _) = result
      return (cls_targets, cls_weights, reg_targets, reg_weights)

    groundtruth_box_corners = np.zeros((0, 4), dtype=np.float32)
    groundtruth_labels = np.zeros((0, 3), dtype=np.float32)
    anchor_means = np.array([[0.0, 0.0, 0.5, 0.5],
                             [0.5, 0.5, 1.0, 0.8],
                             [0, 0.5, .5, 1.0],
                             [.75, 0, 1.0, .25]],
                            dtype=np.float32)
    exp_cls_targets = [[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]]
    exp_cls_weights = [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]
    exp_reg_targets = [[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]]
    exp_reg_weights = [0, 0, 0, 0]
    (cls_targets_out,
     cls_weights_out, reg_targets_out, reg_weights_out) = self.execute(
         graph_fn, [anchor_means, groundtruth_box_corners, groundtruth_labels])
    self.assertAllClose(cls_targets_out, exp_cls_targets)
    self.assertAllClose(cls_weights_out, exp_cls_weights)
    self.assertAllClose(reg_targets_out, exp_reg_targets)
    self.assertAllClose(reg_weights_out, exp_reg_weights)
    self.assertEqual(cls_targets_out.dtype, np.float32)
    self.assertEqual(cls_weights_out.dtype, np.float32)
    self.assertEqual(reg_targets_out.dtype, np.float32)
    self.assertEqual(reg_weights_out.dtype, np.float32)

  def test_raises_error_on_incompatible_groundtruth_boxes_and_labels(self):
    similarity_calc = region_similarity_calculator.NegSqDistSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(0.5)
    box_coder = mean_stddev_box_coder.MeanStddevBoxCoder()
    unmatched_class_label = tf.constant([1, 0, 0, 0, 0, 0, 0], tf.float32)
    target_assigner = targetassigner.TargetAssigner(
        similarity_calc, matcher, box_coder)

    prior_means = tf.constant([[0.0, 0.0, 0.5, 0.5],
                               [0.5, 0.5, 1.0, 0.8],
                               [0, 0.5, .5, 1.0],
                               [.75, 0, 1.0, .25]])
    priors = box_list.BoxList(prior_means)

    box_corners = [[0.0, 0.0, 0.5, 0.5],
                   [0.0, 0.0, 0.5, 0.8],
                   [0.5, 0.5, 0.9, 0.9],
                   [.75, 0, .95, .27]]
    boxes = box_list.BoxList(tf.constant(box_corners))

    groundtruth_labels = tf.constant([[0, 1, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0],
                                      [0, 0, 0, 1, 0, 0, 0]], tf.float32)
    with self.assertRaisesRegexp(ValueError, 'Unequal shapes'):
      target_assigner.assign(
          priors,
          boxes,
          groundtruth_labels,
          unmatched_class_label=unmatched_class_label)

  def test_raises_error_on_invalid_groundtruth_labels(self):
    similarity_calc = region_similarity_calculator.NegSqDistSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(0.5)
    box_coder = mean_stddev_box_coder.MeanStddevBoxCoder(stddev=1.0)
    unmatched_class_label = tf.constant([[0, 0], [0, 0], [0, 0]], tf.float32)
    target_assigner = targetassigner.TargetAssigner(
        similarity_calc, matcher, box_coder)

    prior_means = tf.constant([[0.0, 0.0, 0.5, 0.5]])
    priors = box_list.BoxList(prior_means)

    box_corners = [[0.0, 0.0, 0.5, 0.5],
                   [0.5, 0.5, 0.9, 0.9],
                   [.75, 0, .95, .27]]
    boxes = box_list.BoxList(tf.constant(box_corners))
    groundtruth_labels = tf.constant([[[0, 1], [1, 0]]], tf.float32)

    with self.assertRaises(ValueError):
      target_assigner.assign(
          priors,
          boxes,
          groundtruth_labels,
          unmatched_class_label=unmatched_class_label)


class BatchTargetAssignerTest(test_case.TestCase):

  def _get_target_assigner(self):
    similarity_calc = region_similarity_calculator.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5,
                                           unmatched_threshold=0.5)
    box_coder = mean_stddev_box_coder.MeanStddevBoxCoder(stddev=0.1)
    return targetassigner.TargetAssigner(similarity_calc, matcher, box_coder)

  def test_batch_assign_targets(self):

    def graph_fn(anchor_means, groundtruth_boxlist1, groundtruth_boxlist2):
      box_list1 = box_list.BoxList(groundtruth_boxlist1)
      box_list2 = box_list.BoxList(groundtruth_boxlist2)
      gt_box_batch = [box_list1, box_list2]
      gt_class_targets = [None, None]
      anchors_boxlist = box_list.BoxList(anchor_means)
      agnostic_target_assigner = self._get_target_assigner()
      (cls_targets, cls_weights, reg_targets, reg_weights,
       _) = targetassigner.batch_assign_targets(
           agnostic_target_assigner, anchors_boxlist, gt_box_batch,
           gt_class_targets)
      return (cls_targets, cls_weights, reg_targets, reg_weights)

    groundtruth_boxlist1 = np.array([[0., 0., 0.2, 0.2]], dtype=np.float32)
    groundtruth_boxlist2 = np.array([[0, 0.25123152, 1, 1],
                                     [0.015789, 0.0985, 0.55789, 0.3842]],
                                    dtype=np.float32)
    anchor_means = np.array([[0, 0, .25, .25],
                             [0, .25, 1, 1],
                             [0, .1, .5, .5],
                             [.75, .75, 1, 1]], dtype=np.float32)

    exp_cls_targets = [[[1], [0], [0], [0]],
                       [[0], [1], [1], [0]]]
    exp_cls_weights = [[[1], [1], [1], [1]],
                       [[1], [1], [1], [1]]]
    exp_reg_targets = [[[0, 0, -0.5, -0.5],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0,],
                        [0, 0, 0, 0,],],
                       [[0, 0, 0, 0,],
                        [0, 0.01231521, 0, 0],
                        [0.15789001, -0.01500003, 0.57889998, -1.15799987],
                        [0, 0, 0, 0]]]
    exp_reg_weights = [[1, 0, 0, 0],
                       [0, 1, 1, 0]]

    (cls_targets_out,
     cls_weights_out, reg_targets_out, reg_weights_out) = self.execute(
         graph_fn, [anchor_means, groundtruth_boxlist1, groundtruth_boxlist2])
    self.assertAllClose(cls_targets_out, exp_cls_targets)
    self.assertAllClose(cls_weights_out, exp_cls_weights)
    self.assertAllClose(reg_targets_out, exp_reg_targets)
    self.assertAllClose(reg_weights_out, exp_reg_weights)

  def test_batch_assign_multiclass_targets(self):

    def graph_fn(anchor_means, groundtruth_boxlist1, groundtruth_boxlist2,
                 class_targets1, class_targets2):
      box_list1 = box_list.BoxList(groundtruth_boxlist1)
      box_list2 = box_list.BoxList(groundtruth_boxlist2)
      gt_box_batch = [box_list1, box_list2]
      gt_class_targets = [class_targets1, class_targets2]
      anchors_boxlist = box_list.BoxList(anchor_means)
      multiclass_target_assigner = self._get_target_assigner()
      num_classes = 3
      unmatched_class_label = tf.constant([1] + num_classes * [0], tf.float32)
      (cls_targets, cls_weights, reg_targets, reg_weights,
       _) = targetassigner.batch_assign_targets(
           multiclass_target_assigner, anchors_boxlist, gt_box_batch,
           gt_class_targets, unmatched_class_label)
      return (cls_targets, cls_weights, reg_targets, reg_weights)

    groundtruth_boxlist1 = np.array([[0., 0., 0.2, 0.2]], dtype=np.float32)
    groundtruth_boxlist2 = np.array([[0, 0.25123152, 1, 1],
                                     [0.015789, 0.0985, 0.55789, 0.3842]],
                                    dtype=np.float32)
    class_targets1 = np.array([[0, 1, 0, 0]], dtype=np.float32)
    class_targets2 = np.array([[0, 0, 0, 1],
                               [0, 0, 1, 0]], dtype=np.float32)

    anchor_means = np.array([[0, 0, .25, .25],
                             [0, .25, 1, 1],
                             [0, .1, .5, .5],
                             [.75, .75, 1, 1]], dtype=np.float32)
    exp_cls_targets = [[[0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0]],
                       [[1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0]]]
    exp_cls_weights = [[[1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1]],
                       [[1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1]]]
    exp_reg_targets = [[[0, 0, -0.5, -0.5],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0,],
                        [0, 0, 0, 0,],],
                       [[0, 0, 0, 0,],
                        [0, 0.01231521, 0, 0],
                        [0.15789001, -0.01500003, 0.57889998, -1.15799987],
                        [0, 0, 0, 0]]]
    exp_reg_weights = [[1, 0, 0, 0],
                       [0, 1, 1, 0]]

    (cls_targets_out, cls_weights_out, reg_targets_out,
     reg_weights_out) = self.execute(graph_fn, [
         anchor_means, groundtruth_boxlist1, groundtruth_boxlist2,
         class_targets1, class_targets2
     ])
    self.assertAllClose(cls_targets_out, exp_cls_targets)
    self.assertAllClose(cls_weights_out, exp_cls_weights)
    self.assertAllClose(reg_targets_out, exp_reg_targets)
    self.assertAllClose(reg_weights_out, exp_reg_weights)

  def test_batch_assign_multiclass_targets_with_padded_groundtruth(self):

    def graph_fn(anchor_means, groundtruth_boxlist1, groundtruth_boxlist2,
                 class_targets1, class_targets2, groundtruth_weights1,
                 groundtruth_weights2):
      box_list1 = box_list.BoxList(groundtruth_boxlist1)
      box_list2 = box_list.BoxList(groundtruth_boxlist2)
      gt_box_batch = [box_list1, box_list2]
      gt_class_targets = [class_targets1, class_targets2]
      gt_weights = [groundtruth_weights1, groundtruth_weights2]
      anchors_boxlist = box_list.BoxList(anchor_means)
      multiclass_target_assigner = self._get_target_assigner()
      num_classes = 3
      unmatched_class_label = tf.constant([1] + num_classes * [0], tf.float32)
      (cls_targets, cls_weights, reg_targets, reg_weights,
       _) = targetassigner.batch_assign_targets(
           multiclass_target_assigner, anchors_boxlist, gt_box_batch,
           gt_class_targets, unmatched_class_label, gt_weights)
      return (cls_targets, cls_weights, reg_targets, reg_weights)

    groundtruth_boxlist1 = np.array([[0., 0., 0.2, 0.2],
                                     [0., 0., 0., 0.]], dtype=np.float32)
    groundtruth_weights1 = np.array([1, 0], dtype=np.float32)
    groundtruth_boxlist2 = np.array([[0, 0.25123152, 1, 1],
                                     [0.015789, 0.0985, 0.55789, 0.3842],
                                     [0, 0, 0, 0]],
                                    dtype=np.float32)
    groundtruth_weights2 = np.array([1, 1, 0], dtype=np.float32)
    class_targets1 = np.array([[0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.float32)
    class_targets2 = np.array([[0, 0, 0, 1],
                               [0, 0, 1, 0],
                               [0, 0, 0, 0]], dtype=np.float32)

    anchor_means = np.array([[0, 0, .25, .25],
                             [0, .25, 1, 1],
                             [0, .1, .5, .5],
                             [.75, .75, 1, 1]], dtype=np.float32)

    exp_cls_targets = [[[0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0]],
                       [[1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0]]]
    exp_cls_weights = [[[1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1]],
                       [[1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1]]]
    exp_reg_targets = [[[0, 0, -0.5, -0.5],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0,],
                        [0, 0, 0, 0,],],
                       [[0, 0, 0, 0,],
                        [0, 0.01231521, 0, 0],
                        [0.15789001, -0.01500003, 0.57889998, -1.15799987],
                        [0, 0, 0, 0]]]
    exp_reg_weights = [[1, 0, 0, 0],
                       [0, 1, 1, 0]]

    (cls_targets_out, cls_weights_out, reg_targets_out,
     reg_weights_out) = self.execute(graph_fn, [
         anchor_means, groundtruth_boxlist1, groundtruth_boxlist2,
         class_targets1, class_targets2, groundtruth_weights1,
         groundtruth_weights2
     ])
    self.assertAllClose(cls_targets_out, exp_cls_targets)
    self.assertAllClose(cls_weights_out, exp_cls_weights)
    self.assertAllClose(reg_targets_out, exp_reg_targets)
    self.assertAllClose(reg_weights_out, exp_reg_weights)

  def test_batch_assign_multidimensional_targets(self):

    def graph_fn(anchor_means, groundtruth_boxlist1, groundtruth_boxlist2,
                 class_targets1, class_targets2):
      box_list1 = box_list.BoxList(groundtruth_boxlist1)
      box_list2 = box_list.BoxList(groundtruth_boxlist2)
      gt_box_batch = [box_list1, box_list2]
      gt_class_targets = [class_targets1, class_targets2]
      anchors_boxlist = box_list.BoxList(anchor_means)
      multiclass_target_assigner = self._get_target_assigner()
      target_dimensions = (2, 3)
      unmatched_class_label = tf.constant(np.zeros(target_dimensions),
                                          tf.float32)
      (cls_targets, cls_weights, reg_targets, reg_weights,
       _) = targetassigner.batch_assign_targets(
           multiclass_target_assigner, anchors_boxlist, gt_box_batch,
           gt_class_targets, unmatched_class_label)
      return (cls_targets, cls_weights, reg_targets, reg_weights)

    groundtruth_boxlist1 = np.array([[0., 0., 0.2, 0.2]], dtype=np.float32)
    groundtruth_boxlist2 = np.array([[0, 0.25123152, 1, 1],
                                     [0.015789, 0.0985, 0.55789, 0.3842]],
                                    dtype=np.float32)
    class_targets1 = np.array([[[0, 1, 1],
                                [1, 1, 0]]], dtype=np.float32)
    class_targets2 = np.array([[[0, 1, 1],
                                [1, 1, 0]],
                               [[0, 0, 1],
                                [0, 0, 1]]], dtype=np.float32)

    anchor_means = np.array([[0, 0, .25, .25],
                             [0, .25, 1, 1],
                             [0, .1, .5, .5],
                             [.75, .75, 1, 1]], dtype=np.float32)

    exp_cls_targets = [[[[0., 1., 1.],
                         [1., 1., 0.]],
                        [[0., 0., 0.],
                         [0., 0., 0.]],
                        [[0., 0., 0.],
                         [0., 0., 0.]],
                        [[0., 0., 0.],
                         [0., 0., 0.]]],
                       [[[0., 0., 0.],
                         [0., 0., 0.]],
                        [[0., 1., 1.],
                         [1., 1., 0.]],
                        [[0., 0., 1.],
                         [0., 0., 1.]],
                        [[0., 0., 0.],
                         [0., 0., 0.]]]]
    exp_cls_weights = [[[[1., 1., 1.],
                         [1., 1., 1.]],
                        [[1., 1., 1.],
                         [1., 1., 1.]],
                        [[1., 1., 1.],
                         [1., 1., 1.]],
                        [[1., 1., 1.],
                         [1., 1., 1.]]],
                       [[[1., 1., 1.],
                         [1., 1., 1.]],
                        [[1., 1., 1.],
                         [1., 1., 1.]],
                        [[1., 1., 1.],
                         [1., 1., 1.]],
                        [[1., 1., 1.],
                         [1., 1., 1.]]]]
    exp_reg_targets = [[[0, 0, -0.5, -0.5],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0,],
                        [0, 0, 0, 0,],],
                       [[0, 0, 0, 0,],
                        [0, 0.01231521, 0, 0],
                        [0.15789001, -0.01500003, 0.57889998, -1.15799987],
                        [0, 0, 0, 0]]]
    exp_reg_weights = [[1, 0, 0, 0],
                       [0, 1, 1, 0]]

    (cls_targets_out, cls_weights_out, reg_targets_out,
     reg_weights_out) = self.execute(graph_fn, [
         anchor_means, groundtruth_boxlist1, groundtruth_boxlist2,
         class_targets1, class_targets2
     ])
    self.assertAllClose(cls_targets_out, exp_cls_targets)
    self.assertAllClose(cls_weights_out, exp_cls_weights)
    self.assertAllClose(reg_targets_out, exp_reg_targets)
    self.assertAllClose(reg_weights_out, exp_reg_weights)

  def test_batch_assign_empty_groundtruth(self):

    def graph_fn(anchor_means, groundtruth_box_corners, gt_class_targets):
      groundtruth_boxlist = box_list.BoxList(groundtruth_box_corners)
      gt_box_batch = [groundtruth_boxlist]
      gt_class_targets_batch = [gt_class_targets]
      anchors_boxlist = box_list.BoxList(anchor_means)

      multiclass_target_assigner = self._get_target_assigner()
      num_classes = 3
      unmatched_class_label = tf.constant([1] + num_classes * [0], tf.float32)
      (cls_targets, cls_weights, reg_targets, reg_weights,
       _) = targetassigner.batch_assign_targets(
           multiclass_target_assigner, anchors_boxlist,
           gt_box_batch, gt_class_targets_batch, unmatched_class_label)
      return (cls_targets, cls_weights, reg_targets, reg_weights)

    groundtruth_box_corners = np.zeros((0, 4), dtype=np.float32)
    anchor_means = np.array([[0, 0, .25, .25],
                             [0, .25, 1, 1]], dtype=np.float32)
    exp_cls_targets = [[[1, 0, 0, 0],
                        [1, 0, 0, 0]]]
    exp_cls_weights = [[[1, 1, 1, 1],
                        [1, 1, 1, 1]]]
    exp_reg_targets = [[[0, 0, 0, 0],
                        [0, 0, 0, 0]]]
    exp_reg_weights = [[0, 0]]
    num_classes = 3
    pad = 1
    gt_class_targets = np.zeros((0, num_classes + pad), dtype=np.float32)

    (cls_targets_out,
     cls_weights_out, reg_targets_out, reg_weights_out) = self.execute(
         graph_fn, [anchor_means, groundtruth_box_corners, gt_class_targets])
    self.assertAllClose(cls_targets_out, exp_cls_targets)
    self.assertAllClose(cls_weights_out, exp_cls_weights)
    self.assertAllClose(reg_targets_out, exp_reg_targets)
    self.assertAllClose(reg_weights_out, exp_reg_weights)


class BatchGetTargetsTest(test_case.TestCase):

  def test_scalar_targets(self):
    batch_match = np.array([[1, 0, 1],
                            [-2, -1, 1]], dtype=np.int32)
    groundtruth_tensors_list = np.array([[11, 12], [13, 14]], dtype=np.int32)
    groundtruth_weights_list = np.array([[1.0, 1.0], [1.0, 0.5]],
                                        dtype=np.float32)
    unmatched_value = np.array(99, dtype=np.int32)
    unmatched_weight = np.array(0.0, dtype=np.float32)

    def graph_fn(batch_match, groundtruth_tensors_list,
                 groundtruth_weights_list, unmatched_value, unmatched_weight):
      targets, weights = targetassigner.batch_get_targets(
          batch_match, tf.unstack(groundtruth_tensors_list),
          tf.unstack(groundtruth_weights_list),
          unmatched_value, unmatched_weight)
      return (targets, weights)

    (targets_np, weights_np) = self.execute(graph_fn, [
        batch_match, groundtruth_tensors_list, groundtruth_weights_list,
        unmatched_value, unmatched_weight
    ])
    self.assertAllEqual([[12, 11, 12],
                         [99, 99, 14]], targets_np)
    self.assertAllClose([[1.0, 1.0, 1.0],
                         [0.0, 0.0, 0.5]], weights_np)

  def test_1d_targets(self):
    batch_match = np.array([[1, 0, 1],
                            [-2, -1, 1]], dtype=np.int32)
    groundtruth_tensors_list = np.array([[[11, 12], [12, 13]],
                                         [[13, 14], [14, 15]]],
                                        dtype=np.float32)
    groundtruth_weights_list = np.array([[1.0, 1.0], [1.0, 0.5]],
                                        dtype=np.float32)
    unmatched_value = np.array([99, 99], dtype=np.float32)
    unmatched_weight = np.array(0.0, dtype=np.float32)

    def graph_fn(batch_match, groundtruth_tensors_list,
                 groundtruth_weights_list, unmatched_value, unmatched_weight):
      targets, weights = targetassigner.batch_get_targets(
          batch_match, tf.unstack(groundtruth_tensors_list),
          tf.unstack(groundtruth_weights_list),
          unmatched_value, unmatched_weight)
      return (targets, weights)

    (targets_np, weights_np) = self.execute(graph_fn, [
        batch_match, groundtruth_tensors_list, groundtruth_weights_list,
        unmatched_value, unmatched_weight
    ])
    self.assertAllClose([[[12, 13], [11, 12], [12, 13]],
                         [[99, 99], [99, 99], [14, 15]]], targets_np)
    self.assertAllClose([[1.0, 1.0, 1.0],
                         [0.0, 0.0, 0.5]], weights_np)


class BatchTargetAssignConfidencesTest(test_case.TestCase):

  def _get_target_assigner(self):
    similarity_calc = region_similarity_calculator.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5,
                                           unmatched_threshold=0.5)
    box_coder = mean_stddev_box_coder.MeanStddevBoxCoder(stddev=0.1)
    return targetassigner.TargetAssigner(similarity_calc, matcher, box_coder)

  def test_batch_assign_empty_groundtruth(self):

    def graph_fn(anchor_means, groundtruth_box_corners, gt_class_confidences):
      groundtruth_boxlist = box_list.BoxList(groundtruth_box_corners)
      gt_box_batch = [groundtruth_boxlist]
      gt_class_confidences_batch = [gt_class_confidences]
      anchors_boxlist = box_list.BoxList(anchor_means)

      num_classes = 3
      implicit_class_weight = 0.5
      unmatched_class_label = tf.constant([1] + num_classes * [0], tf.float32)
      multiclass_target_assigner = self._get_target_assigner()
      (cls_targets, cls_weights, reg_targets, reg_weights,
       _) = targetassigner.batch_assign_confidences(
           multiclass_target_assigner,
           anchors_boxlist,
           gt_box_batch,
           gt_class_confidences_batch,
           unmatched_class_label=unmatched_class_label,
           include_background_class=True,
           implicit_class_weight=implicit_class_weight)
      return (cls_targets, cls_weights, reg_targets, reg_weights)

    groundtruth_box_corners = np.zeros((0, 4), dtype=np.float32)
    anchor_means = np.array([[0, 0, .25, .25],
                             [0, .25, 1, 1]], dtype=np.float32)
    num_classes = 3
    pad = 1
    gt_class_confidences = np.zeros((0, num_classes + pad), dtype=np.float32)

    exp_cls_targets = [[[1, 0, 0, 0],
                        [1, 0, 0, 0]]]
    exp_cls_weights = [[[0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5]]]
    exp_reg_targets = [[[0, 0, 0, 0],
                        [0, 0, 0, 0]]]
    exp_reg_weights = [[0, 0]]

    (cls_targets_out,
     cls_weights_out, reg_targets_out, reg_weights_out) = self.execute(
         graph_fn,
         [anchor_means, groundtruth_box_corners, gt_class_confidences])
    self.assertAllClose(cls_targets_out, exp_cls_targets)
    self.assertAllClose(cls_weights_out, exp_cls_weights)
    self.assertAllClose(reg_targets_out, exp_reg_targets)
    self.assertAllClose(reg_weights_out, exp_reg_weights)

  def test_batch_assign_confidences_agnostic(self):

    def graph_fn(anchor_means, groundtruth_boxlist1, groundtruth_boxlist2):
      box_list1 = box_list.BoxList(groundtruth_boxlist1)
      box_list2 = box_list.BoxList(groundtruth_boxlist2)
      gt_box_batch = [box_list1, box_list2]
      gt_class_confidences_batch = [None, None]
      anchors_boxlist = box_list.BoxList(anchor_means)
      agnostic_target_assigner = self._get_target_assigner()
      implicit_class_weight = 0.5
      (cls_targets, cls_weights, reg_targets, reg_weights,
       _) = targetassigner.batch_assign_confidences(
           agnostic_target_assigner,
           anchors_boxlist,
           gt_box_batch,
           gt_class_confidences_batch,
           include_background_class=False,
           implicit_class_weight=implicit_class_weight)
      return (cls_targets, cls_weights, reg_targets, reg_weights)

    groundtruth_boxlist1 = np.array([[0., 0., 0.2, 0.2]], dtype=np.float32)
    groundtruth_boxlist2 = np.array([[0, 0.25123152, 1, 1],
                                     [0.015789, 0.0985, 0.55789, 0.3842]],
                                    dtype=np.float32)
    anchor_means = np.array([[0, 0, .25, .25],
                             [0, .25, 1, 1],
                             [0, .1, .5, .5],
                             [.75, .75, 1, 1]], dtype=np.float32)

    exp_cls_targets = [[[1], [0], [0], [0]],
                       [[0], [1], [1], [0]]]
    exp_cls_weights = [[[1], [0.5], [0.5], [0.5]],
                       [[0.5], [1], [1], [0.5]]]
    exp_reg_targets = [[[0, 0, -0.5, -0.5],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0,],
                        [0, 0, 0, 0,],],
                       [[0, 0, 0, 0,],
                        [0, 0.01231521, 0, 0],
                        [0.15789001, -0.01500003, 0.57889998, -1.15799987],
                        [0, 0, 0, 0]]]
    exp_reg_weights = [[1, 0, 0, 0],
                       [0, 1, 1, 0]]

    (cls_targets_out,
     cls_weights_out, reg_targets_out, reg_weights_out) = self.execute(
         graph_fn, [anchor_means, groundtruth_boxlist1, groundtruth_boxlist2])
    self.assertAllClose(cls_targets_out, exp_cls_targets)
    self.assertAllClose(cls_weights_out, exp_cls_weights)
    self.assertAllClose(reg_targets_out, exp_reg_targets)
    self.assertAllClose(reg_weights_out, exp_reg_weights)

  def test_batch_assign_confidences_multiclass(self):

    def graph_fn(anchor_means, groundtruth_boxlist1, groundtruth_boxlist2,
                 class_targets1, class_targets2):
      box_list1 = box_list.BoxList(groundtruth_boxlist1)
      box_list2 = box_list.BoxList(groundtruth_boxlist2)
      gt_box_batch = [box_list1, box_list2]
      gt_class_confidences_batch = [class_targets1, class_targets2]
      anchors_boxlist = box_list.BoxList(anchor_means)
      multiclass_target_assigner = self._get_target_assigner()
      num_classes = 3
      implicit_class_weight = 0.5
      unmatched_class_label = tf.constant([1] + num_classes * [0], tf.float32)
      (cls_targets, cls_weights, reg_targets, reg_weights,
       _) = targetassigner.batch_assign_confidences(
           multiclass_target_assigner,
           anchors_boxlist,
           gt_box_batch,
           gt_class_confidences_batch,
           unmatched_class_label=unmatched_class_label,
           include_background_class=True,
           implicit_class_weight=implicit_class_weight)
      return (cls_targets, cls_weights, reg_targets, reg_weights)

    groundtruth_boxlist1 = np.array([[0., 0., 0.2, 0.2]], dtype=np.float32)
    groundtruth_boxlist2 = np.array([[0, 0.25123152, 1, 1],
                                     [0.015789, 0.0985, 0.55789, 0.3842]],
                                    dtype=np.float32)
    class_targets1 = np.array([[0, 1, 0, 0]], dtype=np.float32)
    class_targets2 = np.array([[0, 0, 0, 1],
                               [0, 0, -1, 0]], dtype=np.float32)

    anchor_means = np.array([[0, 0, .25, .25],
                             [0, .25, 1, 1],
                             [0, .1, .5, .5],
                             [.75, .75, 1, 1]], dtype=np.float32)
    exp_cls_targets = [[[0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0]],
                       [[1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0]]]
    exp_cls_weights = [[[1, 1, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5]],
                       [[0.5, 0.5, 0.5, 0.5],
                        [1, 0.5, 0.5, 1],
                        [0.5, 0.5, 1, 0.5],
                        [0.5, 0.5, 0.5, 0.5]]]
    exp_reg_targets = [[[0, 0, -0.5, -0.5],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0,],
                        [0, 0, 0, 0,],],
                       [[0, 0, 0, 0,],
                        [0, 0.01231521, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]]]
    exp_reg_weights = [[1, 0, 0, 0],
                       [0, 1, 0, 0]]

    (cls_targets_out, cls_weights_out, reg_targets_out,
     reg_weights_out) = self.execute(graph_fn, [
         anchor_means, groundtruth_boxlist1, groundtruth_boxlist2,
         class_targets1, class_targets2
     ])
    self.assertAllClose(cls_targets_out, exp_cls_targets)
    self.assertAllClose(cls_weights_out, exp_cls_weights)
    self.assertAllClose(reg_targets_out, exp_reg_targets)
    self.assertAllClose(reg_weights_out, exp_reg_weights)

  def test_batch_assign_confidences_multiclass_with_padded_groundtruth(self):

    def graph_fn(anchor_means, groundtruth_boxlist1, groundtruth_boxlist2,
                 class_targets1, class_targets2, groundtruth_weights1,
                 groundtruth_weights2):
      box_list1 = box_list.BoxList(groundtruth_boxlist1)
      box_list2 = box_list.BoxList(groundtruth_boxlist2)
      gt_box_batch = [box_list1, box_list2]
      gt_class_confidences_batch = [class_targets1, class_targets2]
      gt_weights = [groundtruth_weights1, groundtruth_weights2]
      anchors_boxlist = box_list.BoxList(anchor_means)
      multiclass_target_assigner = self._get_target_assigner()
      num_classes = 3
      unmatched_class_label = tf.constant([1] + num_classes * [0], tf.float32)
      implicit_class_weight = 0.5
      (cls_targets, cls_weights, reg_targets, reg_weights,
       _) = targetassigner.batch_assign_confidences(
           multiclass_target_assigner,
           anchors_boxlist,
           gt_box_batch,
           gt_class_confidences_batch,
           gt_weights,
           unmatched_class_label=unmatched_class_label,
           include_background_class=True,
           implicit_class_weight=implicit_class_weight)

      return (cls_targets, cls_weights, reg_targets, reg_weights)

    groundtruth_boxlist1 = np.array([[0., 0., 0.2, 0.2],
                                     [0., 0., 0., 0.]], dtype=np.float32)
    groundtruth_weights1 = np.array([1, 0], dtype=np.float32)
    groundtruth_boxlist2 = np.array([[0, 0.25123152, 1, 1],
                                     [0.015789, 0.0985, 0.55789, 0.3842],
                                     [0, 0, 0, 0]],
                                    dtype=np.float32)
    groundtruth_weights2 = np.array([1, 1, 0], dtype=np.float32)
    class_targets1 = np.array([[0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.float32)
    class_targets2 = np.array([[0, 0, 0, 1],
                               [0, 0, -1, 0],
                               [0, 0, 0, 0]], dtype=np.float32)
    anchor_means = np.array([[0, 0, .25, .25],
                             [0, .25, 1, 1],
                             [0, .1, .5, .5],
                             [.75, .75, 1, 1]], dtype=np.float32)

    exp_cls_targets = [[[0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0]],
                       [[1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0]]]
    exp_cls_weights = [[[1, 1, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5]],
                       [[0.5, 0.5, 0.5, 0.5],
                        [1, 0.5, 0.5, 1],
                        [0.5, 0.5, 1, 0.5],
                        [0.5, 0.5, 0.5, 0.5]]]
    exp_reg_targets = [[[0, 0, -0.5, -0.5],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0,],
                        [0, 0, 0, 0,],],
                       [[0, 0, 0, 0,],
                        [0, 0.01231521, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]]]
    exp_reg_weights = [[1, 0, 0, 0],
                       [0, 1, 0, 0]]

    (cls_targets_out, cls_weights_out, reg_targets_out,
     reg_weights_out) = self.execute(graph_fn, [
         anchor_means, groundtruth_boxlist1, groundtruth_boxlist2,
         class_targets1, class_targets2, groundtruth_weights1,
         groundtruth_weights2
     ])
    self.assertAllClose(cls_targets_out, exp_cls_targets)
    self.assertAllClose(cls_weights_out, exp_cls_weights)
    self.assertAllClose(reg_targets_out, exp_reg_targets)
    self.assertAllClose(reg_weights_out, exp_reg_weights)

  def test_batch_assign_confidences_multidimensional(self):

    def graph_fn(anchor_means, groundtruth_boxlist1, groundtruth_boxlist2,
                 class_targets1, class_targets2):
      box_list1 = box_list.BoxList(groundtruth_boxlist1)
      box_list2 = box_list.BoxList(groundtruth_boxlist2)
      gt_box_batch = [box_list1, box_list2]
      gt_class_confidences_batch = [class_targets1, class_targets2]
      anchors_boxlist = box_list.BoxList(anchor_means)
      multiclass_target_assigner = self._get_target_assigner()
      target_dimensions = (2, 3)
      unmatched_class_label = tf.constant(np.zeros(target_dimensions),
                                          tf.float32)
      implicit_class_weight = 0.5
      (cls_targets, cls_weights, reg_targets, reg_weights,
       _) = targetassigner.batch_assign_confidences(
           multiclass_target_assigner,
           anchors_boxlist,
           gt_box_batch,
           gt_class_confidences_batch,
           unmatched_class_label=unmatched_class_label,
           include_background_class=True,
           implicit_class_weight=implicit_class_weight)
      return (cls_targets, cls_weights, reg_targets, reg_weights)

    groundtruth_boxlist1 = np.array([[0., 0., 0.2, 0.2]], dtype=np.float32)
    groundtruth_boxlist2 = np.array([[0, 0.25123152, 1, 1],
                                     [0.015789, 0.0985, 0.55789, 0.3842]],
                                    dtype=np.float32)
    class_targets1 = np.array([[0, 1, 0, 0]], dtype=np.float32)
    class_targets2 = np.array([[0, 0, 0, 1],
                               [0, 0, 1, 0]], dtype=np.float32)
    class_targets1 = np.array([[[0, 1, 1],
                                [1, 1, 0]]], dtype=np.float32)
    class_targets2 = np.array([[[0, 1, 1],
                                [1, 1, 0]],
                               [[0, 0, 1],
                                [0, 0, 1]]], dtype=np.float32)

    anchor_means = np.array([[0, 0, .25, .25],
                             [0, .25, 1, 1],
                             [0, .1, .5, .5],
                             [.75, .75, 1, 1]], dtype=np.float32)

    with self.assertRaises(ValueError):
      _, _, _, _ = self.execute(graph_fn, [
          anchor_means, groundtruth_boxlist1, groundtruth_boxlist2,
          class_targets1, class_targets2
      ])


class CreateTargetAssignerTest(test_case.TestCase):

  def test_create_target_assigner(self):
    """Tests that named constructor gives working target assigners.

    TODO(rathodv): Make this test more general.
    """
    corners = [[0.0, 0.0, 1.0, 1.0]]
    groundtruth = box_list.BoxList(tf.constant(corners))

    priors = box_list.BoxList(tf.constant(corners))
    if tf_version.is_tf1():
      multibox_ta = (targetassigner
                     .create_target_assigner('Multibox', stage='proposal'))
      multibox_ta.assign(priors, groundtruth)
    # No tests on output, as that may vary arbitrarily as new target assigners
    # are added. As long as it is constructed correctly and runs without errors,
    # tests on the individual assigners cover correctness of the assignments.

    anchors = box_list.BoxList(tf.constant(corners))
    faster_rcnn_proposals_ta = (targetassigner
                                .create_target_assigner('FasterRCNN',
                                                        stage='proposal'))
    faster_rcnn_proposals_ta.assign(anchors, groundtruth)

    fast_rcnn_ta = (targetassigner
                    .create_target_assigner('FastRCNN'))
    fast_rcnn_ta.assign(anchors, groundtruth)

    faster_rcnn_detection_ta = (targetassigner
                                .create_target_assigner('FasterRCNN',
                                                        stage='detection'))
    faster_rcnn_detection_ta.assign(anchors, groundtruth)

    with self.assertRaises(ValueError):
      targetassigner.create_target_assigner('InvalidDetector',
                                            stage='invalid_stage')


def _array_argmax(array):
  return np.unravel_index(np.argmax(array), array.shape)


class CenterNetCenterHeatmapTargetAssignerTest(test_case.TestCase):

  def setUp(self):
    super(CenterNetCenterHeatmapTargetAssignerTest, self).setUp()

    self._box_center = [0.0, 0.0, 1.0, 1.0]
    self._box_center_small = [0.25, 0.25, 0.75, 0.75]
    self._box_lower_left = [0.5, 0.0, 1.0, 0.5]
    self._box_center_offset = [0.1, 0.05, 1.0, 1.0]
    self._box_odd_coordinates = [0.1625, 0.2125, 0.5625, 0.9625]

  def test_center_location(self):
    """Test that the centers are at the correct location."""
    def graph_fn():
      box_batch = [tf.constant([self._box_center, self._box_lower_left])]
      classes = [
          tf.one_hot([0, 1], depth=4),
      ]
      assigner = targetassigner.CenterNetCenterHeatmapTargetAssigner(4)
      targets = assigner.assign_center_targets_from_boxes(80, 80, box_batch,
                                                          classes)
      return targets
    targets = self.execute(graph_fn, [])
    self.assertEqual((10, 10), _array_argmax(targets[0, :, :, 0]))
    self.assertAlmostEqual(1.0, targets[0, 10, 10, 0])
    self.assertEqual((15, 5), _array_argmax(targets[0, :, :, 1]))
    self.assertAlmostEqual(1.0, targets[0, 15, 5, 1])

  def test_center_batch_shape(self):
    """Test that the shape of the target for a batch is correct."""
    def graph_fn():
      box_batch = [
          tf.constant([self._box_center, self._box_lower_left]),
          tf.constant([self._box_center]),
          tf.constant([self._box_center_small]),
      ]
      classes = [
          tf.one_hot([0, 1], depth=4),
          tf.one_hot([2], depth=4),
          tf.one_hot([3], depth=4),
      ]
      assigner = targetassigner.CenterNetCenterHeatmapTargetAssigner(4)
      targets = assigner.assign_center_targets_from_boxes(80, 80, box_batch,
                                                          classes)
      return targets
    targets = self.execute(graph_fn, [])
    self.assertEqual((3, 20, 20, 4), targets.shape)

  def test_center_overlap_maximum(self):
    """Test that when boxes overlap we, are computing the maximum."""
    def graph_fn():
      box_batch = [
          tf.constant([
              self._box_center, self._box_center_offset, self._box_center,
              self._box_center_offset
          ])
      ]
      classes = [
          tf.one_hot([0, 0, 1, 2], depth=4),
      ]

      assigner = targetassigner.CenterNetCenterHeatmapTargetAssigner(4)
      targets = assigner.assign_center_targets_from_boxes(80, 80, box_batch,
                                                          classes)
      return targets
    targets = self.execute(graph_fn, [])
    class0_targets = targets[0, :, :, 0]
    class1_targets = targets[0, :, :, 1]
    class2_targets = targets[0, :, :, 2]
    np.testing.assert_allclose(class0_targets,
                               np.maximum(class1_targets, class2_targets))

  def test_size_blur(self):
    """Test that the heatmap of a larger box is more blurred."""
    def graph_fn():
      box_batch = [tf.constant([self._box_center, self._box_center_small])]

      classes = [
          tf.one_hot([0, 1], depth=4),
      ]
      assigner = targetassigner.CenterNetCenterHeatmapTargetAssigner(4)
      targets = assigner.assign_center_targets_from_boxes(80, 80, box_batch,
                                                          classes)
      return targets
    targets = self.execute(graph_fn, [])
    self.assertGreater(
        np.count_nonzero(targets[:, :, :, 0]),
        np.count_nonzero(targets[:, :, :, 1]))

  def test_weights(self):
    """Test that the weights correctly ignore ground truth."""
    def graph1_fn():
      box_batch = [
          tf.constant([self._box_center, self._box_lower_left]),
          tf.constant([self._box_center]),
          tf.constant([self._box_center_small]),
      ]
      classes = [
          tf.one_hot([0, 1], depth=4),
          tf.one_hot([2], depth=4),
          tf.one_hot([3], depth=4),
      ]
      assigner = targetassigner.CenterNetCenterHeatmapTargetAssigner(4)
      targets = assigner.assign_center_targets_from_boxes(80, 80, box_batch,
                                                          classes)
      return targets

    targets = self.execute(graph1_fn, [])
    self.assertAlmostEqual(1.0, targets[0, :, :, 0].max())
    self.assertAlmostEqual(1.0, targets[0, :, :, 1].max())
    self.assertAlmostEqual(1.0, targets[1, :, :, 2].max())
    self.assertAlmostEqual(1.0, targets[2, :, :, 3].max())
    self.assertAlmostEqual(0.0, targets[0, :, :, [2, 3]].max())
    self.assertAlmostEqual(0.0, targets[1, :, :, [0, 1, 3]].max())
    self.assertAlmostEqual(0.0, targets[2, :, :, :3].max())

    def graph2_fn():
      weights = [
          tf.constant([0., 1.]),
          tf.constant([1.]),
          tf.constant([1.]),
      ]
      box_batch = [
          tf.constant([self._box_center, self._box_lower_left]),
          tf.constant([self._box_center]),
          tf.constant([self._box_center_small]),
      ]
      classes = [
          tf.one_hot([0, 1], depth=4),
          tf.one_hot([2], depth=4),
          tf.one_hot([3], depth=4),
      ]
      assigner = targetassigner.CenterNetCenterHeatmapTargetAssigner(4)
      targets = assigner.assign_center_targets_from_boxes(80, 80, box_batch,
                                                          classes,
                                                          weights)
      return targets
    targets = self.execute(graph2_fn, [])
    self.assertAlmostEqual(1.0, targets[0, :, :, 1].max())
    self.assertAlmostEqual(1.0, targets[1, :, :, 2].max())
    self.assertAlmostEqual(1.0, targets[2, :, :, 3].max())
    self.assertAlmostEqual(0.0, targets[0, :, :, [0, 2, 3]].max())
    self.assertAlmostEqual(0.0, targets[1, :, :, [0, 1, 3]].max())
    self.assertAlmostEqual(0.0, targets[2, :, :, :3].max())

  def test_low_overlap(self):
    def graph1_fn():
      box_batch = [tf.constant([self._box_center])]
      classes = [
          tf.one_hot([0], depth=2),
      ]
      assigner = targetassigner.CenterNetCenterHeatmapTargetAssigner(
          4, min_overlap=0.1)
      targets_low_overlap = assigner.assign_center_targets_from_boxes(
          80, 80, box_batch, classes)
      return targets_low_overlap
    targets_low_overlap = self.execute(graph1_fn, [])
    self.assertLess(1, np.count_nonzero(targets_low_overlap))

    def graph2_fn():
      box_batch = [tf.constant([self._box_center])]
      classes = [
          tf.one_hot([0], depth=2),
      ]
      assigner = targetassigner.CenterNetCenterHeatmapTargetAssigner(
          4, min_overlap=0.6)
      targets_medium_overlap = assigner.assign_center_targets_from_boxes(
          80, 80, box_batch, classes)
      return targets_medium_overlap
    targets_medium_overlap = self.execute(graph2_fn, [])
    self.assertLess(1, np.count_nonzero(targets_medium_overlap))

    def graph3_fn():
      box_batch = [tf.constant([self._box_center])]
      classes = [
          tf.one_hot([0], depth=2),
      ]
      assigner = targetassigner.CenterNetCenterHeatmapTargetAssigner(
          4, min_overlap=0.99)
      targets_high_overlap = assigner.assign_center_targets_from_boxes(
          80, 80, box_batch, classes)
      return targets_high_overlap

    targets_high_overlap = self.execute(graph3_fn, [])
    self.assertTrue(np.all(targets_low_overlap >= targets_medium_overlap))
    self.assertTrue(np.all(targets_medium_overlap >= targets_high_overlap))

  def test_empty_box_list(self):
    """Test that an empty box list gives an all 0 heatmap."""
    def graph_fn():
      box_batch = [
          tf.zeros((0, 4), dtype=tf.float32),
      ]

      classes = [
          tf.zeros((0, 5), dtype=tf.float32),
      ]

      assigner = targetassigner.CenterNetCenterHeatmapTargetAssigner(
          4, min_overlap=0.1)
      targets = assigner.assign_center_targets_from_boxes(
          80, 80, box_batch, classes)
      return targets
    targets = self.execute(graph_fn, [])
    np.testing.assert_allclose(targets, 0.)


class CenterNetBoxTargetAssignerTest(test_case.TestCase):

  def setUp(self):
    super(CenterNetBoxTargetAssignerTest, self).setUp()
    self._box_center = [0.0, 0.0, 1.0, 1.0]
    self._box_center_small = [0.25, 0.25, 0.75, 0.75]
    self._box_lower_left = [0.5, 0.0, 1.0, 0.5]
    self._box_center_offset = [0.1, 0.05, 1.0, 1.0]
    self._box_odd_coordinates = [0.1625, 0.2125, 0.5625, 0.9625]

  def test_max_distance_for_overlap(self):
    """Test that the distance ensures the IoU with random boxes."""

    # TODO(vighneshb) remove this after the `_smallest_positive_root`
    # function if fixed.
    self.skipTest(('Skipping test because we are using an incorrect version of'
                   'the `max_distance_for_overlap` function to reproduce'
                   ' results.'))

    rng = np.random.RandomState(0)
    n_samples = 100

    width = rng.uniform(1, 100, size=n_samples)
    height = rng.uniform(1, 100, size=n_samples)
    min_iou = rng.uniform(0.1, 1.0, size=n_samples)

    def graph_fn():
      max_dist = targetassigner.max_distance_for_overlap(height, width, min_iou)
      return max_dist
    max_dist = self.execute(graph_fn, [])
    xmin1 = np.zeros(n_samples)
    ymin1 = np.zeros(n_samples)
    xmax1 = np.zeros(n_samples) + width
    ymax1 = np.zeros(n_samples) + height

    xmin2 = max_dist * np.cos(rng.uniform(0, 2 * np.pi))
    ymin2 = max_dist * np.sin(rng.uniform(0, 2 * np.pi))
    xmax2 = width + max_dist * np.cos(rng.uniform(0, 2 * np.pi))
    ymax2 = height + max_dist * np.sin(rng.uniform(0, 2 * np.pi))

    boxes1 = np.vstack([ymin1, xmin1, ymax1, xmax1]).T
    boxes2 = np.vstack([ymin2, xmin2, ymax2, xmax2]).T

    iou = np.diag(np_box_ops.iou(boxes1, boxes2))

    self.assertTrue(np.all(iou >= min_iou))

  def test_max_distance_for_overlap_centernet(self):
    """Test the version of the function used in the CenterNet paper."""

    def graph_fn():
      distance = targetassigner.max_distance_for_overlap(10, 5, 0.5)
      return distance
    distance = self.execute(graph_fn, [])
    self.assertAlmostEqual(2.807764064, distance)

  def test_assign_size_and_offset_targets(self):
    """Test the assign_size_and_offset_targets function."""
    def graph_fn():
      box_batch = [
          tf.constant([self._box_center, self._box_lower_left]),
          tf.constant([self._box_center_offset]),
          tf.constant([self._box_center_small, self._box_odd_coordinates]),
      ]

      assigner = targetassigner.CenterNetBoxTargetAssigner(4)
      indices, hw, yx_offset, weights = assigner.assign_size_and_offset_targets(
          80, 80, box_batch)
      return indices, hw, yx_offset, weights
    indices, hw, yx_offset, weights = self.execute(graph_fn, [])
    self.assertEqual(indices.shape, (5, 3))
    self.assertEqual(hw.shape, (5, 2))
    self.assertEqual(yx_offset.shape, (5, 2))
    self.assertEqual(weights.shape, (5,))
    np.testing.assert_array_equal(
        indices,
        [[0, 10, 10], [0, 15, 5], [1, 11, 10], [2, 10, 10], [2, 7, 11]])
    np.testing.assert_array_equal(
        hw, [[20, 20], [10, 10], [18, 19], [10, 10], [8, 15]])
    np.testing.assert_array_equal(
        yx_offset, [[0, 0], [0, 0], [0, 0.5], [0, 0], [0.25, 0.75]])
    np.testing.assert_array_equal(weights, 1)

  def test_assign_size_and_offset_targets_weights(self):
    """Test the assign_size_and_offset_targets function with box weights."""
    def graph_fn():
      box_batch = [
          tf.constant([self._box_center, self._box_lower_left]),
          tf.constant([self._box_lower_left, self._box_center_small]),
          tf.constant([self._box_center_small, self._box_odd_coordinates]),
      ]

      cn_assigner = targetassigner.CenterNetBoxTargetAssigner(4)
      weights_batch = [
          tf.constant([0.0, 1.0]),
          tf.constant([1.0, 1.0]),
          tf.constant([0.0, 0.0])
      ]
      indices, hw, yx_offset, weights = cn_assigner.assign_size_and_offset_targets(
          80, 80, box_batch, weights_batch)
      return indices, hw, yx_offset, weights
    indices, hw, yx_offset, weights = self.execute(graph_fn, [])
    self.assertEqual(indices.shape, (6, 3))
    self.assertEqual(hw.shape, (6, 2))
    self.assertEqual(yx_offset.shape, (6, 2))
    self.assertEqual(weights.shape, (6,))
    np.testing.assert_array_equal(indices,
                                  [[0, 10, 10], [0, 15, 5], [1, 15, 5],
                                   [1, 10, 10], [2, 10, 10], [2, 7, 11]])
    np.testing.assert_array_equal(
        hw, [[20, 20], [10, 10], [10, 10], [10, 10], [10, 10], [8, 15]])
    np.testing.assert_array_equal(
        yx_offset, [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.25, 0.75]])
    np.testing.assert_array_equal(weights, [0, 1, 1, 1, 0, 0])

  def test_get_batch_predictions_from_indices(self):
    """Test the get_batch_predictions_from_indices function.

    This test verifies that the indices returned by
    assign_size_and_offset_targets function work as expected with a predicted
    tensor.

    """
    def graph_fn():
      box_batch = [
          tf.constant([self._box_center, self._box_lower_left]),
          tf.constant([self._box_center_small, self._box_odd_coordinates]),
      ]

      pred_array = np.ones((2, 40, 20, 2), dtype=np.int32) * -1000
      pred_array[0, 20, 10] = [1, 2]
      pred_array[0, 30, 5] = [3, 4]
      pred_array[1, 20, 10] = [5, 6]
      pred_array[1, 14, 11] = [7, 8]

      pred_tensor = tf.constant(pred_array)

      cn_assigner = targetassigner.CenterNetBoxTargetAssigner(4)
      indices, _, _, _ = cn_assigner.assign_size_and_offset_targets(
          160, 80, box_batch)

      preds = targetassigner.get_batch_predictions_from_indices(
          pred_tensor, indices)
      return preds
    preds = self.execute(graph_fn, [])
    np.testing.assert_array_equal(preds, [[1, 2], [3, 4], [5, 6], [7, 8]])


class CenterNetKeypointTargetAssignerTest(test_case.TestCase):

  def test_keypoint_heatmap_targets(self):
    def graph_fn():
      gt_classes_list = [
          tf.one_hot([0, 1, 0, 1], depth=4),
      ]
      coordinates = tf.expand_dims(
          tf.constant(
              np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                        [float('nan'), 0.7, float('nan'), 0.9, 1.0],
                        [0.4, 0.1, 0.4, 0.2, 0.1],
                        [float('nan'), 0.1, 0.5, 0.7, 0.6]]),
              dtype=tf.float32),
          axis=2)
      gt_keypoints_list = [tf.concat([coordinates, coordinates], axis=2)]
      gt_boxes_list = [
          tf.constant(
              np.array([[0.0, 0.0, 0.3, 0.3],
                        [0.0, 0.0, 0.5, 0.5],
                        [0.0, 0.0, 0.5, 0.5],
                        [0.0, 0.0, 1.0, 1.0]]),
              dtype=tf.float32)
      ]

      cn_assigner = targetassigner.CenterNetKeypointTargetAssigner(
          stride=4,
          class_id=1,
          keypoint_indices=[0, 2])
      (targets, num_instances_batch,
       valid_mask) = cn_assigner.assign_keypoint_heatmap_targets(
           120,
           80,
           gt_keypoints_list,
           gt_classes_list,
           gt_boxes_list=gt_boxes_list)
      return targets, num_instances_batch, valid_mask

    targets, num_instances_batch, valid_mask = self.execute(graph_fn, [])
    # keypoint (0.5, 0.5) is selected. The peak is expected to appear at the
    # center of the image.
    self.assertEqual((15, 10), _array_argmax(targets[0, :, :, 1]))
    self.assertAlmostEqual(1.0, targets[0, 15, 10, 1])
    # No peak for the first class since NaN is selected.
    self.assertAlmostEqual(0.0, targets[0, 15, 10, 0])
    # Verify the output heatmap shape.
    self.assertAllEqual([1, 30, 20, 2], targets.shape)
    # Verify the number of instances is correct.
    np.testing.assert_array_almost_equal([[0, 1]],
                                         num_instances_batch)
    # When calling the function, we specify the class id to be 1 (1th and 3rd)
    # instance and the keypoint indices to be [0, 2], meaning that the 1st
    # instance is the target class with no valid keypoints in it. As a result,
    # the region of the 1st instance boxing box should be blacked out
    # (0.0, 0.0, 0.5, 0.5), transfering to (0, 0, 15, 10) in absolute output
    # space.
    self.assertAlmostEqual(np.sum(valid_mask[:, 0:16, 0:11]), 0.0)
    # All other values are 1.0 so the sum is: 30 * 20 - 16 * 11 = 424.
    self.assertAlmostEqual(np.sum(valid_mask), 424.0)

  def test_assign_keypoints_offset_targets(self):
    def graph_fn():
      gt_classes_list = [
          tf.one_hot([0, 1, 0, 1], depth=4),
      ]
      coordinates = tf.expand_dims(
          tf.constant(
              np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                        [float('nan'), 0.7, float('nan'), 0.9, 0.4],
                        [0.4, 0.1, 0.4, 0.2, 0.0],
                        [float('nan'), 0.0, 0.12, 0.7, 0.4]]),
              dtype=tf.float32),
          axis=2)
      gt_keypoints_list = [tf.concat([coordinates, coordinates], axis=2)]

      cn_assigner = targetassigner.CenterNetKeypointTargetAssigner(
          stride=4,
          class_id=1,
          keypoint_indices=[0, 2])
      (indices, offsets, weights) = cn_assigner.assign_keypoints_offset_targets(
          height=120,
          width=80,
          gt_keypoints_list=gt_keypoints_list,
          gt_classes_list=gt_classes_list)
      return indices, weights, offsets
    indices, weights, offsets = self.execute(graph_fn, [])
    # Only the last element has positive weight.
    np.testing.assert_array_almost_equal(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], weights)
    # Validate the last element's indices and offsets.
    np.testing.assert_array_equal([0, 3, 2], indices[7, :])
    np.testing.assert_array_almost_equal([0.6, 0.4], offsets[7, :])

  def test_assign_keypoint_depths_target(self):
    def graph_fn():
      gt_classes_list = [
          tf.one_hot([0, 1, 0, 1], depth=4),
      ]
      coordinates = tf.expand_dims(
          tf.constant(
              np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                        [float('nan'), 0.7, 0.7, 0.9, 0.4],
                        [0.4, 0.1, 0.4, 0.2, 0.0],
                        [float('nan'), 0.0, 0.12, 0.7, 0.4]]),
              dtype=tf.float32),
          axis=2)
      gt_keypoints_list = [tf.concat([coordinates, coordinates], axis=2)]
      depths = tf.constant(
          np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                    [float('nan'), 0.7, float('nan'), 0.9, 0.4],
                    [0.4, 0.1, 0.4, 0.2, 0.0],
                    [0.5, 0.0, 7.0, 0.7, 0.4]]),
          dtype=tf.float32)
      gt_keypoint_depths_list = [depths]

      gt_keypoint_depth_weights = tf.constant(
          np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                    [float('nan'), 0.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.5, 1.0, 1.0]]),
          dtype=tf.float32)
      gt_keypoint_depth_weights_list = [gt_keypoint_depth_weights]

      cn_assigner = targetassigner.CenterNetKeypointTargetAssigner(
          stride=4,
          class_id=1,
          keypoint_indices=[0, 2],
          peak_radius=1)
      (indices, depths, weights) = cn_assigner.assign_keypoints_depth_targets(
          height=120,
          width=80,
          gt_keypoints_list=gt_keypoints_list,
          gt_classes_list=gt_classes_list,
          gt_keypoint_depths_list=gt_keypoint_depths_list,
          gt_keypoint_depth_weights_list=gt_keypoint_depth_weights_list)
      return indices, depths, weights
    indices, depths, weights = self.execute(graph_fn, [])

    # Only the last 5 elements has positive weight.
    np.testing.assert_array_almost_equal([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5
    ], weights)
    # Validate the last 5 elements' depth value.
    np.testing.assert_array_almost_equal(
        [7.0, 7.0, 7.0, 7.0, 7.0], depths[35:, 0])
    self.assertEqual((40, 3), indices.shape)
    np.testing.assert_array_equal([0, 2, 2], indices[35, :])

  def test_assign_keypoint_depths_per_keypoints(self):
    def graph_fn():
      gt_classes_list = [
          tf.one_hot([0, 1, 0, 1], depth=4),
      ]
      coordinates = tf.expand_dims(
          tf.constant(
              np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                        [float('nan'), 0.7, 0.7, 0.9, 0.4],
                        [0.4, 0.1, 0.4, 0.2, 0.0],
                        [float('nan'), 0.0, 0.12, 0.7, 0.4]]),
              dtype=tf.float32),
          axis=2)
      gt_keypoints_list = [tf.concat([coordinates, coordinates], axis=2)]
      depths = tf.constant(
          np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                    [float('nan'), 0.7, float('nan'), 0.9, 0.4],
                    [0.4, 0.1, 0.4, 0.2, 0.0],
                    [0.5, 0.0, 7.0, 0.7, 0.4]]),
          dtype=tf.float32)
      gt_keypoint_depths_list = [depths]

      gt_keypoint_depth_weights = tf.constant(
          np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                    [float('nan'), 0.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.5, 1.0, 1.0]]),
          dtype=tf.float32)
      gt_keypoint_depth_weights_list = [gt_keypoint_depth_weights]

      cn_assigner = targetassigner.CenterNetKeypointTargetAssigner(
          stride=4,
          class_id=1,
          keypoint_indices=[0, 2],
          peak_radius=1,
          per_keypoint_offset=True)
      (indices, depths, weights) = cn_assigner.assign_keypoints_depth_targets(
          height=120,
          width=80,
          gt_keypoints_list=gt_keypoints_list,
          gt_classes_list=gt_classes_list,
          gt_keypoint_depths_list=gt_keypoint_depths_list,
          gt_keypoint_depth_weights_list=gt_keypoint_depth_weights_list)
      return indices, depths, weights
    indices, depths, weights = self.execute(graph_fn, [])

    # Only the last 5 elements has positive weight.
    np.testing.assert_array_almost_equal([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5
    ], weights)
    # Validate the last 5 elements' depth value.
    np.testing.assert_array_almost_equal(
        [7.0, 7.0, 7.0, 7.0, 7.0], depths[35:, 0])
    self.assertEqual((40, 4), indices.shape)
    np.testing.assert_array_equal([0, 2, 2, 1], indices[35, :])

  def test_assign_keypoints_offset_targets_radius(self):
    def graph_fn():
      gt_classes_list = [
          tf.one_hot([0, 1, 0, 1], depth=4),
      ]
      coordinates = tf.expand_dims(
          tf.constant(
              np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                        [float('nan'), 0.7, float('nan'), 0.9, 0.4],
                        [0.4, 0.1, 0.4, 0.2, 0.0],
                        [float('nan'), 0.0, 0.12, 0.7, 0.4]]),
              dtype=tf.float32),
          axis=2)
      gt_keypoints_list = [tf.concat([coordinates, coordinates], axis=2)]

      cn_assigner = targetassigner.CenterNetKeypointTargetAssigner(
          stride=4,
          class_id=1,
          keypoint_indices=[0, 2],
          peak_radius=1,
          per_keypoint_offset=True)
      (indices, offsets, weights) = cn_assigner.assign_keypoints_offset_targets(
          height=120,
          width=80,
          gt_keypoints_list=gt_keypoints_list,
          gt_classes_list=gt_classes_list)
      return indices, weights, offsets
    indices, weights, offsets = self.execute(graph_fn, [])

    # There are total 8 * 5 (neighbors) = 40 targets.
    self.assertAllEqual(indices.shape, [40, 4])
    self.assertAllEqual(offsets.shape, [40, 2])
    self.assertAllEqual(weights.shape, [40])
    # Only the last 5 (radius 1 generates 5 valid points) element has positive
    # weight.
    np.testing.assert_array_almost_equal([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0
    ], weights)
    # Validate the last element's (with neighbors) indices and offsets.
    np.testing.assert_array_equal([0, 2, 2, 1], indices[35, :])
    np.testing.assert_array_equal([0, 3, 1, 1], indices[36, :])
    np.testing.assert_array_equal([0, 3, 2, 1], indices[37, :])
    np.testing.assert_array_equal([0, 3, 3, 1], indices[38, :])
    np.testing.assert_array_equal([0, 4, 2, 1], indices[39, :])
    np.testing.assert_array_almost_equal([1.6, 0.4], offsets[35, :])
    np.testing.assert_array_almost_equal([0.6, 1.4], offsets[36, :])
    np.testing.assert_array_almost_equal([0.6, 0.4], offsets[37, :])
    np.testing.assert_array_almost_equal([0.6, -0.6], offsets[38, :])
    np.testing.assert_array_almost_equal([-0.4, 0.4], offsets[39, :])

  def test_assign_joint_regression_targets(self):
    def graph_fn():
      gt_boxes_list = [
          tf.constant(
              np.array([[0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0]]),
              dtype=tf.float32)
      ]
      gt_classes_list = [
          tf.one_hot([0, 1, 0, 1], depth=4),
      ]
      coordinates = tf.expand_dims(
          tf.constant(
              np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                        [float('nan'), 0.7, float('nan'), 0.9, 0.4],
                        [0.4, 0.1, 0.4, 0.2, 0.0],
                        [float('nan'), 0.0, 0.12, 0.7, 0.4]]),
              dtype=tf.float32),
          axis=2)
      gt_keypoints_list = [tf.concat([coordinates, coordinates], axis=2)]

      cn_assigner = targetassigner.CenterNetKeypointTargetAssigner(
          stride=4,
          class_id=1,
          keypoint_indices=[0, 2])
      (indices, offsets, weights) = cn_assigner.assign_joint_regression_targets(
          height=120,
          width=80,
          gt_keypoints_list=gt_keypoints_list,
          gt_classes_list=gt_classes_list,
          gt_boxes_list=gt_boxes_list)
      return indices, offsets, weights
    indices, offsets, weights = self.execute(graph_fn, [])
    np.testing.assert_array_almost_equal(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], weights)
    np.testing.assert_array_equal([0, 15, 10, 1], indices[7, :])
    np.testing.assert_array_almost_equal([-11.4, -7.6], offsets[7, :])

  def test_assign_joint_regression_targets_radius(self):
    def graph_fn():
      gt_boxes_list = [
          tf.constant(
              np.array([[0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0]]),
              dtype=tf.float32)
      ]
      gt_classes_list = [
          tf.one_hot([0, 1, 0, 1], depth=4),
      ]
      coordinates = tf.expand_dims(
          tf.constant(
              np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                        [float('nan'), 0.7, float('nan'), 0.9, 0.4],
                        [0.4, 0.1, 0.4, 0.2, 0.0],
                        [float('nan'), 0.0, 0.12, 0.7, 0.4]]),
              dtype=tf.float32),
          axis=2)
      gt_keypoints_list = [tf.concat([coordinates, coordinates], axis=2)]

      cn_assigner = targetassigner.CenterNetKeypointTargetAssigner(
          stride=4,
          class_id=1,
          keypoint_indices=[0, 2],
          peak_radius=1)
      (indices, offsets, weights) = cn_assigner.assign_joint_regression_targets(
          height=120,
          width=80,
          gt_keypoints_list=gt_keypoints_list,
          gt_classes_list=gt_classes_list,
          gt_boxes_list=gt_boxes_list)
      return indices, offsets, weights
    indices, offsets, weights = self.execute(graph_fn, [])

    # There are total 8 * 5 (neighbors) = 40 targets.
    self.assertAllEqual(indices.shape, [40, 4])
    self.assertAllEqual(offsets.shape, [40, 2])
    self.assertAllEqual(weights.shape, [40])
    # Only the last 5 (radius 1 generates 5 valid points) element has positive
    # weight.
    np.testing.assert_array_almost_equal([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0
    ], weights)
    # Test the values of the indices and offsets of the last 5 elements.
    np.testing.assert_array_equal([0, 14, 10, 1], indices[35, :])
    np.testing.assert_array_equal([0, 15, 9, 1], indices[36, :])
    np.testing.assert_array_equal([0, 15, 10, 1], indices[37, :])
    np.testing.assert_array_equal([0, 15, 11, 1], indices[38, :])
    np.testing.assert_array_equal([0, 16, 10, 1], indices[39, :])
    np.testing.assert_array_almost_equal([-10.4, -7.6], offsets[35, :])
    np.testing.assert_array_almost_equal([-11.4, -6.6], offsets[36, :])
    np.testing.assert_array_almost_equal([-11.4, -7.6], offsets[37, :])
    np.testing.assert_array_almost_equal([-11.4, -8.6], offsets[38, :])
    np.testing.assert_array_almost_equal([-12.4, -7.6], offsets[39, :])


class CenterNetMaskTargetAssignerTest(test_case.TestCase):

  def test_assign_segmentation_targets(self):
    def graph_fn():
      gt_masks_list = [
          # Example 0.
          tf.constant([
              [
                  [1., 0., 0., 0.],
                  [1., 1., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.],
              ],
              [
                  [0., 0., 0., 0.],
                  [0., 0., 0., 1.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.],
              ],
              [
                  [1., 1., 0., 0.],
                  [1., 1., 0., 0.],
                  [0., 0., 1., 1.],
                  [0., 0., 1., 1.],
              ]
          ], dtype=tf.float32),
          # Example 1.
          tf.constant([
              [
                  [1., 1., 0., 1.],
                  [1., 1., 1., 1.],
                  [0., 0., 1., 1.],
                  [0., 0., 0., 1.],
              ],
              [
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.],
                  [1., 1., 0., 0.],
                  [1., 1., 0., 0.],
              ],
          ], dtype=tf.float32),
      ]
      gt_classes_list = [
          # Example 0.
          tf.constant([[1., 0., 0.],
                       [0., 1., 0.],
                       [1., 0., 0.]], dtype=tf.float32),
          # Example 1.
          tf.constant([[0., 1., 0.],
                       [0., 1., 0.]], dtype=tf.float32)
      ]
      cn_assigner = targetassigner.CenterNetMaskTargetAssigner(stride=2)
      segmentation_target = cn_assigner.assign_segmentation_targets(
          gt_masks_list=gt_masks_list,
          gt_classes_list=gt_classes_list,
          mask_resize_method=targetassigner.ResizeMethod.NEAREST_NEIGHBOR)
      return segmentation_target
    segmentation_target = self.execute(graph_fn, [])

    expected_seg_target = np.array([
        # Example 0  [[class 0, class 1], [background, class 0]]
        [[[1, 0, 0], [0, 1, 0]],
         [[0, 0, 0], [1, 0, 0]]],
        # Example 1  [[class 1, class 1], [class 1, class 1]]
        [[[0, 1, 0], [0, 1, 0]],
         [[0, 1, 0], [0, 1, 0]]],
    ], dtype=np.float32)
    np.testing.assert_array_almost_equal(
        expected_seg_target, segmentation_target)

  def test_assign_segmentation_targets_no_objects(self):
    def graph_fn():
      gt_masks_list = [tf.zeros((0, 5, 5))]
      gt_classes_list = [tf.zeros((0, 10))]
      cn_assigner = targetassigner.CenterNetMaskTargetAssigner(stride=1)
      segmentation_target = cn_assigner.assign_segmentation_targets(
          gt_masks_list=gt_masks_list,
          gt_classes_list=gt_classes_list,
          mask_resize_method=targetassigner.ResizeMethod.NEAREST_NEIGHBOR)
      return segmentation_target

    segmentation_target = self.execute(graph_fn, [])
    expected_seg_target = np.zeros((1, 5, 5, 10))
    np.testing.assert_array_almost_equal(
        expected_seg_target, segmentation_target)


class CenterNetDensePoseTargetAssignerTest(test_case.TestCase):

  def test_assign_part_and_coordinate_targets(self):
    def graph_fn():
      gt_dp_num_points_list = [
          # Example 0.
          tf.constant([2, 0, 3], dtype=tf.int32),
          # Example 1.
          tf.constant([1, 1], dtype=tf.int32),
      ]
      gt_dp_part_ids_list = [
          # Example 0.
          tf.constant([[1, 6, 0],
                       [0, 0, 0],
                       [0, 2, 3]], dtype=tf.int32),
          # Example 1.
          tf.constant([[7, 0, 0],
                       [0, 0, 0]], dtype=tf.int32),
      ]
      gt_dp_surface_coords_list = [
          # Example 0.
          tf.constant(
              [[[0.11, 0.2, 0.3, 0.4],  # Box 0.
                [0.6, 0.4, 0.1, 0.0],
                [0.0, 0.0, 0.0, 0.0]],
               [[0.0, 0.0, 0.0, 0.0],  # Box 1.
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]],
               [[0.22, 0.1, 0.6, 0.8],  # Box 2.
                [0.0, 0.4, 0.5, 1.0],
                [0.3, 0.2, 0.4, 0.1]]],
              dtype=tf.float32),
          # Example 1.
          tf.constant(
              [[[0.5, 0.5, 0.3, 1.0],  # Box 0.
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]],
               [[0.2, 0.2, 0.5, 0.8],  # Box 1.
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]]],
              dtype=tf.float32),
      ]
      gt_weights_list = [
          # Example 0.
          tf.constant([1.0, 1.0, 0.5], dtype=tf.float32),
          # Example 1.
          tf.constant([0.0, 1.0], dtype=tf.float32),
      ]
      cn_assigner = targetassigner.CenterNetDensePoseTargetAssigner(stride=4)
      batch_indices, batch_part_ids, batch_surface_coords, batch_weights = (
          cn_assigner.assign_part_and_coordinate_targets(
              height=120,
              width=80,
              gt_dp_num_points_list=gt_dp_num_points_list,
              gt_dp_part_ids_list=gt_dp_part_ids_list,
              gt_dp_surface_coords_list=gt_dp_surface_coords_list,
              gt_weights_list=gt_weights_list))

      return batch_indices, batch_part_ids, batch_surface_coords, batch_weights
    batch_indices, batch_part_ids, batch_surface_coords, batch_weights = (
        self.execute(graph_fn, []))

    expected_batch_indices = np.array([
        # Example 0. e.g.
        # The first set of indices is calculated as follows:
        # floor(0.11*120/4) = 3, floor(0.2*80/4) = 4.
        [0, 3, 4, 1], [0, 18, 8, 6], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
        [0, 0, 0, 0], [0, 6, 2, 0], [0, 0, 8, 2], [0, 9, 4, 3],
        # Example 1.
        [1, 15, 10, 7], [1, 0, 0, 0], [1, 0, 0, 0], [1, 6, 4, 0], [1, 0, 0, 0],
        [1, 0, 0, 0]
    ], dtype=np.int32)
    expected_batch_part_ids = tf.one_hot(
        [1, 6, 0, 0, 0, 0, 0, 2, 3, 7, 0, 0, 0, 0, 0], depth=24).numpy()
    expected_batch_surface_coords = np.array([
        # Box 0.
        [0.3, 0.4], [0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
        [0.6, 0.8], [0.5, 1.0], [0.4, 0.1],
        # Box 1.
        [0.3, 1.0], [0.0, 0.0], [0.0, 0.0], [0.5, 0.8], [0.0, 0.0], [0.0, 0.0],
    ], np.float32)
    expected_batch_weights = np.array([
        # Box 0.
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5,
        # Box 1.
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0
    ], dtype=np.float32)
    self.assertAllEqual(expected_batch_indices, batch_indices)
    self.assertAllEqual(expected_batch_part_ids, batch_part_ids)
    self.assertAllClose(expected_batch_surface_coords, batch_surface_coords)
    self.assertAllClose(expected_batch_weights, batch_weights)


class CenterNetTrackTargetAssignerTest(test_case.TestCase):

  def setUp(self):
    super(CenterNetTrackTargetAssignerTest, self).setUp()
    self._box_center = [0.0, 0.0, 1.0, 1.0]
    self._box_center_small = [0.25, 0.25, 0.75, 0.75]
    self._box_lower_left = [0.5, 0.0, 1.0, 0.5]
    self._box_center_offset = [0.1, 0.05, 1.0, 1.0]
    self._box_odd_coordinates = [0.1625, 0.2125, 0.5625, 0.9625]

  def test_assign_track_targets(self):
    """Test the assign_track_targets function."""
    def graph_fn():
      box_batch = [
          tf.constant([self._box_center, self._box_lower_left]),
          tf.constant([self._box_lower_left, self._box_center_small]),
          tf.constant([self._box_center_small, self._box_odd_coordinates]),
      ]
      track_id_batch = [
          tf.constant([0, 1]),
          tf.constant([1, 0]),
          tf.constant([0, 2]),
      ]

      assigner = targetassigner.CenterNetTrackTargetAssigner(
          stride=4, num_track_ids=3)

      (batch_indices, batch_weights,
       track_targets) = assigner.assign_track_targets(
           height=80,
           width=80,
           gt_track_ids_list=track_id_batch,
           gt_boxes_list=box_batch)
      return batch_indices, batch_weights, track_targets

    indices, weights, track_ids = self.execute(graph_fn, [])

    self.assertEqual(indices.shape, (3, 2, 3))
    self.assertEqual(track_ids.shape, (3, 2, 3))
    self.assertEqual(weights.shape, (3, 2))

    np.testing.assert_array_equal(indices,
                                  [[[0, 10, 10], [0, 15, 5]],
                                   [[1, 15, 5], [1, 10, 10]],
                                   [[2, 10, 10], [2, 7, 11]]])
    np.testing.assert_array_equal(track_ids,
                                  [[[1, 0, 0], [0, 1, 0]],
                                   [[0, 1, 0], [1, 0, 0]],
                                   [[1, 0, 0], [0, 0, 1]]])
    np.testing.assert_array_equal(weights, [[1, 1], [1, 1], [1, 1]])

  def test_assign_track_targets_weights(self):
    """Test the assign_track_targets function with box weights."""
    def graph_fn():
      box_batch = [
          tf.constant([self._box_center, self._box_lower_left]),
          tf.constant([self._box_lower_left, self._box_center_small]),
          tf.constant([self._box_center_small, self._box_odd_coordinates]),
      ]
      track_id_batch = [
          tf.constant([0, 1]),
          tf.constant([1, 0]),
          tf.constant([0, 2]),
      ]
      weights_batch = [
          tf.constant([0.0, 1.0]),
          tf.constant([1.0, 1.0]),
          tf.constant([0.0, 0.0])
      ]

      assigner = targetassigner.CenterNetTrackTargetAssigner(
          stride=4, num_track_ids=3)

      (batch_indices, batch_weights,
       track_targets) = assigner.assign_track_targets(
           height=80,
           width=80,
           gt_track_ids_list=track_id_batch,
           gt_boxes_list=box_batch,
           gt_weights_list=weights_batch)
      return batch_indices, batch_weights, track_targets

    indices, weights, track_ids = self.execute(graph_fn, [])

    self.assertEqual(indices.shape, (3, 2, 3))
    self.assertEqual(track_ids.shape, (3, 2, 3))
    self.assertEqual(weights.shape, (3, 2))

    np.testing.assert_array_equal(indices,
                                  [[[0, 10, 10], [0, 15, 5]],
                                   [[1, 15, 5], [1, 10, 10]],
                                   [[2, 10, 10], [2, 7, 11]]])
    np.testing.assert_array_equal(track_ids,
                                  [[[1, 0, 0], [0, 1, 0]],
                                   [[0, 1, 0], [1, 0, 0]],
                                   [[1, 0, 0], [0, 0, 1]]])
    np.testing.assert_array_equal(weights, [[0, 1], [1, 1], [0, 0]])
    # TODO(xwwang): Add a test for the case when no objects are detected.


class CornerOffsetTargetAssignerTest(test_case.TestCase):

  def test_filter_overlap_min_area_empty(self):
    """Test that empty masks work on CPU."""
    def graph_fn(masks):
      return targetassigner.filter_mask_overlap_min_area(masks)

    masks = self.execute_cpu(graph_fn, [np.zeros((0, 5, 5), dtype=np.float32)])
    self.assertEqual(masks.shape, (0, 5, 5))

  def test_filter_overlap_min_area(self):
    """Test the object with min. area is selected instead of overlap."""
    def graph_fn(masks):
      return targetassigner.filter_mask_overlap_min_area(masks)

    masks = np.zeros((3, 4, 4), dtype=np.float32)
    masks[0, :2, :2] = 1.0
    masks[1, :3, :3] = 1.0
    masks[2, 3, 3] = 1.0

    masks = self.execute(graph_fn, [masks])

    self.assertAllClose(masks[0],
                        [[1, 1, 0, 0],
                         [1, 1, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]])
    self.assertAllClose(masks[1],
                        [[0, 0, 1, 0],
                         [0, 0, 1, 0],
                         [1, 1, 1, 0],
                         [0, 0, 0, 0]])

    self.assertAllClose(masks[2],
                        [[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 1]])

  def test_assign_corner_offset_single_object(self):
    """Test that corner offsets are correct with a single object."""
    assigner = targetassigner.CenterNetCornerOffsetTargetAssigner(stride=1)

    def graph_fn():
      boxes = [
          tf.constant([[0., 0., 1., 1.]])
      ]
      mask = np.zeros((1, 4, 4), dtype=np.float32)
      mask[0, 1:3, 1:3] = 1.0

      masks = [tf.constant(mask)]
      return assigner.assign_corner_offset_targets(boxes, masks)

    corner_offsets, foreground = self.execute(graph_fn, [])
    self.assertAllClose(foreground[0],
                        [[0, 0, 0, 0],
                         [0, 1, 1, 0],
                         [0, 1, 1, 0],
                         [0, 0, 0, 0]])

    self.assertAllClose(corner_offsets[0, :, :, 0],
                        [[0, 0, 0, 0],
                         [0, -1, -1, 0],
                         [0, -2, -2, 0],
                         [0, 0, 0, 0]])
    self.assertAllClose(corner_offsets[0, :, :, 1],
                        [[0, 0, 0, 0],
                         [0, -1, -2, 0],
                         [0, -1, -2, 0],
                         [0, 0, 0, 0]])
    self.assertAllClose(corner_offsets[0, :, :, 2],
                        [[0, 0, 0, 0],
                         [0, 3, 3, 0],
                         [0, 2, 2, 0],
                         [0, 0, 0, 0]])
    self.assertAllClose(corner_offsets[0, :, :, 3],
                        [[0, 0, 0, 0],
                         [0, 3, 2, 0],
                         [0, 3, 2, 0],
                         [0, 0, 0, 0]])

  def test_assign_corner_offset_multiple_objects(self):
    """Test corner offsets are correct with multiple objects."""
    assigner = targetassigner.CenterNetCornerOffsetTargetAssigner(stride=1)

    def graph_fn():
      boxes = [
          tf.constant([[0., 0., 1., 1.], [0., 0., 0., 0.]]),
          tf.constant([[0., 0., .25, .25], [.25, .25, 1., 1.]])
      ]
      mask1 = np.zeros((2, 4, 4), dtype=np.float32)
      mask1[0, 0, 0] = 1.0
      mask1[0, 3, 3] = 1.0

      mask2 = np.zeros((2, 4, 4), dtype=np.float32)
      mask2[0, :2, :2] = 1.0
      mask2[1, 1:, 1:] = 1.0

      masks = [tf.constant(mask1), tf.constant(mask2)]
      return assigner.assign_corner_offset_targets(boxes, masks)

    corner_offsets, foreground = self.execute(graph_fn, [])
    self.assertEqual(corner_offsets.shape, (2, 4, 4, 4))
    self.assertEqual(foreground.shape, (2, 4, 4))

    self.assertAllClose(foreground[0],
                        [[1, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 1]])

    self.assertAllClose(corner_offsets[0, :, :, 0],
                        [[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, -3]])
    self.assertAllClose(corner_offsets[0, :, :, 1],
                        [[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, -3]])
    self.assertAllClose(corner_offsets[0, :, :, 2],
                        [[4, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 1]])
    self.assertAllClose(corner_offsets[0, :, :, 3],
                        [[4, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 1]])

    self.assertAllClose(foreground[1],
                        [[1, 1, 0, 0],
                         [1, 1, 1, 1],
                         [0, 1, 1, 1],
                         [0, 1, 1, 1]])

    self.assertAllClose(corner_offsets[1, :, :, 0],
                        [[0, 0, 0, 0],
                         [-1, -1, 0, 0],
                         [0, -1, -1, -1],
                         [0, -2, -2, -2]])
    self.assertAllClose(corner_offsets[1, :, :, 1],
                        [[0, -1, 0, 0],
                         [0, -1, -1, -2],
                         [0, 0, -1, -2],
                         [0, 0, -1, -2]])
    self.assertAllClose(corner_offsets[1, :, :, 2],
                        [[1, 1, 0, 0],
                         [0, 0, 3, 3],
                         [0, 2, 2, 2],
                         [0, 1, 1, 1]])
    self.assertAllClose(corner_offsets[1, :, :, 3],
                        [[1, 0, 0, 0],
                         [1, 0, 2, 1],
                         [0, 3, 2, 1],
                         [0, 3, 2, 1]])

  def test_assign_corner_offsets_no_objects(self):
    """Test assignment works with empty input on cpu."""
    assigner = targetassigner.CenterNetCornerOffsetTargetAssigner(stride=1)

    def graph_fn():
      boxes = [
          tf.zeros((0, 4), dtype=tf.float32)
      ]
      masks = [tf.zeros((0, 5, 5), dtype=tf.float32)]
      return assigner.assign_corner_offset_targets(boxes, masks)

    corner_offsets, foreground = self.execute_cpu(graph_fn, [])
    self.assertAllClose(corner_offsets, np.zeros((1, 5, 5, 4)))
    self.assertAllClose(foreground, np.zeros((1, 5, 5)))


class CenterNetTemporalOffsetTargetAssigner(test_case.TestCase):

  def setUp(self):
    super(CenterNetTemporalOffsetTargetAssigner, self).setUp()
    self._box_center = [0.0, 0.0, 1.0, 1.0]
    self._box_center_small = [0.25, 0.25, 0.75, 0.75]
    self._box_lower_left = [0.5, 0.0, 1.0, 0.5]
    self._box_center_offset = [0.1, 0.05, 1.0, 1.0]
    self._box_odd_coordinates = [0.1625, 0.2125, 0.5625, 0.9625]
    self._offset_center = [0.5, 0.4]
    self._offset_center_small = [0.1, 0.1]
    self._offset_lower_left = [-0.1, 0.1]
    self._offset_center_offset = [0.4, 0.3]
    self._offset_odd_coord = [0.125, -0.125]

  def test_assign_empty_groundtruths(self):
    """Tests the assign_offset_targets function with empty inputs."""
    def graph_fn():
      box_batch = [
          tf.zeros((0, 4), dtype=tf.float32),
      ]

      offset_batch = [
          tf.zeros((0, 2), dtype=tf.float32),
      ]

      match_flag_batch = [
          tf.zeros((0), dtype=tf.float32),
      ]

      assigner = targetassigner.CenterNetTemporalOffsetTargetAssigner(4)
      indices, temporal_offset, weights = assigner.assign_temporal_offset_targets(
          80, 80, box_batch, offset_batch, match_flag_batch)
      return indices, temporal_offset, weights
    indices, temporal_offset, weights = self.execute(graph_fn, [])
    self.assertEqual(indices.shape, (0, 3))
    self.assertEqual(temporal_offset.shape, (0, 2))
    self.assertEqual(weights.shape, (0,))

  def test_assign_offset_targets(self):
    """Tests the assign_offset_targets function."""
    def graph_fn():
      box_batch = [
          tf.constant([self._box_center, self._box_lower_left]),
          tf.constant([self._box_center_offset]),
          tf.constant([self._box_center_small, self._box_odd_coordinates]),
      ]

      offset_batch = [
          tf.constant([self._offset_center, self._offset_lower_left]),
          tf.constant([self._offset_center_offset]),
          tf.constant([self._offset_center_small, self._offset_odd_coord]),
      ]

      match_flag_batch = [
          tf.constant([1.0, 1.0]),
          tf.constant([1.0]),
          tf.constant([1.0, 1.0]),
      ]

      assigner = targetassigner.CenterNetTemporalOffsetTargetAssigner(4)
      indices, temporal_offset, weights = assigner.assign_temporal_offset_targets(
          80, 80, box_batch, offset_batch, match_flag_batch)
      return indices, temporal_offset, weights
    indices, temporal_offset, weights = self.execute(graph_fn, [])
    self.assertEqual(indices.shape, (5, 3))
    self.assertEqual(temporal_offset.shape, (5, 2))
    self.assertEqual(weights.shape, (5,))
    np.testing.assert_array_equal(
        indices,
        [[0, 10, 10], [0, 15, 5], [1, 11, 10], [2, 10, 10], [2, 7, 11]])
    np.testing.assert_array_almost_equal(
        temporal_offset,
        [[0.5, 0.4], [-0.1, 0.1], [0.4, 0.3], [0.1, 0.1], [0.125, -0.125]])
    np.testing.assert_array_equal(weights, 1)

  def test_assign_offset_targets_with_match_flags(self):
    """Tests the assign_offset_targets function with match flags."""
    def graph_fn():
      box_batch = [
          tf.constant([self._box_center, self._box_lower_left]),
          tf.constant([self._box_center_offset]),
          tf.constant([self._box_center_small, self._box_odd_coordinates]),
      ]

      offset_batch = [
          tf.constant([self._offset_center, self._offset_lower_left]),
          tf.constant([self._offset_center_offset]),
          tf.constant([self._offset_center_small, self._offset_odd_coord]),
      ]

      match_flag_batch = [
          tf.constant([0.0, 1.0]),
          tf.constant([1.0]),
          tf.constant([1.0, 1.0]),
      ]

      cn_assigner = targetassigner.CenterNetTemporalOffsetTargetAssigner(4)
      weights_batch = [
          tf.constant([1.0, 0.0]),
          tf.constant([1.0]),
          tf.constant([1.0, 1.0])
      ]
      indices, temporal_offset, weights = cn_assigner.assign_temporal_offset_targets(
          80, 80, box_batch, offset_batch, match_flag_batch, weights_batch)
      return indices, temporal_offset, weights
    indices, temporal_offset, weights = self.execute(graph_fn, [])
    self.assertEqual(indices.shape, (5, 3))
    self.assertEqual(temporal_offset.shape, (5, 2))
    self.assertEqual(weights.shape, (5,))

    np.testing.assert_array_equal(
        indices,
        [[0, 10, 10], [0, 15, 5], [1, 11, 10], [2, 10, 10], [2, 7, 11]])
    np.testing.assert_array_almost_equal(
        temporal_offset,
        [[0.5, 0.4], [-0.1, 0.1], [0.4, 0.3], [0.1, 0.1], [0.125, -0.125]])
    np.testing.assert_array_equal(weights, [0, 0, 1, 1, 1])


class DETRTargetAssignerTest(test_case.TestCase):

  def test_assign_detr(self):
    def graph_fn(pred_corners, groundtruth_box_corners,
                 groundtruth_labels, predicted_labels):
      detr_target_assigner = targetassigner.DETRTargetAssigner()
      pred_boxlist = box_list.BoxList(pred_corners)
      groundtruth_boxlist = box_list.BoxList(groundtruth_box_corners)
      result = detr_target_assigner.assign(
          pred_boxlist, groundtruth_boxlist,
          predicted_labels, groundtruth_labels)
      (cls_targets, cls_weights, reg_targets, reg_weights) = result
      return (cls_targets, cls_weights, reg_targets, reg_weights)

    pred_corners = np.array([[0.25, 0.25, 0.4, 0.2],
                             [0.5, 0.8, 1.0, 0.8],
                             [0.9, 0.5, 0.1, 1.0]], dtype=np.float32)
    groundtruth_box_corners = np.array([[0.0, 0.0, 0.5, 0.5],
                                        [0.5, 0.5, 0.9, 0.9]],
                                       dtype=np.float32)
    predicted_labels = np.array([[-3.0, 3.0], [2.0, 9.4], [5.0, 1.0]],
                                dtype=np.float32)
    groundtruth_labels = np.array([[0.0, 1.0], [0.0, 1.0]],
                                  dtype=np.float32)

    exp_cls_targets = [[0, 1], [0, 1], [1, 0]]
    exp_cls_weights = [[1, 1], [1, 1], [1, 1]]
    exp_reg_targets = [[0.25, 0.25, 0.5, 0.5],
                       [0.7, 0.7, 0.4, 0.4],
                       [0, 0, 0, 0]]
    exp_reg_weights = [1, 1, 0]

    (cls_targets_out,
     cls_weights_out, reg_targets_out, reg_weights_out) = self.execute_cpu(
         graph_fn, [pred_corners, groundtruth_box_corners,
                    groundtruth_labels, predicted_labels])

    self.assertAllClose(cls_targets_out, exp_cls_targets)
    self.assertAllClose(cls_weights_out, exp_cls_weights)
    self.assertAllClose(reg_targets_out, exp_reg_targets)
    self.assertAllClose(reg_weights_out, exp_reg_weights)
    self.assertEqual(cls_targets_out.dtype, np.float32)
    self.assertEqual(cls_weights_out.dtype, np.float32)
    self.assertEqual(reg_targets_out.dtype, np.float32)
    self.assertEqual(reg_weights_out.dtype, np.float32)

  def test_batch_assign_detr(self):
    def graph_fn(pred_corners, groundtruth_box_corners,
                 groundtruth_labels, predicted_labels):
      detr_target_assigner = targetassigner.DETRTargetAssigner()
      result = detr_target_assigner.batch_assign(
          pred_corners, groundtruth_box_corners,
          [predicted_labels], [groundtruth_labels])
      (cls_targets, cls_weights, reg_targets, reg_weights) = result
      return (cls_targets, cls_weights, reg_targets, reg_weights)

    pred_corners = np.array([[[0.25, 0.25, 0.4, 0.2],
                              [0.5, 0.8, 1.0, 0.8],
                              [0.9, 0.5, 0.1, 1.0]]], dtype=np.float32)
    groundtruth_box_corners = np.array([[[0.0, 0.0, 0.5, 0.5],
                                         [0.5, 0.5, 0.9, 0.9]]],
                                       dtype=np.float32)
    predicted_labels = np.array([[-3.0, 3.0], [2.0, 9.4], [5.0, 1.0]],
                                dtype=np.float32)
    groundtruth_labels = np.array([[0.0, 1.0], [0.0, 1.0]],
                                  dtype=np.float32)

    exp_cls_targets = [[[0, 1], [0, 1], [1, 0]]]
    exp_cls_weights = [[[1, 1], [1, 1], [1, 1]]]
    exp_reg_targets = [[[0.25, 0.25, 0.5, 0.5],
                        [0.7, 0.7, 0.4, 0.4],
                        [0, 0, 0, 0]]]
    exp_reg_weights = [[1, 1, 0]]

    (cls_targets_out,
     cls_weights_out, reg_targets_out, reg_weights_out) = self.execute_cpu(
         graph_fn, [pred_corners, groundtruth_box_corners,
                    groundtruth_labels, predicted_labels])

    self.assertAllClose(cls_targets_out, exp_cls_targets)
    self.assertAllClose(cls_weights_out, exp_cls_weights)
    self.assertAllClose(reg_targets_out, exp_reg_targets)
    self.assertAllClose(reg_weights_out, exp_reg_weights)
    self.assertEqual(cls_targets_out.dtype, np.float32)
    self.assertEqual(cls_weights_out.dtype, np.float32)
    self.assertEqual(reg_targets_out.dtype, np.float32)
    self.assertEqual(reg_weights_out.dtype, np.float32)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
