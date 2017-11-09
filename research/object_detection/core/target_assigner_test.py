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
import tensorflow as tf

from object_detection.box_coders import mean_stddev_box_coder
from object_detection.core import box_list
from object_detection.core import region_similarity_calculator
from object_detection.core import target_assigner as targetassigner
from object_detection.matchers import argmax_matcher
from object_detection.matchers import bipartite_matcher


class TargetAssignerTest(tf.test.TestCase):

  def test_assign_agnostic(self):
    similarity_calc = region_similarity_calculator.NegSqDistSimilarity()
    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    box_coder = mean_stddev_box_coder.MeanStddevBoxCoder()
    target_assigner = targetassigner.TargetAssigner(
        similarity_calc, matcher, box_coder, unmatched_cls_target=None)

    prior_means = tf.constant([[0.0, 0.0, 0.5, 0.5],
                               [0.5, 0.5, 1.0, 0.8],
                               [0, 0.5, .5, 1.0]])
    prior_stddevs = tf.constant(3 * [4 * [.1]])
    priors = box_list.BoxList(prior_means)
    priors.add_field('stddev', prior_stddevs)

    box_corners = [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.9, 0.9]]
    boxes = box_list.BoxList(tf.constant(box_corners))
    exp_cls_targets = [[1], [1], [0]]
    exp_cls_weights = [1, 1, 1]
    exp_reg_targets = [[0, 0, 0, 0],
                       [0, 0, -1, 1],
                       [0, 0, 0, 0]]
    exp_reg_weights = [1, 1, 0]
    exp_matching_anchors = [0, 1]

    result = target_assigner.assign(priors, boxes, num_valid_rows=2)
    (cls_targets, cls_weights, reg_targets, reg_weights, match) = result

    with self.test_session() as sess:
      (cls_targets_out, cls_weights_out,
       reg_targets_out, reg_weights_out, matching_anchors_out) = sess.run(
           [cls_targets, cls_weights, reg_targets, reg_weights,
            match.matched_column_indices()])

      self.assertAllClose(cls_targets_out, exp_cls_targets)
      self.assertAllClose(cls_weights_out, exp_cls_weights)
      self.assertAllClose(reg_targets_out, exp_reg_targets)
      self.assertAllClose(reg_weights_out, exp_reg_weights)
      self.assertAllClose(matching_anchors_out, exp_matching_anchors)
      self.assertEquals(cls_targets_out.dtype, np.float32)
      self.assertEquals(cls_weights_out.dtype, np.float32)
      self.assertEquals(reg_targets_out.dtype, np.float32)
      self.assertEquals(reg_weights_out.dtype, np.float32)
      self.assertEquals(matching_anchors_out.dtype, np.int32)

  def test_assign_with_ignored_matches(self):
    # Note: test is very similar to above. The third box matched with an IOU
    # of 0.35, which is between the matched and unmatched threshold. This means
    # That like above the expected classification targets are [1, 1, 0].
    # Unlike above, the third target is ignored and therefore expected
    # classification weights are [1, 1, 0].
    similarity_calc = region_similarity_calculator.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5,
                                           unmatched_threshold=0.3)
    box_coder = mean_stddev_box_coder.MeanStddevBoxCoder()
    target_assigner = targetassigner.TargetAssigner(
        similarity_calc, matcher, box_coder)

    prior_means = tf.constant([[0.0, 0.0, 0.5, 0.5],
                               [0.5, 0.5, 1.0, 0.8],
                               [0.0, 0.5, .9, 1.0]])
    prior_stddevs = tf.constant(3 * [4 * [.1]])
    priors = box_list.BoxList(prior_means)
    priors.add_field('stddev', prior_stddevs)

    box_corners = [[0.0, 0.0, 0.5, 0.5],
                   [0.5, 0.5, 0.9, 0.9]]
    boxes = box_list.BoxList(tf.constant(box_corners))
    exp_cls_targets = [[1], [1], [0]]
    exp_cls_weights = [1, 1, 0]
    exp_reg_targets = [[0, 0, 0, 0],
                       [0, 0, -1, 1],
                       [0, 0, 0, 0]]
    exp_reg_weights = [1, 1, 0]
    exp_matching_anchors = [0, 1]

    result = target_assigner.assign(priors, boxes)
    (cls_targets, cls_weights, reg_targets, reg_weights, match) = result
    with self.test_session() as sess:
      (cls_targets_out, cls_weights_out,
       reg_targets_out, reg_weights_out, matching_anchors_out) = sess.run(
           [cls_targets, cls_weights, reg_targets, reg_weights,
            match.matched_column_indices()])

      self.assertAllClose(cls_targets_out, exp_cls_targets)
      self.assertAllClose(cls_weights_out, exp_cls_weights)
      self.assertAllClose(reg_targets_out, exp_reg_targets)
      self.assertAllClose(reg_weights_out, exp_reg_weights)
      self.assertAllClose(matching_anchors_out, exp_matching_anchors)
      self.assertEquals(cls_targets_out.dtype, np.float32)
      self.assertEquals(cls_weights_out.dtype, np.float32)
      self.assertEquals(reg_targets_out.dtype, np.float32)
      self.assertEquals(reg_weights_out.dtype, np.float32)
      self.assertEquals(matching_anchors_out.dtype, np.int32)

  def test_assign_multiclass(self):
    similarity_calc = region_similarity_calculator.NegSqDistSimilarity()
    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    box_coder = mean_stddev_box_coder.MeanStddevBoxCoder()
    unmatched_cls_target = tf.constant([1, 0, 0, 0, 0, 0, 0], tf.float32)
    target_assigner = targetassigner.TargetAssigner(
        similarity_calc, matcher, box_coder,
        unmatched_cls_target=unmatched_cls_target)

    prior_means = tf.constant([[0.0, 0.0, 0.5, 0.5],
                               [0.5, 0.5, 1.0, 0.8],
                               [0, 0.5, .5, 1.0],
                               [.75, 0, 1.0, .25]])
    prior_stddevs = tf.constant(4 * [4 * [.1]])
    priors = box_list.BoxList(prior_means)
    priors.add_field('stddev', prior_stddevs)

    box_corners = [[0.0, 0.0, 0.5, 0.5],
                   [0.5, 0.5, 0.9, 0.9],
                   [.75, 0, .95, .27]]
    boxes = box_list.BoxList(tf.constant(box_corners))

    groundtruth_labels = tf.constant([[0, 1, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0],
                                      [0, 0, 0, 1, 0, 0, 0]], tf.float32)

    exp_cls_targets = [[0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0],
                       [1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0]]
    exp_cls_weights = [1, 1, 1, 1]
    exp_reg_targets = [[0, 0, 0, 0],
                       [0, 0, -1, 1],
                       [0, 0, 0, 0],
                       [0, 0, -.5, .2]]
    exp_reg_weights = [1, 1, 0, 1]
    exp_matching_anchors = [0, 1, 3]

    result = target_assigner.assign(priors, boxes, groundtruth_labels,
                                    num_valid_rows=3)
    (cls_targets, cls_weights, reg_targets, reg_weights, match) = result
    with self.test_session() as sess:
      (cls_targets_out, cls_weights_out,
       reg_targets_out, reg_weights_out, matching_anchors_out) = sess.run(
           [cls_targets, cls_weights, reg_targets, reg_weights,
            match.matched_column_indices()])

      self.assertAllClose(cls_targets_out, exp_cls_targets)
      self.assertAllClose(cls_weights_out, exp_cls_weights)
      self.assertAllClose(reg_targets_out, exp_reg_targets)
      self.assertAllClose(reg_weights_out, exp_reg_weights)
      self.assertAllClose(matching_anchors_out, exp_matching_anchors)
      self.assertEquals(cls_targets_out.dtype, np.float32)
      self.assertEquals(cls_weights_out.dtype, np.float32)
      self.assertEquals(reg_targets_out.dtype, np.float32)
      self.assertEquals(reg_weights_out.dtype, np.float32)
      self.assertEquals(matching_anchors_out.dtype, np.int32)

  def test_assign_multiclass_unequal_class_weights(self):
    similarity_calc = region_similarity_calculator.NegSqDistSimilarity()
    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    box_coder = mean_stddev_box_coder.MeanStddevBoxCoder()
    unmatched_cls_target = tf.constant([1, 0, 0, 0, 0, 0, 0], tf.float32)
    target_assigner = targetassigner.TargetAssigner(
        similarity_calc, matcher, box_coder,
        positive_class_weight=1.0, negative_class_weight=0.5,
        unmatched_cls_target=unmatched_cls_target)

    prior_means = tf.constant([[0.0, 0.0, 0.5, 0.5],
                               [0.5, 0.5, 1.0, 0.8],
                               [0, 0.5, .5, 1.0],
                               [.75, 0, 1.0, .25]])
    prior_stddevs = tf.constant(4 * [4 * [.1]])
    priors = box_list.BoxList(prior_means)
    priors.add_field('stddev', prior_stddevs)

    box_corners = [[0.0, 0.0, 0.5, 0.5],
                   [0.5, 0.5, 0.9, 0.9],
                   [.75, 0, .95, .27]]
    boxes = box_list.BoxList(tf.constant(box_corners))

    groundtruth_labels = tf.constant([[0, 1, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0],
                                      [0, 0, 0, 1, 0, 0, 0]], tf.float32)

    exp_cls_weights = [1, 1, .5, 1]
    result = target_assigner.assign(priors, boxes, groundtruth_labels,
                                    num_valid_rows=3)
    (_, cls_weights, _, _, _) = result
    with self.test_session() as sess:
      cls_weights_out = sess.run(cls_weights)
      self.assertAllClose(cls_weights_out, exp_cls_weights)

  def test_assign_multidimensional_class_targets(self):
    similarity_calc = region_similarity_calculator.NegSqDistSimilarity()
    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    box_coder = mean_stddev_box_coder.MeanStddevBoxCoder()
    unmatched_cls_target = tf.constant([[0, 0], [0, 0]], tf.float32)
    target_assigner = targetassigner.TargetAssigner(
        similarity_calc, matcher, box_coder,
        unmatched_cls_target=unmatched_cls_target)

    prior_means = tf.constant([[0.0, 0.0, 0.5, 0.5],
                               [0.5, 0.5, 1.0, 0.8],
                               [0, 0.5, .5, 1.0],
                               [.75, 0, 1.0, .25]])
    prior_stddevs = tf.constant(4 * [4 * [.1]])
    priors = box_list.BoxList(prior_means)
    priors.add_field('stddev', prior_stddevs)

    box_corners = [[0.0, 0.0, 0.5, 0.5],
                   [0.5, 0.5, 0.9, 0.9],
                   [.75, 0, .95, .27]]
    boxes = box_list.BoxList(tf.constant(box_corners))

    groundtruth_labels = tf.constant([[[0, 1], [1, 0]],
                                      [[1, 0], [0, 1]],
                                      [[0, 1], [1, .5]]], tf.float32)

    exp_cls_targets = [[[0, 1], [1, 0]],
                       [[1, 0], [0, 1]],
                       [[0, 0], [0, 0]],
                       [[0, 1], [1, .5]]]
    exp_cls_weights = [1, 1, 1, 1]
    exp_reg_targets = [[0, 0, 0, 0],
                       [0, 0, -1, 1],
                       [0, 0, 0, 0],
                       [0, 0, -.5, .2]]
    exp_reg_weights = [1, 1, 0, 1]
    exp_matching_anchors = [0, 1, 3]

    result = target_assigner.assign(priors, boxes, groundtruth_labels,
                                    num_valid_rows=3)
    (cls_targets, cls_weights, reg_targets, reg_weights, match) = result
    with self.test_session() as sess:
      (cls_targets_out, cls_weights_out,
       reg_targets_out, reg_weights_out, matching_anchors_out) = sess.run(
           [cls_targets, cls_weights, reg_targets, reg_weights,
            match.matched_column_indices()])

      self.assertAllClose(cls_targets_out, exp_cls_targets)
      self.assertAllClose(cls_weights_out, exp_cls_weights)
      self.assertAllClose(reg_targets_out, exp_reg_targets)
      self.assertAllClose(reg_weights_out, exp_reg_weights)
      self.assertAllClose(matching_anchors_out, exp_matching_anchors)
      self.assertEquals(cls_targets_out.dtype, np.float32)
      self.assertEquals(cls_weights_out.dtype, np.float32)
      self.assertEquals(reg_targets_out.dtype, np.float32)
      self.assertEquals(reg_weights_out.dtype, np.float32)
      self.assertEquals(matching_anchors_out.dtype, np.int32)

  def test_assign_empty_groundtruth(self):
    similarity_calc = region_similarity_calculator.NegSqDistSimilarity()
    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    box_coder = mean_stddev_box_coder.MeanStddevBoxCoder()
    unmatched_cls_target = tf.constant([0, 0, 0], tf.float32)
    target_assigner = targetassigner.TargetAssigner(
        similarity_calc, matcher, box_coder,
        unmatched_cls_target=unmatched_cls_target)

    prior_means = tf.constant([[0.0, 0.0, 0.5, 0.5],
                               [0.5, 0.5, 1.0, 0.8],
                               [0, 0.5, .5, 1.0],
                               [.75, 0, 1.0, .25]])
    prior_stddevs = tf.constant(4 * [4 * [.1]])
    priors = box_list.BoxList(prior_means)
    priors.add_field('stddev', prior_stddevs)

    box_corners_expanded = tf.constant([[0.0, 0.0, 0.0, 0.0]])
    box_corners = tf.slice(box_corners_expanded, [0, 0], [0, 4])
    boxes = box_list.BoxList(box_corners)

    groundtruth_labels_expanded = tf.constant([[0, 0, 0]], tf.float32)
    groundtruth_labels = tf.slice(groundtruth_labels_expanded, [0, 0], [0, 3])

    exp_cls_targets = [[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]]
    exp_cls_weights = [1, 1, 1, 1]
    exp_reg_targets = [[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]]
    exp_reg_weights = [0, 0, 0, 0]
    exp_matching_anchors = []

    result = target_assigner.assign(priors, boxes, groundtruth_labels)
    (cls_targets, cls_weights, reg_targets, reg_weights, match) = result
    with self.test_session() as sess:
      (cls_targets_out, cls_weights_out,
       reg_targets_out, reg_weights_out, matching_anchors_out) = sess.run(
           [cls_targets, cls_weights, reg_targets, reg_weights,
            match.matched_column_indices()])

      self.assertAllClose(cls_targets_out, exp_cls_targets)
      self.assertAllClose(cls_weights_out, exp_cls_weights)
      self.assertAllClose(reg_targets_out, exp_reg_targets)
      self.assertAllClose(reg_weights_out, exp_reg_weights)
      self.assertAllClose(matching_anchors_out, exp_matching_anchors)
      self.assertEquals(cls_targets_out.dtype, np.float32)
      self.assertEquals(cls_weights_out.dtype, np.float32)
      self.assertEquals(reg_targets_out.dtype, np.float32)
      self.assertEquals(reg_weights_out.dtype, np.float32)
      self.assertEquals(matching_anchors_out.dtype, np.int32)

  def test_raises_error_on_incompatible_groundtruth_boxes_and_labels(self):
    similarity_calc = region_similarity_calculator.NegSqDistSimilarity()
    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    box_coder = mean_stddev_box_coder.MeanStddevBoxCoder()
    unmatched_cls_target = tf.constant([1, 0, 0, 0, 0, 0, 0], tf.float32)
    target_assigner = targetassigner.TargetAssigner(
        similarity_calc, matcher, box_coder,
        unmatched_cls_target=unmatched_cls_target)

    prior_means = tf.constant([[0.0, 0.0, 0.5, 0.5],
                               [0.5, 0.5, 1.0, 0.8],
                               [0, 0.5, .5, 1.0],
                               [.75, 0, 1.0, .25]])
    prior_stddevs = tf.constant(4 * [4 * [.1]])
    priors = box_list.BoxList(prior_means)
    priors.add_field('stddev', prior_stddevs)

    box_corners = [[0.0, 0.0, 0.5, 0.5],
                   [0.0, 0.0, 0.5, 0.8],
                   [0.5, 0.5, 0.9, 0.9],
                   [.75, 0, .95, .27]]
    boxes = box_list.BoxList(tf.constant(box_corners))

    groundtruth_labels = tf.constant([[0, 1, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0],
                                      [0, 0, 0, 1, 0, 0, 0]], tf.float32)
    result = target_assigner.assign(priors, boxes, groundtruth_labels,
                                    num_valid_rows=3)
    (cls_targets, cls_weights, reg_targets, reg_weights, _) = result
    with self.test_session() as sess:
      with self.assertRaisesWithPredicateMatch(
          tf.errors.InvalidArgumentError,
          'Groundtruth boxes and labels have incompatible shapes!'):
        sess.run([cls_targets, cls_weights, reg_targets, reg_weights])

  def test_raises_error_on_invalid_groundtruth_labels(self):
    similarity_calc = region_similarity_calculator.NegSqDistSimilarity()
    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    box_coder = mean_stddev_box_coder.MeanStddevBoxCoder()
    unmatched_cls_target = tf.constant([[0, 0], [0, 0], [0, 0]], tf.float32)
    target_assigner = targetassigner.TargetAssigner(
        similarity_calc, matcher, box_coder,
        unmatched_cls_target=unmatched_cls_target)

    prior_means = tf.constant([[0.0, 0.0, 0.5, 0.5]])
    prior_stddevs = tf.constant([[1.0, 1.0, 1.0, 1.0]])
    priors = box_list.BoxList(prior_means)
    priors.add_field('stddev', prior_stddevs)

    box_corners = [[0.0, 0.0, 0.5, 0.5],
                   [0.5, 0.5, 0.9, 0.9],
                   [.75, 0, .95, .27]]
    boxes = box_list.BoxList(tf.constant(box_corners))

    groundtruth_labels = tf.constant([[[0, 1], [1, 0]]], tf.float32)

    with self.assertRaises(ValueError):
      target_assigner.assign(priors, boxes, groundtruth_labels,
                             num_valid_rows=3)


class BatchTargetAssignerTest(tf.test.TestCase):

  def _get_agnostic_target_assigner(self):
    similarity_calc = region_similarity_calculator.NegSqDistSimilarity()
    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    box_coder = mean_stddev_box_coder.MeanStddevBoxCoder()
    return targetassigner.TargetAssigner(
        similarity_calc, matcher, box_coder,
        positive_class_weight=1.0,
        negative_class_weight=1.0,
        unmatched_cls_target=None)

  def _get_multi_class_target_assigner(self, num_classes):
    similarity_calc = region_similarity_calculator.NegSqDistSimilarity()
    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    box_coder = mean_stddev_box_coder.MeanStddevBoxCoder()
    unmatched_cls_target = tf.constant([1] + num_classes * [0], tf.float32)
    return targetassigner.TargetAssigner(
        similarity_calc, matcher, box_coder,
        positive_class_weight=1.0,
        negative_class_weight=1.0,
        unmatched_cls_target=unmatched_cls_target)

  def _get_multi_dimensional_target_assigner(self, target_dimensions):
    similarity_calc = region_similarity_calculator.NegSqDistSimilarity()
    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    box_coder = mean_stddev_box_coder.MeanStddevBoxCoder()
    unmatched_cls_target = tf.constant(np.zeros(target_dimensions),
                                       tf.float32)
    return targetassigner.TargetAssigner(
        similarity_calc, matcher, box_coder,
        positive_class_weight=1.0,
        negative_class_weight=1.0,
        unmatched_cls_target=unmatched_cls_target)

  def test_batch_assign_targets(self):
    box_list1 = box_list.BoxList(tf.constant([[0., 0., 0.2, 0.2]]))
    box_list2 = box_list.BoxList(tf.constant(
        [[0, 0.25123152, 1, 1],
         [0.015789, 0.0985, 0.55789, 0.3842]]
    ))

    gt_box_batch = [box_list1, box_list2]
    gt_class_targets = [None, None]

    prior_means = tf.constant([[0, 0, .25, .25],
                               [0, .25, 1, 1],
                               [0, .1, .5, .5],
                               [.75, .75, 1, 1]])
    prior_stddevs = tf.constant([[.1, .1, .1, .1],
                                 [.1, .1, .1, .1],
                                 [.1, .1, .1, .1],
                                 [.1, .1, .1, .1]])
    priors = box_list.BoxList(prior_means)
    priors.add_field('stddev', prior_stddevs)

    exp_reg_targets = [[[0, 0, -0.5, -0.5],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0,],
                        [0, 0, 0, 0,],],
                       [[0, 0, 0, 0,],
                        [0, 0.01231521, 0, 0],
                        [0.15789001, -0.01500003, 0.57889998, -1.15799987],
                        [0, 0, 0, 0]]]
    exp_cls_weights = [[1, 1, 1, 1],
                       [1, 1, 1, 1]]
    exp_cls_targets = [[[1], [0], [0], [0]],
                       [[0], [1], [1], [0]]]
    exp_reg_weights = [[1, 0, 0, 0],
                       [0, 1, 1, 0]]
    exp_match_0 = [0]
    exp_match_1 = [1, 2]

    agnostic_target_assigner = self._get_agnostic_target_assigner()
    (cls_targets, cls_weights, reg_targets, reg_weights,
     match_list) = targetassigner.batch_assign_targets(
         agnostic_target_assigner, priors, gt_box_batch, gt_class_targets)
    self.assertTrue(isinstance(match_list, list) and len(match_list) == 2)
    with self.test_session() as sess:
      (cls_targets_out, cls_weights_out, reg_targets_out, reg_weights_out,
       match_out_0, match_out_1) = sess.run([
           cls_targets, cls_weights, reg_targets, reg_weights] + [
               match.matched_column_indices() for match in match_list])
      self.assertAllClose(cls_targets_out, exp_cls_targets)
      self.assertAllClose(cls_weights_out, exp_cls_weights)
      self.assertAllClose(reg_targets_out, exp_reg_targets)
      self.assertAllClose(reg_weights_out, exp_reg_weights)
      self.assertAllClose(match_out_0, exp_match_0)
      self.assertAllClose(match_out_1, exp_match_1)

  def test_batch_assign_multiclass_targets(self):
    box_list1 = box_list.BoxList(tf.constant([[0., 0., 0.2, 0.2]]))

    box_list2 = box_list.BoxList(tf.constant(
        [[0, 0.25123152, 1, 1],
         [0.015789, 0.0985, 0.55789, 0.3842]]
    ))

    gt_box_batch = [box_list1, box_list2]

    class_targets1 = tf.constant([[0, 1, 0, 0]], tf.float32)
    class_targets2 = tf.constant([[0, 0, 0, 1],
                                  [0, 0, 1, 0]], tf.float32)

    gt_class_targets = [class_targets1, class_targets2]

    prior_means = tf.constant([[0, 0, .25, .25],
                               [0, .25, 1, 1],
                               [0, .1, .5, .5],
                               [.75, .75, 1, 1]])
    prior_stddevs = tf.constant([[.1, .1, .1, .1],
                                 [.1, .1, .1, .1],
                                 [.1, .1, .1, .1],
                                 [.1, .1, .1, .1]])
    priors = box_list.BoxList(prior_means)
    priors.add_field('stddev', prior_stddevs)

    exp_reg_targets = [[[0, 0, -0.5, -0.5],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],
                       [[0, 0, 0, 0],
                        [0, 0.01231521, 0, 0],
                        [0.15789001, -0.01500003, 0.57889998, -1.15799987],
                        [0, 0, 0, 0]]]
    exp_cls_weights = [[1, 1, 1, 1],
                       [1, 1, 1, 1]]
    exp_cls_targets = [[[0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0]],
                       [[1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0]]]
    exp_reg_weights = [[1, 0, 0, 0],
                       [0, 1, 1, 0]]
    exp_match_0 = [0]
    exp_match_1 = [1, 2]

    multiclass_target_assigner = self._get_multi_class_target_assigner(
        num_classes=3)

    (cls_targets, cls_weights, reg_targets, reg_weights,
     match_list) = targetassigner.batch_assign_targets(
         multiclass_target_assigner, priors, gt_box_batch, gt_class_targets)
    self.assertTrue(isinstance(match_list, list) and len(match_list) == 2)
    with self.test_session() as sess:
      (cls_targets_out, cls_weights_out, reg_targets_out, reg_weights_out,
       match_out_0, match_out_1) = sess.run([
           cls_targets, cls_weights, reg_targets, reg_weights] + [
               match.matched_column_indices() for match in match_list])
      self.assertAllClose(cls_targets_out, exp_cls_targets)
      self.assertAllClose(cls_weights_out, exp_cls_weights)
      self.assertAllClose(reg_targets_out, exp_reg_targets)
      self.assertAllClose(reg_weights_out, exp_reg_weights)
      self.assertAllClose(match_out_0, exp_match_0)
      self.assertAllClose(match_out_1, exp_match_1)

  def test_batch_assign_multidimensional_targets(self):
    box_list1 = box_list.BoxList(tf.constant([[0., 0., 0.2, 0.2]]))

    box_list2 = box_list.BoxList(tf.constant(
        [[0, 0.25123152, 1, 1],
         [0.015789, 0.0985, 0.55789, 0.3842]]
    ))

    gt_box_batch = [box_list1, box_list2]
    class_targets1 = tf.constant([[[0, 1, 1],
                                   [1, 1, 0]]], tf.float32)
    class_targets2 = tf.constant([[[0, 1, 1],
                                   [1, 1, 0]],
                                  [[0, 0, 1],
                                   [0, 0, 1]]], tf.float32)

    gt_class_targets = [class_targets1, class_targets2]

    prior_means = tf.constant([[0, 0, .25, .25],
                               [0, .25, 1, 1],
                               [0, .1, .5, .5],
                               [.75, .75, 1, 1]])
    prior_stddevs = tf.constant([[.1, .1, .1, .1],
                                 [.1, .1, .1, .1],
                                 [.1, .1, .1, .1],
                                 [.1, .1, .1, .1]])
    priors = box_list.BoxList(prior_means)
    priors.add_field('stddev', prior_stddevs)

    exp_reg_targets = [[[0, 0, -0.5, -0.5],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],
                       [[0, 0, 0, 0],
                        [0, 0.01231521, 0, 0],
                        [0.15789001, -0.01500003, 0.57889998, -1.15799987],
                        [0, 0, 0, 0]]]
    exp_cls_weights = [[1, 1, 1, 1],
                       [1, 1, 1, 1]]

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
    exp_reg_weights = [[1, 0, 0, 0],
                       [0, 1, 1, 0]]
    exp_match_0 = [0]
    exp_match_1 = [1, 2]

    multiclass_target_assigner = self._get_multi_dimensional_target_assigner(
        target_dimensions=(2, 3))

    (cls_targets, cls_weights, reg_targets, reg_weights,
     match_list) = targetassigner.batch_assign_targets(
         multiclass_target_assigner, priors, gt_box_batch, gt_class_targets)
    self.assertTrue(isinstance(match_list, list) and len(match_list) == 2)
    with self.test_session() as sess:
      (cls_targets_out, cls_weights_out, reg_targets_out, reg_weights_out,
       match_out_0, match_out_1) = sess.run([
           cls_targets, cls_weights, reg_targets, reg_weights] + [
               match.matched_column_indices() for match in match_list])
      self.assertAllClose(cls_targets_out, exp_cls_targets)
      self.assertAllClose(cls_weights_out, exp_cls_weights)
      self.assertAllClose(reg_targets_out, exp_reg_targets)
      self.assertAllClose(reg_weights_out, exp_reg_weights)
      self.assertAllClose(match_out_0, exp_match_0)
      self.assertAllClose(match_out_1, exp_match_1)

  def test_batch_assign_empty_groundtruth(self):
    box_coords_expanded = tf.zeros((1, 4), tf.float32)
    box_coords = tf.slice(box_coords_expanded, [0, 0], [0, 4])
    box_list1 = box_list.BoxList(box_coords)
    gt_box_batch = [box_list1]

    prior_means = tf.constant([[0, 0, .25, .25],
                               [0, .25, 1, 1]])
    prior_stddevs = tf.constant([[.1, .1, .1, .1],
                                 [.1, .1, .1, .1]])
    priors = box_list.BoxList(prior_means)
    priors.add_field('stddev', prior_stddevs)

    exp_reg_targets = [[[0, 0, 0, 0],
                        [0, 0, 0, 0]]]
    exp_cls_weights = [[1, 1]]
    exp_cls_targets = [[[1, 0, 0, 0],
                        [1, 0, 0, 0]]]
    exp_reg_weights = [[0, 0]]
    exp_match_0 = []

    num_classes = 3
    pad = 1
    gt_class_targets = tf.zeros((0, num_classes + pad))
    gt_class_targets_batch = [gt_class_targets]

    multiclass_target_assigner = self._get_multi_class_target_assigner(
        num_classes=3)

    (cls_targets, cls_weights, reg_targets, reg_weights,
     match_list) = targetassigner.batch_assign_targets(
         multiclass_target_assigner, priors,
         gt_box_batch, gt_class_targets_batch)
    self.assertTrue(isinstance(match_list, list) and len(match_list) == 1)
    with self.test_session() as sess:
      (cls_targets_out, cls_weights_out, reg_targets_out, reg_weights_out,
       match_out_0) = sess.run([
           cls_targets, cls_weights, reg_targets, reg_weights] + [
               match.matched_column_indices() for match in match_list])
      self.assertAllClose(cls_targets_out, exp_cls_targets)
      self.assertAllClose(cls_weights_out, exp_cls_weights)
      self.assertAllClose(reg_targets_out, exp_reg_targets)
      self.assertAllClose(reg_weights_out, exp_reg_weights)
      self.assertAllClose(match_out_0, exp_match_0)


class CreateTargetAssignerTest(tf.test.TestCase):

  def test_create_target_assigner(self):
    """Tests that named constructor gives working target assigners.

    TODO: Make this test more general.
    """
    corners = [[0.0, 0.0, 1.0, 1.0]]
    groundtruth = box_list.BoxList(tf.constant(corners))

    priors = box_list.BoxList(tf.constant(corners))
    prior_stddevs = tf.constant([[1.0, 1.0, 1.0, 1.0]])
    priors.add_field('stddev', prior_stddevs)
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


if __name__ == '__main__':
  tf.test.main()
