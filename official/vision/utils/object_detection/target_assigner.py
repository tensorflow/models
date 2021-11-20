# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Base target assigner module.

The job of a TargetAssigner is, for a given set of anchors (bounding boxes) and
groundtruth detections (bounding boxes), to assign classification and regression
targets to each anchor as well as weights to each anchor (specifying, e.g.,
which anchors should not contribute to training loss).

It assigns classification/regression targets by performing the following steps:
1) Computing pairwise similarity between anchors and groundtruth boxes using a
  provided RegionSimilarity Calculator
2) Computing a matching based on the similarity matrix using a provided Matcher
3) Assigning regression targets based on the matching and a provided BoxCoder
4) Assigning classification targets based on the matching and groundtruth labels

Note that TargetAssigners only operate on detections from a single
image at a time, so any logic for applying a TargetAssigner to multiple
images must be handled externally.
"""

import tensorflow as tf

from official.vision.utils.object_detection import box_list
from official.vision.utils.object_detection import shape_utils

KEYPOINTS_FIELD_NAME = 'keypoints'


class TargetAssigner(object):
  """Target assigner to compute classification and regression targets."""

  def __init__(self,
               similarity_calc,
               matcher,
               box_coder,
               negative_class_weight=1.0,
               unmatched_cls_target=None):
    """Construct Object Detection Target Assigner.

    Args:
      similarity_calc: a RegionSimilarityCalculator
      matcher: Matcher used to match groundtruth to anchors.
      box_coder: BoxCoder used to encode matching groundtruth boxes with respect
        to anchors.
      negative_class_weight: classification weight to be associated to negative
        anchors (default: 1.0). The weight must be in [0., 1.].
      unmatched_cls_target: a float32 tensor with shape [d_1, d_2, ..., d_k]
        which is consistent with the classification target for each anchor (and
        can be empty for scalar targets).  This shape must thus be compatible
        with the groundtruth labels that are passed to the "assign" function
        (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]). If set to None,
        unmatched_cls_target is set to be [0] for each anchor.

    Raises:
      ValueError: if similarity_calc is not a RegionSimilarityCalculator or
        if matcher is not a Matcher or if box_coder is not a BoxCoder
    """
    self._similarity_calc = similarity_calc
    self._matcher = matcher
    self._box_coder = box_coder
    self._negative_class_weight = negative_class_weight
    if unmatched_cls_target is None:
      self._unmatched_cls_target = tf.constant([0], tf.float32)
    else:
      self._unmatched_cls_target = unmatched_cls_target

  @property
  def box_coder(self):
    return self._box_coder

  def assign(self,
             anchors,
             groundtruth_boxes,
             groundtruth_labels=None,
             groundtruth_weights=None,
             **params):
    """Assign classification and regression targets to each anchor.

    For a given set of anchors and groundtruth detections, match anchors
    to groundtruth_boxes and assign classification and regression targets to
    each anchor as well as weights based on the resulting match (specifying,
    e.g., which anchors should not contribute to training loss).

    Anchors that are not matched to anything are given a classification target
    of self._unmatched_cls_target which can be specified via the constructor.

    Args:
      anchors: a BoxList representing N anchors
      groundtruth_boxes: a BoxList representing M groundtruth boxes
      groundtruth_labels:  a tensor of shape [M, d_1, ... d_k] with labels for
        each of the ground_truth boxes. The subshape [d_1, ... d_k] can be empty
        (corresponding to scalar inputs).  When set to None, groundtruth_labels
        assumes a binary problem where all ground_truth boxes get a positive
        label (of 1).
      groundtruth_weights: a float tensor of shape [M] indicating the weight to
        assign to all anchors match to a particular groundtruth box. The weights
        must be in [0., 1.]. If None, all weights are set to 1.
      **params: Additional keyword arguments for specific implementations of the
        Matcher.

    Returns:
      cls_targets: a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k],
        where the subshape [d_1, ..., d_k] is compatible with groundtruth_labels
        which has shape [num_gt_boxes, d_1, d_2, ... d_k].
      cls_weights: a float32 tensor with shape [num_anchors]
      reg_targets: a float32 tensor with shape [num_anchors, box_code_dimension]
      reg_weights: a float32 tensor with shape [num_anchors]
      match: a matcher.Match object encoding the match between anchors and
        groundtruth boxes, with rows corresponding to groundtruth boxes
        and columns corresponding to anchors.

    Raises:
      ValueError: if anchors or groundtruth_boxes are not of type
        box_list.BoxList
    """
    if not isinstance(anchors, box_list.BoxList):
      raise ValueError('anchors must be an BoxList')
    if not isinstance(groundtruth_boxes, box_list.BoxList):
      raise ValueError('groundtruth_boxes must be an BoxList')

    if groundtruth_labels is None:
      groundtruth_labels = tf.ones(
          tf.expand_dims(groundtruth_boxes.num_boxes(), 0))
      groundtruth_labels = tf.expand_dims(groundtruth_labels, -1)
    unmatched_shape_assert = shape_utils.assert_shape_equal(
        shape_utils.combined_static_and_dynamic_shape(groundtruth_labels)[1:],
        shape_utils.combined_static_and_dynamic_shape(
            self._unmatched_cls_target))
    labels_and_box_shapes_assert = shape_utils.assert_shape_equal(
        shape_utils.combined_static_and_dynamic_shape(groundtruth_labels)[:1],
        shape_utils.combined_static_and_dynamic_shape(
            groundtruth_boxes.get())[:1])

    if groundtruth_weights is None:
      num_gt_boxes = groundtruth_boxes.num_boxes_static()
      if not num_gt_boxes:
        num_gt_boxes = groundtruth_boxes.num_boxes()
      groundtruth_weights = tf.ones([num_gt_boxes], dtype=tf.float32)
    with tf.control_dependencies(
        [unmatched_shape_assert, labels_and_box_shapes_assert]):
      match_quality_matrix = self._similarity_calc(
          groundtruth_boxes.get(), anchors.get())
      match = self._matcher.match(match_quality_matrix, **params)
      reg_targets = self._create_regression_targets(anchors, groundtruth_boxes,
                                                    match)
      cls_targets = self._create_classification_targets(groundtruth_labels,
                                                        match)
      reg_weights = self._create_regression_weights(match, groundtruth_weights)
      cls_weights = self._create_classification_weights(match,
                                                        groundtruth_weights)

    num_anchors = anchors.num_boxes_static()
    if num_anchors is not None:
      reg_targets = self._reset_target_shape(reg_targets, num_anchors)
      cls_targets = self._reset_target_shape(cls_targets, num_anchors)
      reg_weights = self._reset_target_shape(reg_weights, num_anchors)
      cls_weights = self._reset_target_shape(cls_weights, num_anchors)

    return cls_targets, cls_weights, reg_targets, reg_weights, match

  def _reset_target_shape(self, target, num_anchors):
    """Sets the static shape of the target.

    Args:
      target: the target tensor. Its first dimension will be overwritten.
      num_anchors: the number of anchors, which is used to override the target's
        first dimension.

    Returns:
      A tensor with the shape info filled in.
    """
    target_shape = target.get_shape().as_list()
    target_shape[0] = num_anchors
    target.set_shape(target_shape)
    return target

  def _create_regression_targets(self, anchors, groundtruth_boxes, match):
    """Returns a regression target for each anchor.

    Args:
      anchors: a BoxList representing N anchors
      groundtruth_boxes: a BoxList representing M groundtruth_boxes
      match: a matcher.Match object

    Returns:
      reg_targets: a float32 tensor with shape [N, box_code_dimension]
    """
    matched_gt_boxes = match.gather_based_on_match(
        groundtruth_boxes.get(),
        unmatched_value=tf.zeros(4),
        ignored_value=tf.zeros(4))
    matched_gt_boxlist = box_list.BoxList(matched_gt_boxes)
    if groundtruth_boxes.has_field(KEYPOINTS_FIELD_NAME):
      groundtruth_keypoints = groundtruth_boxes.get_field(KEYPOINTS_FIELD_NAME)
      matched_keypoints = match.gather_based_on_match(
          groundtruth_keypoints,
          unmatched_value=tf.zeros(groundtruth_keypoints.get_shape()[1:]),
          ignored_value=tf.zeros(groundtruth_keypoints.get_shape()[1:]))
      matched_gt_boxlist.add_field(KEYPOINTS_FIELD_NAME, matched_keypoints)
    matched_reg_targets = self._box_coder.encode(matched_gt_boxlist, anchors)
    match_results_shape = shape_utils.combined_static_and_dynamic_shape(
        match.match_results)

    # Zero out the unmatched and ignored regression targets.
    unmatched_ignored_reg_targets = tf.tile(self._default_regression_target(),
                                            [match_results_shape[0], 1])
    matched_anchors_mask = match.matched_column_indicator()
    # To broadcast matched_anchors_mask to the same shape as
    # matched_reg_targets.
    matched_anchors_mask = tf.tile(
        tf.expand_dims(matched_anchors_mask, 1),
        [1, tf.shape(matched_reg_targets)[1]])
    reg_targets = tf.where(matched_anchors_mask, matched_reg_targets,
                           unmatched_ignored_reg_targets)
    return reg_targets

  def _default_regression_target(self):
    """Returns the default target for anchors to regress to.

    Default regression targets are set to zero (though in
    this implementation what these targets are set to should
    not matter as the regression weight of any box set to
    regress to the default target is zero).

    Returns:
      default_target: a float32 tensor with shape [1, box_code_dimension]
    """
    return tf.constant([self._box_coder.code_size * [0]], tf.float32)

  def _create_classification_targets(self, groundtruth_labels, match):
    """Create classification targets for each anchor.

    Assign a classification target of for each anchor to the matching
    groundtruth label that is provided by match.  Anchors that are not matched
    to anything are given the target self._unmatched_cls_target

    Args:
      groundtruth_labels:  a tensor of shape [num_gt_boxes, d_1, ... d_k] with
        labels for each of the ground_truth boxes. The subshape [d_1, ... d_k]
        can be empty (corresponding to scalar labels).
      match: a matcher.Match object that provides a matching between anchors and
        groundtruth boxes.

    Returns:
      a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k], where the
      subshape [d_1, ..., d_k] is compatible with groundtruth_labels which has
      shape [num_gt_boxes, d_1, d_2, ... d_k].
    """
    return match.gather_based_on_match(
        groundtruth_labels,
        unmatched_value=self._unmatched_cls_target,
        ignored_value=self._unmatched_cls_target)

  def _create_regression_weights(self, match, groundtruth_weights):
    """Set regression weight for each anchor.

    Only positive anchors are set to contribute to the regression loss, so this
    method returns a weight of 1 for every positive anchor and 0 for every
    negative anchor.

    Args:
      match: a matcher.Match object that provides a matching between anchors and
        groundtruth boxes.
      groundtruth_weights: a float tensor of shape [M] indicating the weight to
        assign to all anchors match to a particular groundtruth box.

    Returns:
      a float32 tensor with shape [num_anchors] representing regression weights.
    """
    return match.gather_based_on_match(
        groundtruth_weights, ignored_value=0., unmatched_value=0.)

  def _create_classification_weights(self, match, groundtruth_weights):
    """Create classification weights for each anchor.

    Positive (matched) anchors are associated with a weight of
    positive_class_weight and negative (unmatched) anchors are associated with
    a weight of negative_class_weight. When anchors are ignored, weights are set
    to zero. By default, both positive/negative weights are set to 1.0,
    but they can be adjusted to handle class imbalance (which is almost always
    the case in object detection).

    Args:
      match: a matcher.Match object that provides a matching between anchors and
        groundtruth boxes.
      groundtruth_weights: a float tensor of shape [M] indicating the weight to
        assign to all anchors match to a particular groundtruth box.

    Returns:
      a float32 tensor with shape [num_anchors] representing classification
      weights.
    """
    return match.gather_based_on_match(
        groundtruth_weights,
        ignored_value=0.,
        unmatched_value=self._negative_class_weight)

  def get_box_coder(self):
    """Get BoxCoder of this TargetAssigner.

    Returns:
      BoxCoder object.
    """
    return self._box_coder


class OlnTargetAssigner(TargetAssigner):
  """Target assigner to compute classification and regression targets."""

  def __init__(self,
               similarity_calc,
               matcher,
               box_coder,
               negative_class_weight=1.0,
               unmatched_cls_target=None,
               center_matcher=None):
    """Construct Object Detection Target Assigner.

    Args:
      similarity_calc: a RegionSimilarityCalculator
      matcher: Matcher used to match groundtruth to anchors.
      box_coder: BoxCoder used to encode matching groundtruth boxes with respect
        to anchors.
      negative_class_weight: classification weight to be associated to negative
        anchors (default: 1.0). The weight must be in [0., 1.].
      unmatched_cls_target: a float32 tensor with shape [d_1, d_2, ..., d_k]
        which is consistent with the classification target for each anchor (and
        can be empty for scalar targets).  This shape must thus be compatible
        with the groundtruth labels that are passed to the "assign" function
        (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]). If set to None,
        unmatched_cls_target is set to be [0] for each anchor.
      center_matcher: Matcher used to match groundtruth to anchors to sample and
        assign the regression targets of centerness to each anchor.

    Raises:
      ValueError: if similarity_calc is not a RegionSimilarityCalculator or
        if matcher is not a Matcher or if box_coder is not a BoxCoder
    """
    super(OlnTargetAssigner, self).__init__(
        similarity_calc=similarity_calc,
        matcher=matcher,
        box_coder=box_coder,
        negative_class_weight=negative_class_weight,
        unmatched_cls_target=unmatched_cls_target)

    # centerness-matcher with independent sampling IoU threshold.
    self._center_matcher = center_matcher

  def assign(self,
             anchors,
             groundtruth_boxes,
             groundtruth_labels=None,
             groundtruth_weights=None,
             **params):
    """Assign classification and regression targets to each anchor.

    For a given set of anchors and groundtruth detections, match anchors
    to groundtruth_boxes and assign classification and regression targets to
    each anchor as well as weights based on the resulting match (specifying,
    e.g., which anchors should not contribute to training loss).

    Anchors that are not matched to anything are given a classification target
    of self._unmatched_cls_target which can be specified via the constructor.

    Args:
      anchors: a BoxList representing N anchors
      groundtruth_boxes: a BoxList representing M groundtruth boxes
      groundtruth_labels:  a tensor of shape [M, d_1, ... d_k] with labels for
        each of the ground_truth boxes. The subshape [d_1, ... d_k] can be empty
        (corresponding to scalar inputs).  When set to None, groundtruth_labels
        assumes a binary problem where all ground_truth boxes get a positive
        label (of 1).
      groundtruth_weights: a float tensor of shape [M] indicating the weight to
        assign to all anchors match to a particular groundtruth box. The weights
        must be in [0., 1.]. If None, all weights are set to 1.
      **params: Additional keyword arguments for specific implementations of the
        Matcher.

    Returns:
      cls_targets: a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k],
        where the subshape [d_1, ..., d_k] is compatible with groundtruth_labels
        which has shape [num_gt_boxes, d_1, d_2, ... d_k].
      cls_weights: a float32 tensor with shape [num_anchors]
      reg_targets: a float32 tensor with shape [num_anchors, box_code_dimension]
      reg_weights: a float32 tensor with shape [num_anchors]
      match: a matcher.Match object encoding the match between anchors and
        groundtruth boxes, with rows corresponding to groundtruth boxes
        and columns corresponding to anchors.
      matched_gt_boxlist: a BoxList object with data of float32 tensor with
        shape [num_anchors, box_dimension] which encodes the coordinates of the
        matched groundtruth boxes.
      matched_anchors_mask: a Bool tensor with shape [num_anchors] which
        indicates whether an anchor is matched or not.
      center_matched_gt_boxlist: a BoxList object with data of float32 tensor
        with shape [num_anchors, box_dimension] which encodes the coordinates of
        the groundtruth boxes matched for centerness target assignment.
      center_matched_anchors_mask: a Boolean tensor with shape [num_anchors]
        which indicates whether an anchor is matched or not for centerness
        target assignment.
      matched_ious: a float32 tensor with shape [num_anchors] which encodes the
        ious between each anchor and the matched groundtruth boxes.

    Raises:
      ValueError: if anchors or groundtruth_boxes are not of type
        box_list.BoxList
    """
    if not isinstance(anchors, box_list.BoxList):
      raise ValueError('anchors must be an BoxList')
    if not isinstance(groundtruth_boxes, box_list.BoxList):
      raise ValueError('groundtruth_boxes must be an BoxList')

    if groundtruth_labels is None:
      groundtruth_labels = tf.ones(
          tf.expand_dims(groundtruth_boxes.num_boxes(), 0))
      groundtruth_labels = tf.expand_dims(groundtruth_labels, -1)
    unmatched_shape_assert = shape_utils.assert_shape_equal(
        shape_utils.combined_static_and_dynamic_shape(groundtruth_labels)[1:],
        shape_utils.combined_static_and_dynamic_shape(
            self._unmatched_cls_target))
    labels_and_box_shapes_assert = shape_utils.assert_shape_equal(
        shape_utils.combined_static_and_dynamic_shape(groundtruth_labels)[:1],
        shape_utils.combined_static_and_dynamic_shape(
            groundtruth_boxes.get())[:1])

    if groundtruth_weights is None:
      num_gt_boxes = groundtruth_boxes.num_boxes_static()
      if not num_gt_boxes:
        num_gt_boxes = groundtruth_boxes.num_boxes()
      groundtruth_weights = tf.ones([num_gt_boxes], dtype=tf.float32)
    with tf.control_dependencies(
        [unmatched_shape_assert, labels_and_box_shapes_assert]):
      match_quality_matrix = self._similarity_calc(
          groundtruth_boxes.get(), anchors.get())
      match = self._matcher.match(match_quality_matrix, **params)
      reg_targets, matched_gt_boxlist, matched_anchors_mask = (
          self._create_regression_targets(anchors,
                                          groundtruth_boxes,
                                          match))
      cls_targets = self._create_classification_targets(groundtruth_labels,
                                                        match)
      reg_weights = self._create_regression_weights(match, groundtruth_weights)
      cls_weights = self._create_classification_weights(match,
                                                        groundtruth_weights)
      # Match for creation of centerness regression targets.
      if self._center_matcher is not None:
        center_match = self._center_matcher.match(
            match_quality_matrix, **params)
        center_matched_gt_boxes = center_match.gather_based_on_match(
            groundtruth_boxes.get(),
            unmatched_value=tf.zeros(4),
            ignored_value=tf.zeros(4))
        center_matched_gt_boxlist = box_list.BoxList(center_matched_gt_boxes)
        center_matched_anchors_mask = center_match.matched_column_indicator()

    num_anchors = anchors.num_boxes_static()
    if num_anchors is not None:
      reg_targets = self._reset_target_shape(reg_targets, num_anchors)
      cls_targets = self._reset_target_shape(cls_targets, num_anchors)
      reg_weights = self._reset_target_shape(reg_weights, num_anchors)
      cls_weights = self._reset_target_shape(cls_weights, num_anchors)

    if self._center_matcher is not None:
      matched_ious = tf.reduce_max(match_quality_matrix, 0)
      return (cls_targets, cls_weights, reg_targets, reg_weights, match,
              matched_gt_boxlist, matched_anchors_mask,
              center_matched_gt_boxlist, center_matched_anchors_mask,
              matched_ious)
    else:
      return (cls_targets, cls_weights, reg_targets, reg_weights, match)

  def _create_regression_targets(self, anchors, groundtruth_boxes, match):
    """Returns a regression target for each anchor.

    Args:
      anchors: a BoxList representing N anchors
      groundtruth_boxes: a BoxList representing M groundtruth_boxes
      match: a matcher.Match object

    Returns:
      reg_targets: a float32 tensor with shape [N, box_code_dimension]
    """
    matched_gt_boxes = match.gather_based_on_match(
        groundtruth_boxes.get(),
        unmatched_value=tf.zeros(4),
        ignored_value=tf.zeros(4))
    matched_gt_boxlist = box_list.BoxList(matched_gt_boxes)
    if groundtruth_boxes.has_field(KEYPOINTS_FIELD_NAME):
      groundtruth_keypoints = groundtruth_boxes.get_field(KEYPOINTS_FIELD_NAME)
      matched_keypoints = match.gather_based_on_match(
          groundtruth_keypoints,
          unmatched_value=tf.zeros(groundtruth_keypoints.get_shape()[1:]),
          ignored_value=tf.zeros(groundtruth_keypoints.get_shape()[1:]))
      matched_gt_boxlist.add_field(KEYPOINTS_FIELD_NAME, matched_keypoints)
    matched_reg_targets = self._box_coder.encode(matched_gt_boxlist, anchors)
    match_results_shape = shape_utils.combined_static_and_dynamic_shape(
        match.match_results)

    # Zero out the unmatched and ignored regression targets.
    unmatched_ignored_reg_targets = tf.tile(self._default_regression_target(),
                                            [match_results_shape[0], 1])
    matched_anchors_mask = match.matched_column_indicator()
    # To broadcast matched_anchors_mask to the same shape as
    # matched_reg_targets.
    matched_anchors_mask_tiled = tf.tile(
        tf.expand_dims(matched_anchors_mask, 1),
        [1, tf.shape(matched_reg_targets)[1]])
    reg_targets = tf.where(matched_anchors_mask_tiled,
                           matched_reg_targets,
                           unmatched_ignored_reg_targets)
    return reg_targets, matched_gt_boxlist, matched_anchors_mask
