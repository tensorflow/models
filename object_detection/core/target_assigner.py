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

from object_detection.box_coders import faster_rcnn_box_coder
from object_detection.box_coders import mean_stddev_box_coder
from object_detection.core import box_coder as bcoder
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import matcher as mat
from object_detection.core import region_similarity_calculator as sim_calc
from object_detection.matchers import argmax_matcher
from object_detection.matchers import bipartite_matcher


class TargetAssigner(object):
  """Target assigner to compute classification and regression targets."""

  def __init__(self, similarity_calc, matcher, box_coder,
               positive_class_weight=1.0, negative_class_weight=1.0,
               unmatched_cls_target=None):
    """Construct Multibox Target Assigner.

    Args:
      similarity_calc: a RegionSimilarityCalculator
      matcher: an object_detection.core.Matcher used to match groundtruth to
        anchors.
      box_coder: an object_detection.core.BoxCoder used to encode matching
        groundtruth boxes with respect to anchors.
      positive_class_weight: classification weight to be associated to positive
        anchors (default: 1.0)
      negative_class_weight: classification weight to be associated to negative
        anchors (default: 1.0)
      unmatched_cls_target: a float32 tensor with shape [d_1, d_2, ..., d_k]
        which is consistent with the classification target for each
        anchor (and can be empty for scalar targets).  This shape must thus be
        compatible with the groundtruth labels that are passed to the "assign"
        function (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]).
        If set to None, unmatched_cls_target is set to be [0] for each anchor.

    Raises:
      ValueError: if similarity_calc is not a RegionSimilarityCalculator or
        if matcher is not a Matcher or if box_coder is not a BoxCoder
    """
    if not isinstance(similarity_calc, sim_calc.RegionSimilarityCalculator):
      raise ValueError('similarity_calc must be a RegionSimilarityCalculator')
    if not isinstance(matcher, mat.Matcher):
      raise ValueError('matcher must be a Matcher')
    if not isinstance(box_coder, bcoder.BoxCoder):
      raise ValueError('box_coder must be a BoxCoder')
    self._similarity_calc = similarity_calc
    self._matcher = matcher
    self._box_coder = box_coder
    self._positive_class_weight = positive_class_weight
    self._negative_class_weight = negative_class_weight
    if unmatched_cls_target is None:
      self._unmatched_cls_target = tf.constant([0], tf.float32)
    else:
      self._unmatched_cls_target = unmatched_cls_target

  @property
  def box_coder(self):
    return self._box_coder

  def assign(self, anchors, groundtruth_boxes, groundtruth_labels=None,
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
      groundtruth_labels:  a tensor of shape [num_gt_boxes, d_1, ... d_k]
        with labels for each of the ground_truth boxes. The subshape
        [d_1, ... d_k] can be empty (corresponding to scalar inputs).  When set
        to None, groundtruth_labels assumes a binary problem where all
        ground_truth boxes get a positive label (of 1).
      **params: Additional keyword arguments for specific implementations of
              the Matcher.

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
      groundtruth_labels = tf.ones(tf.expand_dims(groundtruth_boxes.num_boxes(),
                                                  0))
      groundtruth_labels = tf.expand_dims(groundtruth_labels, -1)
    shape_assert = tf.assert_equal(tf.shape(groundtruth_labels)[1:],
                                   tf.shape(self._unmatched_cls_target))

    with tf.control_dependencies([shape_assert]):
      match_quality_matrix = self._similarity_calc.compare(groundtruth_boxes,
                                                           anchors)
      match = self._matcher.match(match_quality_matrix, **params)
      reg_targets = self._create_regression_targets(anchors,
                                                    groundtruth_boxes,
                                                    match)
      cls_targets = self._create_classification_targets(groundtruth_labels,
                                                        match)
      reg_weights = self._create_regression_weights(match)
      cls_weights = self._create_classification_weights(
          match, self._positive_class_weight, self._negative_class_weight)

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
    matched_anchor_indices = match.matched_column_indices()
    unmatched_ignored_anchor_indices = (match.
                                        unmatched_or_ignored_column_indices())
    matched_gt_indices = match.matched_row_indices()
    matched_anchors = box_list_ops.gather(anchors,
                                          matched_anchor_indices)
    matched_gt_boxes = box_list_ops.gather(groundtruth_boxes,
                                           matched_gt_indices)
    matched_reg_targets = self._box_coder.encode(matched_gt_boxes,
                                                 matched_anchors)
    unmatched_ignored_reg_targets = tf.tile(
        self._default_regression_target(),
        tf.stack([tf.size(unmatched_ignored_anchor_indices), 1]))
    reg_targets = tf.dynamic_stitch(
        [matched_anchor_indices, unmatched_ignored_anchor_indices],
        [matched_reg_targets, unmatched_ignored_reg_targets])
    # TODO: summarize the number of matches on average.
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
    return tf.constant([self._box_coder.code_size*[0]], tf.float32)

  def _create_classification_targets(self, groundtruth_labels, match):
    """Create classification targets for each anchor.

    Assign a classification target of for each anchor to the matching
    groundtruth label that is provided by match.  Anchors that are not matched
    to anything are given the target self._unmatched_cls_target

    Args:
      groundtruth_labels:  a tensor of shape [num_gt_boxes, d_1, ... d_k]
        with labels for each of the ground_truth boxes. The subshape
        [d_1, ... d_k] can be empty (corresponding to scalar labels).
      match: a matcher.Match object that provides a matching between anchors
        and groundtruth boxes.

    Returns:
      cls_targets: a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k],
        where the subshape [d_1, ..., d_k] is compatible with groundtruth_labels
        which has shape [num_gt_boxes, d_1, d_2, ... d_k].
    """
    matched_anchor_indices = match.matched_column_indices()
    unmatched_ignored_anchor_indices = (match.
                                        unmatched_or_ignored_column_indices())
    matched_gt_indices = match.matched_row_indices()
    matched_cls_targets = tf.gather(groundtruth_labels, matched_gt_indices)

    ones = self._unmatched_cls_target.shape.ndims * [1]
    unmatched_ignored_cls_targets = tf.tile(
        tf.expand_dims(self._unmatched_cls_target, 0),
        tf.stack([tf.size(unmatched_ignored_anchor_indices)] + ones))

    cls_targets = tf.dynamic_stitch(
        [matched_anchor_indices, unmatched_ignored_anchor_indices],
        [matched_cls_targets, unmatched_ignored_cls_targets])
    return cls_targets

  def _create_regression_weights(self, match):
    """Set regression weight for each anchor.

    Only positive anchors are set to contribute to the regression loss, so this
    method returns a weight of 1 for every positive anchor and 0 for every
    negative anchor.

    Args:
      match: a matcher.Match object that provides a matching between anchors
        and groundtruth boxes.

    Returns:
      reg_weights: a float32 tensor with shape [num_anchors] representing
        regression weights
    """
    reg_weights = tf.cast(match.matched_column_indicator(), tf.float32)
    return reg_weights

  def _create_classification_weights(self,
                                     match,
                                     positive_class_weight=1.0,
                                     negative_class_weight=1.0):
    """Create classification weights for each anchor.

    Positive (matched) anchors are associated with a weight of
    positive_class_weight and negative (unmatched) anchors are associated with
    a weight of negative_class_weight. When anchors are ignored, weights are set
    to zero. By default, both positive/negative weights are set to 1.0,
    but they can be adjusted to handle class imbalance (which is almost always
    the case in object detection).

    Args:
      match: a matcher.Match object that provides a matching between anchors
        and groundtruth boxes.
      positive_class_weight: weight to be associated to positive anchors
      negative_class_weight: weight to be associated to negative anchors

    Returns:
      cls_weights: a float32 tensor with shape [num_anchors] representing
        classification weights.
    """
    matched_indicator = tf.cast(match.matched_column_indicator(), tf.float32)
    ignore_indicator = tf.cast(match.ignored_column_indicator(), tf.float32)
    unmatched_indicator = 1.0 - matched_indicator - ignore_indicator
    cls_weights = (positive_class_weight * matched_indicator
                   + negative_class_weight * unmatched_indicator)
    return cls_weights

  def get_box_coder(self):
    """Get BoxCoder of this TargetAssigner.

    Returns:
      BoxCoder: BoxCoder object.
    """
    return self._box_coder


# TODO: This method pulls in all the implementation dependencies into core.
# Therefore its best to have this factory method outside of core.
def create_target_assigner(reference, stage=None,
                           positive_class_weight=1.0,
                           negative_class_weight=1.0,
                           unmatched_cls_target=None):
  """Factory function for creating standard target assigners.

  Args:
    reference: string referencing the type of TargetAssigner.
    stage: string denoting stage: {proposal, detection}.
    positive_class_weight: classification weight to be associated to positive
      anchors (default: 1.0)
    negative_class_weight: classification weight to be associated to negative
      anchors (default: 1.0)
    unmatched_cls_target: a float32 tensor with shape [d_1, d_2, ..., d_k]
      which is consistent with the classification target for each
      anchor (and can be empty for scalar targets).  This shape must thus be
      compatible with the groundtruth labels that are passed to the Assign
      function (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]).
      If set to None, unmatched_cls_target is set to be 0 for each anchor.

  Returns:
    TargetAssigner: desired target assigner.

  Raises:
    ValueError: if combination reference+stage is invalid.
  """
  if reference == 'Multibox' and stage == 'proposal':
    similarity_calc = sim_calc.NegSqDistSimilarity()
    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    box_coder = mean_stddev_box_coder.MeanStddevBoxCoder()

  elif reference == 'FasterRCNN' and stage == 'proposal':
    similarity_calc = sim_calc.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.7,
                                           unmatched_threshold=0.3,
                                           force_match_for_each_row=True)
    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
        scale_factors=[10.0, 10.0, 5.0, 5.0])

  elif reference == 'FasterRCNN' and stage == 'detection':
    similarity_calc = sim_calc.IouSimilarity()
    # Uses all proposals with IOU < 0.5 as candidate negatives.
    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5,
                                           negatives_lower_than_unmatched=True)
    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
        scale_factors=[10.0, 10.0, 5.0, 5.0])

  elif reference == 'FastRCNN':
    similarity_calc = sim_calc.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5,
                                           unmatched_threshold=0.1,
                                           force_match_for_each_row=False,
                                           negatives_lower_than_unmatched=False)
    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()

  else:
    raise ValueError('No valid combination of reference and stage.')

  return TargetAssigner(similarity_calc, matcher, box_coder,
                        positive_class_weight=positive_class_weight,
                        negative_class_weight=negative_class_weight,
                        unmatched_cls_target=unmatched_cls_target)


def batch_assign_targets(target_assigner,
                         anchors_batch,
                         gt_box_batch,
                         gt_class_targets_batch):
  """Batched assignment of classification and regression targets.

  Args:
    target_assigner: a target assigner.
    anchors_batch: BoxList representing N box anchors or list of BoxList objects
      with length batch_size representing anchor sets.
    gt_box_batch: a list of BoxList objects with length batch_size
      representing groundtruth boxes for each image in the batch
    gt_class_targets_batch: a list of tensors with length batch_size, where
      each tensor has shape [num_gt_boxes_i, classification_target_size] and
      num_gt_boxes_i is the number of boxes in the ith boxlist of
      gt_box_batch.

  Returns:
    batch_cls_targets: a tensor with shape [batch_size, num_anchors,
      num_classes],
    batch_cls_weights: a tensor with shape [batch_size, num_anchors],
    batch_reg_targets: a tensor with shape [batch_size, num_anchors,
      box_code_dimension]
    batch_reg_weights: a tensor with shape [batch_size, num_anchors],
    match_list: a list of matcher.Match objects encoding the match between
      anchors and groundtruth boxes for each image of the batch,
      with rows of the Match objects corresponding to groundtruth boxes
      and columns corresponding to anchors.
  Raises:
    ValueError: if input list lengths are inconsistent, i.e.,
      batch_size == len(gt_box_batch) == len(gt_class_targets_batch)
        and batch_size == len(anchors_batch) unless anchors_batch is a single
        BoxList.
  """
  if not isinstance(anchors_batch, list):
    anchors_batch = len(gt_box_batch) * [anchors_batch]
  if not all(
      isinstance(anchors, box_list.BoxList) for anchors in anchors_batch):
    raise ValueError('anchors_batch must be a BoxList or list of BoxLists.')
  if not (len(anchors_batch)
          == len(gt_box_batch)
          == len(gt_class_targets_batch)):
    raise ValueError('batch size incompatible with lengths of anchors_batch, '
                     'gt_box_batch and gt_class_targets_batch.')
  cls_targets_list = []
  cls_weights_list = []
  reg_targets_list = []
  reg_weights_list = []
  match_list = []
  for anchors, gt_boxes, gt_class_targets in zip(
      anchors_batch, gt_box_batch, gt_class_targets_batch):
    (cls_targets, cls_weights, reg_targets,
     reg_weights, match) = target_assigner.assign(
         anchors, gt_boxes, gt_class_targets)
    cls_targets_list.append(cls_targets)
    cls_weights_list.append(cls_weights)
    reg_targets_list.append(reg_targets)
    reg_weights_list.append(reg_weights)
    match_list.append(match)
  batch_cls_targets = tf.stack(cls_targets_list)
  batch_cls_weights = tf.stack(cls_weights_list)
  batch_reg_targets = tf.stack(reg_targets_list)
  batch_reg_weights = tf.stack(reg_weights_list)
  return (batch_cls_targets, batch_cls_weights, batch_reg_targets,
          batch_reg_weights, match_list)
