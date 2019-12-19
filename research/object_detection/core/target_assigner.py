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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
from six.moves import zip
import tensorflow as tf

from object_detection.box_coders import faster_rcnn_box_coder
from object_detection.box_coders import mean_stddev_box_coder
from object_detection.core import box_coder
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import matcher as mat
from object_detection.core import region_similarity_calculator as sim_calc
from object_detection.core import standard_fields as fields
from object_detection.matchers import argmax_matcher
from object_detection.matchers import bipartite_matcher
from object_detection.utils import shape_utils


class TargetAssigner(object):
  """Target assigner to compute classification and regression targets."""

  def __init__(self,
               similarity_calc,
               matcher,
               box_coder_instance,
               negative_class_weight=1.0):
    """Construct Object Detection Target Assigner.

    Args:
      similarity_calc: a RegionSimilarityCalculator
      matcher: an object_detection.core.Matcher used to match groundtruth to
        anchors.
      box_coder_instance: an object_detection.core.BoxCoder used to encode
        matching groundtruth boxes with respect to anchors.
      negative_class_weight: classification weight to be associated to negative
        anchors (default: 1.0). The weight must be in [0., 1.].

    Raises:
      ValueError: if similarity_calc is not a RegionSimilarityCalculator or
        if matcher is not a Matcher or if box_coder is not a BoxCoder
    """
    if not isinstance(similarity_calc, sim_calc.RegionSimilarityCalculator):
      raise ValueError('similarity_calc must be a RegionSimilarityCalculator')
    if not isinstance(matcher, mat.Matcher):
      raise ValueError('matcher must be a Matcher')
    if not isinstance(box_coder_instance, box_coder.BoxCoder):
      raise ValueError('box_coder must be a BoxCoder')
    self._similarity_calc = similarity_calc
    self._matcher = matcher
    self._box_coder = box_coder_instance
    self._negative_class_weight = negative_class_weight

  @property
  def box_coder(self):
    return self._box_coder

  # TODO(rathodv): move labels, scores, and weights to groundtruth_boxes fields.
  def assign(self,
             anchors,
             groundtruth_boxes,
             groundtruth_labels=None,
             unmatched_class_label=None,
             groundtruth_weights=None):
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
      groundtruth_labels:  a tensor of shape [M, d_1, ... d_k]
        with labels for each of the ground_truth boxes. The subshape
        [d_1, ... d_k] can be empty (corresponding to scalar inputs).  When set
        to None, groundtruth_labels assumes a binary problem where all
        ground_truth boxes get a positive label (of 1).
      unmatched_class_label: a float32 tensor with shape [d_1, d_2, ..., d_k]
        which is consistent with the classification target for each
        anchor (and can be empty for scalar targets).  This shape must thus be
        compatible with the groundtruth labels that are passed to the "assign"
        function (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]).
        If set to None, unmatched_cls_target is set to be [0] for each anchor.
      groundtruth_weights: a float tensor of shape [M] indicating the weight to
        assign to all anchors match to a particular groundtruth box. The weights
        must be in [0., 1.]. If None, all weights are set to 1. Generally no
        groundtruth boxes with zero weight match to any anchors as matchers are
        aware of groundtruth weights. Additionally, `cls_weights` and
        `reg_weights` are calculated using groundtruth weights as an added
        safety.

    Returns:
      cls_targets: a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k],
        where the subshape [d_1, ..., d_k] is compatible with groundtruth_labels
        which has shape [num_gt_boxes, d_1, d_2, ... d_k].
      cls_weights: a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k],
        representing weights for each element in cls_targets.
      reg_targets: a float32 tensor with shape [num_anchors, box_code_dimension]
      reg_weights: a float32 tensor with shape [num_anchors]
      match: an int32 tensor of shape [num_anchors] containing result of anchor
        groundtruth matching. Each position in the tensor indicates an anchor
        and holds the following meaning:
        (1) if match[i] >= 0, anchor i is matched with groundtruth match[i].
        (2) if match[i]=-1, anchor i is marked to be background .
        (3) if match[i]=-2, anchor i is ignored since it is not background and
            does not have sufficient overlap to call it a foreground.

    Raises:
      ValueError: if anchors or groundtruth_boxes are not of type
        box_list.BoxList
    """
    if not isinstance(anchors, box_list.BoxList):
      raise ValueError('anchors must be an BoxList')
    if not isinstance(groundtruth_boxes, box_list.BoxList):
      raise ValueError('groundtruth_boxes must be an BoxList')

    if unmatched_class_label is None:
      unmatched_class_label = tf.constant([0], tf.float32)

    if groundtruth_labels is None:
      groundtruth_labels = tf.ones(tf.expand_dims(groundtruth_boxes.num_boxes(),
                                                  0))
      groundtruth_labels = tf.expand_dims(groundtruth_labels, -1)

    unmatched_shape_assert = shape_utils.assert_shape_equal(
        shape_utils.combined_static_and_dynamic_shape(groundtruth_labels)[1:],
        shape_utils.combined_static_and_dynamic_shape(unmatched_class_label))
    labels_and_box_shapes_assert = shape_utils.assert_shape_equal(
        shape_utils.combined_static_and_dynamic_shape(
            groundtruth_labels)[:1],
        shape_utils.combined_static_and_dynamic_shape(
            groundtruth_boxes.get())[:1])

    if groundtruth_weights is None:
      num_gt_boxes = groundtruth_boxes.num_boxes_static()
      if not num_gt_boxes:
        num_gt_boxes = groundtruth_boxes.num_boxes()
      groundtruth_weights = tf.ones([num_gt_boxes], dtype=tf.float32)

    # set scores on the gt boxes
    scores = 1 - groundtruth_labels[:, 0]
    groundtruth_boxes.add_field(fields.BoxListFields.scores, scores)

    with tf.control_dependencies(
        [unmatched_shape_assert, labels_and_box_shapes_assert]):
      match_quality_matrix = self._similarity_calc.compare(groundtruth_boxes,
                                                           anchors)
      match = self._matcher.match(match_quality_matrix,
                                  valid_rows=tf.greater(groundtruth_weights, 0))
      reg_targets = self._create_regression_targets(anchors,
                                                    groundtruth_boxes,
                                                    match)
      cls_targets = self._create_classification_targets(groundtruth_labels,
                                                        unmatched_class_label,
                                                        match)
      reg_weights = self._create_regression_weights(match, groundtruth_weights)

      cls_weights = self._create_classification_weights(match,
                                                        groundtruth_weights)
      # convert cls_weights from per-anchor to per-class.
      class_label_shape = tf.shape(cls_targets)[1:]
      weights_shape = tf.shape(cls_weights)
      weights_multiple = tf.concat(
          [tf.ones_like(weights_shape), class_label_shape],
          axis=0)
      for _ in range(len(cls_targets.get_shape()[1:])):
        cls_weights = tf.expand_dims(cls_weights, -1)
      cls_weights = tf.tile(cls_weights, weights_multiple)

    num_anchors = anchors.num_boxes_static()
    if num_anchors is not None:
      reg_targets = self._reset_target_shape(reg_targets, num_anchors)
      cls_targets = self._reset_target_shape(cls_targets, num_anchors)
      reg_weights = self._reset_target_shape(reg_weights, num_anchors)
      cls_weights = self._reset_target_shape(cls_weights, num_anchors)

    return (cls_targets, cls_weights, reg_targets, reg_weights,
            match.match_results)

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
    if groundtruth_boxes.has_field(fields.BoxListFields.keypoints):
      groundtruth_keypoints = groundtruth_boxes.get_field(
          fields.BoxListFields.keypoints)
      matched_keypoints = match.gather_based_on_match(
          groundtruth_keypoints,
          unmatched_value=tf.zeros(groundtruth_keypoints.get_shape()[1:]),
          ignored_value=tf.zeros(groundtruth_keypoints.get_shape()[1:]))
      matched_gt_boxlist.add_field(fields.BoxListFields.keypoints,
                                   matched_keypoints)
    matched_reg_targets = self._box_coder.encode(matched_gt_boxlist, anchors)
    match_results_shape = shape_utils.combined_static_and_dynamic_shape(
        match.match_results)

    # Zero out the unmatched and ignored regression targets.
    unmatched_ignored_reg_targets = tf.tile(
        self._default_regression_target(), [match_results_shape[0], 1])
    matched_anchors_mask = match.matched_column_indicator()
    reg_targets = tf.where(matched_anchors_mask,
                           matched_reg_targets,
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
    return tf.constant([self._box_coder.code_size*[0]], tf.float32)

  def _create_classification_targets(self, groundtruth_labels,
                                     unmatched_class_label, match):
    """Create classification targets for each anchor.

    Assign a classification target of for each anchor to the matching
    groundtruth label that is provided by match.  Anchors that are not matched
    to anything are given the target self._unmatched_cls_target

    Args:
      groundtruth_labels:  a tensor of shape [num_gt_boxes, d_1, ... d_k]
        with labels for each of the ground_truth boxes. The subshape
        [d_1, ... d_k] can be empty (corresponding to scalar labels).
      unmatched_class_label: a float32 tensor with shape [d_1, d_2, ..., d_k]
        which is consistent with the classification target for each
        anchor (and can be empty for scalar targets).  This shape must thus be
        compatible with the groundtruth labels that are passed to the "assign"
        function (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]).
      match: a matcher.Match object that provides a matching between anchors
        and groundtruth boxes.

    Returns:
      a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k], where the
      subshape [d_1, ..., d_k] is compatible with groundtruth_labels which has
      shape [num_gt_boxes, d_1, d_2, ... d_k].
    """
    return match.gather_based_on_match(
        groundtruth_labels,
        unmatched_value=unmatched_class_label,
        ignored_value=unmatched_class_label)

  def _create_regression_weights(self, match, groundtruth_weights):
    """Set regression weight for each anchor.

    Only positive anchors are set to contribute to the regression loss, so this
    method returns a weight of 1 for every positive anchor and 0 for every
    negative anchor.

    Args:
      match: a matcher.Match object that provides a matching between anchors
        and groundtruth boxes.
      groundtruth_weights: a float tensor of shape [M] indicating the weight to
        assign to all anchors match to a particular groundtruth box.

    Returns:
      a float32 tensor with shape [num_anchors] representing regression weights.
    """
    return match.gather_based_on_match(
        groundtruth_weights, ignored_value=0., unmatched_value=0.)

  def _create_classification_weights(self,
                                     match,
                                     groundtruth_weights):
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


# TODO(rathodv): This method pulls in all the implementation dependencies into
# core. Therefore its best to have this factory method outside of core.
def create_target_assigner(reference, stage=None,
                           negative_class_weight=1.0, use_matmul_gather=False):
  """Factory function for creating standard target assigners.

  Args:
    reference: string referencing the type of TargetAssigner.
    stage: string denoting stage: {proposal, detection}.
    negative_class_weight: classification weight to be associated to negative
      anchors (default: 1.0)
    use_matmul_gather: whether to use matrix multiplication based gather which
      are better suited for TPUs.

  Returns:
    TargetAssigner: desired target assigner.

  Raises:
    ValueError: if combination reference+stage is invalid.
  """
  if reference == 'Multibox' and stage == 'proposal':
    similarity_calc = sim_calc.NegSqDistSimilarity()
    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    box_coder_instance = mean_stddev_box_coder.MeanStddevBoxCoder()

  elif reference == 'FasterRCNN' and stage == 'proposal':
    similarity_calc = sim_calc.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.7,
                                           unmatched_threshold=0.3,
                                           force_match_for_each_row=True,
                                           use_matmul_gather=use_matmul_gather)
    box_coder_instance = faster_rcnn_box_coder.FasterRcnnBoxCoder(
        scale_factors=[10.0, 10.0, 5.0, 5.0])

  elif reference == 'FasterRCNN' and stage == 'detection':
    similarity_calc = sim_calc.IouSimilarity()
    # Uses all proposals with IOU < 0.5 as candidate negatives.
    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5,
                                           negatives_lower_than_unmatched=True,
                                           use_matmul_gather=use_matmul_gather)
    box_coder_instance = faster_rcnn_box_coder.FasterRcnnBoxCoder(
        scale_factors=[10.0, 10.0, 5.0, 5.0])

  elif reference == 'FastRCNN':
    similarity_calc = sim_calc.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5,
                                           unmatched_threshold=0.1,
                                           force_match_for_each_row=False,
                                           negatives_lower_than_unmatched=False,
                                           use_matmul_gather=use_matmul_gather)
    box_coder_instance = faster_rcnn_box_coder.FasterRcnnBoxCoder()

  else:
    raise ValueError('No valid combination of reference and stage.')

  return TargetAssigner(similarity_calc, matcher, box_coder_instance,
                        negative_class_weight=negative_class_weight)


def batch_assign(target_assigner,
                 anchors_batch,
                 gt_box_batch,
                 gt_class_targets_batch,
                 unmatched_class_label=None,
                 gt_weights_batch=None):
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
    unmatched_class_label: a float32 tensor with shape [d_1, d_2, ..., d_k]
      which is consistent with the classification target for each
      anchor (and can be empty for scalar targets).  This shape must thus be
      compatible with the groundtruth labels that are passed to the "assign"
      function (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]).
    gt_weights_batch: A list of 1-D tf.float32 tensors of shape
      [num_boxes] containing weights for groundtruth boxes.

  Returns:
    batch_cls_targets: a tensor with shape [batch_size, num_anchors,
      num_classes],
    batch_cls_weights: a tensor with shape [batch_size, num_anchors,
      num_classes],
    batch_reg_targets: a tensor with shape [batch_size, num_anchors,
      box_code_dimension]
    batch_reg_weights: a tensor with shape [batch_size, num_anchors],
    match: an int32 tensor of shape [batch_size, num_anchors] containing result
      of anchor groundtruth matching. Each position in the tensor indicates an
      anchor and holds the following meaning:
      (1) if match[x, i] >= 0, anchor i is matched with groundtruth match[x, i].
      (2) if match[x, i]=-1, anchor i is marked to be background .
      (3) if match[x, i]=-2, anchor i is ignored since it is not background and
          does not have sufficient overlap to call it a foreground.

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
  if gt_weights_batch is None:
    gt_weights_batch = [None] * len(gt_class_targets_batch)
  for anchors, gt_boxes, gt_class_targets, gt_weights in zip(
      anchors_batch, gt_box_batch, gt_class_targets_batch, gt_weights_batch):
    (cls_targets, cls_weights,
     reg_targets, reg_weights, match) = target_assigner.assign(
         anchors, gt_boxes, gt_class_targets, unmatched_class_label, gt_weights)
    cls_targets_list.append(cls_targets)
    cls_weights_list.append(cls_weights)
    reg_targets_list.append(reg_targets)
    reg_weights_list.append(reg_weights)
    match_list.append(match)
  batch_cls_targets = tf.stack(cls_targets_list)
  batch_cls_weights = tf.stack(cls_weights_list)
  batch_reg_targets = tf.stack(reg_targets_list)
  batch_reg_weights = tf.stack(reg_weights_list)
  batch_match = tf.stack(match_list)
  return (batch_cls_targets, batch_cls_weights, batch_reg_targets,
          batch_reg_weights, batch_match)


# Assign an alias to avoid large refactor of existing users.
batch_assign_targets = batch_assign


def batch_get_targets(batch_match, groundtruth_tensor_list,
                      groundtruth_weights_list, unmatched_value,
                      unmatched_weight):
  """Returns targets based on anchor-groundtruth box matching results.

  Args:
    batch_match: An int32 tensor of shape [batch, num_anchors] containing the
      result of target assignment returned by TargetAssigner.assign(..).
    groundtruth_tensor_list: A list of groundtruth tensors of shape
      [num_groundtruth, d_1, d_2, ..., d_k]. The tensors can be of any type.
    groundtruth_weights_list: A list of weights, one per groundtruth tensor, of
      shape [num_groundtruth].
    unmatched_value: A tensor of shape [d_1, d_2, ..., d_k] of the same type as
      groundtruth tensor containing target value for anchors that remain
      unmatched.
    unmatched_weight: Scalar weight to assign to anchors that remain unmatched.

  Returns:
    targets: A tensor of shape [batch, num_anchors, d_1, d_2, ..., d_k]
      containing targets for anchors.
    weights: A float tensor of shape [batch, num_anchors] containing the weights
      to assign to each target.
  """
  match_list = tf.unstack(batch_match)
  targets_list = []
  weights_list = []
  for match_tensor, groundtruth_tensor, groundtruth_weight in zip(
      match_list, groundtruth_tensor_list, groundtruth_weights_list):
    match_object = mat.Match(match_tensor)
    targets = match_object.gather_based_on_match(
        groundtruth_tensor,
        unmatched_value=unmatched_value,
        ignored_value=unmatched_value)
    targets_list.append(targets)
    weights = match_object.gather_based_on_match(
        groundtruth_weight,
        unmatched_value=unmatched_weight,
        ignored_value=tf.zeros_like(unmatched_weight))
    weights_list.append(weights)
  return tf.stack(targets_list), tf.stack(weights_list)


def batch_assign_confidences(target_assigner,
                             anchors_batch,
                             gt_box_batch,
                             gt_class_confidences_batch,
                             gt_weights_batch=None,
                             unmatched_class_label=None,
                             include_background_class=True,
                             implicit_class_weight=1.0):
  """Batched assignment of classification and regression targets.

  This differences between batch_assign_confidences and batch_assign_targets:
   - 'batch_assign_targets' supports scalar (agnostic), vector (multiclass) and
     tensor (high-dimensional) targets. 'batch_assign_confidences' only support
     scalar (agnostic) and vector (multiclass) targets.
   - 'batch_assign_targets' assumes the input class tensor using the binary
     one/K-hot encoding. 'batch_assign_confidences' takes the class confidence
     scores as the input, where 1 means positive classes, 0 means implicit
     negative classes, and -1 means explicit negative classes.
   - 'batch_assign_confidences' assigns the targets in the similar way as
     'batch_assign_targets' except that it gives different weights for implicit
     and explicit classes. This allows user to control the negative gradients
     pushed differently for implicit and explicit examples during the training.

  Args:
    target_assigner: a target assigner.
    anchors_batch: BoxList representing N box anchors or list of BoxList objects
      with length batch_size representing anchor sets.
    gt_box_batch: a list of BoxList objects with length batch_size
      representing groundtruth boxes for each image in the batch
    gt_class_confidences_batch: a list of tensors with length batch_size, where
      each tensor has shape [num_gt_boxes_i, classification_target_size] and
      num_gt_boxes_i is the number of boxes in the ith boxlist of
      gt_box_batch. Note that in this tensor, 1 means explicit positive class,
      -1 means explicit negative class, and 0 means implicit negative class.
    gt_weights_batch: A list of 1-D tf.float32 tensors of shape
      [num_gt_boxes_i] containing weights for groundtruth boxes.
    unmatched_class_label: a float32 tensor with shape [d_1, d_2, ..., d_k]
      which is consistent with the classification target for each
      anchor (and can be empty for scalar targets).  This shape must thus be
      compatible with the groundtruth labels that are passed to the "assign"
      function (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]).
    include_background_class: whether or not gt_class_confidences_batch includes
      the background class.
    implicit_class_weight: the weight assigned to implicit examples.

  Returns:
    batch_cls_targets: a tensor with shape [batch_size, num_anchors,
      num_classes],
    batch_cls_weights: a tensor with shape [batch_size, num_anchors,
      num_classes],
    batch_reg_targets: a tensor with shape [batch_size, num_anchors,
      box_code_dimension]
    batch_reg_weights: a tensor with shape [batch_size, num_anchors],
    match: an int32 tensor of shape [batch_size, num_anchors] containing result
      of anchor groundtruth matching. Each position in the tensor indicates an
      anchor and holds the following meaning:
      (1) if match[x, i] >= 0, anchor i is matched with groundtruth match[x, i].
      (2) if match[x, i]=-1, anchor i is marked to be background .
      (3) if match[x, i]=-2, anchor i is ignored since it is not background and
          does not have sufficient overlap to call it a foreground.

  Raises:
    ValueError: if input list lengths are inconsistent, i.e.,
      batch_size == len(gt_box_batch) == len(gt_class_targets_batch)
      and batch_size == len(anchors_batch) unless anchors_batch is a single
      BoxList, or if any element in gt_class_confidences_batch has rank > 2.
  """
  if not isinstance(anchors_batch, list):
    anchors_batch = len(gt_box_batch) * [anchors_batch]
  if not all(
      isinstance(anchors, box_list.BoxList) for anchors in anchors_batch):
    raise ValueError('anchors_batch must be a BoxList or list of BoxLists.')
  if not (len(anchors_batch)
          == len(gt_box_batch)
          == len(gt_class_confidences_batch)):
    raise ValueError('batch size incompatible with lengths of anchors_batch, '
                     'gt_box_batch and gt_class_confidences_batch.')

  cls_targets_list = []
  cls_weights_list = []
  reg_targets_list = []
  reg_weights_list = []
  match_list = []
  if gt_weights_batch is None:
    gt_weights_batch = [None] * len(gt_class_confidences_batch)
  for anchors, gt_boxes, gt_class_confidences, gt_weights in zip(
      anchors_batch, gt_box_batch, gt_class_confidences_batch,
      gt_weights_batch):

    if (gt_class_confidences is not None and
        len(gt_class_confidences.get_shape().as_list()) > 2):
      raise ValueError('The shape of the class target is not supported. ',
                       gt_class_confidences.get_shape())

    cls_targets, _, reg_targets, _, match = target_assigner.assign(
        anchors, gt_boxes, gt_class_confidences, unmatched_class_label,
        groundtruth_weights=gt_weights)

    if include_background_class:
      cls_targets_without_background = tf.slice(
          cls_targets, [0, 1], [-1, -1])
    else:
      cls_targets_without_background = cls_targets

    positive_mask = tf.greater(cls_targets_without_background, 0.0)
    negative_mask = tf.less(cls_targets_without_background, 0.0)
    explicit_example_mask = tf.logical_or(positive_mask, negative_mask)
    positive_anchors = tf.reduce_any(positive_mask, axis=-1)

    regression_weights = tf.cast(positive_anchors, dtype=tf.float32)
    regression_targets = (
        reg_targets * tf.expand_dims(regression_weights, axis=-1))
    regression_weights_expanded = tf.expand_dims(regression_weights, axis=-1)

    cls_targets_without_background = (
        cls_targets_without_background *
        (1 - tf.cast(negative_mask, dtype=tf.float32)))
    cls_weights_without_background = ((1 - implicit_class_weight) * tf.cast(
        explicit_example_mask, dtype=tf.float32) + implicit_class_weight)

    if include_background_class:
      cls_weights_background = (
          (1 - implicit_class_weight) * regression_weights_expanded
          + implicit_class_weight)
      classification_weights = tf.concat(
          [cls_weights_background, cls_weights_without_background], axis=-1)
      cls_targets_background = 1 - regression_weights_expanded
      classification_targets = tf.concat(
          [cls_targets_background, cls_targets_without_background], axis=-1)
    else:
      classification_targets = cls_targets_without_background
      classification_weights = cls_weights_without_background

    cls_targets_list.append(classification_targets)
    cls_weights_list.append(classification_weights)
    reg_targets_list.append(regression_targets)
    reg_weights_list.append(regression_weights)
    match_list.append(match)
  batch_cls_targets = tf.stack(cls_targets_list)
  batch_cls_weights = tf.stack(cls_weights_list)
  batch_reg_targets = tf.stack(reg_targets_list)
  batch_reg_weights = tf.stack(reg_weights_list)
  batch_match = tf.stack(match_list)
  return (batch_cls_targets, batch_cls_weights, batch_reg_targets,
          batch_reg_weights, batch_match)


