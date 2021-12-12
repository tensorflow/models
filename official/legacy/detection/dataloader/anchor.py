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

"""Anchor box and labeler definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
from official.legacy.detection.utils import box_utils
from official.vision.beta.ops import iou_similarity
from official.vision.utils.object_detection import argmax_matcher
from official.vision.utils.object_detection import balanced_positive_negative_sampler
from official.vision.utils.object_detection import box_list
from official.vision.utils.object_detection import faster_rcnn_box_coder
from official.vision.utils.object_detection import target_assigner


class Anchor(object):
  """Anchor class for anchor-based object detectors."""

  def __init__(self, min_level, max_level, num_scales, aspect_ratios,
               anchor_size, image_size):
    """Constructs multiscale anchors.

    Args:
      min_level: integer number of minimum level of the output feature pyramid.
      max_level: integer number of maximum level of the output feature pyramid.
      num_scales: integer number representing intermediate scales added on each
        level. For instances, num_scales=2 adds one additional intermediate
        anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: list of float numbers representing the aspect ratio anchors
        added on each level. The number indicates the ratio of width to height.
        For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors on each
        scale level.
      anchor_size: float number representing the scale of size of the base
        anchor to the feature stride 2^level.
      image_size: a list of integer numbers or Tensors representing [height,
        width] of the input image size.The image_size should be divisible by the
        largest feature stride 2^max_level.
    """
    self.min_level = min_level
    self.max_level = max_level
    self.num_scales = num_scales
    self.aspect_ratios = aspect_ratios
    self.anchor_size = anchor_size
    self.image_size = image_size
    self.boxes = self._generate_boxes()

  def _generate_boxes(self):
    """Generates multiscale anchor boxes.

    Returns:
      a Tensor of shape [N, 4], represneting anchor boxes of all levels
      concatenated together.
    """
    boxes_all = []
    for level in range(self.min_level, self.max_level + 1):
      boxes_l = []
      for scale in range(self.num_scales):
        for aspect_ratio in self.aspect_ratios:
          stride = 2**level
          intermediate_scale = 2**(scale / float(self.num_scales))
          base_anchor_size = self.anchor_size * stride * intermediate_scale
          aspect_x = aspect_ratio**0.5
          aspect_y = aspect_ratio**-0.5
          half_anchor_size_x = base_anchor_size * aspect_x / 2.0
          half_anchor_size_y = base_anchor_size * aspect_y / 2.0
          x = tf.range(stride / 2, self.image_size[1], stride)
          y = tf.range(stride / 2, self.image_size[0], stride)
          xv, yv = tf.meshgrid(x, y)
          xv = tf.cast(tf.reshape(xv, [-1]), dtype=tf.float32)
          yv = tf.cast(tf.reshape(yv, [-1]), dtype=tf.float32)
          # Tensor shape Nx4.
          boxes = tf.stack([
              yv - half_anchor_size_y, xv - half_anchor_size_x,
              yv + half_anchor_size_y, xv + half_anchor_size_x
          ],
                           axis=1)
          boxes_l.append(boxes)
      # Concat anchors on the same level to tensor shape NxAx4.
      boxes_l = tf.stack(boxes_l, axis=1)
      boxes_l = tf.reshape(boxes_l, [-1, 4])
      boxes_all.append(boxes_l)
    return tf.concat(boxes_all, axis=0)

  def unpack_labels(self, labels):
    """Unpacks an array of labels into multiscales labels."""
    unpacked_labels = collections.OrderedDict()
    count = 0
    for level in range(self.min_level, self.max_level + 1):
      feat_size_y = tf.cast(self.image_size[0] / 2**level, tf.int32)
      feat_size_x = tf.cast(self.image_size[1] / 2**level, tf.int32)
      steps = feat_size_y * feat_size_x * self.anchors_per_location
      unpacked_labels[level] = tf.reshape(labels[count:count + steps],
                                          [feat_size_y, feat_size_x, -1])
      count += steps
    return unpacked_labels

  @property
  def anchors_per_location(self):
    return self.num_scales * len(self.aspect_ratios)

  @property
  def multilevel_boxes(self):
    return self.unpack_labels(self.boxes)


class AnchorLabeler(object):
  """Labeler for dense object detector."""

  def __init__(self, anchor, match_threshold=0.5, unmatched_threshold=0.5):
    """Constructs anchor labeler to assign labels to anchors.

    Args:
      anchor: an instance of class Anchors.
      match_threshold: a float number between 0 and 1 representing the
        lower-bound threshold to assign positive labels for anchors. An anchor
        with a score over the threshold is labeled positive.
      unmatched_threshold: a float number between 0 and 1 representing the
        upper-bound threshold to assign negative labels for anchors. An anchor
        with a score below the threshold is labeled negative.
    """
    similarity_calc = iou_similarity.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(
        match_threshold,
        unmatched_threshold=unmatched_threshold,
        negatives_lower_than_unmatched=True,
        force_match_for_each_row=True)
    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()

    self._target_assigner = target_assigner.TargetAssigner(
        similarity_calc, matcher, box_coder)
    self._anchor = anchor
    self._match_threshold = match_threshold
    self._unmatched_threshold = unmatched_threshold

  def label_anchors(self, gt_boxes, gt_labels):
    """Labels anchors with ground truth inputs.

    Args:
      gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.
        For each row, it stores [y0, x0, y1, x1] for four corners of a box.
      gt_labels: A integer tensor with shape [N, 1] representing groundtruth
        classes.

    Returns:
      cls_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors_per_location]. The height_l and
        width_l represent the dimension of class logits at l-th level.
      box_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors_per_location * 4]. The height_l
        and width_l represent the dimension of bounding box regression output at
        l-th level.
      num_positives: scalar tensor storing number of positives in an image.
    """
    gt_box_list = box_list.BoxList(gt_boxes)
    anchor_box_list = box_list.BoxList(self._anchor.boxes)

    # The cls_weights, box_weights are not used.
    cls_targets, _, box_targets, _, matches = self._target_assigner.assign(
        anchor_box_list, gt_box_list, gt_labels)

    # Labels definition in matches.match_results:
    # (1) match_results[i]>=0, meaning that column i is matched with row
    #     match_results[i].
    # (2) match_results[i]=-1, meaning that column i is not matched.
    # (3) match_results[i]=-2, meaning that column i is ignored.
    match_results = tf.expand_dims(matches.match_results, axis=1)
    cls_targets = tf.cast(cls_targets, tf.int32)
    cls_targets = tf.where(
        tf.equal(match_results, -1), -tf.ones_like(cls_targets), cls_targets)
    cls_targets = tf.where(
        tf.equal(match_results, -2), -2 * tf.ones_like(cls_targets),
        cls_targets)

    # Unpacks labels into multi-level representations.
    cls_targets_dict = self._anchor.unpack_labels(cls_targets)
    box_targets_dict = self._anchor.unpack_labels(box_targets)
    num_positives = tf.reduce_sum(
        input_tensor=tf.cast(tf.greater(matches.match_results, -1), tf.float32))

    return cls_targets_dict, box_targets_dict, num_positives


class RpnAnchorLabeler(AnchorLabeler):
  """Labeler for Region Proposal Network."""

  def __init__(self,
               anchor,
               match_threshold=0.7,
               unmatched_threshold=0.3,
               rpn_batch_size_per_im=256,
               rpn_fg_fraction=0.5):
    AnchorLabeler.__init__(
        self, anchor, match_threshold=0.7, unmatched_threshold=0.3)
    self._rpn_batch_size_per_im = rpn_batch_size_per_im
    self._rpn_fg_fraction = rpn_fg_fraction

  def _get_rpn_samples(self, match_results):
    """Computes anchor labels.

    This function performs subsampling for foreground (fg) and background (bg)
    anchors.
    Args:
      match_results: A integer tensor with shape [N] representing the matching
        results of anchors. (1) match_results[i]>=0, meaning that column i is
        matched with row match_results[i]. (2) match_results[i]=-1, meaning that
        column i is not matched. (3) match_results[i]=-2, meaning that column i
        is ignored.

    Returns:
      score_targets: a integer tensor with the a shape of [N].
        (1) score_targets[i]=1, the anchor is a positive sample.
        (2) score_targets[i]=0, negative. (3) score_targets[i]=-1, the anchor is
        don't care (ignore).
    """
    sampler = (
        balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(
            positive_fraction=self._rpn_fg_fraction, is_static=False))
    # indicator includes both positive and negative labels.
    # labels includes only positives labels.
    # positives = indicator & labels.
    # negatives = indicator & !labels.
    # ignore = !indicator.
    indicator = tf.greater(match_results, -2)
    labels = tf.greater(match_results, -1)

    samples = sampler.subsample(indicator, self._rpn_batch_size_per_im, labels)
    positive_labels = tf.where(
        tf.logical_and(samples, labels),
        tf.constant(2, dtype=tf.int32, shape=match_results.shape),
        tf.constant(0, dtype=tf.int32, shape=match_results.shape))
    negative_labels = tf.where(
        tf.logical_and(samples, tf.logical_not(labels)),
        tf.constant(1, dtype=tf.int32, shape=match_results.shape),
        tf.constant(0, dtype=tf.int32, shape=match_results.shape))
    ignore_labels = tf.fill(match_results.shape, -1)

    return (ignore_labels + positive_labels + negative_labels, positive_labels,
            negative_labels)

  def label_anchors(self, gt_boxes, gt_labels):
    """Labels anchors with ground truth inputs.

    Args:
      gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.
        For each row, it stores [y0, x0, y1, x1] for four corners of a box.
      gt_labels: A integer tensor with shape [N, 1] representing groundtruth
        classes.

    Returns:
      score_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors]. The height_l and width_l
        represent the dimension of class logits at l-th level.
      box_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors * 4]. The height_l and
        width_l represent the dimension of bounding box regression output at
        l-th level.
    """
    gt_box_list = box_list.BoxList(gt_boxes)
    anchor_box_list = box_list.BoxList(self._anchor.boxes)

    # cls_targets, cls_weights, box_weights are not used.
    _, _, box_targets, _, matches = self._target_assigner.assign(
        anchor_box_list, gt_box_list, gt_labels)

    # score_targets contains the subsampled positive and negative anchors.
    score_targets, _, _ = self._get_rpn_samples(matches.match_results)

    # Unpacks labels.
    score_targets_dict = self._anchor.unpack_labels(score_targets)
    box_targets_dict = self._anchor.unpack_labels(box_targets)

    return score_targets_dict, box_targets_dict


class OlnAnchorLabeler(RpnAnchorLabeler):
  """Labeler for Region Proposal Network."""

  def __init__(self,
               anchor,
               match_threshold=0.7,
               unmatched_threshold=0.3,
               rpn_batch_size_per_im=256,
               rpn_fg_fraction=0.5,
               has_centerness=False,
               center_match_iou_threshold=0.3,
               center_unmatched_iou_threshold=0.1,
               num_center_samples_per_im=256):
    """Constructs rpn anchor labeler to assign labels and centerness to anchors.

    Args:
      anchor: an instance of class Anchors.
      match_threshold: a float number between 0 and 1 representing the
        lower-bound threshold to assign positive labels for anchors. An anchor
        with a score over the threshold is labeled positive.
      unmatched_threshold: a float number between 0 and 1 representing the
        upper-bound threshold to assign negative labels for anchors. An anchor
        with a score below the threshold is labeled negative.
      rpn_batch_size_per_im: number of anchors that are sampled per image.
      rpn_fg_fraction:
      has_centerness: whether to include centerness target creation. An anchor
        is paired with one centerness score.
      center_match_iou_threshold: a float number between 0 and 1 representing
        the lower-bound threshold to sample foreground anchors for centerness
        regression. An anchor with a score over the threshold is sampled as
        foreground sample for centerness regression. We sample mostly from the
        foreground region (255 out of 256 samples). That is, we sample 255 vs 1
        (foreground vs background) anchor points to learn centerness regression.
      center_unmatched_iou_threshold: a float number between 0 and 1
        representing the lower-bound threshold to sample background anchors for
        centerness regression. An anchor with a score over the threshold is
        sampled as foreground sample for centerness regression. We sample very
        sparsely from the background region (1 out of 256 samples). That is, we
        sample 255 vs 1 (foreground vs background) anchor points to learn
        centerness regression.
      num_center_samples_per_im: number of anchor points per image that are
        sampled as centerness targets.
    """
    super(OlnAnchorLabeler, self).__init__(
        anchor, match_threshold=match_threshold,
        unmatched_threshold=unmatched_threshold,
        rpn_batch_size_per_im=rpn_batch_size_per_im,
        rpn_fg_fraction=rpn_fg_fraction)
    similarity_calc = iou_similarity.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(
        match_threshold,
        unmatched_threshold=unmatched_threshold,
        negatives_lower_than_unmatched=True,
        force_match_for_each_row=True)
    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()
    if has_centerness:
      center_matcher = argmax_matcher.ArgMaxMatcher(
          center_match_iou_threshold,
          unmatched_threshold=center_match_iou_threshold,
          negatives_lower_than_unmatched=True,
          force_match_for_each_row=True,)
    else:
      center_matcher = None

    self._target_assigner = target_assigner.OlnTargetAssigner(
        similarity_calc, matcher, box_coder,
        center_matcher=center_matcher)
    self._num_center_samples_per_im = num_center_samples_per_im
    self._center_unmatched_iou_threshold = center_unmatched_iou_threshold
    self._rpn_batch_size_per_im = rpn_batch_size_per_im
    self._rpn_fg_fraction = rpn_fg_fraction

  def label_anchors_lrtb(self, gt_boxes, gt_labels):
    """Labels anchors with ground truth inputs.

    Args:
      gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.
        For each row, it stores [y0, x0, y1, x1] for four corners of a box.
      gt_labels: A integer tensor with shape [N, 1] representing groundtruth
        classes.

    Returns:
      score_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors]. The height_l and width_l
        represent the dimension of class logits at l-th level.
      box_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors * 4]. The height_l and
        width_l represent the dimension of bounding box regression output at
        l-th level.
      lrtb_targets_dict: Same strucure to box_target_dict, except the regression
        targets are converted from xyhw to lrtb format. Ordered dictionary with
        keys [min_level, min_level+1, ..., max_level]. The values are tensor
        with shape [height_l, width_l, num_anchors * 4]. The height_l and
        width_l represent the dimension of bounding box regression output at
        l-th level.
      center_targets_dict: Same structure to score_tragets_dict, except the
        scores are centerness values ranging from 0 to 1. Ordered dictionary
        with keys [min_level, min_level+1, ..., max_level]. The values are
        tensor with shape [height_l, width_l, num_anchors]. The height_l and
        width_l represent the dimension of class logits at l-th level.
    """
    gt_box_list = box_list.BoxList(gt_boxes)
    anchor_box_list = box_list.BoxList(self._anchor.boxes)

    # cls_targets, cls_weights, box_weights are not used.
    (_, _, box_targets, _, matches,
     matched_gt_box_list, matched_anchors_mask,
     center_matched_gt_box_list, center_matched_anchors_mask,
     matched_ious) = self._target_assigner.assign(
         anchor_box_list, gt_box_list, gt_labels)
    # Box lrtb_targets.
    lrtb_targets, _ = box_utils.encode_boxes_lrtb(
        matched_gt_box_list.data['boxes'],
        anchor_box_list.data['boxes'],
        weights=[1.0, 1.0, 1.0, 1.0])
    lrtb_sanity = tf.logical_and(
        tf.greater(tf.reduce_min(lrtb_targets, -1), 0.),
        matched_anchors_mask)
    # To broadcast lrtb_sanity to the same shape as lrtb_targets.
    lrtb_sanity = tf.tile(tf.expand_dims(lrtb_sanity, 1),
                          [1, tf.shape(lrtb_targets)[1]])
    lrtb_targets = tf.where(lrtb_sanity,
                            lrtb_targets,
                            tf.zeros_like(lrtb_targets))
    # RPN anchor-gtbox iou values.
    iou_targets = tf.where(tf.greater(matched_ious, 0.0),
                           matched_ious,
                           tf.zeros_like(matched_ious))
    # Centerness_targets.
    _, center_targets = box_utils.encode_boxes_lrtb(
        center_matched_gt_box_list.data['boxes'],
        anchor_box_list.data['boxes'],
        weights=[1.0, 1.0, 1.0, 1.0])
    # Positive-negative centerness sampler.
    num_center_samples_per_im = self._num_center_samples_per_im
    center_pos_neg_sampler = (
        balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(
            positive_fraction=(1.- 1./num_center_samples_per_im),
            is_static=False))
    center_pos_neg_indicator = tf.logical_or(
        center_matched_anchors_mask,
        tf.less(iou_targets, self._center_unmatched_iou_threshold))
    center_pos_labels = center_matched_anchors_mask
    center_samples = center_pos_neg_sampler.subsample(
        center_pos_neg_indicator, num_center_samples_per_im, center_pos_labels)
    is_valid = center_samples
    center_targets = tf.where(is_valid,
                              center_targets,
                              (-1) * tf.ones_like(center_targets))

    # score_targets contains the subsampled positive and negative anchors.
    score_targets, _, _ = self._get_rpn_samples(matches.match_results)

    # Unpacks labels.
    score_targets_dict = self._anchor.unpack_labels(score_targets)
    box_targets_dict = self._anchor.unpack_labels(box_targets)
    lrtb_targets_dict = self._anchor.unpack_labels(lrtb_targets)
    center_targets_dict = self._anchor.unpack_labels(center_targets)

    return (score_targets_dict, box_targets_dict,
            lrtb_targets_dict, center_targets_dict)
