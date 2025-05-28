# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

import collections
import math
from typing import Dict, Optional, Tuple

import tensorflow as tf, tf_keras

from official.vision.ops import box_matcher
from official.vision.ops import iou_similarity
from official.vision.ops import target_gather
from official.vision.utils.object_detection import balanced_positive_negative_sampler
from official.vision.utils.object_detection import box_list
from official.vision.utils.object_detection import faster_rcnn_box_coder


class Anchor(object):
  """Anchor class for anchor-based object detectors.

  Example:
  ```python
  anchor_boxes = Anchor(
      min_level=3,
      max_level=4,
      num_scales=2,
      aspect_ratios=[0.5, 1., 2.],
      anchor_size=4.,
      image_size=[256, 256],
  ).multilevel_boxes
  ```

  Attributes:
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
      width] of the input image size.
    multilevel_boxes: an OrderedDict from level to the generated anchor boxes of
      shape [height_l, width_l, num_anchors_per_location * 4].
    anchors_per_location: number of anchors per pixel location.
  """

  def __init__(
      self,
      min_level,
      max_level,
      num_scales,
      aspect_ratios,
      anchor_size,
      image_size,
  ):
    """Initializes the instance."""
    self.min_level = min_level
    self.max_level = max_level
    self.num_scales = num_scales
    self.aspect_ratios = aspect_ratios
    self.anchor_size = anchor_size
    self.image_size = image_size
    self.multilevel_boxes = self._generate_multilevel_boxes()

  def _generate_multilevel_boxes(self) -> Dict[str, tf.Tensor]:
    """Generates multi-scale anchor boxes.

    Returns:
      An OrderedDict from level to anchor boxes of shape [height_l, width_l,
      num_anchors_per_location * 4].
    """
    multilevel_boxes = collections.OrderedDict()
    for level in range(self.min_level, self.max_level + 1):
      boxes_l = []
      feat_size_y = math.ceil(self.image_size[0] / 2**level)
      feat_size_x = math.ceil(self.image_size[1] / 2**level)
      stride_y = tf.cast(self.image_size[0] / feat_size_y, tf.float32)
      stride_x = tf.cast(self.image_size[1] / feat_size_x, tf.float32)
      x = tf.range(stride_x / 2, self.image_size[1], stride_x)
      y = tf.range(stride_y / 2, self.image_size[0], stride_y)
      xv, yv = tf.meshgrid(x, y)
      for scale in range(self.num_scales):
        for aspect_ratio in self.aspect_ratios:
          intermidate_scale = 2 ** (scale / self.num_scales)
          base_anchor_size = self.anchor_size * 2**level * intermidate_scale
          aspect_x = aspect_ratio**0.5
          aspect_y = aspect_ratio**-0.5
          half_anchor_size_x = base_anchor_size * aspect_x / 2.0
          half_anchor_size_y = base_anchor_size * aspect_y / 2.0
          # Tensor shape Nx4.
          boxes = tf.stack(
              [
                  yv - half_anchor_size_y,
                  xv - half_anchor_size_x,
                  yv + half_anchor_size_y,
                  xv + half_anchor_size_x,
              ],
              axis=-1,
          )
          boxes_l.append(boxes)
      # Concat anchors on the same level to tensor shape HxWx(Ax4).
      boxes_l = tf.concat(boxes_l, axis=-1)
      multilevel_boxes[str(level)] = boxes_l
    return multilevel_boxes

  @property
  def anchors_per_location(self) -> int:
    return self.num_scales * len(self.aspect_ratios)


class AnchorLabeler(object):
  """Labeler for dense object detector."""

  def __init__(
      self,
      match_threshold=0.5,
      unmatched_threshold=0.5,
      box_coder_weights=None,
  ):
    """Constructs anchor labeler to assign labels to anchors.

    Args:
      match_threshold: a float number between 0 and 1 representing the
        lower-bound threshold to assign positive labels for anchors. An anchor
        with a score over the threshold is labeled positive.
      unmatched_threshold: a float number between 0 and 1 representing the
        upper-bound threshold to assign negative labels for anchors. An anchor
        with a score below the threshold is labeled negative.
      box_coder_weights: Optional `list` of 4 positive floats to scale y, x, h,
        and w when encoding box coordinates. If set to None, does not perform
        scaling. For Faster RCNN, the open-source implementation recommends
        using [10.0, 10.0, 5.0, 5.0].
    """
    self.similarity_calc = iou_similarity.IouSimilarity()
    self.target_gather = target_gather.TargetGather()
    self.matcher = box_matcher.BoxMatcher(
        thresholds=[unmatched_threshold, match_threshold],
        indicators=[-1, -2, 1],
        force_match_for_each_col=True,
    )
    self.box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
        scale_factors=box_coder_weights,
    )

  def label_anchors(
      self,
      anchor_boxes: Dict[str, tf.Tensor],
      gt_boxes: tf.Tensor,
      gt_labels: tf.Tensor,
      gt_attributes: Optional[Dict[str, tf.Tensor]] = None,
      gt_weights: Optional[tf.Tensor] = None,
  ) -> Tuple[
      Dict[str, tf.Tensor],
      Dict[str, tf.Tensor],
      Dict[str, Dict[str, tf.Tensor]],
      tf.Tensor,
      tf.Tensor,
  ]:
    """Labels anchors with ground truth inputs.

    Args:
      anchor_boxes: An ordered dictionary with keys [min_level, min_level+1,
        ..., max_level]. The values are tensor with shape [height_l, width_l,
        num_anchors_per_location * 4]. The height_l and width_l represent the
        dimension of the feature pyramid at l-th level. For each anchor box, the
        tensor stores [y0, x0, y1, x1] for the four corners.
      gt_boxes: A float tensor with shape [N, 4] representing ground-truth
        boxes. For each row, it stores [y0, x0, y1, x1] for four corners of a
        box.
      gt_labels: A integer tensor with shape [N, 1] representing ground-truth
        classes.
      gt_attributes: If not None, a dict of (name, gt_attribute) pairs.
        `gt_attribute` is a float tensor with shape [N, attribute_size]
        representing ground-truth attributes.
      gt_weights: If not None, a float tensor with shape [N] representing
        ground-truth weights.

    Returns:
      cls_targets_dict: An ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors_per_location]. The height_l and
        width_l represent the dimension of class logits at l-th level.
      box_targets_dict: An ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors_per_location * 4]. The height_l
        and width_l represent the dimension of bounding box regression output at
        l-th level.
      attribute_targets_dict: A dict with (name, attribute_targets) pairs. Each
        `attribute_targets` represents an ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors_per_location * attribute_size].
        The height_l and width_l represent the dimension of attribute prediction
        output at l-th level.
      cls_weights: A flattened Tensor with shape [num_anchors], that serves as
        masking / sample weight for classification loss. Its value is 1.0 for
        positive and negative matched anchors, and 0.0 for ignored anchors.
      box_weights: A flattened Tensor with shape [num_anchors], that serves as
        masking / sample weight for regression loss. Its value is 1.0 for
        positive matched anchors, and 0.0 for negative and ignored anchors.
    """
    flattened_anchor_boxes = []
    for anchors in anchor_boxes.values():
      flattened_anchor_boxes.append(tf.reshape(anchors, [-1, 4]))
    flattened_anchor_boxes = tf.concat(flattened_anchor_boxes, axis=0)
    similarity_matrix = self.similarity_calc(flattened_anchor_boxes, gt_boxes)
    match_indices, match_indicators = self.matcher(similarity_matrix)

    mask = tf.less_equal(match_indicators, 0)
    cls_mask = tf.expand_dims(mask, -1)
    cls_targets = self.target_gather(gt_labels, match_indices, cls_mask, -1)
    box_mask = tf.tile(cls_mask, [1, 4])
    box_targets = self.target_gather(gt_boxes, match_indices, box_mask)
    att_targets = {}
    if gt_attributes:
      for k, v in gt_attributes.items():
        att_size = v.get_shape().as_list()[-1]
        att_mask = tf.tile(cls_mask, [1, att_size])
        att_targets[k] = self.target_gather(v, match_indices, att_mask, 0.0)

    # When there is no ground truth labels, we force the weight to be 1 so that
    # negative matched anchors get non-zero weights.
    num_gt_labels = tf.shape(gt_labels)[0]
    weights = tf.cond(
        tf.greater(num_gt_labels, 0),
        lambda: tf.ones_like(gt_labels, dtype=tf.float32)[..., -1],
        lambda: tf.ones([1], dtype=tf.float32),
    )
    if gt_weights is not None:
      weights = tf.cond(
          tf.greater(num_gt_labels, 0),
          lambda: tf.math.multiply(weights, gt_weights),
          lambda: weights,
      )
    box_weights = self.target_gather(weights, match_indices, mask)
    ignore_mask = tf.equal(match_indicators, -2)
    cls_weights = self.target_gather(weights, match_indices, ignore_mask)
    box_targets = box_list.BoxList(box_targets)
    anchor_box = box_list.BoxList(flattened_anchor_boxes)
    box_targets = self.box_coder.encode(box_targets, anchor_box)

    # Unpacks labels into multi-level representations.
    cls_targets = unpack_targets(cls_targets, anchor_boxes)
    box_targets = unpack_targets(box_targets, anchor_boxes)
    attribute_targets = {
        k: unpack_targets(v, anchor_boxes) for k, v in att_targets.items()
    }

    return (
        cls_targets,
        box_targets,
        attribute_targets,
        cls_weights,
        box_weights,
    )


class RpnAnchorLabeler(AnchorLabeler):
  """Labeler for Region Proposal Network."""

  def __init__(
      self,
      match_threshold=0.7,
      unmatched_threshold=0.3,
      rpn_batch_size_per_im=256,
      rpn_fg_fraction=0.5,
  ):
    AnchorLabeler.__init__(
        self,
        match_threshold=match_threshold,
        unmatched_threshold=unmatched_threshold,
    )
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
            positive_fraction=self._rpn_fg_fraction, is_static=False
        )
    )
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
        tf.constant(0, dtype=tf.int32, shape=match_results.shape),
    )
    negative_labels = tf.where(
        tf.logical_and(samples, tf.logical_not(labels)),
        tf.constant(1, dtype=tf.int32, shape=match_results.shape),
        tf.constant(0, dtype=tf.int32, shape=match_results.shape),
    )
    ignore_labels = tf.fill(match_results.shape, -1)

    return (
        ignore_labels + positive_labels + negative_labels,
        positive_labels,
        negative_labels,
    )

  def label_anchors(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self,
      anchor_boxes: Dict[str, tf.Tensor],
      gt_boxes: tf.Tensor,
      gt_labels: tf.Tensor,
  ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """Labels anchors with ground truth inputs.

    Args:
      anchor_boxes: An ordered dictionary with keys [min_level, min_level+1,
        ..., max_level]. The values are tensor with shape [height_l, width_l,
        num_anchors_per_location * 4]. The height_l and width_l represent the
        dimension of the feature pyramid at l-th level. For each anchor box, the
        tensor stores [y0, x0, y1, x1] for the four corners.
      gt_boxes: A float tensor with shape [N, 4] representing ground-truth
        boxes. For each row, it stores [y0, x0, y1, x1] for four corners of a
        box.
      gt_labels: A integer tensor with shape [N, 1] representing ground-truth
        classes.

    Returns:
      score_targets_dict: An ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors_per_location]. The height_l and
        width_l represent the dimension of class logits at l-th level.
      box_targets_dict: An ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors_per_location * 4]. The height_l
        and width_l represent the dimension of bounding box regression output at
        l-th level.
    """
    flattened_anchor_boxes = []
    for anchors in anchor_boxes.values():
      flattened_anchor_boxes.append(tf.reshape(anchors, [-1, 4]))
    flattened_anchor_boxes = tf.concat(flattened_anchor_boxes, axis=0)
    similarity_matrix = self.similarity_calc(flattened_anchor_boxes, gt_boxes)
    match_indices, match_indicators = self.matcher(similarity_matrix)
    box_mask = tf.tile(
        tf.expand_dims(tf.less_equal(match_indicators, 0), -1), [1, 4]
    )
    box_targets = self.target_gather(gt_boxes, match_indices, box_mask)
    box_targets_list = box_list.BoxList(box_targets)
    anchor_box_list = box_list.BoxList(flattened_anchor_boxes)
    box_targets = self.box_coder.encode(box_targets_list, anchor_box_list)

    # Zero out the unmatched and ignored regression targets.
    num_matches = match_indices.shape.as_list()[0] or tf.shape(match_indices)[0]
    unmatched_ignored_box_targets = tf.zeros([num_matches, 4], dtype=tf.float32)
    matched_anchors_mask = tf.greater_equal(match_indicators, 0)
    # To broadcast matched_anchors_mask to the same shape as
    # matched_reg_targets.
    matched_anchors_mask = tf.tile(
        tf.expand_dims(matched_anchors_mask, 1), [1, tf.shape(box_targets)[1]]
    )
    box_targets = tf.where(
        matched_anchors_mask, box_targets, unmatched_ignored_box_targets
    )

    # score_targets contains the subsampled positive and negative anchors.
    score_targets, _, _ = self._get_rpn_samples(match_indicators)

    # Unpacks labels.
    score_targets_dict = unpack_targets(score_targets, anchor_boxes)
    box_targets_dict = unpack_targets(box_targets, anchor_boxes)

    return score_targets_dict, box_targets_dict


class AnchorGeneratorv2:
  """Utility to generate anchors for a multiple feature maps.

  Attributes:
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
  """

  def __init__(
      self,
      min_level,
      max_level,
      num_scales,
      aspect_ratios,
      anchor_size,
    ):
    """Initializes the instance."""
    self.min_level = min_level
    self.max_level = max_level
    self.num_scales = num_scales
    self.aspect_ratios = aspect_ratios
    self.anchor_size = anchor_size

  def __call__(self, image_size):
    """Generate multilevel anchor boxes.

    Args:
      image_size: a list of integer numbers or Tensors representing [height,
        width] of the input image size.
    Returns:
      An ordered dictionary from level to anchor boxes of shape [height_l,
      width_l, num_anchors_per_location * 4].
    """
    return Anchor(
        min_level=self.min_level,
        max_level=self.max_level,
        num_scales=self.num_scales,
        aspect_ratios=self.aspect_ratios,
        anchor_size=self.anchor_size,
        image_size=image_size,
    ).multilevel_boxes


def build_anchor_generator(
    min_level, max_level, num_scales, aspect_ratios, anchor_size
):
  """Build anchor generator from levels."""
  anchor_gen = AnchorGeneratorv2(
      min_level=min_level,
      max_level=max_level,
      num_scales=num_scales,
      aspect_ratios=aspect_ratios,
      anchor_size=anchor_size,
  )
  return anchor_gen


def unpack_targets(
    targets: tf.Tensor, anchor_boxes_dict: Dict[str, tf.Tensor]
) -> Dict[str, tf.Tensor]:
  """Unpacks an array of labels into multi-scales labels.

  Args:
    targets: A tensor with shape [num_anchors, M] representing the packed
      targets with M values stored for each anchor.
    anchor_boxes_dict: An ordered dictionary with keys [min_level, min_level+1,
      ..., max_level]. The values are tensor with shape [height_l, width_l,
      num_anchors_per_location * 4]. The height_l and width_l represent the
      dimension of the feature pyramid at l-th level. For each anchor box, the
      tensor stores [y0, x0, y1, x1] for the four corners.

  Returns:
    unpacked_targets: An ordered dictionary with keys
      [min_level, min_level+1, ..., max_level]. The values are tensor with shape
      [height_l, width_l, num_anchors_per_location * M]. The height_l and
      width_l represent the dimension of the feature pyramid at l-th level. M is
      the number of values stored for each anchor.
  """
  unpacked_targets = collections.OrderedDict()
  count = 0
  for level, anchor_boxes in anchor_boxes_dict.items():
    feat_size_shape = anchor_boxes.shape.as_list()
    feat_size_y = feat_size_shape[0]
    feat_size_x = feat_size_shape[1]
    anchors_per_location = int(feat_size_shape[2] / 4)
    steps = feat_size_y * feat_size_x * anchors_per_location
    unpacked_targets[level] = tf.reshape(
        targets[count : count + steps], [feat_size_y, feat_size_x, -1]
    )
    count += steps
  return unpacked_targets
