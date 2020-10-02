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
"""Anchor box and labeler definition."""

import collections
# Import libraries
import tensorflow as tf
from official.vision import keras_cv
from official.vision.detection.utils.object_detection import balanced_positive_negative_sampler
from official.vision.detection.utils.object_detection import box_list
from official.vision.detection.utils.object_detection import faster_rcnn_box_coder


class Anchor(object):
  """Anchor class for anchor-based object detectors."""

  def __init__(self,
               min_level,
               max_level,
               num_scales,
               aspect_ratios,
               anchor_size,
               image_size):
    """Constructs multiscale anchors.

    Args:
      min_level: integer number of minimum level of the output feature pyramid.
      max_level: integer number of maximum level of the output feature pyramid.
      num_scales: integer number representing intermediate scales added
        on each level. For instances, num_scales=2 adds one additional
        intermediate anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: list of float numbers representing the aspect raito anchors
        added on each level. The number indicates the ratio of width to height.
        For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors on each
        scale level.
      anchor_size: float number representing the scale of size of the base
        anchor to the feature stride 2^level.
      image_size: a list of integer numbers or Tensors representing
        [height, width] of the input image size.The image_size should be divided
        by the largest feature stride 2^max_level.
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
      a Tensor of shape [N, 4], representing anchor boxes of all levels
      concatenated together.
    """
    boxes_all = []
    for level in range(self.min_level, self.max_level + 1):
      boxes_l = []
      for scale in range(self.num_scales):
        for aspect_ratio in self.aspect_ratios:
          stride = 2 ** level
          intermidate_scale = 2 ** (scale / float(self.num_scales))
          base_anchor_size = self.anchor_size * stride * intermidate_scale
          aspect_x = aspect_ratio ** 0.5
          aspect_y = aspect_ratio ** -0.5
          half_anchor_size_x = base_anchor_size * aspect_x / 2.0
          half_anchor_size_y = base_anchor_size * aspect_y / 2.0
          x = tf.range(stride / 2, self.image_size[1], stride)
          y = tf.range(stride / 2, self.image_size[0], stride)
          xv, yv = tf.meshgrid(x, y)
          xv = tf.cast(tf.reshape(xv, [-1]), dtype=tf.float32)
          yv = tf.cast(tf.reshape(yv, [-1]), dtype=tf.float32)
          # Tensor shape Nx4.
          boxes = tf.stack([yv - half_anchor_size_y, xv - half_anchor_size_x,
                            yv + half_anchor_size_y, xv + half_anchor_size_x],
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
      feat_size_y = tf.cast(self.image_size[0] / 2 ** level, tf.int32)
      feat_size_x = tf.cast(self.image_size[1] / 2 ** level, tf.int32)
      steps = feat_size_y * feat_size_x * self.anchors_per_location
      unpacked_labels[str(level)] = tf.reshape(
          labels[count:count + steps], [feat_size_y, feat_size_x, -1])
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

  def __init__(self,
               match_threshold=0.5,
               unmatched_threshold=0.5):
    """Constructs anchor labeler to assign labels to anchors.

    Args:
      match_threshold: a float number between 0 and 1 representing the
        lower-bound threshold to assign positive labels for anchors. An anchor
        with a score over the threshold is labeled positive.
      unmatched_threshold: a float number between 0 and 1 representing the
        upper-bound threshold to assign negative labels for anchors. An anchor
        with a score below the threshold is labeled negative.
    """
    self.similarity_calc = keras_cv.ops.IouSimilarity()
    self.anchor_labeler = keras_cv.ops.AnchorLabeler()
    self.matcher = keras_cv.ops.BoxMatcher(
        positive_threshold=match_threshold,
        negative_threshold=unmatched_threshold,
        force_match_for_each_col=True)
    self.box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()

  def label_anchors(self, anchor_boxes, gt_boxes, gt_labels):
    """Labels anchors with ground truth inputs.

    Args:
      anchor_boxes: A float tensor with shape [N, 4] representing anchor boxes.
        For each row, it stores [y0, x0, y1, x1] for four corners of a box.
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
      cls_weights: A flattened Tensor with shape [batch_size, num_anchors], that
        serves as masking / sample weight for classification loss. Its value
        is 1.0 for positive and negative matched anchors, and 0.0 for ignored
        anchors.
      box_weights: A flattened Tensor with shape [batch_size, num_anchors], that
        serves as masking / sample weight for regression loss. Its value is
        1.0 for positive matched anchors, and 0.0 for negative and ignored
        anchors.
    """
    flattened_anchor_boxes = []
    for anchors in anchor_boxes.values():
      flattened_anchor_boxes.append(tf.reshape(anchors, [-1, 4]))
    flattened_anchor_boxes = tf.concat(flattened_anchor_boxes, axis=0)
    similarity_matrix = self.similarity_calc(flattened_anchor_boxes, gt_boxes)
    match_indices, match_indicators = self.matcher(similarity_matrix)
    mask = tf.less_equal(match_indicators, 0)
    cls_mask = tf.expand_dims(mask, -1)
    cls_targets = self.anchor_labeler(gt_labels, match_indices, cls_mask, -1)
    box_mask = tf.tile(cls_mask, [1, 4])
    box_targets = self.anchor_labeler(gt_boxes, match_indices, box_mask)
    weights = tf.squeeze(tf.ones_like(gt_labels, dtype=tf.float32), -1)
    box_weights = self.anchor_labeler(weights, match_indices, mask)
    ignore_mask = tf.equal(match_indicators, -2)
    cls_weights = self.anchor_labeler(weights, match_indices, ignore_mask)
    box_targets_list = box_list.BoxList(box_targets)
    anchor_box_list = box_list.BoxList(flattened_anchor_boxes)
    box_targets = self.box_coder.encode(box_targets_list, anchor_box_list)

    # Unpacks labels into multi-level representations.
    cls_targets_dict = unpack_targets(cls_targets, anchor_boxes)
    box_targets_dict = unpack_targets(box_targets, anchor_boxes)

    return cls_targets_dict, box_targets_dict, cls_weights, box_weights


class RpnAnchorLabeler(AnchorLabeler):
  """Labeler for Region Proposal Network."""

  def __init__(self,
               match_threshold=0.7,
               unmatched_threshold=0.3,
               rpn_batch_size_per_im=256,
               rpn_fg_fraction=0.5):
    AnchorLabeler.__init__(self, match_threshold=0.7, unmatched_threshold=0.3)
    self._rpn_batch_size_per_im = rpn_batch_size_per_im
    self._rpn_fg_fraction = rpn_fg_fraction

  def _get_rpn_samples(self, match_results):
    """Computes anchor labels.

    This function performs subsampling for foreground (fg) and background (bg)
    anchors.
    Args:
      match_results: A integer tensor with shape [N] representing the
        matching results of anchors. (1) match_results[i]>=0,
        meaning that column i is matched with row match_results[i].
        (2) match_results[i]=-1, meaning that column i is not matched.
        (3) match_results[i]=-2, meaning that column i is ignored.
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

    samples = sampler.subsample(
        indicator, self._rpn_batch_size_per_im, labels)
    positive_labels = tf.where(
        tf.logical_and(samples, labels),
        tf.constant(2, dtype=tf.int32, shape=match_results.shape),
        tf.constant(0, dtype=tf.int32, shape=match_results.shape))
    negative_labels = tf.where(
        tf.logical_and(samples, tf.logical_not(labels)),
        tf.constant(1, dtype=tf.int32, shape=match_results.shape),
        tf.constant(0, dtype=tf.int32, shape=match_results.shape))
    ignore_labels = tf.fill(match_results.shape, -1)

    return (ignore_labels + positive_labels + negative_labels,
            positive_labels, negative_labels)

  def label_anchors(self, anchor_boxes, gt_boxes, gt_labels):
    """Labels anchors with ground truth inputs.

    Args:
      anchor_boxes: A float tensor with shape [N, 4] representing anchor boxes.
        For each row, it stores [y0, x0, y1, x1] for four corners of a box.
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
    flattened_anchor_boxes = []
    for anchors in anchor_boxes.values():
      flattened_anchor_boxes.append(tf.reshape(anchors, [-1, 4]))
    flattened_anchor_boxes = tf.concat(flattened_anchor_boxes, axis=0)
    similarity_matrix = self.similarity_calc(flattened_anchor_boxes, gt_boxes)
    match_indices, match_indicators = self.matcher(similarity_matrix)
    box_mask = tf.tile(tf.expand_dims(tf.less_equal(match_indicators, 0), -1),
                       [1, 4])
    box_targets = self.anchor_labeler(gt_boxes, match_indices, box_mask)
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
        tf.expand_dims(matched_anchors_mask, 1),
        [1, tf.shape(box_targets)[1]])
    box_targets = tf.where(matched_anchors_mask, box_targets,
                           unmatched_ignored_box_targets)

    # score_targets contains the subsampled positive and negative anchors.
    score_targets, _, _ = self._get_rpn_samples(match_indicators)

    # Unpacks labels.
    score_targets_dict = unpack_targets(score_targets, anchor_boxes)
    box_targets_dict = unpack_targets(box_targets, anchor_boxes)

    return score_targets_dict, box_targets_dict


def build_anchor_generator(min_level, max_level, num_scales, aspect_ratios,
                           anchor_size):
  """Build anchor generator from levels."""
  anchor_sizes = collections.OrderedDict()
  strides = collections.OrderedDict()
  scales = []
  for scale in range(num_scales):
    scales.append(2**(scale / float(num_scales)))
  for level in range(min_level, max_level + 1):
    stride = 2**level
    strides[str(level)] = stride
    anchor_sizes[str(level)] = anchor_size * stride
  anchor_gen = keras_cv.ops.AnchorGenerator(
      anchor_sizes=anchor_sizes,
      scales=scales,
      aspect_ratios=aspect_ratios,
      strides=strides)
  return anchor_gen


def unpack_targets(targets, anchor_boxes_dict):
  """Unpacks an array of labels into multiscales labels."""
  unpacked_targets = collections.OrderedDict()
  count = 0
  for level, anchor_boxes in anchor_boxes_dict.items():
    feat_size_shape = anchor_boxes.shape.as_list()
    feat_size_y = feat_size_shape[0]
    feat_size_x = feat_size_shape[1]
    anchors_per_location = int(feat_size_shape[2] / 4)
    steps = feat_size_y * feat_size_x * anchors_per_location
    unpacked_targets[level] = tf.reshape(targets[count:count + steps],
                                         [feat_size_y, feat_size_x, -1])
    count += steps
  return unpacked_targets
