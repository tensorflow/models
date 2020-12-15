# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Target and sampling related ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from official.vision.detection.ops import spatial_transform_ops
from official.vision.detection.utils import box_utils
from official.vision.detection.utils.object_detection import balanced_positive_negative_sampler


def box_matching(boxes, gt_boxes, gt_classes):
  """Match boxes to groundtruth boxes.

  Given the proposal boxes and the groundtruth boxes and classes, perform the
  groundtruth matching by taking the argmax of the IoU between boxes and
  groundtruth boxes.

  Args:
    boxes: a tensor of shape of [batch_size, N, 4] representing the box
      coordiantes to be matched to groundtruth boxes.
    gt_boxes: a tensor of shape of [batch_size, MAX_INSTANCES, 4] representing
      the groundtruth box coordinates. It is padded with -1s to indicate the
      invalid boxes.
    gt_classes: [batch_size, MAX_INSTANCES] representing the groundtruth box
      classes. It is padded with -1s to indicate the invalid classes.

  Returns:
    matched_gt_boxes: a tensor of shape of [batch_size, N, 4], representing
      the matched groundtruth box coordinates for each input box. If the box
      does not overlap with any groundtruth boxes, the matched boxes of it
      will be set to all 0s.
    matched_gt_classes: a tensor of shape of [batch_size, N], representing
      the matched groundtruth classes for each input box. If the box does not
      overlap with any groundtruth boxes, the matched box classes of it will
      be set to 0, which corresponds to the background class.
    matched_gt_indices: a tensor of shape of [batch_size, N], representing
      the indices of the matched groundtruth boxes in the original gt_boxes
      tensor. If the box does not overlap with any groundtruth boxes, the
      index of the matched groundtruth will be set to -1.
    matched_iou: a tensor of shape of [batch_size, N], representing the IoU
      between the box and its matched groundtruth box. The matched IoU is the
      maximum IoU of the box and all the groundtruth boxes.
    iou: a tensor of shape of [batch_size, N, K], representing the IoU matrix
      between boxes and the groundtruth boxes. The IoU between a box and the
      invalid groundtruth boxes whose coordinates are [-1, -1, -1, -1] is -1.
  """
  # Compute IoU between boxes and gt_boxes.
  # iou <- [batch_size, N, K]
  iou = box_utils.bbox_overlap(boxes, gt_boxes)

  # max_iou <- [batch_size, N]
  # 0.0 -> no match to gt, or -1.0 match to no gt
  matched_iou = tf.reduce_max(iou, axis=-1)

  # background_box_mask <- bool, [batch_size, N]
  background_box_mask = tf.less_equal(matched_iou, 0.0)

  argmax_iou_indices = tf.argmax(iou, axis=-1, output_type=tf.int32)

  argmax_iou_indices_shape = tf.shape(argmax_iou_indices)
  batch_indices = (
      tf.expand_dims(tf.range(argmax_iou_indices_shape[0]), axis=-1) *
      tf.ones([1, argmax_iou_indices_shape[-1]], dtype=tf.int32))
  gather_nd_indices = tf.stack([batch_indices, argmax_iou_indices], axis=-1)

  matched_gt_boxes = tf.gather_nd(gt_boxes, gather_nd_indices)
  matched_gt_boxes = tf.where(
      tf.tile(tf.expand_dims(background_box_mask, axis=-1), [1, 1, 4]),
      tf.zeros_like(matched_gt_boxes, dtype=matched_gt_boxes.dtype),
      matched_gt_boxes)

  matched_gt_classes = tf.gather_nd(gt_classes, gather_nd_indices)
  matched_gt_classes = tf.where(background_box_mask,
                                tf.zeros_like(matched_gt_classes),
                                matched_gt_classes)

  matched_gt_indices = tf.where(background_box_mask,
                                -tf.ones_like(argmax_iou_indices),
                                argmax_iou_indices)

  return (matched_gt_boxes, matched_gt_classes, matched_gt_indices, matched_iou,
          iou)


def assign_and_sample_proposals(proposed_boxes,
                                gt_boxes,
                                gt_classes,
                                num_samples_per_image=512,
                                mix_gt_boxes=True,
                                fg_fraction=0.25,
                                fg_iou_thresh=0.5,
                                bg_iou_thresh_hi=0.5,
                                bg_iou_thresh_lo=0.0):
  """Assigns the proposals with groundtruth classes and performs subsmpling.

  Given `proposed_boxes`, `gt_boxes`, and `gt_classes`, the function uses the
  following algorithm to generate the final `num_samples_per_image` RoIs.
    1. Calculates the IoU between each proposal box and each gt_boxes.
    2. Assigns each proposed box with a groundtruth class and box by choosing
       the largest IoU overlap.
    3. Samples `num_samples_per_image` boxes from all proposed boxes, and
       returns box_targets, class_targets, and RoIs.

  Args:
    proposed_boxes: a tensor of shape of [batch_size, N, 4]. N is the number of
      proposals before groundtruth assignment. The last dimension is the box
      coordinates w.r.t. the scaled images in [ymin, xmin, ymax, xmax] format.
    gt_boxes: a tensor of shape of [batch_size, MAX_NUM_INSTANCES, 4]. The
      coordinates of gt_boxes are in the pixel coordinates of the scaled image.
      This tensor might have padding of values -1 indicating the invalid box
      coordinates.
    gt_classes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES]. This
      tensor might have paddings with values of -1 indicating the invalid
      classes.
    num_samples_per_image: a integer represents RoI minibatch size per image.
    mix_gt_boxes: a bool indicating whether to mix the groundtruth boxes before
      sampling proposals.
    fg_fraction: a float represents the target fraction of RoI minibatch that is
      labeled foreground (i.e., class > 0).
    fg_iou_thresh: a float represents the IoU overlap threshold for an RoI to be
      considered foreground (if >= fg_iou_thresh).
    bg_iou_thresh_hi: a float represents the IoU overlap threshold for an RoI to
      be considered background (class = 0 if overlap in [LO, HI)).
    bg_iou_thresh_lo: a float represents the IoU overlap threshold for an RoI to
      be considered background (class = 0 if overlap in [LO, HI)).

  Returns:
    sampled_rois: a tensor of shape of [batch_size, K, 4], representing the
      coordinates of the sampled RoIs, where K is the number of the sampled
      RoIs, i.e. K = num_samples_per_image.
    sampled_gt_boxes: a tensor of shape of [batch_size, K, 4], storing the
      box coordinates of the matched groundtruth boxes of the samples RoIs.
    sampled_gt_classes: a tensor of shape of [batch_size, K], storing the
      classes of the matched groundtruth boxes of the sampled RoIs.
    sampled_gt_indices: a tensor of shape of [batch_size, K], storing the
      indices of the sampled groudntruth boxes in the original `gt_boxes`
      tensor, i.e. gt_boxes[sampled_gt_indices[:, i]] = sampled_gt_boxes[:, i].
  """

  with tf.name_scope('sample_proposals'):
    if mix_gt_boxes:
      boxes = tf.concat([proposed_boxes, gt_boxes], axis=1)
    else:
      boxes = proposed_boxes

    (matched_gt_boxes, matched_gt_classes, matched_gt_indices, matched_iou,
     _) = box_matching(boxes, gt_boxes, gt_classes)

    positive_match = tf.greater(matched_iou, fg_iou_thresh)
    negative_match = tf.logical_and(
        tf.greater_equal(matched_iou, bg_iou_thresh_lo),
        tf.less(matched_iou, bg_iou_thresh_hi))
    ignored_match = tf.less(matched_iou, 0.0)

    # re-assign negatively matched boxes to the background class.
    matched_gt_classes = tf.where(negative_match,
                                  tf.zeros_like(matched_gt_classes),
                                  matched_gt_classes)
    matched_gt_indices = tf.where(negative_match,
                                  tf.zeros_like(matched_gt_indices),
                                  matched_gt_indices)

    sample_candidates = tf.logical_and(
        tf.logical_or(positive_match, negative_match),
        tf.logical_not(ignored_match))

    sampler = (
        balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(
            positive_fraction=fg_fraction, is_static=True))

    batch_size, _ = sample_candidates.get_shape().as_list()
    sampled_indicators = []
    for i in range(batch_size):
      sampled_indicator = sampler.subsample(sample_candidates[i],
                                            num_samples_per_image,
                                            positive_match[i])
      sampled_indicators.append(sampled_indicator)
    sampled_indicators = tf.stack(sampled_indicators)
    _, sampled_indices = tf.nn.top_k(
        tf.cast(sampled_indicators, dtype=tf.int32),
        k=num_samples_per_image,
        sorted=True)

    sampled_indices_shape = tf.shape(sampled_indices)
    batch_indices = (
        tf.expand_dims(tf.range(sampled_indices_shape[0]), axis=-1) *
        tf.ones([1, sampled_indices_shape[-1]], dtype=tf.int32))
    gather_nd_indices = tf.stack([batch_indices, sampled_indices], axis=-1)

    sampled_rois = tf.gather_nd(boxes, gather_nd_indices)
    sampled_gt_boxes = tf.gather_nd(matched_gt_boxes, gather_nd_indices)
    sampled_gt_classes = tf.gather_nd(matched_gt_classes, gather_nd_indices)
    sampled_gt_indices = tf.gather_nd(matched_gt_indices, gather_nd_indices)

    return (sampled_rois, sampled_gt_boxes, sampled_gt_classes,
            sampled_gt_indices)


def sample_and_crop_foreground_masks(candidate_rois,
                                     candidate_gt_boxes,
                                     candidate_gt_classes,
                                     candidate_gt_indices,
                                     gt_masks,
                                     num_mask_samples_per_image=128,
                                     mask_target_size=28):
  """Samples and creates cropped foreground masks for training.

  Args:
    candidate_rois: a tensor of shape of [batch_size, N, 4], where N is the
      number of candidate RoIs to be considered for mask sampling. It includes
      both positive and negative RoIs. The `num_mask_samples_per_image` positive
      RoIs will be sampled to create mask training targets.
    candidate_gt_boxes: a tensor of shape of [batch_size, N, 4], storing the
      corresponding groundtruth boxes to the `candidate_rois`.
    candidate_gt_classes: a tensor of shape of [batch_size, N], storing the
      corresponding groundtruth classes to the `candidate_rois`. 0 in the tensor
      corresponds to the background class, i.e. negative RoIs.
    candidate_gt_indices: a tensor of shape [batch_size, N], storing the
      corresponding groundtruth instance indices to the `candidate_gt_boxes`,
      i.e. gt_boxes[candidate_gt_indices[:, i]] = candidate_gt_boxes[:, i] and
        gt_boxes which is of shape [batch_size, MAX_INSTANCES, 4], M >= N, is
        the superset of candidate_gt_boxes.
    gt_masks: a tensor of [batch_size, MAX_INSTANCES, mask_height, mask_width]
      containing all the groundtruth masks which sample masks are drawn from.
    num_mask_samples_per_image: an integer which specifies the number of masks
      to sample.
    mask_target_size: an integer which specifies the final cropped mask size
      after sampling. The output masks are resized w.r.t the sampled RoIs.

  Returns:
    foreground_rois: a tensor of shape of [batch_size, K, 4] storing the RoI
      that corresponds to the sampled foreground masks, where
      K = num_mask_samples_per_image.
    foreground_classes: a tensor of shape of [batch_size, K] storing the classes
      corresponding to the sampled foreground masks.
    cropoped_foreground_masks: a tensor of shape of
      [batch_size, K, mask_target_size, mask_target_size] storing the cropped
      foreground masks used for training.
  """
  with tf.name_scope('sample_and_crop_foreground_masks'):
    _, fg_instance_indices = tf.nn.top_k(
        tf.cast(tf.greater(candidate_gt_classes, 0), dtype=tf.int32),
        k=num_mask_samples_per_image)

    fg_instance_indices_shape = tf.shape(fg_instance_indices)
    batch_indices = (
        tf.expand_dims(tf.range(fg_instance_indices_shape[0]), axis=-1) *
        tf.ones([1, fg_instance_indices_shape[-1]], dtype=tf.int32))

    gather_nd_instance_indices = tf.stack([batch_indices, fg_instance_indices],
                                          axis=-1)
    foreground_rois = tf.gather_nd(candidate_rois, gather_nd_instance_indices)
    foreground_boxes = tf.gather_nd(candidate_gt_boxes,
                                    gather_nd_instance_indices)
    foreground_classes = tf.gather_nd(candidate_gt_classes,
                                      gather_nd_instance_indices)
    foreground_gt_indices = tf.gather_nd(candidate_gt_indices,
                                         gather_nd_instance_indices)

    foreground_gt_indices_shape = tf.shape(foreground_gt_indices)
    batch_indices = (
        tf.expand_dims(tf.range(foreground_gt_indices_shape[0]), axis=-1) *
        tf.ones([1, foreground_gt_indices_shape[-1]], dtype=tf.int32))
    gather_nd_gt_indices = tf.stack([batch_indices, foreground_gt_indices],
                                    axis=-1)
    foreground_masks = tf.gather_nd(gt_masks, gather_nd_gt_indices)

    cropped_foreground_masks = spatial_transform_ops.crop_mask_in_target_box(
        foreground_masks,
        foreground_boxes,
        foreground_rois,
        mask_target_size,
        sample_offset=0.5)

    return foreground_rois, foreground_classes, cropped_foreground_masks


class ROISampler(object):
  """Samples RoIs and creates training targets."""

  def __init__(self, params):
    self._num_samples_per_image = params.num_samples_per_image
    self._fg_fraction = params.fg_fraction
    self._fg_iou_thresh = params.fg_iou_thresh
    self._bg_iou_thresh_hi = params.bg_iou_thresh_hi
    self._bg_iou_thresh_lo = params.bg_iou_thresh_lo
    self._mix_gt_boxes = params.mix_gt_boxes

  def __call__(self, rois, gt_boxes, gt_classes):
    """Sample and assign RoIs for training.

    Args:
      rois: a tensor of shape of [batch_size, N, 4]. N is the number of
        proposals before groundtruth assignment. The last dimension is the box
        coordinates w.r.t. the scaled images in [ymin, xmin, ymax, xmax] format.
      gt_boxes: a tensor of shape of [batch_size, MAX_NUM_INSTANCES, 4]. The
        coordinates of gt_boxes are in the pixel coordinates of the scaled
        image. This tensor might have padding of values -1 indicating the
        invalid box coordinates.
      gt_classes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES]. This
        tensor might have paddings with values of -1 indicating the invalid
        classes.

    Returns:
      sampled_rois: a tensor of shape of [batch_size, K, 4], representing the
        coordinates of the sampled RoIs, where K is the number of the sampled
        RoIs, i.e. K = num_samples_per_image.
      sampled_gt_boxes: a tensor of shape of [batch_size, K, 4], storing the
        box coordinates of the matched groundtruth boxes of the samples RoIs.
      sampled_gt_classes: a tensor of shape of [batch_size, K], storing the
        classes of the matched groundtruth boxes of the sampled RoIs.
    """
    sampled_rois, sampled_gt_boxes, sampled_gt_classes, sampled_gt_indices = (
        assign_and_sample_proposals(
            rois,
            gt_boxes,
            gt_classes,
            num_samples_per_image=self._num_samples_per_image,
            mix_gt_boxes=self._mix_gt_boxes,
            fg_fraction=self._fg_fraction,
            fg_iou_thresh=self._fg_iou_thresh,
            bg_iou_thresh_hi=self._bg_iou_thresh_hi,
            bg_iou_thresh_lo=self._bg_iou_thresh_lo))
    return (sampled_rois, sampled_gt_boxes, sampled_gt_classes,
            sampled_gt_indices)


class ROIScoreSampler(ROISampler):
  """Samples RoIs, RoI-scores and creates training targets."""

  def __call__(self, rois, roi_scores, gt_boxes, gt_classes):
    """Sample and assign RoIs for training.

    Args:
      rois: a tensor of shape of [batch_size, N, 4]. N is the number of
        proposals before groundtruth assignment. The last dimension is the box
        coordinates w.r.t. the scaled images in [ymin, xmin, ymax, xmax] format.
      roi_scores:
      gt_boxes: a tensor of shape of [batch_size, MAX_NUM_INSTANCES, 4]. The
        coordinates of gt_boxes are in the pixel coordinates of the scaled
        image. This tensor might have padding of values -1 indicating the
        invalid box coordinates.
      gt_classes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES]. This
        tensor might have paddings with values of -1 indicating the invalid
        classes.

    Returns:
      sampled_rois: a tensor of shape of [batch_size, K, 4], representing the
        coordinates of the sampled RoIs, where K is the number of the sampled
        RoIs, i.e. K = num_samples_per_image.
      sampled_roi_scores:
      sampled_gt_boxes: a tensor of shape of [batch_size, K, 4], storing the
        box coordinates of the matched groundtruth boxes of the samples RoIs.
      sampled_gt_classes: a tensor of shape of [batch_size, K], storing the
        classes of the matched groundtruth boxes of the sampled RoIs.
    """
    (sampled_rois, sampled_roi_scores, sampled_gt_boxes, sampled_gt_classes,
     sampled_gt_indices) = (
         self.assign_and_sample_proposals_and_scores(
             rois,
             roi_scores,
             gt_boxes,
             gt_classes,
             num_samples_per_image=self._num_samples_per_image,
             mix_gt_boxes=self._mix_gt_boxes,
             fg_fraction=self._fg_fraction,
             fg_iou_thresh=self._fg_iou_thresh,
             bg_iou_thresh_hi=self._bg_iou_thresh_hi,
             bg_iou_thresh_lo=self._bg_iou_thresh_lo))
    return (sampled_rois, sampled_roi_scores, sampled_gt_boxes,
            sampled_gt_classes, sampled_gt_indices)

  def assign_and_sample_proposals_and_scores(self,
                                             proposed_boxes,
                                             proposed_scores,
                                             gt_boxes,
                                             gt_classes,
                                             num_samples_per_image=512,
                                             mix_gt_boxes=True,
                                             fg_fraction=0.25,
                                             fg_iou_thresh=0.5,
                                             bg_iou_thresh_hi=0.5,
                                             bg_iou_thresh_lo=0.0):
    """Assigns the proposals with groundtruth classes and performs subsmpling.

    Given `proposed_boxes`, `gt_boxes`, and `gt_classes`, the function uses the
    following algorithm to generate the final `num_samples_per_image` RoIs.
      1. Calculates the IoU between each proposal box and each gt_boxes.
      2. Assigns each proposed box with a groundtruth class and box by choosing
         the largest IoU overlap.
      3. Samples `num_samples_per_image` boxes from all proposed boxes, and
         returns box_targets, class_targets, and RoIs.

    Args:
      proposed_boxes: a tensor of shape of [batch_size, N, 4]. N is the number
        of proposals before groundtruth assignment. The last dimension is the
        box coordinates w.r.t. the scaled images in [ymin, xmin, ymax, xmax]
        format.
      proposed_scores: a tensor of shape of [batch_size, N]. N is the number of
        proposals before groundtruth assignment. It is the rpn scores for all
        proposed boxes which can be either their classification or centerness
        scores.
      gt_boxes: a tensor of shape of [batch_size, MAX_NUM_INSTANCES, 4]. The
        coordinates of gt_boxes are in the pixel coordinates of the scaled
        image. This tensor might have padding of values -1 indicating the
        invalid box coordinates.
      gt_classes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES]. This
        tensor might have paddings with values of -1 indicating the invalid
        classes.
      num_samples_per_image: a integer represents RoI minibatch size per image.
      mix_gt_boxes: a bool indicating whether to mix the groundtruth boxes
      before sampling proposals.
      fg_fraction: a float represents the target fraction of RoI minibatch that
        is labeled foreground (i.e., class > 0).
      fg_iou_thresh: a float represents the IoU overlap threshold for an RoI to
        be considered foreground (if >= fg_iou_thresh).
      bg_iou_thresh_hi: a float represents the IoU overlap threshold for an RoI
        to be considered background (class = 0 if overlap in [LO, HI)).
      bg_iou_thresh_lo: a float represents the IoU overlap threshold for an RoI
        to be considered background (class = 0 if overlap in [LO, HI)).

    Returns:
      sampled_rois: a tensor of shape of [batch_size, K, 4], representing the
        coordinates of the sampled RoIs, where K is the number of the sampled
        RoIs, i.e. K = num_samples_per_image.
      sampled_scores: a tensor of shape of [batch_size, K], representing the
        confidence score of the sampled RoIs, where K is the number of the
        sampled RoIs, i.e. K = num_samples_per_image.
      sampled_gt_boxes: a tensor of shape of [batch_size, K, 4], storing the
        box coordinates of the matched groundtruth boxes of the samples RoIs.
      sampled_gt_classes: a tensor of shape of [batch_size, K], storing the
        classes of the matched groundtruth boxes of the sampled RoIs.
      sampled_gt_indices: a tensor of shape of [batch_size, K], storing the
        indices of the sampled groudntruth boxes in the original `gt_boxes`
        tensor, i.e. gt_boxes[sampled_gt_indices[:, i]] =
        sampled_gt_boxes[:, i].
    """

    with tf.name_scope('sample_proposals_and_scores'):
      if mix_gt_boxes:
        boxes = tf.concat([proposed_boxes, gt_boxes], axis=1)
        gt_scores = tf.ones_like(gt_boxes[:, :, 0])
        scores = tf.concat([proposed_scores, gt_scores], axis=1)
      else:
        boxes = proposed_boxes
        scores = proposed_scores

      (matched_gt_boxes, matched_gt_classes, matched_gt_indices, matched_iou,
       _) = box_matching(boxes, gt_boxes, gt_classes)

      positive_match = tf.greater(matched_iou, fg_iou_thresh)
      negative_match = tf.logical_and(
          tf.greater_equal(matched_iou, bg_iou_thresh_lo),
          tf.less(matched_iou, bg_iou_thresh_hi))
      ignored_match = tf.less(matched_iou, 0.0)

      # re-assign negatively matched boxes to the background class.
      matched_gt_classes = tf.where(negative_match,
                                    tf.zeros_like(matched_gt_classes),
                                    matched_gt_classes)
      matched_gt_indices = tf.where(negative_match,
                                    tf.zeros_like(matched_gt_indices),
                                    matched_gt_indices)

      sample_candidates = tf.logical_and(
          tf.logical_or(positive_match, negative_match),
          tf.logical_not(ignored_match))

      sampler = (
          balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(
              positive_fraction=fg_fraction, is_static=True))

      batch_size, _ = sample_candidates.get_shape().as_list()
      sampled_indicators = []
      for i in range(batch_size):
        sampled_indicator = sampler.subsample(sample_candidates[i],
                                              num_samples_per_image,
                                              positive_match[i])
        sampled_indicators.append(sampled_indicator)
      sampled_indicators = tf.stack(sampled_indicators)
      _, sampled_indices = tf.nn.top_k(
          tf.cast(sampled_indicators, dtype=tf.int32),
          k=num_samples_per_image,
          sorted=True)

      sampled_indices_shape = tf.shape(sampled_indices)
      batch_indices = (
          tf.expand_dims(tf.range(sampled_indices_shape[0]), axis=-1) *
          tf.ones([1, sampled_indices_shape[-1]], dtype=tf.int32))
      gather_nd_indices = tf.stack([batch_indices, sampled_indices], axis=-1)

      sampled_rois = tf.gather_nd(boxes, gather_nd_indices)
      sampled_roi_scores = tf.gather_nd(scores, gather_nd_indices)
      sampled_gt_boxes = tf.gather_nd(matched_gt_boxes, gather_nd_indices)
      sampled_gt_classes = tf.gather_nd(matched_gt_classes, gather_nd_indices)
      sampled_gt_indices = tf.gather_nd(matched_gt_indices, gather_nd_indices)

      return (sampled_rois, sampled_roi_scores, sampled_gt_boxes,
              sampled_gt_classes, sampled_gt_indices)


class MaskSampler(object):
  """Samples and creates mask training targets."""

  def __init__(self, mask_target_size, num_mask_samples_per_image):
    self._mask_target_size = mask_target_size
    self._num_mask_samples_per_image = num_mask_samples_per_image

  def __call__(self, candidate_rois, candidate_gt_boxes, candidate_gt_classes,
               candidate_gt_indices, gt_masks):
    """Sample and create mask targets for training.

    Args:
      candidate_rois: a tensor of shape of [batch_size, N, 4], where N is the
        number of candidate RoIs to be considered for mask sampling. It includes
        both positive and negative RoIs. The `num_mask_samples_per_image`
        positive RoIs will be sampled to create mask training targets.
      candidate_gt_boxes: a tensor of shape of [batch_size, N, 4], storing the
        corresponding groundtruth boxes to the `candidate_rois`.
      candidate_gt_classes: a tensor of shape of [batch_size, N], storing the
        corresponding groundtruth classes to the `candidate_rois`. 0 in the
        tensor corresponds to the background class, i.e. negative RoIs.
      candidate_gt_indices: a tensor of shape [batch_size, N], storing the
        corresponding groundtruth instance indices to the `candidate_gt_boxes`,
        i.e. gt_boxes[candidate_gt_indices[:, i]] = candidate_gt_boxes[:, i],
          where gt_boxes which is of shape [batch_size, MAX_INSTANCES, 4], M >=
          N, is the superset of candidate_gt_boxes.
      gt_masks: a tensor of [batch_size, MAX_INSTANCES, mask_height, mask_width]
        containing all the groundtruth masks which sample masks are drawn from.
        after sampling. The output masks are resized w.r.t the sampled RoIs.

    Returns:
      foreground_rois: a tensor of shape of [batch_size, K, 4] storing the RoI
        that corresponds to the sampled foreground masks, where
        K = num_mask_samples_per_image.
      foreground_classes: a tensor of shape of [batch_size, K] storing the
        classes corresponding to the sampled foreground masks.
      cropoped_foreground_masks: a tensor of shape of
        [batch_size, K, mask_target_size, mask_target_size] storing the
        cropped foreground masks used for training.
    """
    foreground_rois, foreground_classes, cropped_foreground_masks = (
        sample_and_crop_foreground_masks(candidate_rois, candidate_gt_boxes,
                                         candidate_gt_classes,
                                         candidate_gt_indices, gt_masks,
                                         self._num_mask_samples_per_image,
                                         self._mask_target_size))
    return foreground_rois, foreground_classes, cropped_foreground_masks
