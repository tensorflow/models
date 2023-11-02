# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Contains definitions of ROI sampler."""
from typing import Optional, Tuple, Union
# Import libraries
import tensorflow as tf, tf_keras

from official.vision.modeling.layers import box_sampler
from official.vision.ops import box_matcher
from official.vision.ops import iou_similarity
from official.vision.ops import target_gather

# The return type can be a tuple of 4 or 5 tf.Tensor.
ROISamplerReturnType = Union[
    Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]]


@tf_keras.utils.register_keras_serializable(package='Vision')
class ROISampler(tf_keras.layers.Layer):
  """Samples ROIs and assigns targets to the sampled ROIs."""

  def __init__(self,
               mix_gt_boxes: bool = True,
               num_sampled_rois: int = 512,
               foreground_fraction: float = 0.25,
               foreground_iou_threshold: float = 0.5,
               background_iou_high_threshold: float = 0.5,
               background_iou_low_threshold: float = 0,
               skip_subsampling: bool = False,
               **kwargs):
    """Initializes a ROI sampler.

    Args:
      mix_gt_boxes: A `bool` of whether to mix the groundtruth boxes with
        proposed ROIs.
      num_sampled_rois: An `int` of the number of sampled ROIs per image.
      foreground_fraction: A `float` in [0, 1], what percentage of proposed ROIs
        should be sampled from the foreground boxes.
      foreground_iou_threshold: A `float` that represents the IoU threshold for
        a box to be considered as positive (if >= `foreground_iou_threshold`).
      background_iou_high_threshold: A `float` that represents the IoU threshold
        for a box to be considered as negative (if overlap in
        [`background_iou_low_threshold`, `background_iou_high_threshold`]).
      background_iou_low_threshold: A `float` that represents the IoU threshold
        for a box to be considered as negative (if overlap in
        [`background_iou_low_threshold`, `background_iou_high_threshold`])
      skip_subsampling: a bool that determines if we want to skip the sampling
        procedure than balances the fg/bg classes. Used for upper frcnn layers
        in cascade RCNN.
      **kwargs: Additional keyword arguments passed to Layer.
    """
    self._config_dict = {
        'mix_gt_boxes': mix_gt_boxes,
        'num_sampled_rois': num_sampled_rois,
        'foreground_fraction': foreground_fraction,
        'foreground_iou_threshold': foreground_iou_threshold,
        'background_iou_high_threshold': background_iou_high_threshold,
        'background_iou_low_threshold': background_iou_low_threshold,
        'skip_subsampling': skip_subsampling,
    }

    self._sim_calc = iou_similarity.IouSimilarity()
    self._box_matcher = box_matcher.BoxMatcher(
        thresholds=[
            background_iou_low_threshold, background_iou_high_threshold,
            foreground_iou_threshold
        ],
        indicators=[-3, -1, -2, 1])
    self._target_gather = target_gather.TargetGather()

    self._sampler = box_sampler.BoxSampler(
        num_sampled_rois, foreground_fraction)
    super().__init__(**kwargs)

  def call(
      self,
      boxes: tf.Tensor,
      gt_boxes: tf.Tensor,
      gt_classes: tf.Tensor,
      gt_outer_boxes: Optional[tf.Tensor] = None) -> ROISamplerReturnType:
    """Assigns the proposals with groundtruth classes and performs subsmpling.

    Given `proposed_boxes`, `gt_boxes`, and `gt_classes`, the function uses the
    following algorithm to generate the final `num_samples_per_image` RoIs.
      1. Calculates the IoU between each proposal box and each gt_boxes.
      2. Assigns each proposed box with a groundtruth class and box by choosing
         the largest IoU overlap.
      3. Samples `num_samples_per_image` boxes from all proposed boxes, and
         returns box_targets, class_targets, and RoIs.

    Args:
      boxes: A `tf.Tensor` of shape of [batch_size, N, 4]. N is the number of
        proposals before groundtruth assignment. The last dimension is the
        box coordinates w.r.t. the scaled images in [ymin, xmin, ymax, xmax]
        format.
      gt_boxes: A `tf.Tensor` of shape of [batch_size, MAX_NUM_INSTANCES, 4].
        The coordinates of gt_boxes are in the pixel coordinates of the scaled
        image. This tensor might have padding of values -1 indicating the
        invalid box coordinates.
      gt_classes: A `tf.Tensor` with a shape of [batch_size, MAX_NUM_INSTANCES].
        This tensor might have paddings with values of -1 indicating the invalid
        classes.
      gt_outer_boxes: A `tf.Tensor` of shape of [batch_size, MAX_NUM_INSTANCES,
        4]. The corrdinates of gt_outer_boxes are in the pixel coordinates of
        the scaled image. This tensor might have padding of values -1 indicating
        the invalid box coordinates. Ignored if not provided.

    Returns:
      sampled_rois: A `tf.Tensor` of shape of [batch_size, K, 4], representing
        the coordinates of the sampled RoIs, where K is the number of the
        sampled RoIs, i.e. K = num_samples_per_image.
      sampled_gt_boxes: A `tf.Tensor` of shape of [batch_size, K, 4], storing
        the box coordinates of the matched groundtruth boxes of the samples
        RoIs.
      sampled_gt_outer_boxes: A `tf.Tensor` of shape of [batch_size, K, 4],
        storing the box coordinates of the matched groundtruth outer boxes of
        the samples RoIs. This field is missing if gt_outer_boxes is None.
      sampled_gt_classes: A `tf.Tensor` of shape of [batch_size, K], storing the
        classes of the matched groundtruth boxes of the sampled RoIs.
      sampled_gt_indices: A `tf.Tensor` of shape of [batch_size, K], storing the
        indices of the sampled groudntruth boxes in the original `gt_boxes`
        tensor, i.e.,
        gt_boxes[sampled_gt_indices[:, i]] = sampled_gt_boxes[:, i].
    """
    gt_boxes = tf.cast(gt_boxes, dtype=boxes.dtype)
    if self._config_dict['mix_gt_boxes']:
      boxes = tf.concat([boxes, gt_boxes], axis=1)

    boxes_invalid_mask = tf.less(
        tf.reduce_max(boxes, axis=-1, keepdims=True), 0.0)
    gt_invalid_mask = tf.less(
        tf.reduce_max(gt_boxes, axis=-1, keepdims=True), 0.0)
    similarity_matrix = self._sim_calc(boxes, gt_boxes, boxes_invalid_mask,
                                       gt_invalid_mask)
    matched_gt_indices, match_indicators = self._box_matcher(similarity_matrix)
    positive_matches = tf.greater_equal(match_indicators, 0)
    negative_matches = tf.equal(match_indicators, -1)
    ignored_matches = tf.equal(match_indicators, -2)
    invalid_matches = tf.equal(match_indicators, -3)

    background_mask = tf.expand_dims(
        tf.logical_or(negative_matches, invalid_matches), -1)
    gt_classes = tf.expand_dims(gt_classes, axis=-1)
    matched_gt_classes = self._target_gather(gt_classes, matched_gt_indices,
                                             background_mask)
    matched_gt_classes = tf.where(background_mask,
                                  tf.zeros_like(matched_gt_classes),
                                  matched_gt_classes)
    matched_gt_boxes = self._target_gather(gt_boxes, matched_gt_indices,
                                           tf.tile(background_mask, [1, 1, 4]))
    matched_gt_boxes = tf.where(background_mask,
                                tf.zeros_like(matched_gt_boxes),
                                matched_gt_boxes)
    if gt_outer_boxes is not None:
      matched_gt_outer_boxes = self._target_gather(
          gt_outer_boxes, matched_gt_indices, tf.tile(background_mask,
                                                      [1, 1, 4]))
      matched_gt_outer_boxes = tf.where(background_mask,
                                        tf.zeros_like(matched_gt_outer_boxes),
                                        matched_gt_outer_boxes)
    matched_gt_indices = tf.where(
        tf.squeeze(background_mask, -1), -tf.ones_like(matched_gt_indices),
        matched_gt_indices)

    if self._config_dict['skip_subsampling']:
      matched_gt_classes = tf.squeeze(matched_gt_classes, axis=-1)
      if gt_outer_boxes is None:
        return (boxes, matched_gt_boxes, matched_gt_classes, matched_gt_indices)
      return (boxes, matched_gt_boxes, matched_gt_outer_boxes,
              matched_gt_classes, matched_gt_indices)

    sampled_indices = self._sampler(
        positive_matches, negative_matches, ignored_matches)

    sampled_rois = self._target_gather(boxes, sampled_indices)
    sampled_gt_boxes = self._target_gather(matched_gt_boxes, sampled_indices)
    sampled_gt_classes = tf.squeeze(self._target_gather(
        matched_gt_classes, sampled_indices), axis=-1)
    sampled_gt_indices = tf.squeeze(self._target_gather(
        tf.expand_dims(matched_gt_indices, -1), sampled_indices), axis=-1)
    if gt_outer_boxes is None:
      return (sampled_rois, sampled_gt_boxes, sampled_gt_classes,
              sampled_gt_indices)
    sampled_gt_outer_boxes = self._target_gather(matched_gt_outer_boxes,
                                                 sampled_indices)
    return (sampled_rois, sampled_gt_boxes, sampled_gt_outer_boxes,
            sampled_gt_classes, sampled_gt_indices)

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)
