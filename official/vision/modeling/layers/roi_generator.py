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

"""Contains definitions of ROI generator."""
from typing import Optional, Mapping
# Import libraries
import tensorflow as tf, tf_keras

from official.vision.ops import box_ops
from official.vision.ops import nms


def _multilevel_propose_rois(raw_boxes: Mapping[str, tf.Tensor],
                             raw_scores: Mapping[str, tf.Tensor],
                             anchor_boxes: Mapping[str, tf.Tensor],
                             image_shape: tf.Tensor,
                             pre_nms_top_k: int = 2000,
                             pre_nms_score_threshold: float = 0.0,
                             pre_nms_min_size_threshold: float = 0.0,
                             nms_iou_threshold: float = 0.7,
                             num_proposals: int = 1000,
                             use_batched_nms: bool = False,
                             decode_boxes: bool = True,
                             clip_boxes: bool = True,
                             apply_sigmoid_to_score: bool = True):
  """Proposes RoIs given a group of candidates from different FPN levels.

  The following describes the steps:
    1. For each individual level:
      a. Apply sigmoid transform if specified.
      b. Decode boxes if specified.
      c. Clip boxes if specified.
      d. Filter small boxes and those fall outside image if specified.
      e. Apply pre-NMS filtering including pre-NMS top k and score thresholding.
      f. Apply NMS.
    2. Aggregate post-NMS boxes from each level.
    3. Apply an overall top k to generate the final selected RoIs.

  Args:
    raw_boxes: A `dict` with keys representing FPN levels and values
      representing box tenors of shape
      [batch_size, feature_h, feature_w, num_anchors * 4].
    raw_scores: A `dict` with keys representing FPN levels and values
      representing logit tensors of shape
      [batch_size, feature_h, feature_w, num_anchors].
    anchor_boxes: A `dict` with keys representing FPN levels and values
      representing anchor box tensors of shape
      [batch_size, feature_h * feature_w * num_anchors, 4].
    image_shape: A `tf.Tensor` of shape [batch_size, 2] where the last dimension
      are [height, width] of the scaled image.
    pre_nms_top_k: An `int` of top scoring RPN proposals *per level* to keep
      before applying NMS. Default: 2000.
    pre_nms_score_threshold: A `float` between 0 and 1 representing the minimal
      box score to keep before applying NMS. This is often used as a
      pre-filtering step for better performance. Default: 0, no filtering is
      applied.
    pre_nms_min_size_threshold: A `float` representing the minimal box size in
      each side (w.r.t. the scaled image) to keep before applying NMS. This is
      often used as a pre-filtering step for better performance. Default: 0, no
      filtering is applied.
    nms_iou_threshold: A `float` between 0 and 1 representing the IoU threshold
      used for NMS. If 0.0, no NMS is applied. Default: 0.7.
    num_proposals: An `int` of top scoring RPN proposals *in total* to keep
      after applying NMS. Default: 1000.
    use_batched_nms: A `bool` indicating whether NMS is applied in batch using
      `tf.image.combined_non_max_suppression`. Currently only available in
      CPU/GPU. Default is False.
    decode_boxes: A `bool` indicating whether `raw_boxes` needs to be decoded
      using `anchor_boxes`. If False, use `raw_boxes` directly and ignore
      `anchor_boxes`. Default is True.
    clip_boxes: A `bool` indicating whether boxes are first clipped to the
      scaled image size before appliying NMS. If False, no clipping is applied
      and `image_shape` is ignored. Default is True.
    apply_sigmoid_to_score: A `bool` indicating whether apply sigmoid to
      `raw_scores` before applying NMS. Default is True.

  Returns:
    selected_rois: A `tf.Tensor` of shape [batch_size, num_proposals, 4],
      representing the box coordinates of the selected proposals w.r.t. the
      scaled image.
    selected_roi_scores: A `tf.Tensor` of shape [batch_size, num_proposals, 1],
      representing the scores of the selected proposals.
  """
  with tf.name_scope('multilevel_propose_rois'):
    rois = []
    roi_scores = []
    image_shape = tf.expand_dims(image_shape, axis=1)
    for level in sorted(raw_scores.keys()):
      with tf.name_scope('level_%s' % level):
        _, feature_h, feature_w, num_anchors_per_location = (
            raw_scores[level].get_shape().as_list())

        num_boxes = feature_h * feature_w * num_anchors_per_location
        this_level_scores = tf.reshape(raw_scores[level], [-1, num_boxes])
        this_level_boxes = tf.reshape(raw_boxes[level], [-1, num_boxes, 4])
        this_level_anchors = tf.cast(
            tf.reshape(anchor_boxes[level], [-1, num_boxes, 4]),
            dtype=this_level_scores.dtype)

        if apply_sigmoid_to_score:
          this_level_scores = tf.sigmoid(this_level_scores)

        if decode_boxes:
          this_level_boxes = box_ops.decode_boxes(
              this_level_boxes, this_level_anchors)
        if clip_boxes:
          this_level_boxes = box_ops.clip_boxes(
              this_level_boxes, image_shape)

        if pre_nms_min_size_threshold > 0.0:
          this_level_boxes, this_level_scores = box_ops.filter_boxes(
              this_level_boxes,
              this_level_scores,
              image_shape,
              pre_nms_min_size_threshold)

        this_level_pre_nms_top_k = min(num_boxes, pre_nms_top_k)
        this_level_post_nms_top_k = min(num_boxes, num_proposals)
        if nms_iou_threshold > 0.0:
          if use_batched_nms:
            this_level_rois, this_level_roi_scores, _, _ = (
                tf.image.combined_non_max_suppression(
                    tf.expand_dims(this_level_boxes, axis=2),
                    tf.expand_dims(this_level_scores, axis=-1),
                    max_output_size_per_class=this_level_pre_nms_top_k,
                    max_total_size=this_level_post_nms_top_k,
                    iou_threshold=nms_iou_threshold,
                    score_threshold=pre_nms_score_threshold,
                    pad_per_class=False,
                    clip_boxes=False))
          else:
            if pre_nms_score_threshold > 0.0:
              this_level_boxes, this_level_scores = (
                  box_ops.filter_boxes_by_scores(
                      this_level_boxes,
                      this_level_scores,
                      pre_nms_score_threshold))
            this_level_boxes, this_level_scores = box_ops.top_k_boxes(
                this_level_boxes, this_level_scores, k=this_level_pre_nms_top_k)
            this_level_roi_scores, this_level_rois = (
                nms.sorted_non_max_suppression_padded(
                    this_level_scores,
                    this_level_boxes,
                    max_output_size=this_level_post_nms_top_k,
                    iou_threshold=nms_iou_threshold))
        else:
          this_level_rois, this_level_roi_scores = box_ops.top_k_boxes(
              this_level_boxes,
              this_level_scores,
              k=this_level_post_nms_top_k)

        rois.append(this_level_rois)
        roi_scores.append(this_level_roi_scores)

    all_rois = tf.concat(rois, axis=1)
    all_roi_scores = tf.concat(roi_scores, axis=1)

    with tf.name_scope('top_k_rois'):
      _, num_valid_rois = all_roi_scores.get_shape().as_list()
      overall_top_k = min(num_valid_rois, num_proposals)

      selected_rois, selected_roi_scores = box_ops.top_k_boxes(
          all_rois, all_roi_scores, k=overall_top_k)

    return selected_rois, selected_roi_scores


@tf_keras.utils.register_keras_serializable(package='Vision')
class MultilevelROIGenerator(tf_keras.layers.Layer):
  """Proposes RoIs for the second stage processing."""

  def __init__(self,
               pre_nms_top_k: int = 2000,
               pre_nms_score_threshold: float = 0.0,
               pre_nms_min_size_threshold: float = 0.0,
               nms_iou_threshold: float = 0.7,
               num_proposals: int = 1000,
               test_pre_nms_top_k: int = 1000,
               test_pre_nms_score_threshold: float = 0.0,
               test_pre_nms_min_size_threshold: float = 0.0,
               test_nms_iou_threshold: float = 0.7,
               test_num_proposals: int = 1000,
               use_batched_nms: bool = False,
               **kwargs):
    """Initializes a ROI generator.

    The ROI generator transforms the raw predictions from RPN to ROIs.

    Args:
      pre_nms_top_k: An `int` of the number of top scores proposals to be kept
        before applying NMS.
      pre_nms_score_threshold: A `float` of the score threshold to apply before
        applying NMS. Proposals whose scores are below this threshold are
        thrown away.
      pre_nms_min_size_threshold: A `float` of the threshold of each side of the
        box (w.r.t. the scaled image). Proposals whose sides are below this
        threshold are thrown away.
      nms_iou_threshold: A `float` in [0, 1], the NMS IoU threshold.
      num_proposals: An `int` of the final number of proposals to generate.
      test_pre_nms_top_k: An `int` of the number of top scores proposals to be
        kept before applying NMS in testing.
      test_pre_nms_score_threshold: A `float` of the score threshold to apply
        before applying NMS in testing. Proposals whose scores are below this
        threshold are thrown away.
      test_pre_nms_min_size_threshold: A `float` of the threshold of each side
        of the box (w.r.t. the scaled image) in testing. Proposals whose sides
        are below this threshold are thrown away.
      test_nms_iou_threshold: A `float` in [0, 1] of the NMS IoU threshold in
        testing.
      test_num_proposals: An `int` of the final number of proposals to generate
        in testing.
      use_batched_nms: A `bool` of whether or not use
        `tf.image.combined_non_max_suppression`.
      **kwargs: Additional keyword arguments passed to Layer.
    """
    self._config_dict = {
        'pre_nms_top_k': pre_nms_top_k,
        'pre_nms_score_threshold': pre_nms_score_threshold,
        'pre_nms_min_size_threshold': pre_nms_min_size_threshold,
        'nms_iou_threshold': nms_iou_threshold,
        'num_proposals': num_proposals,
        'test_pre_nms_top_k': test_pre_nms_top_k,
        'test_pre_nms_score_threshold': test_pre_nms_score_threshold,
        'test_pre_nms_min_size_threshold': test_pre_nms_min_size_threshold,
        'test_nms_iou_threshold': test_nms_iou_threshold,
        'test_num_proposals': test_num_proposals,
        'use_batched_nms': use_batched_nms,
    }
    super(MultilevelROIGenerator, self).__init__(**kwargs)

  def call(self,
           raw_boxes: Mapping[str, tf.Tensor],
           raw_scores: Mapping[str, tf.Tensor],
           anchor_boxes: Mapping[str, tf.Tensor],
           image_shape: tf.Tensor,
           training: Optional[bool] = None):
    """Proposes RoIs given a group of candidates from different FPN levels.

    The following describes the steps:
      1. For each individual level:
        a. Apply sigmoid transform if specified.
        b. Decode boxes if specified.
        c. Clip boxes if specified.
        d. Filter small boxes and those fall outside image if specified.
        e. Apply pre-NMS filtering including pre-NMS top k and score
           thresholding.
        f. Apply NMS.
      2. Aggregate post-NMS boxes from each level.
      3. Apply an overall top k to generate the final selected RoIs.

    Args:
      raw_boxes: A `dict` with keys representing FPN levels and values
        representing box tenors of shape
        [batch, feature_h, feature_w, num_anchors * 4].
      raw_scores: A `dict` with keys representing FPN levels and values
        representing logit tensors of shape
        [batch, feature_h, feature_w, num_anchors].
      anchor_boxes: A `dict` with keys representing FPN levels and values
        representing anchor box tensors of shape
        [batch, feature_h * feature_w * num_anchors, 4].
      image_shape: A `tf.Tensor` of shape [batch, 2] where the last dimension
        are [height, width] of the scaled image.
      training: A `bool` that indicates whether it is in training mode.

    Returns:
      roi_boxes: A `tf.Tensor` of shape [batch, num_proposals, 4], the proposed
        ROIs in the scaled image coordinate.
      roi_scores: A `tf.Tensor` of shape [batch, num_proposals], scores of the
        proposed ROIs.
    """
    roi_boxes, roi_scores = _multilevel_propose_rois(
        raw_boxes,
        raw_scores,
        anchor_boxes,
        image_shape,
        pre_nms_top_k=(
            self._config_dict['pre_nms_top_k'] if training
            else self._config_dict['test_pre_nms_top_k']),
        pre_nms_score_threshold=(
            self._config_dict['pre_nms_score_threshold'] if training
            else self._config_dict['test_pre_nms_score_threshold']),
        pre_nms_min_size_threshold=(
            self._config_dict['pre_nms_min_size_threshold'] if training
            else self._config_dict['test_pre_nms_min_size_threshold']),
        nms_iou_threshold=(
            self._config_dict['nms_iou_threshold'] if training
            else self._config_dict['test_nms_iou_threshold']),
        num_proposals=(
            self._config_dict['num_proposals'] if training
            else self._config_dict['test_num_proposals']),
        use_batched_nms=self._config_dict['use_batched_nms'],
        decode_boxes=True,
        clip_boxes=True,
        apply_sigmoid_to_score=True)
    return roi_boxes, roi_scores

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
