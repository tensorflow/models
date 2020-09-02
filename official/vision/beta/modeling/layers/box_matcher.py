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
"""Box matcher."""

# Import libraries
import tensorflow as tf

from official.vision.beta.ops import box_ops


@tf.keras.utils.register_keras_serializable(package='Vision')
class BoxMatcher(tf.keras.layers.Layer):
  """Match boxes with groundtruth boxes."""

  def __init__(self,
               foreground_iou_threshold=0.5,
               background_iou_high_threshold=0.5,
               background_iou_low_threshold=0,
               **kwargs):
    """Initializes a box matcher.

    Args:
      foreground_iou_threshold: float, represent the IoU threshold for a box to
        be considered as positive (if >= `foreground_iou_threshold`).
      background_iou_high_threshold: float, represent the IoU threshold for a
        box to be considered as negative (if overlap in
        [`background_iou_low_threshold`, `background_iou_high_threshold`]).
      background_iou_low_threshold: float, represent the IoU threshold for a box
        to be considered as negative (if overlap in
        [`background_iou_low_threshold`, `background_iou_high_threshold`])
      **kwargs: other key word arguments passed to Layer.
    """
    self._config_dict = {
        'foreground_iou_threshold': foreground_iou_threshold,
        'background_iou_high_threshold': background_iou_high_threshold,
        'background_iou_low_threshold': background_iou_low_threshold,
    }
    super(BoxMatcher, self).__init__(**kwargs)

  def call(self, boxes, gt_boxes, gt_classes):
    """Match boxes to groundtruth boxes.

    Given the proposal boxes and the groundtruth boxes and classes, perform the
    groundtruth matching by taking the argmax of the IoU between boxes and
    groundtruth boxes.

    Args:
      boxes: a tensor of shape of [batch_size, N, 4] representing the box
        coordianates to be matched to groundtruth boxes.
      gt_boxes: a tensor of shape of [batch_size, MAX_INSTANCES, 4] representing
        the groundtruth box coordinates. It is padded with -1s to indicate the
        invalid boxes.
      gt_classes: [batch_size, MAX_INSTANCES] representing the groundtruth box
        classes. It is padded with -1s to indicate the invalid classes.

    Returns:
      matched_gt_boxes: a tensor of shape of [batch, N, 4], representing
        the matched groundtruth box coordinates for each input box. The box is
        considered to match to a groundtruth box only if the IoU overlap is
        greater than `foreground_iou_threshold`. If the box is a negative match,
        or does not overlap with any groundtruth boxes, the matched boxes will
        be set to all 0s.
      matched_gt_classes: a tensor of shape of [batch, N], representing
        the matched groundtruth classes for each input box. If the box is a
        negative match or does not overlap with any groundtruth boxes, the
        matched classes of it will be set to 0, which corresponds to the
        background class.
      matched_gt_indices: a tensor of shape of [batch, N], representing the
        indices of the matched groundtruth boxes in the original gt_boxes
        tensor. If the box is a negative match or does not overlap with any
        groundtruth boxes, the index of the matched groundtruth will be set to
        -1.
      positive_matches: a bool tensor of shape of [batch, N], representing
        whether each box is a positive matches or not. A positive match is the
        case where IoU of a box with any groundtruth box is greater than
        `foreground_iou_threshold`.
      negative_matches: a bool tensor of shape of [batch, N], representing
        whether each box is a negative matches or not. A negative match is the
        case where IoU of a box with any groundtruth box is greater than
        `background_iou_low_threshold` and less than
        `background_iou_low_threshold`.
      ignored_matches: a bool tensor of shape of [batch, N], representing
        whether each box is an ignored matches or not. An ignored matches is the
        match that is neither positive or negative.
    """
    matched_gt_boxes, matched_gt_classes, matched_gt_indices, matched_iou, _ = (
        box_ops.box_matching(boxes, gt_boxes, gt_classes))

    positive_matches = tf.greater(
        matched_iou, self._config_dict['foreground_iou_threshold'])
    negative_matches = tf.logical_and(
        tf.greater_equal(
            matched_iou, self._config_dict['background_iou_low_threshold']),
        tf.less(
            matched_iou, self._config_dict['background_iou_high_threshold']))
    ignored_matches = tf.logical_and(
        tf.less(matched_iou, 0.0),
        tf.greater_equal(
            matched_iou, self._config_dict['background_iou_high_threshold']))
    ignored_matches = tf.logical_and(
        ignored_matches,
        tf.less(
            matched_iou, self._config_dict['foreground_iou_threshold']))

    background_indicator = tf.logical_or(negative_matches, ignored_matches)

    # re-assign negatively matched boxes to the background class.
    matched_gt_boxes = tf.where(
        tf.tile(tf.expand_dims(background_indicator, -1), [1, 1, 4]),
        tf.zeros_like(matched_gt_boxes),
        matched_gt_boxes)
    matched_gt_classes = tf.where(
        background_indicator,
        tf.zeros_like(matched_gt_classes),
        matched_gt_classes)
    matched_gt_indices = tf.where(
        background_indicator,
        -tf.ones_like(matched_gt_indices),
        matched_gt_indices)

    return (matched_gt_boxes, matched_gt_classes, matched_gt_indices,
            positive_matches, negative_matches, ignored_matches)

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)
