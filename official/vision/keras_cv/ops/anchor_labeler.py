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
"""Definition of anchor labeler, which assigns ground truth boxes to anchors."""

import tensorflow as tf


class AnchorLabeler:
  """Labeler for dense object detector."""

  def __call__(self, labels, match_indices, mask, mask_val=0.0):
    """Labels anchors with ground truth inputs.

    B: batch_size
    N: number of groundtruth boxes.

    Args:
      labels: An integer tensor with shape [N, 1] or [B, N, 1] representing
        groundtruth labels.
      match_indices: An integer tensor with shape [N] or [B, N] representing
        match label index.
      mask: An integer tensor with shape [N] or [B, N] representing match
        labels, e.g., 1 for positive, -1 for negative, -2 for ignore.
      mask_val: An integer to fill in for mask.

    Returns:
      class_targets: A integer Tensor with shape [num_anchors].
      box_targets: A float Tensor with shape [num_anchors, 4].
      class_weights: A float Tensor with shape [num_anchors], that
        serves as masking / sample weight for classification loss. Its value
        is 1.0 for positive and negative matched anchors, and 0.0 for ignored
        anchors.
      box_weights: A float Tensor with shape [num_anchors], that
        serves as masking / sample weight for regression loss. Its value is
        1.0 for positive matched anchors, and 0.0 for negative and ignored
        anchors.
    """
    if len(labels.shape) <= 2:
      return self._gather_unbatched(labels, match_indices, mask, mask_val)
    elif len(labels.shape) == 3:
      return self._gather_batched(labels, match_indices, mask, mask_val)

  def _gather_unbatched(self, labels, match_indices, mask, mask_val):
    """Gather based on unbatched labels and boxes."""
    num_gt_boxes = tf.shape(labels)[0]
    masked_targets = tf.cast(mask_val, labels.dtype) * tf.ones_like(
        mask, dtype=labels.dtype)

    def _assign_when_rows_empty():
      return masked_targets

    def _assign_when_rows_not_empty():
      targets = tf.gather(labels, match_indices)
      return tf.where(mask, masked_targets, targets)

    return tf.cond(tf.greater(num_gt_boxes, 0),
                   _assign_when_rows_not_empty,
                   _assign_when_rows_empty)

  def _gather_batched(self, labels, match_indices, mask, mask_val):
    """Gather based on batched labels."""
    batch_size = labels.shape[0]
    if batch_size == 1:
      result = self._gather_unbatched(
          tf.squeeze(labels, axis=0), tf.squeeze(match_indices, axis=0),
          tf.squeeze(mask, axis=0), mask_val)
      return tf.expand_dims(result, axis=0)
    else:
      indices_shape = tf.shape(match_indices)
      indices_dtype = match_indices.dtype
      batch_indices = (tf.expand_dims(
          tf.range(indices_shape[0], dtype=indices_dtype), axis=-1) *
                       tf.ones([1, indices_shape[-1]], dtype=indices_dtype))
      gather_nd_indices = tf.stack(
          [batch_indices, match_indices], axis=-1)
      targets = tf.gather_nd(labels, gather_nd_indices)
      return targets
