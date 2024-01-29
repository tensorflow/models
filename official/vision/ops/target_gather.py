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

"""Definition of target gather, which gathers targets from indices."""

import tensorflow as tf


class TargetGather:
  """Targer gather for dense object detector."""

  def __call__(self, labels, match_indices, mask=None, mask_val=0.0):
    """Labels anchors with ground truth inputs.

    B: batch_size
    N: number of groundtruth boxes.

    Args:
      labels: An integer tensor with shape [N, dims] or [B, N, ...] representing
        groundtruth labels.
      match_indices: An integer tensor with shape [M] or [B, M] representing
        match label index.
      mask: An boolean tensor with shape [M, dims] or [B, M,...] representing
        match labels.
      mask_val: An integer to fill in for mask.

    Returns:
      target: An integer Tensor with shape [M] or [B, M]
    Raises:
      ValueError: If `labels` is higher than rank 3.
    """
    if len(labels.shape) <= 2:
      return self._gather_unbatched(labels, match_indices, mask, mask_val)
    elif len(labels.shape) == 3:
      return self._gather_batched(labels, match_indices, mask, mask_val)
    else:
      raise ValueError("`TargetGather` does not support `labels` with rank "
                       "larger than 3, got {}".format(len(labels.shape)))

  def _gather_unbatched(self, labels, match_indices, mask, mask_val):
    """Gather based on unbatched labels and boxes."""
    num_gt_boxes = tf.shape(labels)[0]

    def _assign_when_rows_empty():
      if len(labels.shape) > 1:
        mask_shape = [match_indices.shape[0], labels.shape[-1]]
      else:
        mask_shape = [match_indices.shape[0]]
      return tf.cast(mask_val, labels.dtype) * tf.ones(
          mask_shape, dtype=labels.dtype)

    def _assign_when_rows_not_empty():
      targets = tf.gather(labels, match_indices)
      if mask is None:
        return targets
      else:
        masked_targets = tf.cast(mask_val, labels.dtype) * tf.ones_like(
            mask, dtype=labels.dtype)
        return tf.where(mask, masked_targets, targets)

    return tf.cond(tf.greater(num_gt_boxes, 0),
                   _assign_when_rows_not_empty,
                   _assign_when_rows_empty)

  def _gather_batched(self, labels, match_indices, mask, mask_val):
    """Gather based on batched labels."""
    batch_size = labels.shape[0]
    if batch_size == 1:
      if mask is not None:
        result = self._gather_unbatched(
            tf.squeeze(labels, axis=0), tf.squeeze(match_indices, axis=0),
            tf.squeeze(mask, axis=0), mask_val)
      else:
        result = self._gather_unbatched(
            tf.squeeze(labels, axis=0), tf.squeeze(match_indices, axis=0),
            None, mask_val)
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
      if mask is None:
        return targets
      else:
        masked_targets = tf.cast(mask_val, labels.dtype) * tf.ones_like(
            mask, dtype=labels.dtype)
        return tf.where(mask, masked_targets, targets)
