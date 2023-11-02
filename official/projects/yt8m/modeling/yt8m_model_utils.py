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

"""Contains a collection of util functions for model construction."""

from typing import Optional

import tensorflow as tf, tf_keras


def _large_compatible_negative(tensor_type):
  """Large negative number as Tensor.

  This function is necessary because the standard value for epsilon
  in this module (-1e9) cannot be represented using `tf.float16`.

  Args:
    tensor_type: A dtype to determine the type.

  Returns:
    A large negative number.
  """
  if tensor_type == tf.float16:
    return tf.float16.min
  return -1e9


def weighted_average_pooling(features, weights, axis):
  """Weighted average pooling.

  Args:
    features: a tensor of at least rank 1.
    weights: a weight tensor whose shape is broadcast compatible with features.
      It doesn't have to be normalized.
    axis: the dimensions to reduce.

  Returns:
    The reduced tensor.
  """
  return tf.math.divide_no_nan(
      tf.reduce_sum(weights * features, axis),  # numerator.
      tf.reduce_sum(weights, axis),  # denominator.
  )


def frame_swap(
    frames: tf.Tensor, frame_mask: Optional[tf.Tensor] = None
) -> tf.Tensor:
  """Self-weighted average pooling over all frames of a video.

  It does the following operation independently for each feature:
    x_pooled = (sum_i x_i * |x_i|) / (sum_i |x_i|).
  Basically the weight for the feature in each frame is determined by the
  magnitude of the feature itself.

  Paper: https://research.google/pubs/pub48351/

  Args:
    frames: A tensor with shape [batch_size, max_frames, feature_size].
    frame_mask:  A tensor with shape [batch_size, max_frames, 1].

  Returns:
    A tensor with shape [batch_size, feature_size].
  """
  weights = tf.abs(frames)
  if frame_mask is not None:
    weights *= tf.cast(frame_mask, weights.dtype)
  # We set axis to 1 to reduce the dimension corresponding to max_frames.
  return weighted_average_pooling(frames, weights, axis=1)


def frame_pooling(frames, method="average", num_frames=None):
  """Pools over the frames of a video.

  Args:
    frames: tensor of shape [batch_size, num_frames, feature_size].
    method: string indicating pooling method, one of: "average", "max",
      "attention", or "none".
    num_frames: optional tensor of shape [batch_size] indicating valid number of
      frames for each video.

  Returns:
    tensor of shape [batch_size, feature_size] for average, max, or
    attention pooling, and shape [batch_size*num_frames, feature_size]
    for none pooling.
  Raises:
    ValueError: if method is other than "average", "max", "attention", or
    "none".
  """
  frame_mask = None
  if num_frames is not None:
    max_frames = frames.shape.as_list()[1]
    # Generate binary mask from number of frames.
    frame_mask = tf.sequence_mask(num_frames, max_frames, frames.dtype)
    frame_mask = tf.expand_dims(frame_mask, axis=2)

  if method == "average":
    if num_frames is None:
      reduced = tf.reduce_mean(frames, 1)
    else:
      num_frames = tf.reshape(tf.cast(num_frames, frames.dtype), [-1, 1])
      reduced = tf.reduce_sum(frames * frame_mask, 1) / num_frames
  elif method == "max":
    if num_frames is not None:
      frame_mask = tf.cast(frame_mask, tf.bool)
      frames = tf.where(
          frame_mask,
          frames,
          tf.ones_like(frames, dtype=frames.dtype)
          * _large_compatible_negative(frames.dtype),
      )
    reduced = tf.reduce_max(frames, 1)
  elif method == "swap":
    # Note we assume the frames are in the shape of
    # [batch_size, num_frames, feature_size]. Otherwise this function might
    # fail.
    reduced = frame_swap(frames, frame_mask)
  elif method == "none":
    feature_size = frames.shape.as_list()[2]
    reduced = tf.reshape(frames, [-1, feature_size])
  else:
    raise ValueError("Unrecognized pooling method: %s" % method)

  return reduced
