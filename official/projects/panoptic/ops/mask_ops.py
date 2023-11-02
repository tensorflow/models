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

"""Utility functions for masks."""

import tensorflow as tf, tf_keras


def resize_and_rescale_offsets(input_tensor: tf.Tensor, target_size):
  """Bilinearly resizes and rescales the offsets.

    Reference:
    https://github.com/google-research/deeplab2/blob/main/model/utils.py#L157

  Args:
    input_tensor: A tf.Tensor of shape [batch, height, width, 2].
    target_size: A list or tuple or 1D tf.Tensor that specifies the height and
      width after resizing.

  Returns:
    The input_tensor resized to shape `[batch, target_height, target_width, 2]`.
      Moreover, the offsets along the y-axis are rescaled by a factor equal to
      (target_height - 1) / (reference_height - 1) and the offsets along the
      x-axis are rescaled by a factor equal to
      (target_width - 1) / (reference_width - 1).
  """
  input_size_y = tf.shape(input_tensor)[1]
  input_size_x = tf.shape(input_tensor)[2]
  dtype = input_tensor.dtype

  scale_y = tf.cast(target_size[0] - 1, dtype=dtype) / tf.cast(
      input_size_y - 1, dtype=dtype)
  scale_x = tf.cast(target_size[1] - 1, dtype=dtype) / tf.cast(
      input_size_x - 1, dtype=dtype)

  target_y, target_x = tf.split(
      value=input_tensor, num_or_size_splits=2, axis=3)
  target_y *= scale_y
  target_x *= scale_x
  _ = tf.concat([target_y, target_x], 3)
  return tf.image.resize(
      input_tensor,
      size=target_size,
      method=tf.image.ResizeMethod.BILINEAR)
