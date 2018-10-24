# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Ops for a StarGAN model.

This module contains basic ops to build a StarGAN model.

See https://arxiv.org/abs/1711.09020 for details about the model.

See https://github.com/yunjey/StarGAN for the original pytorvh implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf


def _padding_arg(h, w, input_format):
  """Calculate the padding shape for tf.pad().

  Args:
    h: (int) padding on the height dim.
    w: (int) padding on the width dim.
    input_format: (string) the input format as in 'NHWC' or 'HWC'.
  Raises:
    ValueError: If input_format is not 'NHWC' or 'HWC'.

  Returns:
    A two dimension array representing the padding argument.
  """
  if input_format == 'NHWC':
    return [[0, 0], [h, h], [w, w], [0, 0]]
  elif input_format == 'HWC':
    return [[h, h], [w, w], [0, 0]]
  else:
    raise ValueError('Input Format %s is not supported.' % input_format)


def pad(input_net, padding_size):
  """Padding the tensor with padding_size on both the height and width dim.

  Args:
    input_net: Tensor in 3D ('HWC') or 4D ('NHWC').
    padding_size: (int) the size of the padding.

  Notes:
    Original StarGAN use zero padding instead of mirror padding.

  Raises:
    ValueError: If input_net Tensor is not 3D or 4D.

  Returns:
    Tensor with same rank as input_net but with padding on the height and width
    dim.
  """
  if len(input_net.shape) == 4:
    return tf.pad(input_net, _padding_arg(padding_size, padding_size, 'NHWC'))
  elif len(input_net.shape) == 3:
    return tf.pad(input_net, _padding_arg(padding_size, padding_size, 'HWC'))
  else:
    raise ValueError('The input tensor need to be either 3D or 4D.')


def condition_input_with_pixel_padding(input_tensor, condition_tensor):
  """Pad image tensor with condition tensor as additional color channel.

  Args:
    input_tensor: Tensor of shape (batch_size, h, w, c) representing images.
    condition_tensor: Tensor of shape (batch_size, num_domains) representing the
      associated domain for the image in input_tensor.

  Returns:
    Tensor of shape (batch_size, h, w, c + num_domains) representing the
    conditioned data.

  Raises:
    ValueError: If `input_tensor` isn't rank 4.
    ValueError: If `condition_tensor` isn't rank 2.
    ValueError: If dimension 1 of the input_tensor and condition_tensor is not
      the same.
  """

  input_tensor.shape.assert_has_rank(4)
  condition_tensor.shape.assert_has_rank(2)
  input_tensor.shape[:1].assert_is_compatible_with(condition_tensor.shape[:1])
  condition_tensor = tf.expand_dims(condition_tensor, axis=1)
  condition_tensor = tf.expand_dims(condition_tensor, axis=1)
  condition_tensor = tf.tile(
      condition_tensor, [1, input_tensor.shape[1], input_tensor.shape[2], 1])

  return tf.concat([input_tensor, condition_tensor], -1)
