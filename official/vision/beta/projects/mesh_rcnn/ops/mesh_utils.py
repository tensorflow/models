# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Utility functions for mesh input processing."""

from typing import List, Tuple

import tensorflow as tf

from official.vision.detection.utils.input_utils import pad_to_fixed_size


def list_to_padded_tensor(x: List[tf.Tensor],
                          pad_value: int = -1) -> Tuple[tf.Tensor]: 
  """Converts a list of tensors into a padded tensor

  Each tensor in the input is padded to a fixed size and stacked to create
  the final output tensor.

  Args:
    x: A `list` of 1D tensors of any sizes.
    pad_value: `int` value assigned to the paddings.

  Returns:
    `Tensor` with shape [n, max_size] where n is the number of tensors in x and
      max_size is the length of the longest tensor in x.
  """
  pad_size = max([len(y) for y in x])
  padded_list = []

  for y in x:
    padded_list.append(pad_to_fixed_size(
        input_tensor=y, size=pad_size, constant_values=pad_value))

  padded_verts = tf.stack(padded_list, axis=0)

  return padded_verts

def list_to_packed_tensor(x: List[tf.Tensor]) -> Tuple[tf.Tensor]:
  """Converts a list of tensors into a packed tensor.

  Args:
    x: A `list` of 1D tensors of any sizes.

  Returns:
    `Tensor` with shape [sum_size] where sum_size is the sum of the lengths of
     each tensor in x.
  """
  num_tensors = len(x)

  # Packing the tensors from list x
  x_packed = tf.concat(x, axis=0)

  # Get the number of items in each tensor
  num_items = tf.convert_to_tensor([tf.shape(y)[0] for y in x])

  # tf.cumsum gives the cumulative sum over the values in each tensor from x.
  # To get the start indices, we subtract the values from the cumulative sum
  items_packed_first_idx = tf.cumsum(num_items, axis=0) - num_items

  # This gives the original list idx of each tensor in the input
  items_packed_to_list_idx = tf.repeat(tf.range(num_tensors), num_items)

  return x_packed, num_items, items_packed_first_idx, items_packed_to_list_idx
