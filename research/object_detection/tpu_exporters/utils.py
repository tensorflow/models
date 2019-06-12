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
"""Utilities for TPU inference."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def bfloat16_to_float32(tensor):
  """Converts a tensor to tf.float32 only if it is tf.bfloat16."""
  if tensor.dtype == tf.bfloat16:
    return tf.cast(tensor, dtype=tf.float32)
  else:
    return tensor


def bfloat16_to_float32_nested(bfloat16_tensor_dict):
  """Converts bfloat16 tensors in a nested structure to float32.

  Other tensors not of dtype bfloat16 will be left as is.

  Args:
    bfloat16_tensor_dict: A Python dict, values being Tensor or Python
      list/tuple of Tensor.

  Returns:
    A Python dict with the same structure as `bfloat16_tensor_dict`,
    with all bfloat16 tensors converted to float32.
  """
  float32_tensor_dict = {}
  for k, v in bfloat16_tensor_dict.items():
    if isinstance(v, tf.Tensor):
      float32_tensor_dict[k] = bfloat16_to_float32(v)
    elif isinstance(v, (list, tuple)):
      float32_tensor_dict[k] = [bfloat16_to_float32(t) for t in v]
  return float32_tensor_dict
