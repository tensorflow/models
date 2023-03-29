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

"""Losses utilities for detection models."""

import tensorflow as tf


def multi_level_flatten(multi_level_inputs, last_dim=None):
  """Flattens a multi-level input.

  Args:
    multi_level_inputs: Ordered Dict with level to [batch, d1, ..., dm].
    last_dim: Whether the output should be [batch_size, None], or [batch_size,
      None, last_dim]. Defaults to `None`.

  Returns:
    Concatenated output [batch_size, None], or [batch_size, None, dm]
  """
  flattened_inputs = []
  batch_size = None
  for level in multi_level_inputs.keys():
    single_input = multi_level_inputs[level]
    if batch_size is None:
      batch_size = single_input.shape[0] or tf.shape(single_input)[0]
    if last_dim is not None:
      flattened_input = tf.reshape(single_input, [batch_size, -1, last_dim])
    else:
      flattened_input = tf.reshape(single_input, [batch_size, -1])
    flattened_inputs.append(flattened_input)
  return tf.concat(flattened_inputs, axis=1)
