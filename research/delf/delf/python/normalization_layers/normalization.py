# Copyright 2021 The TensorFlow Authors All Rights Reserved.
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
"""Normalization layer definitions."""

import tensorflow as tf


class L2Normalization(tf.keras.layers.Layer):
  """Normalization layer using L2 norm."""

  def __init__(self):
    """Initialization of the L2Normalization layer."""
    super(L2Normalization, self).__init__()
    # A lower bound value for the norm.
    self.eps = 1e-6

  def call(self, x, axis=1):
    """Invokes the L2Normalization instance.

    Args:
      x: A Tensor.
      axis: Dimension along which to normalize. A scalar or a vector of
        integers.

    Returns:
      norm: A Tensor with the same shape as `x`.
    """
    return tf.nn.l2_normalize(x, axis, epsilon=self.eps)
