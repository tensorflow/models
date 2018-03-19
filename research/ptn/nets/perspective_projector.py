# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""3D->2D projector model as used in PTN (NIPS16)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import perspective_transform


def model(voxels, transform_matrix, params, is_training):
  """Model transforming the 3D voxels into 2D projections.

  Args:
    voxels: A tensor of size [batch, depth, height, width, channel]
      representing the input of projection layer (tf.float32).
    transform_matrix: A tensor of size [batch, 16] representing
      the flattened 4-by-4 matrix for transformation (tf.float32).
    params: Model parameters (dict).
    is_training: Set to True if while training (boolean).

  Returns:
    A transformed tensor (tf.float32)

  """
  del is_training  # Doesn't make a difference for projector
  # Rearrangement (batch, z, y, x, channel) --> (batch, y, z, x, channel).
  # By the standard, projection happens along z-axis but the voxels
  # are stored in a different way. So we need to switch the y and z
  # axis for transformation operation.
  voxels = tf.transpose(voxels, [0, 2, 1, 3, 4])
  z_near = params.focal_length
  z_far = params.focal_length + params.focal_range
  transformed_voxels = perspective_transform.transformer(
      voxels, transform_matrix, [params.vox_size] * 3, z_near, z_far)
  views = tf.reduce_max(transformed_voxels, [1])
  views = tf.reverse(views, [1])
  return views
