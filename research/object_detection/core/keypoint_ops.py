# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Keypoint operations.

Keypoints are represented as tensors of shape [num_instances, num_keypoints, 2],
where the last dimension holds rank 2 tensors of the form [y, x] representing
the coordinates of the keypoint.
"""
import numpy as np
import tensorflow as tf


def scale(keypoints, y_scale, x_scale, scope=None):
  """Scales keypoint coordinates in x and y dimensions.

  Args:
    keypoints: a tensor of shape [num_instances, num_keypoints, 2]
    y_scale: (float) scalar tensor
    x_scale: (float) scalar tensor
    scope: name scope.

  Returns:
    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]
  """
  with tf.name_scope(scope, 'Scale'):
    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    new_keypoints = keypoints * [[[y_scale, x_scale]]]
    return new_keypoints


def clip_to_window(keypoints, window, scope=None):
  """Clips keypoints to a window.

  This op clips any input keypoints to a window.

  Args:
    keypoints: a tensor of shape [num_instances, num_keypoints, 2]
    window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]
      window to which the op should clip the keypoints.
    scope: name scope.

  Returns:
    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]
  """
  with tf.name_scope(scope, 'ClipToWindow'):
    y, x = tf.split(value=keypoints, num_or_size_splits=2, axis=2)
    win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
    y = tf.maximum(tf.minimum(y, win_y_max), win_y_min)
    x = tf.maximum(tf.minimum(x, win_x_max), win_x_min)
    new_keypoints = tf.concat([y, x], 2)
    return new_keypoints


def prune_outside_window(keypoints, window, scope=None):
  """Prunes keypoints that fall outside a given window.

  This function replaces keypoints that fall outside the given window with nan.
  See also clip_to_window which clips any keypoints that fall outside the given
  window.

  Args:
    keypoints: a tensor of shape [num_instances, num_keypoints, 2]
    window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]
      window outside of which the op should prune the keypoints.
    scope: name scope.

  Returns:
    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]
  """
  with tf.name_scope(scope, 'PruneOutsideWindow'):
    y, x = tf.split(value=keypoints, num_or_size_splits=2, axis=2)
    win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)

    valid_indices = tf.logical_and(
        tf.logical_and(y >= win_y_min, y <= win_y_max),
        tf.logical_and(x >= win_x_min, x <= win_x_max))

    new_y = tf.where(valid_indices, y, np.nan * tf.ones_like(y))
    new_x = tf.where(valid_indices, x, np.nan * tf.ones_like(x))
    new_keypoints = tf.concat([new_y, new_x], 2)

    return new_keypoints


def change_coordinate_frame(keypoints, window, scope=None):
  """Changes coordinate frame of the keypoints to be relative to window's frame.

  Given a window of the form [y_min, x_min, y_max, x_max], changes keypoint
  coordinates from keypoints of shape [num_instances, num_keypoints, 2]
  to be relative to this window.

  An example use case is data augmentation: where we are given groundtruth
  keypoints and would like to randomly crop the image to some window. In this
  case we need to change the coordinate frame of each groundtruth keypoint to be
  relative to this new window.

  Args:
    keypoints: a tensor of shape [num_instances, num_keypoints, 2]
    window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]
      window we should change the coordinate frame to.
    scope: name scope.

  Returns:
    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]
  """
  with tf.name_scope(scope, 'ChangeCoordinateFrame'):
    win_height = window[2] - window[0]
    win_width = window[3] - window[1]
    new_keypoints = scale(keypoints - [window[0], window[1]], 1.0 / win_height,
                          1.0 / win_width)
    return new_keypoints


def to_normalized_coordinates(keypoints, height, width,
                              check_range=True, scope=None):
  """Converts absolute keypoint coordinates to normalized coordinates in [0, 1].

  Usually one uses the dynamic shape of the image or conv-layer tensor:
    keypoints = keypoint_ops.to_normalized_coordinates(keypoints,
                                                       tf.shape(images)[1],
                                                       tf.shape(images)[2]),

  This function raises an assertion failed error at graph execution time when
  the maximum coordinate is smaller than 1.01 (which means that coordinates are
  already normalized). The value 1.01 is to deal with small rounding errors.

  Args:
    keypoints: A tensor of shape [num_instances, num_keypoints, 2].
    height: Maximum value for y coordinate of absolute keypoint coordinates.
    width: Maximum value for x coordinate of absolute keypoint coordinates.
    check_range: If True, checks if the coordinates are normalized.
    scope: name scope.

  Returns:
    tensor of shape [num_instances, num_keypoints, 2] with normalized
    coordinates in [0, 1].
  """
  with tf.name_scope(scope, 'ToNormalizedCoordinates'):
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)

    if check_range:
      max_val = tf.reduce_max(keypoints)
      max_assert = tf.Assert(tf.greater(max_val, 1.01),
                             ['max value is lower than 1.01: ', max_val])
      with tf.control_dependencies([max_assert]):
        width = tf.identity(width)

    return scale(keypoints, 1.0 / height, 1.0 / width)


def to_absolute_coordinates(keypoints, height, width,
                            check_range=True, scope=None):
  """Converts normalized keypoint coordinates to absolute pixel coordinates.

  This function raises an assertion failed error when the maximum keypoint
  coordinate value is larger than 1.01 (in which case coordinates are already
  absolute).

  Args:
    keypoints: A tensor of shape [num_instances, num_keypoints, 2]
    height: Maximum value for y coordinate of absolute keypoint coordinates.
    width: Maximum value for x coordinate of absolute keypoint coordinates.
    check_range: If True, checks if the coordinates are normalized or not.
    scope: name scope.

  Returns:
    tensor of shape [num_instances, num_keypoints, 2] with absolute coordinates
    in terms of the image size.

  """
  with tf.name_scope(scope, 'ToAbsoluteCoordinates'):
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)

    # Ensure range of input keypoints is correct.
    if check_range:
      max_val = tf.reduce_max(keypoints)
      max_assert = tf.Assert(tf.greater_equal(1.01, max_val),
                             ['maximum keypoint coordinate value is larger '
                              'than 1.01: ', max_val])
      with tf.control_dependencies([max_assert]):
        width = tf.identity(width)

    return scale(keypoints, height, width)


def flip_horizontal(keypoints, flip_point, flip_permutation, scope=None):
  """Flips the keypoints horizontally around the flip_point.

  This operation flips the x coordinate for each keypoint around the flip_point
  and also permutes the keypoints in a manner specified by flip_permutation.

  Args:
    keypoints: a tensor of shape [num_instances, num_keypoints, 2]
    flip_point:  (float) scalar tensor representing the x coordinate to flip the
      keypoints around.
    flip_permutation: rank 1 int32 tensor containing the keypoint flip
      permutation. This specifies the mapping from original keypoint indices
      to the flipped keypoint indices. This is used primarily for keypoints
      that are not reflection invariant. E.g. Suppose there are 3 keypoints
      representing ['head', 'right_eye', 'left_eye'], then a logical choice for
      flip_permutation might be [0, 2, 1] since we want to swap the 'left_eye'
      and 'right_eye' after a horizontal flip.
    scope: name scope.

  Returns:
    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]
  """
  with tf.name_scope(scope, 'FlipHorizontal'):
    keypoints = tf.transpose(keypoints, [1, 0, 2])
    keypoints = tf.gather(keypoints, flip_permutation)
    v, u = tf.split(value=keypoints, num_or_size_splits=2, axis=2)
    u = flip_point * 2.0 - u
    new_keypoints = tf.concat([v, u], 2)
    new_keypoints = tf.transpose(new_keypoints, [1, 0, 2])
    return new_keypoints


def flip_vertical(keypoints, flip_point, flip_permutation, scope=None):
  """Flips the keypoints vertically around the flip_point.

  This operation flips the y coordinate for each keypoint around the flip_point
  and also permutes the keypoints in a manner specified by flip_permutation.

  Args:
    keypoints: a tensor of shape [num_instances, num_keypoints, 2]
    flip_point:  (float) scalar tensor representing the y coordinate to flip the
      keypoints around.
    flip_permutation: rank 1 int32 tensor containing the keypoint flip
      permutation. This specifies the mapping from original keypoint indices
      to the flipped keypoint indices. This is used primarily for keypoints
      that are not reflection invariant. E.g. Suppose there are 3 keypoints
      representing ['head', 'right_eye', 'left_eye'], then a logical choice for
      flip_permutation might be [0, 2, 1] since we want to swap the 'left_eye'
      and 'right_eye' after a horizontal flip.
    scope: name scope.

  Returns:
    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]
  """
  with tf.name_scope(scope, 'FlipVertical'):
    keypoints = tf.transpose(keypoints, [1, 0, 2])
    keypoints = tf.gather(keypoints, flip_permutation)
    v, u = tf.split(value=keypoints, num_or_size_splits=2, axis=2)
    v = flip_point * 2.0 - v
    new_keypoints = tf.concat([v, u], 2)
    new_keypoints = tf.transpose(new_keypoints, [1, 0, 2])
    return new_keypoints


def rot90(keypoints, scope=None):
  """Rotates the keypoints counter-clockwise by 90 degrees.

  Args:
    keypoints: a tensor of shape [num_instances, num_keypoints, 2]
    scope: name scope.

  Returns:
    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]
  """
  with tf.name_scope(scope, 'Rot90'):
    keypoints = tf.transpose(keypoints, [1, 0, 2])
    v, u = tf.split(value=keypoints[:, :, ::-1], num_or_size_splits=2, axis=2)
    v = 1.0 - v
    new_keypoints = tf.concat([v, u], 2)
    new_keypoints = tf.transpose(new_keypoints, [1, 0, 2])
    return new_keypoints
