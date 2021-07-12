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

"""DensePose operations.

DensePose part ids are represented as tensors of shape
[num_instances, num_points] and coordinates are represented as tensors of shape
[num_instances, num_points, 4] where each point holds (y, x, v, u). The location
of the DensePose sampled point is (y, x) in normalized coordinates. The surface
coordinate (in the part coordinate frame) is (v, u). Note that dim 1 of both
tensors may contain padding, since the number of sampled points per instance
is not fixed. The value `num_points` represents the maximum number of sampled
points for an instance in the example.
"""
import os

import numpy as np
import scipy.io
import tensorflow.compat.v1 as tf

from object_detection.utils import shape_utils

PART_NAMES = [
    b'torso_back', b'torso_front', b'right_hand', b'left_hand', b'left_foot',
    b'right_foot', b'right_upper_leg_back', b'left_upper_leg_back',
    b'right_upper_leg_front', b'left_upper_leg_front', b'right_lower_leg_back',
    b'left_lower_leg_back', b'right_lower_leg_front', b'left_lower_leg_front',
    b'left_upper_arm_back', b'right_upper_arm_back', b'left_upper_arm_front',
    b'right_upper_arm_front', b'left_lower_arm_back', b'right_lower_arm_back',
    b'left_lower_arm_front', b'right_lower_arm_front', b'right_face',
    b'left_face',
]


def scale(dp_surface_coords, y_scale, x_scale, scope=None):
  """Scales DensePose coordinates in y and x dimensions.

  Args:
    dp_surface_coords: a tensor of shape [num_instances, num_points, 4], with
      coordinates in (y, x, v, u) format.
    y_scale: (float) scalar tensor
    x_scale: (float) scalar tensor
    scope: name scope.

  Returns:
    new_dp_surface_coords: a tensor of shape [num_instances, num_points, 4]
  """
  with tf.name_scope(scope, 'DensePoseScale'):
    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    new_keypoints = dp_surface_coords * [[[y_scale, x_scale, 1, 1]]]
    return new_keypoints


def clip_to_window(dp_surface_coords, window, scope=None):
  """Clips DensePose points to a window.

  This op clips any input DensePose points to a window.

  Args:
    dp_surface_coords: a tensor of shape [num_instances, num_points, 4] with
      DensePose surface coordinates in (y, x, v, u) format.
    window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]
      window to which the op should clip the keypoints.
    scope: name scope.

  Returns:
    new_dp_surface_coords: a tensor of shape [num_instances, num_points, 4].
  """
  with tf.name_scope(scope, 'DensePoseClipToWindow'):
    y, x, v, u = tf.split(value=dp_surface_coords, num_or_size_splits=4, axis=2)
    win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
    y = tf.maximum(tf.minimum(y, win_y_max), win_y_min)
    x = tf.maximum(tf.minimum(x, win_x_max), win_x_min)
    new_dp_surface_coords = tf.concat([y, x, v, u], 2)
    return new_dp_surface_coords


def prune_outside_window(dp_num_points, dp_part_ids, dp_surface_coords, window,
                         scope=None):
  """Prunes DensePose points that fall outside a given window.

  This function replaces points that fall outside the given window with zeros.
  See also clip_to_window which clips any DensePose points that fall outside the
  given window.

  Note that this operation uses dynamic shapes, and therefore is not currently
  suitable for TPU.

  Args:
    dp_num_points: a tensor of shape [num_instances] that indicates how many
      (non-padded) DensePose points there are per instance.
    dp_part_ids: a tensor of shape [num_instances, num_points] with DensePose
      part ids. These part_ids are 0-indexed, where the first non-background
      part has index 0.
    dp_surface_coords: a tensor of shape [num_instances, num_points, 4] with
      DensePose surface coordinates in (y, x, v, u) format.
    window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]
      window outside of which the op should prune the points.
    scope: name scope.

  Returns:
    new_dp_num_points: a tensor of shape [num_instances] that indicates how many
      (non-padded) DensePose points there are per instance after pruning.
    new_dp_part_ids: a tensor of shape [num_instances, num_points] with
      DensePose part ids. These part_ids are 0-indexed, where the first
      non-background part has index 0.
    new_dp_surface_coords: a tensor of shape [num_instances, num_points, 4] with
      DensePose surface coordinates after pruning.
  """
  with tf.name_scope(scope, 'DensePosePruneOutsideWindow'):
    y, x, _, _ = tf.unstack(dp_surface_coords, axis=-1)
    win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)

    num_instances, num_points = shape_utils.combined_static_and_dynamic_shape(
        dp_part_ids)
    dp_num_points_tiled = tf.tile(dp_num_points[:, tf.newaxis],
                                  multiples=[1, num_points])
    range_tiled = tf.tile(tf.range(num_points)[tf.newaxis, :],
                          multiples=[num_instances, 1])
    valid_initial = range_tiled < dp_num_points_tiled
    valid_in_window = tf.logical_and(
        tf.logical_and(y >= win_y_min, y <= win_y_max),
        tf.logical_and(x >= win_x_min, x <= win_x_max))
    valid_indices = tf.logical_and(valid_initial, valid_in_window)

    new_dp_num_points = tf.math.reduce_sum(
        tf.cast(valid_indices, tf.int32), axis=1)
    max_num_points = tf.math.reduce_max(new_dp_num_points)

    def gather_and_reshuffle(elems):
      dp_part_ids, dp_surface_coords, valid_indices = elems
      locs = tf.where(valid_indices)[:, 0]
      valid_part_ids = tf.gather(dp_part_ids, locs, axis=0)
      valid_part_ids_padded = shape_utils.pad_or_clip_nd(
          valid_part_ids, output_shape=[max_num_points])
      valid_surface_coords = tf.gather(dp_surface_coords, locs, axis=0)
      valid_surface_coords_padded = shape_utils.pad_or_clip_nd(
          valid_surface_coords, output_shape=[max_num_points, 4])
      return [valid_part_ids_padded, valid_surface_coords_padded]

    new_dp_part_ids, new_dp_surface_coords = (
        shape_utils.static_or_dynamic_map_fn(
            gather_and_reshuffle,
            elems=[dp_part_ids, dp_surface_coords, valid_indices],
            dtype=[tf.int32, tf.float32],
            back_prop=False))
    return new_dp_num_points, new_dp_part_ids, new_dp_surface_coords


def change_coordinate_frame(dp_surface_coords, window, scope=None):
  """Changes coordinate frame of the points to be relative to window's frame.

  Given a window of the form [y_min, x_min, y_max, x_max] in normalized
  coordinates, changes DensePose coordinates to be relative to this window.

  An example use case is data augmentation: where we are given groundtruth
  points and would like to randomly crop the image to some window. In this
  case we need to change the coordinate frame of each sampled point to be
  relative to this new window.

  Args:
    dp_surface_coords: a tensor of shape [num_instances, num_points, 4] with
      DensePose surface coordinates in (y, x, v, u) format.
    window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]
      window we should change the coordinate frame to.
    scope: name scope.

  Returns:
    new_dp_surface_coords: a tensor of shape [num_instances, num_points, 4].
  """
  with tf.name_scope(scope, 'DensePoseChangeCoordinateFrame'):
    win_height = window[2] - window[0]
    win_width = window[3] - window[1]
    new_dp_surface_coords = scale(
        dp_surface_coords - [window[0], window[1], 0, 0],
        1.0 / win_height, 1.0 / win_width)
    return new_dp_surface_coords


def to_normalized_coordinates(dp_surface_coords, height, width,
                              check_range=True, scope=None):
  """Converts absolute DensePose coordinates to normalized in range [0, 1].

  This function raises an assertion failed error at graph execution time when
  the maximum coordinate is smaller than 1.01 (which means that coordinates are
  already normalized). The value 1.01 is to deal with small rounding errors.

  Args:
    dp_surface_coords: a tensor of shape [num_instances, num_points, 4] with
      DensePose absolute surface coordinates in (y, x, v, u) format.
    height: Height of image.
    width: Width of image.
    check_range: If True, checks if the coordinates are already normalized.
    scope: name scope.

  Returns:
    A tensor of shape [num_instances, num_points, 4] with normalized
    coordinates.
  """
  with tf.name_scope(scope, 'DensePoseToNormalizedCoordinates'):
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)

    if check_range:
      max_val = tf.reduce_max(dp_surface_coords[:, :, :2])
      max_assert = tf.Assert(tf.greater(max_val, 1.01),
                             ['max value is lower than 1.01: ', max_val])
      with tf.control_dependencies([max_assert]):
        width = tf.identity(width)

    return scale(dp_surface_coords, 1.0 / height, 1.0 / width)


def to_absolute_coordinates(dp_surface_coords, height, width,
                            check_range=True, scope=None):
  """Converts normalized DensePose coordinates to absolute pixel coordinates.

  This function raises an assertion failed error when the maximum
  coordinate value is larger than 1.01 (in which case coordinates are already
  absolute).

  Args:
    dp_surface_coords: a tensor of shape [num_instances, num_points, 4] with
      DensePose normalized surface coordinates in (y, x, v, u) format.
    height: Height of image.
    width: Width of image.
    check_range: If True, checks if the coordinates are normalized or not.
    scope: name scope.

  Returns:
    A tensor of shape [num_instances, num_points, 4] with absolute coordinates.
  """
  with tf.name_scope(scope, 'DensePoseToAbsoluteCoordinates'):
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)

    if check_range:
      max_val = tf.reduce_max(dp_surface_coords[:, :, :2])
      max_assert = tf.Assert(tf.greater_equal(1.01, max_val),
                             ['maximum coordinate value is larger than 1.01: ',
                              max_val])
      with tf.control_dependencies([max_assert]):
        width = tf.identity(width)

    return scale(dp_surface_coords, height, width)


class DensePoseHorizontalFlip(object):
  """Class responsible for horizontal flipping of parts and surface coords."""

  def __init__(self):
    """Constructor."""

    path = os.path.dirname(os.path.abspath(__file__))
    uv_symmetry_transforms_path = tf.resource_loader.get_path_to_datafile(
        os.path.join(path, '..', 'dataset_tools', 'densepose',
                     'UV_symmetry_transforms.mat'))
    tf.logging.info('Loading DensePose symmetry transforms file from {}'.format(
        uv_symmetry_transforms_path))
    with tf.io.gfile.GFile(uv_symmetry_transforms_path, 'rb') as f:
      data = scipy.io.loadmat(f)

    # Create lookup maps which indicate how a VU coordinate changes after a
    # horizontal flip.
    uv_symmetry_map = {}
    for key in ('U_transforms', 'V_transforms'):
      uv_symmetry_map_per_part = []
      for i in range(data[key].shape[1]):
        # The following tensor has shape [256, 256]. The raw data is stored as
        # uint8 values, so convert to float and scale to the range [0., 1.]
        data_normalized = data[key][0, i].astype(np.float32) / 255.
        map_per_part = tf.constant(data_normalized, dtype=tf.float32)
        uv_symmetry_map_per_part.append(map_per_part)
      uv_symmetry_map[key] = tf.reshape(
          tf.stack(uv_symmetry_map_per_part, axis=0), [-1])
    # The following dictionary contains flattened lookup maps for the U and V
    # coordinates separately. The shape of each is [24 * 256 * 256].
    self.uv_symmetries = uv_symmetry_map

    # Create a list of that maps part index to flipped part index (0-indexed).
    part_symmetries = []
    for i, part_name in enumerate(PART_NAMES):
      if b'left' in part_name:
        part_symmetries.append(PART_NAMES.index(
            part_name.replace(b'left', b'right')))
      elif b'right' in part_name:
        part_symmetries.append(PART_NAMES.index(
            part_name.replace(b'right', b'left')))
      else:
        part_symmetries.append(i)
    self.part_symmetries = part_symmetries

  def flip_parts_and_coords(self, part_ids, vu):
    """Flips part ids and coordinates.

    Args:
      part_ids: a [num_instances, num_points] int32 tensor with pre-flipped part
        ids. These part_ids are 0-indexed, where the first non-background part
        has index 0.
      vu: a [num_instances, num_points, 2] float32 tensor with pre-flipped vu
        normalized coordinates.

    Returns:
      new_part_ids: a [num_instances, num_points] int32 tensor with post-flipped
        part ids. These part_ids are 0-indexed, where the first non-background
        part has index 0.
      new_vu: a [num_instances, num_points, 2] float32 tensor with post-flipped
        vu coordinates.
    """
    num_instances, num_points = shape_utils.combined_static_and_dynamic_shape(
        part_ids)
    part_ids_flattened = tf.reshape(part_ids, [-1])
    new_part_ids_flattened = tf.gather(self.part_symmetries, part_ids_flattened)
    new_part_ids = tf.reshape(new_part_ids_flattened,
                              [num_instances, num_points])

    # Convert VU floating point coordinates to values in [256, 256] grid.
    vu = tf.math.minimum(tf.math.maximum(vu, 0.0), 1.0)
    vu_locs = tf.cast(vu * 256., dtype=tf.int32)
    vu_locs_flattened = tf.reshape(vu_locs, [-1, 2])
    v_locs_flattened, u_locs_flattened = tf.unstack(vu_locs_flattened, axis=1)

    # Convert vu_locs into lookup indices (in flattened part symmetries map).
    symmetry_lookup_inds = (
        part_ids_flattened * 65536 + 256 * v_locs_flattened + u_locs_flattened)

    # New VU coordinates.
    v_new = tf.gather(self.uv_symmetries['V_transforms'], symmetry_lookup_inds)
    u_new = tf.gather(self.uv_symmetries['U_transforms'], symmetry_lookup_inds)
    new_vu_flattened = tf.stack([v_new, u_new], axis=1)
    new_vu = tf.reshape(new_vu_flattened, [num_instances, num_points, 2])

    return new_part_ids, new_vu


def flip_horizontal(dp_part_ids, dp_surface_coords, scope=None):
  """Flips the DensePose points horizontally around the flip_point.

  This operation flips dense pose annotations horizontally. Note that part ids
  and surface coordinates may or may not change as a result of the flip.

  Args:
    dp_part_ids: a tensor of shape [num_instances, num_points] with DensePose
      part ids. These part_ids are 0-indexed, where the first non-background
      part has index 0.
    dp_surface_coords: a tensor of shape [num_instances, num_points, 4] with
      DensePose surface coordinates in (y, x, v, u) normalized format.
    scope: name scope.

  Returns:
    new_dp_part_ids: a tensor of shape [num_instances, num_points] with
      DensePose part ids after flipping.
    new_dp_surface_coords: a tensor of shape [num_instances, num_points, 4] with
      DensePose surface coordinates after flipping.
  """
  with tf.name_scope(scope, 'DensePoseFlipHorizontal'):
    # First flip x coordinate.
    y, x, vu = tf.split(dp_surface_coords, num_or_size_splits=[1, 1, 2], axis=2)
    xflipped = 1.0 - x

    # Flip part ids and surface coordinates.
    horizontal_flip = DensePoseHorizontalFlip()
    new_dp_part_ids, new_vu = horizontal_flip.flip_parts_and_coords(
        dp_part_ids, vu)
    new_dp_surface_coords = tf.concat([y, xflipped, new_vu], axis=2)
    return new_dp_part_ids, new_dp_surface_coords

