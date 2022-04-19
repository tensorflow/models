# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

from typing import Union, List
import tensorflow as tf
from official.vision.beta.ops import preprocess_ops

def horizontal_flip_coords(coords: tf.Tensor) -> tf.Tensor:
  """Flips coordinates horizontally.

  Args:
    coords: A `Tensor` of shape [V, 3] where V is the number of coordinates and
      each coordinate is given in x, y, z order.
  Returns:
    flipped_coords: A `Tensor` of shape [N, 3] with the x-coordinates flipped.
  
  """
  flipped_coords = tf.stack(
    [-1 * coords[:, 0], coords[:, 1], coords[:, 2]], axis=1
  )

  return flipped_coords

def resize_coords(coords: tf.Tensor,
                  image_scale: Union[tf.Tensor, List]) -> tf.Tensor:
  """Resizes coordinates vertically and horizontally.

  Args:
    coords: A `Tensor` of shape [V, 3] where V is the number of coordinates and
      each coordinate is given in x, y, z order.
    image_scale: A `Tensor` of `list` of shape [2] containing the vertical and horizontal
      scaling factors to multiply the coordinates by. 
  Returns:
    resized_coords: A `Tensor` of shape [V,  3] with the scaled coordinates.
  
  """
  y_scale, x_scale = image_scale[0], image_scale[1]
  
  resized_coords = tf.stack(
    [coords[:, 0] * x_scale, coords[:, 1] * y_scale, coords[:, 2]], axis=1
  )

  return resized_coords

def random_horizontal_flip(image,
                           normalized_boxes=None,
                           masks=None,
                           verts=None,
                           voxel_verts=None,
                           seed=1):
  """Randomly flips input image, bounding boxes, masks, and coordinates."""
  with tf.name_scope('random_horizontal_flip'):
    do_flip = tf.greater(tf.random.uniform([], seed=seed), 0.5)

    image = tf.cond(
        do_flip,
        lambda: preprocess_ops.horizontal_flip_image(image),
        lambda: image)

    if normalized_boxes is not None:
      normalized_boxes = tf.cond(
          do_flip,
          lambda: preprocess_ops.horizontal_flip_boxes(normalized_boxes),
          lambda: normalized_boxes)

    if masks is not None:
      masks = tf.cond(
          do_flip,
          lambda: preprocess_ops.horizontal_flip_masks(masks),
          lambda: masks)
    
    if verts is not None:
      verts = tf.cond(
          do_flip,
          lambda: horizontal_flip_coords(verts),
          lambda: verts)
    
    if voxel_verts is not None:
      voxel_verts = tf.cond(
          do_flip,
          lambda: horizontal_flip_coords(voxel_verts),
          lambda: voxel_verts)

    return image, normalized_boxes, masks, verts, voxel_verts

def voxel_to_verts(voxel: tf.Tensor) -> tf.Tensor:
  """Converts from voxel occupancy grid to x, y, z coordinates.

  The conversion is performed in 3 steps. First, the voxels are rotated. 
  Then the locations of the occupied voxels are found and converted into
  coordinates. Finally, the coordinates are centered and normalized.

  Args:
    voxel: A `Tensor` of shape [H, W, D] that represents the voxel occupancy of
      an object.
  Returns:
    verts: A `Tensor` of shape [V, 3] containing the coordinates of the occupied
      voxels.
  """

  # Perform 270 degree rotation about axes 1 and 2
  voxel = tf.transpose(voxel, perm=[1, 2, 0])
  voxel = tf.image.rot90(voxel, k=3)
  voxel = tf.transpose(voxel, perm=[2, 0, 1])

  # Determine locations of occupied voxels
  occupied_indices = tf.where(voxel > 0)
  occupied_indices = tf.cast(occupied_indices, dtype=tf.float32)
  
  # Center the coordinates
  min_x = tf.math.reduce_min(occupied_indices[:, 0])
  max_x = tf.math.reduce_max(occupied_indices[:, 0])
  min_y = tf.math.reduce_min(occupied_indices[:, 1])
  max_y = tf.math.reduce_max(occupied_indices[:, 1])
  min_z = tf.math.reduce_min(occupied_indices[:, 2])
  max_z = tf.math.reduce_max(occupied_indices[:, 2])
 
  new_x = occupied_indices[:, 0] - (max_x + min_x) / 2
  new_y = occupied_indices[:, 1] - (max_y + min_y) / 2
  new_z = occupied_indices[:, 2] - (max_z + min_z) / 2
  verts = tf.stack([new_x, new_y, new_z], axis=1)

  # Normalizing
  scale = tf.math.sqrt(
      tf.math.reduce_max(tf.math.reduce_sum(verts ** 2, axis=1))) * 2
  verts /= scale

  return verts

def apply_3d_transformations(verts: tf.Tensor,
                             rot_mat: Union[tf.Tensor, None] = None,
                             trans_mat: Union[tf.Tensor, None] = None
    ) -> tf.Tensor: 
  """Performs rotation and translation transformations on 3d coordinates. 

  Args:
    verts: A `Tensor` of shape [V, 3] containing the coordinates to transform.
    rot_mat: A `Tensor` of shape [3, 3] used to rotate the coordinates.
    trans_mat: A `Tensor` of shape [3] used to translate the coordinates.
  Returns:
    transformed_verts: A `Tensor` of shape [V, 3] containing the transformed
    coordinates.
  """
  verts = tf.transpose(verts)
  if rot_mat is not None:
    transformed_verts = tf.linalg.matmul(rot_mat, verts)
  
  if trans_mat is not None:
    transformed_verts = transformed_verts + tf.expand_dims(trans_mat, axis=1)

  transformed_verts = tf.transpose(transformed_verts)
  return transformed_verts
