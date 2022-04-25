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

"""Tensorflow decoder for the Pix3D dataset."""

from typing import Union

import tensorflow as tf

from official.vision.dataloaders import tf_example_decoder


def _generate_source_id(image_bytes):
  # Hashing using 22 bits since float32 has only 23 mantissa bits.
  return tf.strings.as_string(
      tf.strings.to_hash_bucket_fast(image_bytes, 2 ** 22 - 1))

class Pix3dDecoder(tf_example_decoder.TfExampleDecoder):
  """Pix3D Decoder."""

  def __init__(self,
               include_mask: bool = False,
               regenerate_source_id: bool = False,
               mask_binarize_threshold: Union[float, None] = None):
    super(Pix3dDecoder, self).__init__(
        include_mask, regenerate_source_id, mask_binarize_threshold)

    # Additional keys and features for mesh labels
    self._keys_to_features.update({
        'model/vertices':
            tf.io.FixedLenFeature((), tf.string),
        'model/faces':
            tf.io.FixedLenFeature((), tf.string),
        'model/voxel_indices':
            tf.io.FixedLenFeature((), tf.string),
        'camera/rot_mat':
            tf.io.FixedLenFeature((), tf.string),
        'camera/trans_mat':
            tf.io.VarLenFeature(tf.float32),
        'camera/intrinstic_mat':
            tf.io.VarLenFeature(tf.float32),
    })

  def _decode_mesh(self, parsed_tensors: dict):
    """Decode the mesh data into tensors.

    Args:
      parsed_tensors: A `dict` mapping feature keys to corresponding `Tensors`.
    Returns:
      verts: A `Tensor` of shape [V, 3] containing the vertices data
        of the groundtruth mesh, where V is the number of vertices.
      faces: A `Tensor` of shape [F, 3] containing the faces data
        of the groundtruth mesh, where F is the number of vertices.
    """
    verts = tf.io.parse_tensor(
        parsed_tensors['model/vertices'], out_type=tf.float64)
    verts = tf.ensure_shape(verts, [None, 3])

    faces = tf.io.parse_tensor(
        parsed_tensors['model/faces'], out_type=tf.int32)
    faces = tf.ensure_shape(faces, [None, 3])

    return verts, faces

  def _decode_voxel(self, parsed_tensors: dict):
    """Decode the voxel data into a tensor.

    Args:
      parsed_tensors: A `dict` mapping feature keys to corresponding `Tensors`.
    Returns:
      voxel: A `Tensor` of shape [H, W, D] that represents the groundtruth
        voxel occupancies of object.
    """
    voxel_indices = tf.io.parse_tensor(
        parsed_tensors['model/voxel_indices'], out_type=tf.int64)
    voxel_indices = tf.ensure_shape(voxel_indices, [None, 3])

    return voxel_indices

  def _decode_camera(self, parsed_tensors: dict):
    """Decode the camera properties into tensors.

    Args:
      parsed_tensors: A `dict` mapping feature keys to corresponding `Tensors`.
    Returns:
      rot_mat: A `Tensor` of shape [3, 3] that represents the matrix to be
        multiplied with the voxels and vertices to align them to the input image
        object.
      trans_mat: A `Tensor` of shape [3] that represents the vector to be added
        to the voxels and vertices to align them to the input image object.
      intrinsic_mat: A `Tensor` of shape [3] that contains the intrinsic camera
        parameters in the following order: focal length, principal point x-
        coordinate, principal point y-coordinate.
    """
    rot_mat = tf.io.parse_tensor(
        parsed_tensors['camera/rot_mat'], out_type=tf.float32)
    rot_mat = tf.ensure_shape(rot_mat, [3, 3])

    trans_mat = parsed_tensors['camera/trans_mat']
    trans_mat = tf.ensure_shape(trans_mat, [3])

    intrinstic_mat = parsed_tensors['camera/intrinstic_mat']
    intrinstic_mat = tf.ensure_shape(intrinstic_mat, [3])

    return rot_mat, trans_mat, intrinstic_mat

  def decode(self, serialized_example: tf.train.Example):
    """Decode the serialized example.

    Args:
      serialized_example: a single serialized tf.Example string.
    Returns:
      decoded_tensors: a dictionary of tensors with the following fields:
        - source_id: a string scalar tensor.
        - image: a uint8 tensor of shape [None, None, 3].
        - height: an integer scalar tensor.
        - width: an integer scalar tensor.
        - groundtruth_classes: a int64 tensor of shape [None].
        - groundtruth_is_crowd: a bool tensor of shape [None].
        - groundtruth_area: a float32 tensor of shape [None].
        - groundtruth_boxes: a float32 tensor of shape [None, 4].
        - groundtruth_verts: a float32 tensor of shape [None, 3].
        - groundtruth_faces: a int64 tensor of shape [None, 3].
        - groundtruth_voxel_indices: a int64 tensor of shape [None, 3].
        - rot_mat: a float32 tensor of shape [3, 3].
        - trans_mat: a float32 tensor of shape [3].
        - intrinstic_mat: a float32 tensor of shape [3].
        - groundtruth_instance_masks: a float32 tensor of shape
            [None, None, None].
        - groundtruth_instance_masks_png: a string tensor of shape [None].
    """
    parsed_tensors = tf.io.parse_single_example(
        serialized=serialized_example, features=self._keys_to_features)
    for k in parsed_tensors:
      if isinstance(parsed_tensors[k], tf.SparseTensor):
        if parsed_tensors[k].dtype == tf.string:
          parsed_tensors[k] = tf.sparse.to_dense(
              parsed_tensors[k], default_value='')
        else:
          parsed_tensors[k] = tf.sparse.to_dense(
              parsed_tensors[k], default_value=0)

    if self._regenerate_source_id:
      source_id = _generate_source_id(parsed_tensors['image/encoded'])
    else:
      source_id = tf.cond(
          tf.greater(tf.strings.length(parsed_tensors['image/source_id']), 0),
          lambda: parsed_tensors['image/source_id'],
          lambda: _generate_source_id(parsed_tensors['image/encoded']))
    image = self._decode_image(parsed_tensors)
    boxes = self._decode_boxes(parsed_tensors)
    classes = self._decode_classes(parsed_tensors)
    areas = self._decode_areas(parsed_tensors)
    is_crowds = tf.cond(
        tf.greater(tf.shape(parsed_tensors['image/object/is_crowd'])[0], 0),
        lambda: tf.cast(parsed_tensors['image/object/is_crowd'], dtype=tf.bool),
        lambda: tf.zeros_like(classes, dtype=tf.bool))
    if self._include_mask:
      masks = self._decode_masks(parsed_tensors)

      if self._mask_binarize_threshold is not None:
        masks = tf.cast(masks > self._mask_binarize_threshold, tf.float32)

    verts, faces = self._decode_mesh(parsed_tensors)
    voxel_indices = self._decode_voxel(parsed_tensors)
    rot_mat, trans_mat, intrinstic_mat = self._decode_camera(parsed_tensors)

    decoded_tensors = {
        'source_id': source_id,
        'image': image,
        'height': parsed_tensors['image/height'],
        'width': parsed_tensors['image/width'],
        'groundtruth_classes': classes,
        'groundtruth_is_crowd': is_crowds,
        'groundtruth_area': areas,
        'groundtruth_boxes': boxes,
        'groundtruth_verts': verts,
        'groundtruth_faces': faces,
        'groundtruth_voxel_indices': voxel_indices,
        'rot_mat': rot_mat,
        'trans_mat': trans_mat,
        'intrinstic_mat': intrinstic_mat
    }
    if self._include_mask:
      decoded_tensors.update({
          'groundtruth_instance_masks': masks,
          'groundtruth_instance_masks_png': parsed_tensors['image/object/mask'],
      })
    return decoded_tensors
