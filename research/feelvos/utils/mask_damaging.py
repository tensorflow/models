# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Utilities for artificially damaging segmentation masks."""

import numpy as np
from scipy.ndimage import interpolation
from skimage import morphology
from skimage import transform
import tensorflow as tf


def damage_masks(labels, shift=True, scale=True, rotate=True, dilate=True):
  """Damages segmentation masks by random transformations.

  Args:
    labels: Int32 labels tensor of shape (height, width, 1).
    shift: Boolean, whether to damage the masks by shifting.
    scale: Boolean, whether to damage the masks by scaling.
    rotate: Boolean, whether to damage the masks by rotation.
    dilate: Boolean, whether to damage the masks by dilation.

  Returns:
    The damaged version of labels.
  """
  def _damage_masks_np(labels_):
    return damage_masks_np(labels_, shift, scale, rotate, dilate)
  damaged_masks = tf.py_func(_damage_masks_np, [labels], tf.int32,
                             name='damage_masks')
  damaged_masks.set_shape(labels.get_shape())
  return damaged_masks


def damage_masks_np(labels, shift=True, scale=True, rotate=True, dilate=True):
  """Performs the actual mask damaging in numpy.

  Args:
    labels: Int32 numpy array of shape (height, width, 1).
    shift: Boolean, whether to damage the masks by shifting.
    scale: Boolean, whether to damage the masks by scaling.
    rotate: Boolean, whether to damage the masks by rotation.
    dilate: Boolean, whether to damage the masks by dilation.

  Returns:
    The damaged version of labels.
  """
  unique_labels = np.unique(labels)
  unique_labels = np.setdiff1d(unique_labels, [0])
  # Shuffle to get random depth ordering when combining together.
  np.random.shuffle(unique_labels)
  damaged_labels = np.zeros_like(labels)
  for l in unique_labels:
    obj_mask = (labels == l)
    damaged_obj_mask = _damage_single_object_mask(obj_mask, shift, scale,
                                                  rotate, dilate)
    damaged_labels[damaged_obj_mask] = l
  return damaged_labels


def _damage_single_object_mask(mask, shift, scale, rotate, dilate):
  """Performs mask damaging in numpy for a single object.

  Args:
    mask: Boolean numpy array of shape(height, width, 1).
    shift: Boolean, whether to damage the masks by shifting.
    scale: Boolean, whether to damage the masks by scaling.
    rotate: Boolean, whether to damage the masks by rotation.
    dilate: Boolean, whether to damage the masks by dilation.

  Returns:
    The damaged version of mask.
  """
  # For now we just do shifting and scaling. Better would be Affine or thin
  # spline plate transformations.
  if shift:
    mask = _shift_mask(mask)
  if scale:
    mask = _scale_mask(mask)
  if rotate:
    mask = _rotate_mask(mask)
  if dilate:
    mask = _dilate_mask(mask)
  return mask


def _shift_mask(mask, max_shift_factor=0.05):
  """Damages a mask for a single object by randomly shifting it in numpy.

  Args:
    mask: Boolean numpy array of shape(height, width, 1).
    max_shift_factor: Float scalar, the maximum factor for random shifting.

  Returns:
    The shifted version of mask.
  """
  nzy, nzx, _ = mask.nonzero()
  h = nzy.max() - nzy.min()
  w = nzx.max() - nzx.min()
  size = np.sqrt(h * w)
  offset = np.random.uniform(-size * max_shift_factor, size * max_shift_factor,
                             2)
  shifted_mask = interpolation.shift(np.squeeze(mask, axis=2),
                                     offset, order=0).astype('bool')[...,
                                                                     np.newaxis]
  return shifted_mask


def _scale_mask(mask, scale_amount=0.025):
  """Damages a mask for a single object by randomly scaling it in numpy.

  Args:
    mask: Boolean numpy array of shape(height, width, 1).
    scale_amount: Float scalar, the maximum factor for random scaling.

  Returns:
    The scaled version of mask.
  """
  nzy, nzx, _ = mask.nonzero()
  cy = 0.5 * (nzy.max() - nzy.min())
  cx = 0.5 * (nzx.max() - nzx.min())
  scale_factor = np.random.uniform(1.0 - scale_amount, 1.0 + scale_amount)
  shift = transform.SimilarityTransform(translation=[-cx, -cy])
  inv_shift = transform.SimilarityTransform(translation=[cx, cy])
  s = transform.SimilarityTransform(scale=[scale_factor, scale_factor])
  m = (shift + (s + inv_shift)).inverse
  scaled_mask = transform.warp(mask, m) > 0.5
  return scaled_mask


def _rotate_mask(mask, max_rot_degrees=3.0):
  """Damages a mask for a single object by randomly rotating it in numpy.

  Args:
    mask: Boolean numpy array of shape(height, width, 1).
    max_rot_degrees: Float scalar, the maximum number of degrees to rotate.

  Returns:
    The scaled version of mask.
  """
  cy = 0.5 * mask.shape[0]
  cx = 0.5 * mask.shape[1]
  rot_degrees = np.random.uniform(-max_rot_degrees, max_rot_degrees)
  shift = transform.SimilarityTransform(translation=[-cx, -cy])
  inv_shift = transform.SimilarityTransform(translation=[cx, cy])
  r = transform.SimilarityTransform(rotation=np.deg2rad(rot_degrees))
  m = (shift + (r + inv_shift)).inverse
  scaled_mask = transform.warp(mask, m) > 0.5
  return scaled_mask


def _dilate_mask(mask, dilation_radius=5):
  """Damages a mask for a single object by dilating it in numpy.

  Args:
    mask: Boolean numpy array of shape(height, width, 1).
    dilation_radius: Integer, the radius of the used disk structure element.

  Returns:
    The dilated version of mask.
  """
  disk = morphology.disk(dilation_radius, dtype=np.bool)
  dilated_mask = morphology.binary_dilation(
      np.squeeze(mask, axis=2), selem=disk)[..., np.newaxis]
  return dilated_mask
