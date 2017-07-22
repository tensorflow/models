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

"""Defines the various loss functions in use by the PTN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def add_rotator_image_loss(inputs, outputs, step_size, weight_scale):
  """Computes the image loss of deep rotator model.

  Args:
    inputs: Input dictionary to the model containing keys
      such as `images_k'.
    outputs: Output dictionary returned by the model containing keys
      such as `images_k'.
    step_size: A scalar representing the number of recurrent
      steps (number of repeated out-of-plane rotations)
      in the deep rotator network (int).
    weight_scale: A reweighting factor applied over the image loss (float).

  Returns:
    A `Tensor' scalar that returns averaged L2 loss
      (divided by batch_size and step_size) between the
      ground-truth images (RGB) and predicted images (tf.float32).

  """
  batch_size = tf.shape(inputs['images_0'])[0]
  image_loss = 0
  for k in range(1, step_size + 1):
    image_loss += tf.nn.l2_loss(
        inputs['images_%d' % k] - outputs['images_%d' % k])

  image_loss /= tf.to_float(step_size * batch_size)
  slim.summaries.add_scalar_summary(
      image_loss, 'image_loss', prefix='losses')
  image_loss *= weight_scale
  return image_loss


def add_rotator_mask_loss(inputs, outputs, step_size, weight_scale):
  """Computes the mask loss of deep rotator model.

  Args:
    inputs: Input dictionary to the model containing keys
      such as `masks_k'.
    outputs: Output dictionary returned by the model containing
      keys such as `masks_k'.
    step_size: A scalar representing the number of recurrent
      steps (number of repeated out-of-plane rotations)
      in the deep rotator network (int).
    weight_scale: A reweighting factor applied over the mask loss (float).

  Returns:
    A `Tensor' that returns averaged L2 loss
      (divided by batch_size and step_size) between the ground-truth masks
      (object silhouettes) and predicted masks (tf.float32).

  """
  batch_size = tf.shape(inputs['images_0'])[0]
  mask_loss = 0
  for k in range(1, step_size + 1):
    mask_loss += tf.nn.l2_loss(
        inputs['masks_%d' % k] - outputs['masks_%d' % k])

  mask_loss /= tf.to_float(step_size * batch_size)
  slim.summaries.add_scalar_summary(
      mask_loss, 'mask_loss', prefix='losses')
  mask_loss *= weight_scale
  return mask_loss


def add_volume_proj_loss(inputs, outputs, num_views, weight_scale):
  """Computes the projection loss of voxel generation model.

  Args:
    inputs: Input dictionary to the model containing keys such as
      `images_1'.
    outputs: Output dictionary returned by the model containing keys
      such as `masks_k' and ``projs_k'.
    num_views: A integer scalar represents the total number of
      viewpoints for each of the object (int).
    weight_scale: A reweighting factor applied over the projection loss (float).

  Returns:
    A `Tensor' that returns the averaged L2 loss
      (divided by batch_size and num_views) between the ground-truth
      masks (object silhouettes) and predicted masks (tf.float32).

  """
  batch_size = tf.shape(inputs['images_1'])[0]
  proj_loss = 0
  for k in range(num_views):
    proj_loss += tf.nn.l2_loss(
        outputs['masks_%d' % (k + 1)] - outputs['projs_%d' % (k + 1)])
  proj_loss /= tf.to_float(num_views * batch_size)
  slim.summaries.add_scalar_summary(
      proj_loss, 'proj_loss', prefix='losses')
  proj_loss *= weight_scale
  return proj_loss


def add_volume_loss(inputs, outputs, num_views, weight_scale):
  """Computes the volume loss of voxel generation model.

  Args:
    inputs: Input dictionary to the model containing keys such as
      `images_1' and `voxels'.
    outputs: Output dictionary returned by the model containing keys
      such as `voxels_k'.
    num_views: A scalar representing the total number of
      viewpoints for each object (int).
    weight_scale: A reweighting factor applied over the volume
      loss (tf.float32).

  Returns:
    A `Tensor' that returns the averaged L2 loss
      (divided by batch_size and num_views) between the ground-truth
      volumes and predicted volumes (tf.float32).

  """
  batch_size = tf.shape(inputs['images_1'])[0]
  vol_loss = 0
  for k in range(num_views):
    vol_loss += tf.nn.l2_loss(
        inputs['voxels'] - outputs['voxels_%d' % (k + 1)])
  vol_loss /= tf.to_float(num_views * batch_size)
  slim.summaries.add_scalar_summary(
      vol_loss, 'vol_loss', prefix='losses')
  vol_loss *= weight_scale
  return vol_loss


def regularization_loss(scopes, params):
  """Computes the weight decay as regularization during training.

  Args:
    scopes: A list of different components of the model such as
      ``encoder'', ``decoder'' and ``projector''.
    params: Parameters of the model.

  Returns:
    Regularization loss (tf.float32).
  """

  reg_loss = tf.zeros(dtype=tf.float32, shape=[])
  if params.weight_decay > 0:
    is_trainable = lambda x: x in tf.trainable_variables()
    is_weights = lambda x: 'weights' in x.name
    for scope in scopes:
      scope_vars = filter(is_trainable,
                          tf.contrib.framework.get_model_variables(scope))
      scope_vars = filter(is_weights, scope_vars)
      if scope_vars:
        reg_loss += tf.add_n([tf.nn.l2_loss(var) for var in scope_vars])

  slim.summaries.add_scalar_summary(
      reg_loss, 'reg_loss', prefix='losses')
  reg_loss *= params.weight_decay
  return reg_loss
