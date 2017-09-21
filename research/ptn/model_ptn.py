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

"""Implementations for Im2Vox PTN (NIPS16) model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import losses
import metrics
import model_voxel_generation
import utils
from nets import im2vox_factory

slim = tf.contrib.slim


class model_PTN(model_voxel_generation.Im2Vox):  # pylint:disable=invalid-name
  """Inherits the generic Im2Vox model class and implements the functions."""

  def __init__(self, params):
    super(model_PTN, self).__init__(params)

  # For testing, this selects all views in input
  def preprocess_with_all_views(self, raw_inputs):
    (quantity, num_views) = raw_inputs['images'].get_shape().as_list()[:2]

    inputs = dict()
    inputs['voxels'] = []
    inputs['images_1'] = []
    for k in xrange(num_views):
      inputs['matrix_%d' % (k + 1)] = []
    inputs['matrix_1'] = []
    for n in xrange(quantity):
      for k in xrange(num_views):
        inputs['images_1'].append(raw_inputs['images'][n, k, :, :, :])
        inputs['voxels'].append(raw_inputs['voxels'][n, :, :, :, :])
        tf_matrix = self.get_transform_matrix(k)
        inputs['matrix_%d' % (k + 1)].append(tf_matrix)

    inputs['images_1'] = tf.stack(inputs['images_1'])
    inputs['voxels'] = tf.stack(inputs['voxels'])
    for k in xrange(num_views):
      inputs['matrix_%d' % (k + 1)] = tf.stack(inputs['matrix_%d' % (k + 1)])

    return inputs

  def get_model_fn(self, is_training=True, reuse=False, run_projection=True):
    return im2vox_factory.get(self._params, is_training, reuse, run_projection)

  def get_regularization_loss(self, scopes):
    return losses.regularization_loss(scopes, self._params)

  def get_loss(self, inputs, outputs):
    """Computes the loss used for PTN paper (projection + volume loss)."""
    g_loss = tf.zeros(dtype=tf.float32, shape=[])

    if self._params.proj_weight:
      g_loss += losses.add_volume_proj_loss(
          inputs, outputs, self._params.step_size, self._params.proj_weight)

    if self._params.volume_weight:
      g_loss += losses.add_volume_loss(inputs, outputs, 1,
                                       self._params.volume_weight)

    slim.summaries.add_scalar_summary(g_loss, 'im2vox_loss', prefix='losses')

    return g_loss

  def get_metrics(self, inputs, outputs):
    """Aggregate the metrics for voxel generation model.

    Args:
      inputs: Input dictionary of the voxel generation model.
      outputs: Output dictionary returned by the voxel generation model.

    Returns:
      names_to_values: metrics->values (dict).
      names_to_updates: metrics->ops (dict).
    """
    names_to_values = dict()
    names_to_updates = dict()

    tmp_values, tmp_updates = metrics.add_volume_iou_metrics(inputs, outputs)

    names_to_values.update(tmp_values)
    names_to_updates.update(tmp_updates)

    for name, value in names_to_values.iteritems():
      slim.summaries.add_scalar_summary(
          value, name, prefix='eval', print_summary=True)

    return names_to_values, names_to_updates

  def write_disk_grid(self,
                      global_step,
                      log_dir,
                      input_images,
                      gt_projs,
                      pred_projs,
                      input_voxels=None,
                      output_voxels=None):
    """Function called by TF to save the prediction periodically."""
    summary_freq = self._params.save_every

    def write_grid(input_images, gt_projs, pred_projs, global_step,
                   input_voxels, output_voxels):
      """Native python function to call for writing images to files."""
      grid = _build_image_grid(
          input_images,
          gt_projs,
          pred_projs,
          input_voxels=input_voxels,
          output_voxels=output_voxels)

      if global_step % summary_freq == 0:
        img_path = os.path.join(log_dir, '%s.jpg' % str(global_step))
        utils.save_image(grid, img_path)
      return grid

    save_op = tf.py_func(write_grid, [
        input_images, gt_projs, pred_projs, global_step, input_voxels,
        output_voxels
    ], [tf.uint8], 'write_grid')[0]
    slim.summaries.add_image_summary(
        tf.expand_dims(save_op, axis=0), name='grid_vis')
    return save_op

  def get_transform_matrix(self, view_out):
    """Get the 4x4 Perspective Transfromation matrix used for PTN."""
    num_views = self._params.num_views
    focal_length = self._params.focal_length
    focal_range = self._params.focal_range
    phi = 30
    theta_interval = 360.0 / num_views
    theta = theta_interval * view_out

    #  pylint: disable=invalid-name
    camera_matrix = np.zeros((4, 4), dtype=np.float32)
    intrinsic_matrix = np.eye(4, dtype=np.float32)
    extrinsic_matrix = np.eye(4, dtype=np.float32)

    sin_phi = np.sin(float(phi) / 180.0 * np.pi)
    cos_phi = np.cos(float(phi) / 180.0 * np.pi)
    sin_theta = np.sin(float(-theta) / 180.0 * np.pi)
    cos_theta = np.cos(float(-theta) / 180.0 * np.pi)

    rotation_azimuth = np.zeros((3, 3), dtype=np.float32)
    rotation_azimuth[0, 0] = cos_theta
    rotation_azimuth[2, 2] = cos_theta
    rotation_azimuth[0, 2] = -sin_theta
    rotation_azimuth[2, 0] = sin_theta
    rotation_azimuth[1, 1] = 1.0

    ## rotation axis -- x
    rotation_elevation = np.zeros((3, 3), dtype=np.float32)
    rotation_elevation[0, 0] = cos_phi
    rotation_elevation[0, 1] = sin_phi
    rotation_elevation[1, 0] = -sin_phi
    rotation_elevation[1, 1] = cos_phi
    rotation_elevation[2, 2] = 1.0

    rotation_matrix = np.matmul(rotation_azimuth, rotation_elevation)
    displacement = np.zeros((3, 1), dtype=np.float32)
    displacement[0, 0] = float(focal_length) + float(focal_range) / 2.0
    displacement = np.matmul(rotation_matrix, displacement)

    extrinsic_matrix[0:3, 0:3] = rotation_matrix
    extrinsic_matrix[0:3, 3:4] = -displacement

    intrinsic_matrix[2, 2] = 1.0 / float(focal_length)
    intrinsic_matrix[1, 1] = 1.0 / float(focal_length)

    camera_matrix = np.matmul(extrinsic_matrix, intrinsic_matrix)
    return camera_matrix


def _build_image_grid(input_images,
                      gt_projs,
                      pred_projs,
                      input_voxels,
                      output_voxels,
                      vis_size=128):
  """Builds a grid image by concatenating the input images."""
  quantity = input_images.shape[0]

  for row in xrange(int(quantity / 3)):
    for col in xrange(3):
      index = row * 3 + col
      input_img_ = utils.resize_image(input_images[index, :, :, :], vis_size,
                                      vis_size)
      gt_proj_ = utils.resize_image(gt_projs[index, :, :, :], vis_size,
                                    vis_size)
      pred_proj_ = utils.resize_image(pred_projs[index, :, :, :], vis_size,
                                      vis_size)
      gt_voxel_vis = utils.resize_image(
          utils.display_voxel(input_voxels[index, :, :, :, 0]), vis_size,
          vis_size)
      pred_voxel_vis = utils.resize_image(
          utils.display_voxel(output_voxels[index, :, :, :, 0]), vis_size,
          vis_size)
      if col == 0:
        tmp_ = np.concatenate(
            [input_img_, gt_proj_, pred_proj_, gt_voxel_vis, pred_voxel_vis], 1)
      else:
        tmp_ = np.concatenate([
            tmp_, input_img_, gt_proj_, pred_proj_, gt_voxel_vis, pred_voxel_vis
        ], 1)
    if row == 0:
      out_grid = tmp_
    else:
      out_grid = np.concatenate([out_grid, tmp_], 0)

  return out_grid
