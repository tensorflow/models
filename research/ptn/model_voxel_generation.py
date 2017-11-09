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

"""Base class for voxel generation model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os

import numpy as np
import tensorflow as tf

import input_generator
import utils

slim = tf.contrib.slim


class Im2Vox(object):
  """Defines the voxel generation model."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, params):
    self._params = params

  @abc.abstractmethod
  def get_metrics(self, inputs, outputs):
    """Gets dictionaries from metrics to value `Tensors` & update `Tensors`."""
    pass

  @abc.abstractmethod
  def get_loss(self, inputs, outputs):
    pass

  @abc.abstractmethod
  def get_regularization_loss(self, scopes):
    pass

  def set_params(self, params):
    self._params = params

  def get_inputs(self,
                 dataset_dir,
                 dataset_name,
                 split_name,
                 batch_size,
                 image_size,
                 vox_size,
                 is_training=True):
    """Loads data for a specified dataset and split."""
    del image_size, vox_size
    with tf.variable_scope('data_loading_%s/%s' % (dataset_name, split_name)):
      common_queue_min = 64
      common_queue_capacity = 256
      num_readers = 4

      inputs = input_generator.get(
          dataset_dir,
          dataset_name,
          split_name,
          shuffle=is_training,
          num_readers=num_readers,
          common_queue_min=common_queue_min,
          common_queue_capacity=common_queue_capacity)

      images, voxels = tf.train.batch(
          [inputs['image'], inputs['voxel']],
          batch_size=batch_size,
          num_threads=8,
          capacity=8 * batch_size,
          name='batching_queues/%s/%s' % (dataset_name, split_name))

      outputs = dict()
      outputs['images'] = images
      outputs['voxels'] = voxels
      outputs['num_samples'] = inputs['num_samples']

    return outputs

  def preprocess(self, raw_inputs, step_size):
    """Selects the subset of viewpoints to train on."""
    (quantity, num_views) = raw_inputs['images'].get_shape().as_list()[:2]

    inputs = dict()
    inputs['voxels'] = raw_inputs['voxels']

    for k in xrange(step_size):
      inputs['images_%d' % (k + 1)] = []
      inputs['matrix_%d' % (k + 1)] = []

    for n in xrange(quantity):
      selected_views = np.random.choice(num_views, step_size, replace=False)
      for k in xrange(step_size):
        view_selected = selected_views[k]
        inputs['images_%d' %
               (k + 1)].append(raw_inputs['images'][n, view_selected, :, :, :])
        tf_matrix = self.get_transform_matrix(view_selected)
        inputs['matrix_%d' % (k + 1)].append(tf_matrix)

    for k in xrange(step_size):
      inputs['images_%d' % (k + 1)] = tf.stack(inputs['images_%d' % (k + 1)])
      inputs['matrix_%d' % (k + 1)] = tf.stack(inputs['matrix_%d' % (k + 1)])

    return inputs

  def get_init_fn(self, scopes):
    """Initialization assignment operator function used while training."""
    if not self._params.init_model:
      return None

    is_trainable = lambda x: x in tf.trainable_variables()
    var_list = []
    for scope in scopes:
      var_list.extend(
          filter(is_trainable, tf.contrib.framework.get_model_variables(scope)))

    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
        self._params.init_model, var_list)

    def init_assign_function(sess):
      sess.run(init_assign_op, init_feed_dict)

    return init_assign_function

  def get_train_op_for_scope(self, loss, optimizer, scopes):
    """Train operation function for the given scope used file training."""
    is_trainable = lambda x: x in tf.trainable_variables()

    var_list = []
    update_ops = []

    for scope in scopes:
      var_list.extend(
          filter(is_trainable, tf.contrib.framework.get_model_variables(scope)))
      update_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))

    return slim.learning.create_train_op(
        loss,
        optimizer,
        update_ops=update_ops,
        variables_to_train=var_list,
        clip_gradient_norm=self._params.clip_gradient_norm)

  def write_disk_grid(self,
                      global_step,
                      log_dir,
                      input_images,
                      gt_projs,
                      pred_projs,
                      pred_voxels=None):
    """Function called by TF to save the prediction periodically."""
    summary_freq = self._params.save_every

    def write_grid(input_images, gt_projs, pred_projs, pred_voxels,
                   global_step):
      """Native python function to call for writing images to files."""
      grid = _build_image_grid(input_images, gt_projs, pred_projs, pred_voxels)

      if global_step % summary_freq == 0:
        img_path = os.path.join(log_dir, '%s.jpg' % str(global_step))
        utils.save_image(grid, img_path)
        with open(
            os.path.join(log_dir, 'pred_voxels_%s' % str(global_step)),
            'w') as fout:
          np.save(fout, pred_voxels)
        with open(
            os.path.join(log_dir, 'input_images_%s' % str(global_step)),
            'w') as fout:
          np.save(fout, input_images)

      return grid

    py_func_args = [
        input_images, gt_projs, pred_projs, pred_voxels, global_step
    ]
    save_grid_op = tf.py_func(write_grid, py_func_args, [tf.uint8],
                              'wrtie_grid')[0]
    slim.summaries.add_image_summary(
        tf.expand_dims(save_grid_op, axis=0), name='grid_vis')
    return save_grid_op


def _build_image_grid(input_images, gt_projs, pred_projs, pred_voxels):
  """Build the visualization grid with py_func."""
  quantity, img_height, img_width = input_images.shape[:3]
  for row in xrange(int(quantity / 3)):
    for col in xrange(3):
      index = row * 3 + col
      input_img_ = input_images[index, :, :, :]
      gt_proj_ = gt_projs[index, :, :, :]
      pred_proj_ = pred_projs[index, :, :, :]
      pred_voxel_ = utils.display_voxel(pred_voxels[index, :, :, :, 0])
      pred_voxel_ = utils.resize_image(pred_voxel_, img_height, img_width)
      if col == 0:
        tmp_ = np.concatenate([input_img_, gt_proj_, pred_proj_, pred_voxel_],
                              1)
      else:
        tmp_ = np.concatenate(
            [tmp_, input_img_, gt_proj_, pred_proj_, pred_voxel_], 1)
    if row == 0:
      out_grid = tmp_
    else:
      out_grid = np.concatenate([out_grid, tmp_], 0)

  out_grid = out_grid.astype(np.uint8)
  return out_grid
