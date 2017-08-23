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

"""Helper functions for pretraining (rotator) as described in PTN paper."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import input_generator
import losses
import metrics
import utils
from nets import deeprotator_factory

slim = tf.contrib.slim


def _get_data_from_provider(inputs, batch_size, split_name):
  """Returns dictionary of batch input data processed by tf.train.batch."""
  images, masks = tf.train.batch(
      [inputs['image'], inputs['mask']],
      batch_size=batch_size,
      num_threads=8,
      capacity=8 * batch_size,
      name='batching_queues/%s' % (split_name))

  outputs = dict()
  outputs['images'] = images
  outputs['masks'] = masks
  outputs['num_samples'] = inputs['num_samples']

  return outputs


def get_inputs(dataset_dir, dataset_name, split_name, batch_size, image_size,
               is_training):
  """Loads the given dataset and split."""
  del image_size  # Unused
  with tf.variable_scope('data_loading_%s/%s' % (dataset_name, split_name)):
    common_queue_min = 50
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

    return _get_data_from_provider(inputs, batch_size, split_name)


def preprocess(raw_inputs, step_size):
  """Selects the subset of viewpoints to train on."""
  shp = raw_inputs['images'].get_shape().as_list()
  quantity = shp[0]
  num_views = shp[1]
  image_size = shp[2]
  del image_size  # Unused

  batch_rot = np.zeros((quantity, 3), dtype=np.float32)
  inputs = dict()
  for n in xrange(step_size + 1):
    inputs['images_%d' % n] = []
    inputs['masks_%d' % n] = []

  for n in xrange(quantity):
    view_in = np.random.randint(0, num_views)
    rng_rot = np.random.randint(0, 2)
    if step_size == 1:
      rng_rot = np.random.randint(0, 3)

    delta = 0
    if rng_rot == 0:
      delta = -1
      batch_rot[n, 2] = 1
    elif rng_rot == 1:
      delta = 1
      batch_rot[n, 0] = 1
    else:
      delta = 0
      batch_rot[n, 1] = 1

    inputs['images_0'].append(raw_inputs['images'][n, view_in, :, :, :])
    inputs['masks_0'].append(raw_inputs['masks'][n, view_in, :, :, :])

    view_out = view_in
    for k in xrange(1, step_size + 1):
      view_out += delta
      if view_out >= num_views:
        view_out = 0
      if view_out < 0:
        view_out = num_views - 1

      inputs['images_%d' % k].append(raw_inputs['images'][n, view_out, :, :, :])
      inputs['masks_%d' % k].append(raw_inputs['masks'][n, view_out, :, :, :])

  for n in xrange(step_size + 1):
    inputs['images_%d' % n] = tf.stack(inputs['images_%d' % n])
    inputs['masks_%d' % n] = tf.stack(inputs['masks_%d' % n])

  inputs['actions'] = tf.constant(batch_rot, dtype=tf.float32)
  return inputs


def get_init_fn(scopes, params):
  """Initialization assignment operator function used while training."""
  if not params.init_model:
    return None

  is_trainable = lambda x: x in tf.trainable_variables()
  var_list = []
  for scope in scopes:
    var_list.extend(
        filter(is_trainable, tf.contrib.framework.get_model_variables(scope)))

  init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
      params.init_model, var_list)

  def init_assign_function(sess):
    sess.run(init_assign_op, init_feed_dict)

  return init_assign_function


def get_model_fn(params, is_training, reuse=False):
  return deeprotator_factory.get(params, is_training, reuse)


def get_regularization_loss(scopes, params):
  return losses.regularization_loss(scopes, params)


def get_loss(inputs, outputs, params):
  """Computes the rotator loss."""
  g_loss = tf.zeros(dtype=tf.float32, shape=[])

  if hasattr(params, 'image_weight'):
    g_loss += losses.add_rotator_image_loss(inputs, outputs, params.step_size,
                                            params.image_weight)

  if hasattr(params, 'mask_weight'):
    g_loss += losses.add_rotator_mask_loss(inputs, outputs, params.step_size,
                                           params.mask_weight)

  slim.summaries.add_scalar_summary(
      g_loss, 'rotator_loss', prefix='losses')

  return g_loss


def get_train_op_for_scope(loss, optimizer, scopes, params):
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
      clip_gradient_norm=params.clip_gradient_norm)


def get_metrics(inputs, outputs, params):
  """Aggregate the metrics for rotator model.
  
  Args:
    inputs: Input dictionary of the rotator model.
    outputs: Output dictionary returned by the rotator model.
    params: Hyperparameters of the rotator model.
  
  Returns:
    names_to_values: metrics->values (dict).
    names_to_updates: metrics->ops (dict).
  """
  names_to_values = dict()
  names_to_updates = dict()
  
  tmp_values, tmp_updates = metrics.add_image_pred_metrics(
      inputs, outputs, params.num_views, 3*params.image_size**2)
  names_to_values.update(tmp_values)
  names_to_updates.update(tmp_updates)

  tmp_values, tmp_updates = metrics.add_mask_pred_metrics(
      inputs, outputs, params.num_views, params.image_size**2)
  names_to_values.update(tmp_values)
  names_to_updates.update(tmp_updates)

  for name, value in names_to_values.iteritems():
    slim.summaries.add_scalar_summary(
        value, name, prefix='eval', print_summary=True)
 
  return names_to_values, names_to_updates


def write_disk_grid(global_step, summary_freq, log_dir, input_images,
                    output_images, pred_images, pred_masks):
  """Function called by TF to save the prediction periodically."""

  def write_grid(grid, global_step):
    """Native python function to call for writing images to files."""
    if global_step % summary_freq == 0:
      img_path = os.path.join(log_dir, '%s.jpg' % str(global_step))
      utils.save_image(grid, img_path)
    return 0

  grid = _build_image_grid(input_images, output_images, pred_images, pred_masks)
  slim.summaries.add_image_summary(
      tf.expand_dims(grid, axis=0), name='grid_vis')
  save_op = tf.py_func(write_grid, [grid, global_step], [tf.int64],
                       'write_grid')[0]
  return save_op


def _build_image_grid(input_images, output_images, pred_images, pred_masks):
  """Builds a grid image by concatenating the input images."""
  quantity = input_images.get_shape().as_list()[0]

  for row in xrange(int(quantity / 4)):
    for col in xrange(4):
      index = row * 4 + col
      input_img_ = input_images[index, :, :, :]
      output_img_ = output_images[index, :, :, :]
      pred_img_ = pred_images[index, :, :, :]
      pred_mask_ = tf.tile(pred_masks[index, :, :, :], [1, 1, 3])
      if col == 0:
        tmp_ = tf.concat([input_img_, output_img_, pred_img_, pred_mask_],
                         1)  ## to the right
      else:
        tmp_ = tf.concat([tmp_, input_img_, output_img_, pred_img_, pred_mask_],
                         1)
    if row == 0:
      out_grid = tmp_
    else:
      out_grid = tf.concat([out_grid, tmp_], 0)

  return out_grid
