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

"""Provides metrics used by PTN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf

slim = tf.contrib.slim


def add_image_pred_metrics(
    inputs, outputs, num_views, upscale_factor):
  """Computes the image prediction metrics.

  Args:
    inputs: Input dictionary of the deep rotator model (model_rotator.py).
    outputs: Output dictionary of the deep rotator model (model_rotator.py).
    num_views: An integer scalar representing the total number
      of different viewpoints for each object in the dataset.
    upscale_factor: A float scalar representing the number of pixels
      per image (num_channels x image_height x image_width).

  Returns:
    names_to_values: A dictionary representing the current value
      of the metric.
    names_to_updates: A dictionary representing the operation
      that accumulates the error from a batch of data.
  """
  names_to_values = dict()
  names_to_updates = dict()
  for k in xrange(num_views):
    tmp_value, tmp_update = tf.contrib.metrics.streaming_mean_squared_error(
        outputs['images_%d' % (k + 1)], inputs['images_%d' % (k + 1)])
    name = 'image_pred/rnn_%d' % (k + 1)
    names_to_values.update({name: tmp_value * upscale_factor})
    names_to_updates.update({name: tmp_update})
  return names_to_values, names_to_updates


def add_mask_pred_metrics(
    inputs, outputs, num_views, upscale_factor):
  """Computes the mask prediction metrics.

  Args:
    inputs: Input dictionary of the deep rotator model (model_rotator.py).
    outputs: Output dictionary of the deep rotator model (model_rotator.py).
    num_views: An integer scalar representing the total number
      of different viewpoints for each object in the dataset.
    upscale_factor: A float scalar representing the number of pixels
      per image (num_channels x image_height x image_width).

  Returns:
    names_to_values: A dictionary representing the current value
      of the metric.
    names_to_updates: A dictionary representing the operation
      that accumulates the error from a batch of data.

  """
  names_to_values = dict()
  names_to_updates = dict()
  for k in xrange(num_views):
    tmp_value, tmp_update = tf.contrib.metrics.streaming_mean_squared_error(
        outputs['masks_%d' % (k + 1)], inputs['masks_%d' % (k + 1)])
    name = 'mask_pred/rnn_%d' % (k + 1)
    names_to_values.update({name: tmp_value * upscale_factor})
    names_to_updates.update({name: tmp_update})
  return names_to_values, names_to_updates


def add_volume_iou_metrics(inputs, outputs):
  """Computes the per-instance volume IOU.

  Args:
    inputs: Input dictionary of the voxel generation model.
    outputs: Output dictionary returned by the voxel generation model.

  Returns:
    names_to_values: metrics->values (dict).
    names_to_updates: metrics->ops (dict).

  """
  names_to_values = dict()
  names_to_updates = dict()
  labels = tf.greater_equal(inputs['voxels'], 0.5)
  predictions = tf.greater_equal(outputs['voxels_1'], 0.5)
  labels = (2 - tf.to_int32(labels)) - 1
  predictions = (3 - tf.to_int32(predictions) * 2) - 1
  tmp_values, tmp_updates = tf.metrics.mean_iou(
      labels=labels,
      predictions=predictions,
      num_classes=3)
  names_to_values['volume_iou'] = tmp_values * 3.0
  names_to_updates['volume_iou'] = tmp_updates
  return names_to_values, names_to_updates
