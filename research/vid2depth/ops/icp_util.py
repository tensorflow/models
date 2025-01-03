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

"""Utility functions for transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# Sample pointcloud with shape (1568, 3).
LIDAR_CLOUD_PATH = 'ops/testdata/pointcloud.npy'


def get_transformation_matrix(transform):
  """Converts [tx, ty, tz, rx, ry, rz] to a transform matrix."""
  rx = transform[3]
  ry = transform[4]
  rz = transform[5]

  rz = tf.clip_by_value(rz, -np.pi, np.pi)
  ry = tf.clip_by_value(ry, -np.pi, np.pi)
  rx = tf.clip_by_value(rx, -np.pi, np.pi)

  cos_rx = tf.cos(rx)
  sin_rx = tf.sin(rx)
  rotx_1 = tf.stack([1.0, 0.0, 0.0])
  rotx_2 = tf.stack([0.0, cos_rx, -sin_rx])
  rotx_3 = tf.stack([0.0, sin_rx, cos_rx])
  xmat = tf.stack([rotx_1, rotx_2, rotx_3])

  cos_ry = tf.cos(ry)
  sin_ry = tf.sin(ry)
  roty_1 = tf.stack([cos_ry, 0.0, sin_ry])
  roty_2 = tf.stack([0.0, 1.0, 0.0])
  roty_3 = tf.stack([-sin_ry, 0.0, cos_ry])
  ymat = tf.stack([roty_1, roty_2, roty_3])

  cos_rz = tf.cos(rz)
  sin_rz = tf.sin(rz)
  rotz_1 = tf.stack([cos_rz, -sin_rz, 0.0])
  rotz_2 = tf.stack([sin_rz, cos_rz, 0.0])
  rotz_3 = tf.stack([0.0, 0.0, 1.0])
  zmat = tf.stack([rotz_1, rotz_2, rotz_3])

  rotate = tf.matmul(tf.matmul(xmat, ymat), zmat)

  translate = transform[:3]
  mat = tf.concat([rotate, tf.expand_dims(translate, 1)], axis=1)

  hom_filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 4], dtype=tf.float32)
  mat = tf.concat([mat, hom_filler], axis=0)
  return mat


def np_get_transformation_matrix(transform):
  """Converts [tx, ty, tz, rx, ry, rz] to a transform matrix."""
  rx = transform[3]
  ry = transform[4]
  rz = transform[5]

  rz = np.clip(rz, -np.pi, np.pi)
  ry = np.clip(ry, -np.pi, np.pi)
  rx = np.clip(rx, -np.pi, np.pi)

  cos_rx = np.cos(rx)
  sin_rx = np.sin(rx)
  rotx_1 = np.stack([1.0, 0.0, 0.0])
  rotx_2 = np.stack([0.0, cos_rx, -sin_rx])
  rotx_3 = np.stack([0.0, sin_rx, cos_rx])
  xmat = np.stack([rotx_1, rotx_2, rotx_3])

  cos_ry = np.cos(ry)
  sin_ry = np.sin(ry)
  roty_1 = np.stack([cos_ry, 0.0, sin_ry])
  roty_2 = np.stack([0.0, 1.0, 0.0])
  roty_3 = np.stack([-sin_ry, 0.0, cos_ry])
  ymat = np.stack([roty_1, roty_2, roty_3])

  cos_rz = np.cos(rz)
  sin_rz = np.sin(rz)
  rotz_1 = np.stack([cos_rz, -sin_rz, 0.0])
  rotz_2 = np.stack([sin_rz, cos_rz, 0.0])
  rotz_3 = np.stack([0.0, 0.0, 1.0])
  zmat = np.stack([rotz_1, rotz_2, rotz_3])

  rotate = np.dot(np.dot(xmat, ymat), zmat)

  translate = transform[:3]
  mat = np.concatenate((rotate, np.expand_dims(translate, 1)), axis=1)

  hom_filler = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
  mat = np.concatenate((mat, hom_filler), axis=0)
  return mat


def transform_cloud_xyz(cloud, transform):
  num_points = cloud.shape.as_list()[0]
  ones = tf.ones(shape=[num_points, 1], dtype=tf.float32)
  hom_cloud = tf.concat([cloud, ones], axis=1)
  hom_cloud_t = tf.transpose(hom_cloud)
  mat = get_transformation_matrix(transform)
  transformed_cloud = tf.matmul(mat, hom_cloud_t)
  transformed_cloud = tf.transpose(transformed_cloud)
  transformed_cloud = transformed_cloud[:, :3]
  return transformed_cloud


def np_transform_cloud_xyz(cloud, transform):
  num_points = cloud.shape[0]
  ones = np.ones(shape=[num_points, 1], dtype=np.float32)
  hom_cloud = np.concatenate((cloud, ones), axis=1)
  hom_cloud_t = np.transpose(hom_cloud)
  mat = np_get_transformation_matrix(transform)
  transformed_cloud = np.dot(mat, hom_cloud_t)
  transformed_cloud = np.transpose(transformed_cloud)
  transformed_cloud = transformed_cloud[:, :3]
  return transformed_cloud


def batch_transform_cloud_xyz(cloud, transform):
  results = []
  cloud_items = tf.unstack(cloud)
  if len(transform.shape.as_list()) == 2:
    transform_items = tf.unstack(transform)
  else:
    transform_items = [transform] * len(cloud_items)
  for cloud_item, transform_item in zip(cloud_items, transform_items):
    results.append(transform_cloud_xyz(cloud_item, transform_item))
  return tf.stack(results)
