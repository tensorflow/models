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

"""Tests for icp op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

from absl import flags
from absl import logging
from icp_op import icp
import icp_util
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

PRINT_CAP = 6


class IcpOpTestBase(tf.test.TestCase):
  """Classed used by IcpOpTest, IcpOpGradTest."""

  def setUp(self):
    self.small_cloud = tf.constant([[[0.352222, -0.151883, -0.106395],
                                     [-0.397406, -0.473106, 0.292602],
                                     [-0.731898, 0.667105, 0.441304],
                                     [-0.734766, 0.854581, -0.0361733],
                                     [-0.4607, -0.277468, -0.916762]]],
                                   dtype=tf.float32)
    self.random_cloud = self._generate_random_cloud()
    self.organized_cloud = self._generate_organized_cloud()
    self.lidar_cloud = self._load_lidar_cloud()
    self.identity_transform = tf.constant([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                                          dtype=tf.float32)
    self.index_translation = 0
    self.index_rotation = 3

  def _run_icp(self, cloud_source, ego_motion, cloud_target):
    transform, residual = icp(cloud_source, ego_motion, cloud_target)
    logging.info('Running ICP:')
    logging.info('ego_motion: %s\n%s', ego_motion, ego_motion.eval())
    logging.info('transform: %s\n%s', transform, transform.eval())
    logging.info('residual: %s\n%s', residual,
                 residual[0, :PRINT_CAP, :].eval())
    return transform, residual

  def _generate_random_cloud(self):
    self.random_cloud_size = 50
    tf.set_random_seed(11)
    return tf.truncated_normal(
        [1, self.random_cloud_size, 3], mean=0.0, stddev=1.0, dtype=tf.float32)

  def _generate_organized_cloud(self):
    res = 10
    scale = 7
    # [B, 10, 10, 3]
    cloud = np.zeros(shape=(1, res, res, 3))
    for i in range(res):
      for j in range(res):
        # For scale == 1.0, x and y range from -0.5 to 0.4.
        y = scale / 2 - scale * (res - i) / res
        x = scale / 2 - scale * (res - j) / res
        z = math.sin(x * x + y * y)
        cloud[0, i, j, :] = (x, y, z)
    return tf.constant(cloud, dtype=tf.float32)

  def _load_lidar_cloud(self):
    lidar_cloud_path = os.path.join(FLAGS.test_srcdir,
                                    icp_util.LIDAR_CLOUD_PATH)
    lidar_cloud = np.load(lidar_cloud_path)
    lidar_cloud = tf.expand_dims(lidar_cloud, axis=0)  # Add batch.
    logging.info('lidar_cloud.shape: %s', lidar_cloud.shape)
    return lidar_cloud


class IcpOpTest(IcpOpTestBase):

  def test_translate_small_cloud(self):
    with self.test_session():
      tx = 0.1
      cloud_source = self.small_cloud
      cloud_target = cloud_source + [tx, 0, 0]
      transform, residual = self._run_icp(cloud_source, self.identity_transform,
                                          cloud_target)
      self.assertAlmostEqual(transform.eval()[0, self.index_translation], tx,
                             places=6)
      self.assertAllClose(residual.eval(), tf.zeros_like(residual).eval(),
                          atol=1e-4)

  def test_translate_random_cloud(self):
    with self.test_session():
      tx = 0.1
      cloud_source = self.random_cloud
      cloud_target = cloud_source + [tx, 0, 0]
      transform, residual = self._run_icp(cloud_source, self.identity_transform,
                                          cloud_target)
      self.assertAlmostEqual(transform.eval()[0, self.index_translation], tx,
                             places=4)
      self.assertAllClose(residual.eval(), tf.zeros_like(residual).eval(),
                          atol=1e-4)

  def test_rotate_random_cloud(self):
    with self.test_session():
      ego_motion = tf.constant([[0.0, 0.0, 0.0,
                                 np.pi / 32, np.pi / 64, np.pi / 24]],
                               dtype=tf.float32)
      cloud_source = self.random_cloud
      cloud_target = icp_util.batch_transform_cloud_xyz(cloud_source,
                                                        ego_motion)
      unused_transform, residual = self._run_icp(cloud_source,
                                                 self.identity_transform,
                                                 cloud_target)
      self.assertAllClose(residual.eval(), tf.zeros_like(residual).eval(),
                          atol=1e-4)

  def test_translate_organized_cloud(self):
    with self.test_session():
      tx = 0.1
      cloud_source = self.organized_cloud
      cloud_target = cloud_source + [tx, 0, 0]
      transform, residual = self._run_icp(cloud_source, self.identity_transform,
                                          cloud_target)
      self.assertAlmostEqual(transform.eval()[0, self.index_translation], tx,
                             places=4)
      self.assertAllClose(residual.eval(), tf.zeros_like(residual).eval(),
                          atol=1e-4)

  def test_rotate_organized_cloud(self):
    with self.test_session():
      ego_motion = tf.constant([[0.0, 0.0, 0.0,
                                 np.pi / 16, np.pi / 32, np.pi / 12]],
                               dtype=tf.float32)
      cloud_source = self.organized_cloud
      cloud_shape = cloud_source.shape.as_list()
      flat_shape = (cloud_shape[0],
                    cloud_shape[1] * cloud_shape[2],
                    cloud_shape[3])
      cloud_source = tf.reshape(cloud_source, shape=flat_shape)
      cloud_target = icp_util.batch_transform_cloud_xyz(cloud_source,
                                                        ego_motion)
      cloud_source = tf.reshape(cloud_source, cloud_shape)
      cloud_target = tf.reshape(cloud_target, cloud_shape)

      unused_transform, residual = self._run_icp(cloud_source,
                                                 self.identity_transform,
                                                 cloud_target)
      self.assertAllClose(residual.eval(), tf.zeros_like(residual).eval(),
                          atol=1e-4)

  def test_translate_lidar_cloud(self):
    with self.test_session():
      tx = 0.1
      cloud_source = self.lidar_cloud
      cloud_target = cloud_source + [tx, 0, 0]
      transform, residual = self._run_icp(cloud_source, self.identity_transform,
                                          cloud_target)
      self.assertAlmostEqual(transform.eval()[0, self.index_translation], tx,
                             places=4)
      self.assertAllClose(residual.eval(), tf.zeros_like(residual).eval(),
                          atol=1e-4)

  def test_translate_lidar_cloud_ego_motion(self):
    with self.test_session():
      tx = 0.2
      ego_motion = tf.constant([[tx, 0.0, 0.0,
                                 0.0, 0.0, 0.0]], dtype=tf.float32)
      cloud_source = self.lidar_cloud
      cloud_target = cloud_source + [tx, 0, 0]
      transform, residual = self._run_icp(cloud_source, ego_motion,
                                          cloud_target)
      self.assertAllClose(transform.eval(), tf.zeros_like(transform).eval(),
                          atol=1e-4)
      self.assertAllClose(residual.eval(), tf.zeros_like(residual).eval(),
                          atol=1e-4)

  def test_rotate_lidar_cloud_ego_motion(self):
    with self.test_session():
      transform = [0.0, 0.0, 0.0, np.pi / 16, np.pi / 32, np.pi / 12]
      ego_motion = tf.constant([transform], dtype=tf.float32)
      cloud_source = self.lidar_cloud
      cloud_target = icp_util.batch_transform_cloud_xyz(cloud_source,
                                                        ego_motion)
      transform, residual = self._run_icp(cloud_source, ego_motion,
                                          cloud_target)
      self.assertAllClose(transform.eval(), tf.zeros_like(transform).eval(),
                          atol=1e-4)
      self.assertAllClose(residual.eval(), tf.zeros_like(residual).eval(),
                          atol=1e-3)

  def test_no_change_lidar_cloud(self):
    with self.test_session():
      cloud_source = self.lidar_cloud
      transform, residual = self._run_icp(cloud_source, self.identity_transform,
                                          cloud_source)
      self.assertAlmostEqual(transform.eval()[0, self.index_translation], 0,
                             places=4)
      self.assertAllClose(residual.eval(), tf.zeros_like(residual).eval(),
                          atol=1e-4)

  def test_translate_lidar_cloud_batch_size_2(self):
    with self.test_session():
      batch_size = 2
      tx = 0.1
      self.assertEqual(len(self.lidar_cloud.shape), 3)
      cloud_source = tf.tile(self.lidar_cloud, [batch_size, 1, 1])
      cloud_target = cloud_source + [tx, 0, 0]
      self.assertEqual(len(self.identity_transform.shape), 2)
      ego_motion = tf.tile(self.identity_transform, [batch_size, 1])
      logging.info('cloud_source.shape: %s', cloud_source.shape)
      logging.info('cloud_target.shape: %s', cloud_target.shape)
      transform, residual = self._run_icp(cloud_source, ego_motion,
                                          cloud_target)
      for b in range(batch_size):
        self.assertAlmostEqual(transform.eval()[b, self.index_translation], tx,
                               places=4)
      self.assertAllClose(residual.eval(), tf.zeros_like(residual).eval(),
                          atol=1e-4)


if __name__ == '__main__':
  tf.test.main()
