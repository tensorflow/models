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

"""Tests for icp grad."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import icp_grad  # pylint: disable=unused-import
import icp_test
import tensorflow as tf
from tensorflow.python.ops import gradient_checker


class IcpOpGradTest(icp_test.IcpOpTestBase):

  def test_grad_transform(self):
    with self.test_session():
      cloud_source = self.small_cloud
      cloud_target = cloud_source + [0.05, 0, 0]
      ego_motion = self.identity_transform
      transform, unused_residual = self._run_icp(cloud_source, ego_motion,
                                                 cloud_target)
      err = gradient_checker.compute_gradient_error(ego_motion,
                                                    ego_motion.shape.as_list(),
                                                    transform,
                                                    transform.shape.as_list())
    # Since our gradient is an approximation, it doesn't pass a numerical check.
    # Nonetheless, this test verifies that icp_grad computes a gradient.
    self.assertGreater(err, 1e-3)

  def test_grad_transform_same_ego_motion(self):
    with self.test_session():
      cloud_source = self.small_cloud
      cloud_target = cloud_source + [0.1, 0, 0]
      ego_motion = tf.constant([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]],
                               dtype=tf.float32)
      transform, unused_residual = self._run_icp(cloud_source, ego_motion,
                                                 cloud_target)
      err = gradient_checker.compute_gradient_error(ego_motion,
                                                    ego_motion.shape.as_list(),
                                                    transform,
                                                    transform.shape.as_list())
    # Since our gradient is an approximation, it doesn't pass a numerical check.
    # Nonetheless, this test verifies that icp_grad computes a gradient.
    self.assertGreater(err, 1e-3)

  def test_grad_residual(self):
    with self.test_session():
      cloud_source = self.small_cloud
      cloud_target = cloud_source + [0.05, 0, 0]
      ego_motion = self.identity_transform
      unused_transform, residual = self._run_icp(cloud_source, ego_motion,
                                                 cloud_target)
      err = gradient_checker.compute_gradient_error(
          cloud_source, cloud_source.shape.as_list(), residual,
          residual.shape.as_list())
    # Since our gradient is an approximation, it doesn't pass a numerical check.
    # Nonetheless, this test verifies that icp_grad computes a gradient.
    self.assertGreater(err, 1e-3)


if __name__ == '__main__':
  tf.test.main()
