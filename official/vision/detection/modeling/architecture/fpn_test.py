# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for fpn.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow.compat.v2 as tf
from official.vision.detection.modeling.architecture import fpn


class FpnTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (3, 7),
      (3, 4),
  )
  def testFPNOutputShape(self, min_level, max_level):
    backbone_min_level = 2
    backbone_max_level = 5
    fpn_feat_dims = 256
    image_size = 256

    inputs = {}
    for level in range(backbone_min_level, backbone_max_level + 1):
      inputs[level] = tf.zeros(
          [1, image_size // 2**level, image_size // 2**level, fpn_feat_dims])

    with tf.name_scope('min_level_%d_max_level_%d' % (min_level, max_level)):
      fpn_fn = fpn.Fpn(min_level=min_level,
                       max_level=max_level,
                       fpn_feat_dims=fpn_feat_dims)
      features = fpn_fn(inputs)

      for level in range(min_level, max_level):
        self.assertEqual(features[level].get_shape().as_list(),
                         [1, image_size // 2**level,
                          image_size // 2**level, fpn_feat_dims])
      self.assertEqual(sorted(features.keys()), range(min_level, max_level + 1))


if __name__ == '__main__':
  assert tf.version.VERSION.startswith('2.')
  tf.test.main()
