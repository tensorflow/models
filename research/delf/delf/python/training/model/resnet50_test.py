# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the ResNet backbone."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from delf.python.training.model import resnet50


class Resnet50Test(tf.test.TestCase):

  def test_gem_pooling_works(self):
    # Input feature map: Batch size = 2, height = 1, width = 2, depth = 2.
    feature_map = tf.constant([[[[.0, 2.0], [1.0, -1.0]]],
                               [[[1.0, 100.0], [1.0, .0]]]],
                              dtype=tf.float32)
    power = 2.0
    threshold = .0

    # Run tested function.
    pooled_feature_map = resnet50.gem_pooling(feature_map=feature_map,
                                              axis=[1, 2],
                                              power=power,
                                              threshold=threshold)

    # Define expected result.
    expected_pooled_feature_map = np.array([[0.707107, 1.414214],
                                            [1.0, 70.710678]],
                                           dtype=float)

    # Compare actual and expected.
    self.assertAllClose(pooled_feature_map, expected_pooled_feature_map)


if __name__ == '__main__':
  tf.test.main()
