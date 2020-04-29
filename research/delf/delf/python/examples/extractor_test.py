# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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
"""Tests for DELF feature extractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from delf import delf_config_pb2
from delf import extractor


class ExtractorTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('Max-1Min-1', -1, -1, 1.0, False, [4, 2, 3], [1.0, 1.0]),
      ('Max-1Min-1Square', -1, -1, 1.0, True, [4, 4, 3], [1.0, 2.0]),
      ('Max2Min-1', 2, -1, 1.0, False, [2, 1, 3], [0.5, 0.5]),
      ('Max2Min-1Square', 2, -1, 1.0, True, [2, 2, 3], [0.5, 1.0]),
      ('Max8Min-1', 8, -1, 1.0, False, [4, 2, 3], [1.0, 1.0]),
      ('Max8Min-1Square', 8, -1, 1.0, True, [4, 4, 3], [1.0, 2.0]),
      ('Max-1Min1', -1, 1, 1.0, False, [4, 2, 3], [1.0, 1.0]),
      ('Max-1Min1Square', -1, 1, 1.0, True, [4, 4, 3], [1.0, 2.0]),
      ('Max-1Min8', -1, 8, 1.0, False, [8, 4, 3], [2.0, 2.0]),
      ('Max-1Min8Square', -1, 8, 1.0, True, [8, 8, 3], [2.0, 4.0]),
      ('Max16Min8', 16, 8, 1.0, False, [8, 4, 3], [2.0, 2.0]),
      ('Max16Min8Square', 16, 8, 1.0, True, [8, 8, 3], [2.0, 4.0]),
      ('Max2Min2', 2, 2, 1.0, False, [2, 1, 3], [0.5, 0.5]),
      ('Max2Min2Square', 2, 2, 1.0, True, [2, 2, 3], [0.5, 1.0]),
      ('Max-1Min-1Factor0.5', -1, -1, 0.5, False, [4, 2, 3], [1.0, 1.0]),
      ('Max-1Min-1Factor0.5Square', -1, -1, 0.5, True, [4, 4, 3], [1.0, 2.0]),
      ('Max2Min-1Factor2.0', 2, -1, 2.0, False, [4, 2, 3], [1.0, 1.0]),
      ('Max2Min-1Factor2.0Square', 2, -1, 2.0, True, [4, 4, 3], [1.0, 2.0]),
      ('Max-1Min8Factor0.5', -1, 8, 0.5, False, [4, 2, 3], [1.0, 1.0]),
      ('Max-1Min8Factor0.5Square', -1, 8, 0.5, True, [4, 4, 3], [1.0, 2.0]),
      ('Max-1Min8Factor0.25', -1, 8, 0.25, False, [4, 2, 3], [1.0, 1.0]),
      ('Max-1Min8Factor0.25Square', -1, 8, 0.25, True, [4, 4, 3], [1.0, 2.0]),
      ('Max2Min2Factor2.0', 2, 2, 2.0, False, [4, 2, 3], [1.0, 1.0]),
      ('Max2Min2Factor2.0Square', 2, 2, 2.0, True, [4, 4, 3], [1.0, 2.0]),
      ('Max16Min8Factor0.5', 16, 8, 0.5, False, [4, 2, 3], [1.0, 1.0]),
      ('Max16Min8Factor0.5Square', 16, 8, 0.5, True, [4, 4, 3], [1.0, 2.0]),
  )
  def testResizeImageWorks(self, max_image_size, min_image_size, resize_factor,
                           square_output, expected_shape,
                           expected_scale_factors):
    # Construct image of size 4x2x3.
    image = np.array([[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]],
                      [[4, 4, 4], [5, 5, 5]], [[6, 6, 6], [7, 7, 7]]],
                     dtype='uint8')

    # Set up config.
    config = delf_config_pb2.DelfConfig(
        max_image_size=max_image_size, min_image_size=min_image_size)

    resized_image, scale_factors = extractor.ResizeImage(
        image, config, resize_factor, square_output)
    self.assertAllEqual(resized_image.shape, expected_shape)
    self.assertAllClose(scale_factors, expected_scale_factors)

  @parameterized.named_parameters(
      ('Max2Min2', 2, 2, 1.0, False, [2, 1, 3], [0.666666, 0.5]),
      ('Max2Min2Square', 2, 2, 1.0, True, [2, 2, 3], [0.666666, 1.0]),
  )
  def testResizeImageRoundingWorks(self, max_image_size, min_image_size,
                                   resize_factor, square_output, expected_shape,
                                   expected_scale_factors):
    # Construct image of size 3x2x3.
    image = np.array([[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]],
                      [[4, 4, 4], [5, 5, 5]]],
                     dtype='uint8')

    # Set up config.
    config = delf_config_pb2.DelfConfig(
        max_image_size=max_image_size, min_image_size=min_image_size)

    resized_image, scale_factors = extractor.ResizeImage(
        image, config, resize_factor, square_output)
    self.assertAllEqual(resized_image.shape, expected_shape)
    self.assertAllClose(scale_factors, expected_scale_factors)


if __name__ == '__main__':
  tf.test.main()
