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
"""Tests for DELF feature extractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from delf import feature_extractor


class FeatureExtractorTest(tf.test.TestCase):

  def testNormalizePixelValues(self):
    image = tf.constant(
        [[[3, 255, 0], [34, 12, 5]], [[45, 5, 65], [56, 77, 89]]],
        dtype=tf.uint8)
    normalized_image = feature_extractor.NormalizePixelValues(
        image, pixel_value_offset=5.0, pixel_value_scale=2.0)
    exp_normalized_image = [[[-1.0, 125.0, -2.5], [14.5, 3.5, 0.0]],
                            [[20.0, 0.0, 30.0], [25.5, 36.0, 42.0]]]

    self.assertAllEqual(normalized_image, exp_normalized_image)

  def testCalculateReceptiveBoxes(self):
    boxes = feature_extractor.CalculateReceptiveBoxes(
        height=1, width=2, rf=291, stride=32, padding=145)
    exp_boxes = [[-145., -145., 145., 145.], [-145., -113., 145., 177.]]

    self.assertAllEqual(exp_boxes, boxes)

  def testCalculateKeypointCenters(self):
    boxes = [[-10.0, 0.0, 11.0, 21.0], [-2.5, 5.0, 18.5, 26.0],
             [45.0, -2.5, 66.0, 18.5]]
    centers = feature_extractor.CalculateKeypointCenters(boxes)

    exp_centers = [[0.5, 10.5], [8.0, 15.5], [55.5, 8.0]]

    self.assertAllEqual(exp_centers, centers)

  def testPcaWhitening(self):
    data = tf.constant([[1.0, 2.0, -2.0], [-5.0, 0.0, 3.0], [-1.0, 2.0, 0.0],
                        [0.0, 4.0, -1.0]])
    pca_matrix = tf.constant([[2.0, 0.0, -1.0], [0.0, 1.0, 1.0],
                              [-1.0, 1.0, 3.0]])
    pca_mean = tf.constant([1.0, 2.0, 3.0])
    output_dim = 2
    use_whitening = True
    pca_variances = tf.constant([4.0, 1.0])

    output = feature_extractor.ApplyPcaAndWhitening(data, pca_matrix, pca_mean,
                                                    output_dim, use_whitening,
                                                    pca_variances)

    exp_output = [[2.5, -5.0], [-6.0, -2.0], [-0.5, -3.0], [1.0, -2.0]]

    self.assertAllEqual(exp_output, output)


if __name__ == '__main__':
  tf.test.main()
