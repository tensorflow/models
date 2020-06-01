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

import numpy as np
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
    with self.session() as sess:
      normalized_image_out = sess.run(normalized_image)

    self.assertAllEqual(normalized_image_out, exp_normalized_image)

  def testCalculateReceptiveBoxes(self):
    boxes = feature_extractor.CalculateReceptiveBoxes(
        height=1, width=2, rf=291, stride=32, padding=145)
    exp_boxes = [[-145., -145., 145., 145.], [-145., -113., 145., 177.]]
    with self.session() as sess:
      boxes_out = sess.run(boxes)

    self.assertAllEqual(exp_boxes, boxes_out)

  def testCalculateKeypointCenters(self):
    boxes = [[-10.0, 0.0, 11.0, 21.0], [-2.5, 5.0, 18.5, 26.0],
             [45.0, -2.5, 66.0, 18.5]]
    centers = feature_extractor.CalculateKeypointCenters(boxes)
    with self.session() as sess:
      centers_out = sess.run(centers)

    exp_centers = [[0.5, 10.5], [8.0, 15.5], [55.5, 8.0]]

    self.assertAllEqual(exp_centers, centers_out)

  def testExtractKeypointDescriptor(self):
    image = tf.constant(
        [[[0, 255, 255], [128, 64, 196]], [[0, 0, 32], [32, 128, 16]]],
        dtype=tf.uint8)

    # Arbitrary model function used to test ExtractKeypointDescriptor. The
    # generated feature_map is a replicated version of the image, concatenated
    # with zeros to achieve the required dimensionality. The attention is simply
    # the norm of the input image pixels.
    def _test_model_fn(image, normalized_image, reuse):
      del normalized_image, reuse  # Unused variables in the test.
      image_shape = tf.shape(image)
      attention = tf.squeeze(tf.norm(image, axis=3))
      feature_map = tf.concat([
          tf.tile(image, [1, 1, 1, 341]),
          tf.zeros([1, image_shape[1], image_shape[2], 1])
      ],
                              axis=3)
      return attention, feature_map

    boxes, feature_scales, features, scores = (
        feature_extractor.ExtractKeypointDescriptor(
            image,
            layer_name='resnet_v1_50/block3',
            image_scales=tf.constant([1.0]),
            iou=1.0,
            max_feature_num=10,
            abs_thres=1.5,
            model_fn=_test_model_fn))

    exp_boxes = [[-145.0, -145.0, 145.0, 145.0], [-113.0, -145.0, 177.0, 145.0]]
    exp_feature_scales = [1.0, 1.0]
    exp_features = np.array(
        np.concatenate(
            (np.tile([[-1.0, 127.0 / 128.0, 127.0 / 128.0], [-1.0, -1.0, -0.75]
                     ], [1, 341]), np.zeros([2, 1])),
            axis=1))
    exp_scores = [[1.723042], [1.600781]]

    with self.session() as sess:
      boxes_out, feature_scales_out, features_out, scores_out = sess.run(
          [boxes, feature_scales, features, scores])

    self.assertAllEqual(exp_boxes, boxes_out)
    self.assertAllEqual(exp_feature_scales, feature_scales_out)
    self.assertAllClose(exp_features, features_out)
    self.assertAllClose(exp_scores, scores_out)

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

    with self.session() as sess:
      output_out = sess.run(output)

    self.assertAllEqual(exp_output, output_out)


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
