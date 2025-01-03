# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Tests for dense_prediction_cell."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from deeplab.core import dense_prediction_cell


class DensePredictionCellTest(tf.test.TestCase):

  def setUp(self):
    self.segmentation_layer = dense_prediction_cell.DensePredictionCell(
        config=[
            {
                dense_prediction_cell._INPUT: -1,
                dense_prediction_cell._OP: dense_prediction_cell._CONV,
                dense_prediction_cell._KERNEL: 1,
            },
            {
                dense_prediction_cell._INPUT: 0,
                dense_prediction_cell._OP: dense_prediction_cell._CONV,
                dense_prediction_cell._KERNEL: 3,
                dense_prediction_cell._RATE: [1, 3],
            },
            {
                dense_prediction_cell._INPUT: 1,
                dense_prediction_cell._OP: (
                    dense_prediction_cell._PYRAMID_POOLING),
                dense_prediction_cell._GRID_SIZE: [1, 2],
            },
        ],
        hparams={'conv_rate_multiplier': 2})

  def testPyramidPoolingArguments(self):
    features_size, pooled_kernel = (
        self.segmentation_layer._get_pyramid_pooling_arguments(
            crop_size=[513, 513],
            output_stride=16,
            image_grid=[4, 4]))
    self.assertListEqual(features_size, [33, 33])
    self.assertListEqual(pooled_kernel, [9, 9])

  def testPyramidPoolingArgumentsWithImageGrid1x1(self):
    features_size, pooled_kernel = (
        self.segmentation_layer._get_pyramid_pooling_arguments(
            crop_size=[257, 257],
            output_stride=16,
            image_grid=[1, 1]))
    self.assertListEqual(features_size, [17, 17])
    self.assertListEqual(pooled_kernel, [17, 17])

  def testParseOperationStringWithConv1x1(self):
    operation = self.segmentation_layer._parse_operation(
        config={
            dense_prediction_cell._OP: dense_prediction_cell._CONV,
            dense_prediction_cell._KERNEL: [1, 1],
        },
        crop_size=[513, 513], output_stride=16)
    self.assertEqual(operation[dense_prediction_cell._OP],
                     dense_prediction_cell._CONV)
    self.assertListEqual(operation[dense_prediction_cell._KERNEL], [1, 1])

  def testParseOperationStringWithConv3x3(self):
    operation = self.segmentation_layer._parse_operation(
        config={
            dense_prediction_cell._OP: dense_prediction_cell._CONV,
            dense_prediction_cell._KERNEL: [3, 3],
            dense_prediction_cell._RATE: [9, 6],
        },
        crop_size=[513, 513], output_stride=16)
    self.assertEqual(operation[dense_prediction_cell._OP],
                     dense_prediction_cell._CONV)
    self.assertListEqual(operation[dense_prediction_cell._KERNEL], [3, 3])
    self.assertEqual(operation[dense_prediction_cell._RATE], [9, 6])

  def testParseOperationStringWithPyramidPooling2x2(self):
    operation = self.segmentation_layer._parse_operation(
        config={
            dense_prediction_cell._OP: dense_prediction_cell._PYRAMID_POOLING,
            dense_prediction_cell._GRID_SIZE: [2, 2],
        },
        crop_size=[513, 513],
        output_stride=16)
    self.assertEqual(operation[dense_prediction_cell._OP],
                     dense_prediction_cell._PYRAMID_POOLING)
    # The feature maps of size [33, 33] should be covered by 2x2 kernels with
    # size [17, 17].
    self.assertListEqual(
        operation[dense_prediction_cell._TARGET_SIZE], [33, 33])
    self.assertListEqual(operation[dense_prediction_cell._KERNEL], [17, 17])

  def testBuildCell(self):
    with self.test_session(graph=tf.Graph()) as sess:
      features = tf.random_normal([2, 33, 33, 5])
      concat_logits = self.segmentation_layer.build_cell(
          features,
          output_stride=8,
          crop_size=[257, 257])
      sess.run(tf.global_variables_initializer())
      concat_logits = sess.run(concat_logits)
      self.assertTrue(concat_logits.any())

  def testBuildCellWithImagePoolingCropSize(self):
    with self.test_session(graph=tf.Graph()) as sess:
      features = tf.random_normal([2, 33, 33, 5])
      concat_logits = self.segmentation_layer.build_cell(
          features,
          output_stride=8,
          crop_size=[257, 257],
          image_pooling_crop_size=[129, 129])
      sess.run(tf.global_variables_initializer())
      concat_logits = sess.run(concat_logits)
      self.assertTrue(concat_logits.any())


if __name__ == '__main__':
  tf.test.main()
