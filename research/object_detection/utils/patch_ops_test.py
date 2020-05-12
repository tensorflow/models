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

"""Tests for object_detection.utils.patch_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from object_detection.utils import patch_ops
from object_detection.utils import test_case


class GetPatchMaskTest(test_case.TestCase, parameterized.TestCase):

  def testMaskShape(self):
    image_shape = [15, 10]
    mask = patch_ops.get_patch_mask(
        10, 5, patch_size=3, image_shape=image_shape)
    self.assertListEqual(mask.shape.as_list(), image_shape)

  def testHandleImageShapeWithChannels(self):
    image_shape = [15, 10, 3]
    mask = patch_ops.get_patch_mask(
        10, 5, patch_size=3, image_shape=image_shape)
    self.assertListEqual(mask.shape.as_list(), image_shape[:2])

  def testMaskDType(self):
    mask = patch_ops.get_patch_mask(2, 3, patch_size=2, image_shape=[6, 7])
    self.assertDTypeEqual(mask, bool)

  def testMaskAreaWithEvenPatchSize(self):
    image_shape = [6, 7]
    mask = patch_ops.get_patch_mask(2, 3, patch_size=2, image_shape=image_shape)
    expected_mask = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]).reshape(image_shape).astype(bool)
    self.assertAllEqual(mask, expected_mask)

  def testMaskAreaWithEvenPatchSize4(self):
    image_shape = [6, 7]
    mask = patch_ops.get_patch_mask(2, 3, patch_size=4, image_shape=image_shape)
    expected_mask = np.array([
        [0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]).reshape(image_shape).astype(bool)
    self.assertAllEqual(mask, expected_mask)

  def testMaskAreaWithOddPatchSize(self):
    image_shape = [6, 7]
    mask = patch_ops.get_patch_mask(2, 3, patch_size=3, image_shape=image_shape)
    expected_mask = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]).reshape(image_shape).astype(bool)
    self.assertAllEqual(mask, expected_mask)

  def testMaskAreaPartiallyOutsideImage(self):
    image_shape = [6, 7]
    mask = patch_ops.get_patch_mask(5, 6, patch_size=5, image_shape=image_shape)
    expected_mask = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1],
    ]).reshape(image_shape).astype(bool)
    self.assertAllEqual(mask, expected_mask)

  @parameterized.parameters(
      {'y': 0, 'x': -1},
      {'y': -1, 'x': 0},
      {'y': 0, 'x': 11},
      {'y': 16, 'x': 0},
  )
  def testStaticCoordinatesOutsideImageRaisesError(self, y, x):
    image_shape = [15, 10]
    with self.assertRaises(tf.errors.InvalidArgumentError):
      patch_ops.get_patch_mask(y, x, patch_size=3, image_shape=image_shape)

  def testDynamicCoordinatesOutsideImageRaisesError(self):

    def graph_fn():
      image_shape = [15, 10]
      x = tf.random_uniform([], minval=-2, maxval=-1, dtype=tf.int32)
      y = tf.random_uniform([], minval=0, maxval=1, dtype=tf.int32)
      mask = patch_ops.get_patch_mask(
          y, x, patch_size=3, image_shape=image_shape)
      return mask

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.execute(graph_fn, [])

  @parameterized.parameters(
      {'patch_size': 0},
      {'patch_size': -1},
  )
  def testStaticNonPositivePatchSizeRaisesError(self, patch_size):
    image_shape = [6, 7]
    with self.assertRaises(tf.errors.InvalidArgumentError):
      patch_ops.get_patch_mask(
          0, 0, patch_size=patch_size, image_shape=image_shape)

  def testDynamicNonPositivePatchSizeRaisesError(self):

    def graph_fn():
      image_shape = [6, 7]
      patch_size = -1 * tf.random_uniform([], minval=0, maxval=3,
                                          dtype=tf.int32)
      mask = patch_ops.get_patch_mask(
          0, 0, patch_size=patch_size, image_shape=image_shape)
      return mask

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.execute(graph_fn, [])


if __name__ == '__main__':
  tf.test.main()
