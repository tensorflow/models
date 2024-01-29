# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for preprocessing_ops.py."""
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.projects.yolo.ops import preprocessing_ops
from official.vision.ops import box_ops as bbox_ops


class InputUtilsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(([1, 2], 20, 0), ([13, 2, 4], 15, 0))
  def testPadMaxInstances(self, input_shape, instances, pad_axis):
    expected_output_shape = input_shape
    expected_output_shape[pad_axis] = instances
    output = preprocessing_ops.pad_max_instances(
        np.ones(input_shape), instances, pad_axis=pad_axis)
    self.assertAllEqual(expected_output_shape, tf.shape(output).numpy())

  @parameterized.parameters((100, 200))
  def testGetImageShape(self, image_height, image_width):
    image = tf.convert_to_tensor(np.random.rand(image_height, image_width, 3))
    image_shape = preprocessing_ops.get_image_shape(image)
    self.assertAllEqual((image_height, image_width), image_shape)

  @parameterized.parameters((400, 600, .5, .5, .0, True),
                            (100, 200, .5, .5, .5))
  def testImageRandHSV(self,
                       image_height,
                       image_width,
                       rh,
                       rs,
                       rv,
                       is_darknet=False):
    image = tf.convert_to_tensor(np.random.rand(image_height, image_width, 3))
    processed_image = preprocessing_ops.image_rand_hsv(
        image, rh, rs, rv, darknet=is_darknet)
    processed_image_shape = tf.shape(processed_image)
    self.assertAllEqual([image_height, image_width, 3],
                        processed_image_shape.numpy())

  @parameterized.parameters((100, 200, [50, 100]))
  def testResizeAndJitterImage(self, image_height, image_width, desired_size):
    image = tf.convert_to_tensor(np.random.rand(image_height, image_width, 3))
    processed_image, _, _ = preprocessing_ops.resize_and_jitter_image(
        image, desired_size)
    processed_image_shape = tf.shape(processed_image)
    self.assertAllEqual([desired_size[0], desired_size[1], 3],
                        processed_image_shape.numpy())

  @parameterized.parameters((400, 600, [200, 300]))
  def testAffineWarpImage(self,
                          image_height,
                          image_width,
                          desired_size,
                          degrees=7.0,
                          scale_min=0.1,
                          scale_max=1.9):
    image = tf.convert_to_tensor(np.random.rand(image_height, image_width, 3))
    processed_image, _, _ = preprocessing_ops.affine_warp_image(
        image,
        desired_size,
        degrees=degrees,
        scale_min=scale_min,
        scale_max=scale_max)
    processed_image_shape = tf.shape(processed_image)
    self.assertAllEqual([desired_size[0], desired_size[1], 3],
                        processed_image_shape.numpy())

  # Working Test
  @parameterized.parameters(([[400, 600], [200, 300],
                              [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], 50))
  def testAffineWarpBoxes(self, affine, num_boxes):
    boxes = tf.convert_to_tensor(np.random.rand(num_boxes, 4))
    boxes = bbox_ops.denormalize_boxes(boxes, affine[0])
    processed_boxes, _ = preprocessing_ops.affine_warp_boxes(
        tf.cast(affine[2], tf.double), boxes, affine[1], box_history=boxes)
    processed_boxes_shape = tf.shape(processed_boxes)
    self.assertAllEqual([num_boxes, 4], processed_boxes_shape.numpy())

  # Working Test
  @parameterized.parameters(([100, 100], [[-0.489, 51.28, 0.236, 51.686],
                                          [65, 100, 200, 150],
                                          [150, 80, 200, 130]]))
  def testBoxCandidates(self, output_size, boxes):
    boxes = tf.cast(bbox_ops.denormalize_boxes(boxes, output_size), tf.double)
    clipped_ind = preprocessing_ops.boxes_candidates(
        boxes, boxes, ar_thr=1e32, wh_thr=0, area_thr=tf.cast(0, tf.double))
    clipped_ind_shape = tf.shape(clipped_ind)
    self.assertAllEqual([3], clipped_ind_shape.numpy())
    self.assertAllEqual([0, 1, 2], clipped_ind.numpy())

  # Working Test
  @parameterized.parameters((
      50,
      [0.5, 0.5],
      [0, 0],  # Clipping all boxes
      [0.0, 0.0]))
  def testResizeAndCropBoxes(self, num_boxes, image_scale, output_size, offset):
    boxes = tf.convert_to_tensor(np.random.rand(num_boxes, 4))
    processed_boxes, _ = preprocessing_ops.resize_and_crop_boxes(
        boxes, tf.cast(image_scale, tf.double), output_size,
        tf.cast(offset, tf.double), boxes)
    processed_boxes_shape = tf.shape(processed_boxes)
    self.assertAllEqual([num_boxes, 4], processed_boxes_shape.numpy())
    self.assertAllEqual(
        tf.math.reduce_sum(processed_boxes), tf.convert_to_tensor(0))


if __name__ == '__main__':
  tf.test.main()
