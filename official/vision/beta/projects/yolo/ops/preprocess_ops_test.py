# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""preprocess_ops tests."""
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.vision.beta.projects.yolo.ops import preprocess_ops


class PreprocessOpsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((416, 416, 5, 300, 300), (100, 200, 6, 50, 50))
  def test_resize_crop_filter(self, default_width, default_height, num_boxes,
                              target_width, target_height):
    image = tf.convert_to_tensor(
        np.random.rand(default_width, default_height, 3))
    boxes = tf.convert_to_tensor(np.random.rand(num_boxes, 4))
    resized_image, resized_boxes = preprocess_ops.resize_crop_filter(
        image, boxes, default_width, default_height, target_width,
        target_height)
    resized_image_shape = tf.shape(resized_image)
    resized_boxes_shape = tf.shape(resized_boxes)
    self.assertAllEqual([default_height, default_width, 3],
                        resized_image_shape.numpy())
    self.assertAllEqual([num_boxes, 4], resized_boxes_shape.numpy())

  @parameterized.parameters((7, 7., 5.), (25, 35., 45.))
  def test_translate_boxes(self, num_boxes, translate_x, translate_y):
    boxes = tf.convert_to_tensor(np.random.rand(num_boxes, 4))
    translated_boxes = preprocess_ops.translate_boxes(
        boxes, translate_x, translate_y)
    translated_boxes_shape = tf.shape(translated_boxes)
    self.assertAllEqual([num_boxes, 4], translated_boxes_shape.numpy())

  @parameterized.parameters((100, 200, 75., 25.), (400, 600, 25., 75.))
  def test_translate_image(self, image_height, image_width, translate_x,
                           translate_y):
    image = tf.convert_to_tensor(np.random.rand(image_height, image_width, 4))
    translated_image = preprocess_ops.translate_image(
        image, translate_x, translate_y)
    translated_image_shape = tf.shape(translated_image)
    self.assertAllEqual([image_height, image_width, 4],
                        translated_image_shape.numpy())

  @parameterized.parameters(([1, 2], 20, 0), ([13, 2, 4], 15, 0))
  def test_pad_max_instances(self, input_shape, instances, pad_axis):
    expected_output_shape = input_shape
    expected_output_shape[pad_axis] = instances
    output = preprocess_ops.pad_max_instances(
        np.ones(input_shape), instances, pad_axis=pad_axis)
    self.assertAllEqual(expected_output_shape, tf.shape(output).numpy())


if __name__ == '__main__':
  tf.test.main()
