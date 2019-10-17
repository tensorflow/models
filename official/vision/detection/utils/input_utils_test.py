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

"""Tests input_utils.py."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
from official.vision.detection.utils import input_utils


class InputUtilsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ([1], 10),
      ([1, 2], 10),
      ([1, 2, 3,], 10),
  )
  def testPadToFixedSize(self, input_shape, output_size):
    # Copies input shape to padding shape.
    padding_shape = input_shape[:]
    padding_shape[0] = output_size - input_shape[0]
    expected_outputs = np.concatenate(
        [np.ones(input_shape), np.zeros(padding_shape)], axis=0)

    data = tf.ones(input_shape)
    output_data = input_utils.pad_to_fixed_size(
        data, output_size, constant_values=0)
    output_data = output_data.numpy()
    self.assertAllClose(output_size, output_data.shape[0])
    self.assertAllClose(expected_outputs, output_data)

  @parameterized.parameters(
      (100, 200, 100, 200, 32, 1.0, 1.0, 100, 200, 128, 224),
      (100, 256, 128, 256, 32, 1.0, 1.0, 100, 256, 128, 256),
      (200, 512, 200, 128, 32, 0.25, 0.25, 50, 128, 224, 128),
  )
  def testResizeAndCropImageRectangluarCase(self,
                                            input_height,
                                            input_width,
                                            desired_height,
                                            desired_width,
                                            stride,
                                            scale_y,
                                            scale_x,
                                            scaled_height,
                                            scaled_width,
                                            output_height,
                                            output_width):
    image = tf.convert_to_tensor(
        value=np.random.rand(input_height, input_width, 3))

    desired_size = (desired_height, desired_width)
    resized_image, image_info = input_utils.resize_and_crop_image(
        image,
        desired_size=desired_size,
        padded_size=input_utils.compute_padded_size(desired_size, stride))
    resized_image_shape = tf.shape(input=resized_image)
    resized_shape_np = resized_image_shape.numpy()
    image_info_np = image_info.numpy()

    self.assertAllEqual([output_height, output_width, 3], resized_shape_np)
    self.assertNDArrayNear(
        [[input_height, input_width], [scaled_height, scaled_width],
         [scale_y, scale_x], [0.0, 0.0]], image_info_np, 1e-5)

  @parameterized.parameters(
      (100, 200, 220, 220, 32, 1.1, 1.1, 110, 220, 224, 224),
      (512, 512, 1024, 1024, 32, 2.0, 2.0, 1024, 1024, 1024, 1024),
  )
  def testResizeAndCropImageSquareCase(self,
                                       input_height,
                                       input_width,
                                       desired_height,
                                       desired_width,
                                       stride,
                                       scale_y,
                                       scale_x,
                                       scaled_height,
                                       scaled_width,
                                       output_height,
                                       output_width):
    image = tf.convert_to_tensor(
        value=np.random.rand(input_height, input_width, 3))

    desired_size = (desired_height, desired_width)
    resized_image, image_info = input_utils.resize_and_crop_image(
        image,
        desired_size=desired_size,
        padded_size=input_utils.compute_padded_size(desired_size, stride))
    resized_image_shape = tf.shape(input=resized_image)

    resized_shape_np, image_info_np = ([
        resized_image_shape.numpy(),
        image_info.numpy()
    ])
    self.assertAllEqual([output_height, output_width, 3], resized_shape_np)
    self.assertNDArrayNear(
        [[input_height, input_width], [scaled_height, scaled_width],
         [scale_y, scale_x], [0.0, 0.0]], image_info_np, 1e-5)


if __name__ == '__main__':
  assert tf.version.VERSION.startswith('2.')
  tf.test.main()
