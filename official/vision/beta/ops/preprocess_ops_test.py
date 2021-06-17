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


"""Tests for preprocess_ops.py."""

import io
# Import libraries
from absl.testing import parameterized
import numpy as np
from PIL import Image
import tensorflow as tf

from official.vision.beta.ops import preprocess_ops


def _encode_image(image_array, fmt):
  image = Image.fromarray(image_array)
  with io.BytesIO() as output:
    image.save(output, format=fmt)
    return output.getvalue()


class InputUtilsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ([1], 10),
      ([1, 2], 10),
      ([1, 2, 3], 10),
      ([11], 10),
      ([12, 2], 10),
      ([13, 2, 3], 10),
  )
  def testPadToFixedSize(self, input_shape, output_size):
    # Copies input shape to padding shape.
    clip_shape = input_shape[:]
    clip_shape[0] = min(output_size, clip_shape[0])
    padding_shape = input_shape[:]
    padding_shape[0] = max(output_size - input_shape[0], 0)
    expected_outputs = np.concatenate(
        [np.ones(clip_shape), np.zeros(padding_shape)], axis=0)

    data = tf.ones(input_shape)
    output_data = preprocess_ops.clip_or_pad_to_fixed_size(
        data, output_size, constant_values=0)
    output_data = output_data.numpy()
    self.assertAllClose(output_size, output_data.shape[0])
    self.assertAllClose(expected_outputs, output_data)

  @parameterized.parameters(
      (100, 200, 100, 200, 32, 1.0, 1.0, 128, 224),
      (100, 256, 128, 256, 32, 1.0, 1.0, 128, 256),
      (200, 512, 200, 128, 32, 0.25, 0.25, 224, 128),
  )
  def testResizeAndCropImageRectangluarCase(self,
                                            input_height,
                                            input_width,
                                            desired_height,
                                            desired_width,
                                            stride,
                                            scale_y,
                                            scale_x,
                                            output_height,
                                            output_width):
    image = tf.convert_to_tensor(
        np.random.rand(input_height, input_width, 3))

    desired_size = (desired_height, desired_width)
    resized_image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        desired_size=desired_size,
        padded_size=preprocess_ops.compute_padded_size(desired_size, stride))
    resized_image_shape = tf.shape(resized_image)

    self.assertAllEqual(
        [output_height, output_width, 3],
        resized_image_shape.numpy())
    self.assertNDArrayNear(
        [[input_height, input_width],
         [desired_height, desired_width],
         [scale_y, scale_x],
         [0.0, 0.0]],
        image_info.numpy(),
        1e-5)

  @parameterized.parameters(
      (100, 200, 220, 220, 32, 1.1, 1.1, 224, 224),
      (512, 512, 1024, 1024, 32, 2.0, 2.0, 1024, 1024),
  )
  def testResizeAndCropImageSquareCase(self,
                                       input_height,
                                       input_width,
                                       desired_height,
                                       desired_width,
                                       stride,
                                       scale_y,
                                       scale_x,
                                       output_height,
                                       output_width):
    image = tf.convert_to_tensor(
        np.random.rand(input_height, input_width, 3))

    desired_size = (desired_height, desired_width)
    resized_image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        desired_size=desired_size,
        padded_size=preprocess_ops.compute_padded_size(desired_size, stride))
    resized_image_shape = tf.shape(resized_image)

    self.assertAllEqual(
        [output_height, output_width, 3],
        resized_image_shape.numpy())
    self.assertNDArrayNear(
        [[input_height, input_width],
         [desired_height, desired_width],
         [scale_y, scale_x],
         [0.0, 0.0]],
        image_info.numpy(),
        1e-5)

  @parameterized.parameters(
      (100, 200, 100, 300, 32, 1.0, 1.0, 100, 200, 128, 320),
      (200, 100, 100, 300, 32, 1.0, 1.0, 200, 100, 320, 128),
      (100, 200, 80, 100, 32, 0.5, 0.5, 50, 100, 96, 128),
      (200, 100, 80, 100, 32, 0.5, 0.5, 100, 50, 128, 96),
  )
  def testResizeAndCropImageV2(self,
                               input_height,
                               input_width,
                               short_side,
                               long_side,
                               stride,
                               scale_y,
                               scale_x,
                               desired_height,
                               desired_width,
                               output_height,
                               output_width):
    image = tf.convert_to_tensor(
        np.random.rand(input_height, input_width, 3))
    image_shape = tf.shape(image)[0:2]

    desired_size = tf.where(
        tf.greater(image_shape[0], image_shape[1]),
        tf.constant([long_side, short_side], dtype=tf.int32),
        tf.constant([short_side, long_side], dtype=tf.int32))
    resized_image, image_info = preprocess_ops.resize_and_crop_image_v2(
        image,
        short_side=short_side,
        long_side=long_side,
        padded_size=preprocess_ops.compute_padded_size(desired_size, stride))
    resized_image_shape = tf.shape(resized_image)

    self.assertAllEqual(
        [output_height, output_width, 3],
        resized_image_shape.numpy())
    self.assertNDArrayNear(
        [[input_height, input_width],
         [desired_height, desired_width],
         [scale_y, scale_x],
         [0.0, 0.0]],
        image_info.numpy(),
        1e-5)

  @parameterized.parameters(
      (400, 600), (600, 400),
  )
  def testCenterCropImage(self,
                          input_height,
                          input_width):
    image = tf.convert_to_tensor(
        np.random.rand(input_height, input_width, 3))
    cropped_image = preprocess_ops.center_crop_image(image)
    cropped_image_shape = tf.shape(cropped_image)
    self.assertAllEqual([350, 350, 3], cropped_image_shape.numpy())

  @parameterized.parameters(
      (400, 600), (600, 400),
  )
  def testCenterCropImageV2(self,
                            input_height,
                            input_width):
    image_bytes = tf.constant(
        _encode_image(
            np.uint8(np.random.rand(input_height, input_width, 3) * 255),
            fmt='JPEG'),
        dtype=tf.string)
    cropped_image = preprocess_ops.center_crop_image_v2(
        image_bytes, tf.constant([input_height, input_width, 3], tf.int32))
    cropped_image_shape = tf.shape(cropped_image)
    self.assertAllEqual([350, 350, 3], cropped_image_shape.numpy())

  @parameterized.parameters(
      (400, 600), (600, 400),
  )
  def testRandomCropImage(self,
                          input_height,
                          input_width):
    image = tf.convert_to_tensor(
        np.random.rand(input_height, input_width, 3))
    _ = preprocess_ops.random_crop_image(image)

  @parameterized.parameters(
      (400, 600), (600, 400),
  )
  def testRandomCropImageV2(self,
                            input_height,
                            input_width):
    image_bytes = tf.constant(
        _encode_image(
            np.uint8(np.random.rand(input_height, input_width, 3) * 255),
            fmt='JPEG'),
        dtype=tf.string)
    _ = preprocess_ops.random_crop_image_v2(
        image_bytes, tf.constant([input_height, input_width, 3], tf.int32))


if __name__ == '__main__':
  tf.test.main()
