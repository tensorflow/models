# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for image_utils."""
import imghdr
from unittest import mock
from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.vision.data import fake_feature_generator
from official.vision.data import image_utils


class ImageUtilsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('RGB_PNG', 128, 64, 3, 'PNG'), ('RGB_JPEG', 2, 1, 3, 'JPEG'),
      ('GREY_BMP', 32, 32, 1, 'BMP'), ('GREY_PNG', 128, 128, 1, 'png'))
  def test_encode_image_then_decode_image(self, height, width, num_channels,
                                          image_format):
    image_np = fake_feature_generator.generate_image_np(height, width,
                                                        num_channels)
    image_str = image_utils.encode_image(image_np, image_format)
    actual_image_np = image_utils.decode_image(image_str)

    # JPEG encoding does not keep the pixel value.
    if image_format != 'JPEG':
      self.assertAllClose(actual_image_np, image_np)
    self.assertEqual(actual_image_np.shape, image_np.shape)

  @parameterized.named_parameters(
      ('RGB_RAW', 128, 64, 3, tf.bfloat16.as_numpy_dtype),
      ('GREY_RAW', 32, 32, 1, tf.uint8.as_numpy_dtype))
  def test_encode_raw_image_then_decode_raw_image(self, height, width,
                                                  num_channels, image_dtype):
    image_np = fake_feature_generator.generate_image_np(height, width,
                                                        num_channels)
    image_np = image_np.astype(image_dtype)
    image_str = image_utils.encode_image(image_np, 'RAW')
    actual_image_np = image_utils.decode_image(image_str, 'RAW', image_dtype)
    actual_image_np = actual_image_np.reshape([height, width, num_channels])

    self.assertAllClose(actual_image_np, image_np)
    self.assertEqual(actual_image_np.shape, image_np.shape)

  @parameterized.named_parameters(
      ('RGB_PNG', 128, 64, 3, 'PNG'), ('RGB_JPEG', 64, 128, 3, 'JPEG'),
      ('GREY_BMP', 32, 32, 1, 'BMP'), ('GREY_PNG', 128, 128, 1, 'png'))
  def test_encode_image_then_decode_image_metadata(self, height, width,
                                                   num_channels, image_format):
    image_np = fake_feature_generator.generate_image_np(height, width,
                                                        num_channels)
    image_str = image_utils.encode_image(image_np, image_format)
    (actual_height, actual_width, actual_num_channels, actual_format) = (
        image_utils.decode_image_metadata(image_str))

    self.assertEqual(actual_height, height)
    self.assertEqual(actual_width, width)
    self.assertEqual(actual_num_channels, num_channels)
    self.assertEqual(actual_format, image_format.upper())

  def test_encode_image_raise_error_with_invalid_image_format(self):
    with self.assertRaisesRegex(ValueError, 'Image format is invalid: foo'):
      image_np = fake_feature_generator.generate_image_np(2, 2, 1)
      image_utils.encode_image(image_np, 'foo')

  @mock.patch.object(imghdr, 'what', return_value='foo', autospec=True)
  def test_decode_image_raise_error_with_invalid_image_format(self, _):
    image_np = fake_feature_generator.generate_image_np(1, 1, 3)
    image_str = image_utils.encode_image(image_np, 'PNG')
    with self.assertRaisesRegex(ValueError, 'Image format is invalid: foo'):
      image_utils.decode_image_metadata(image_str)


if __name__ == '__main__':
  tf.test.main()
