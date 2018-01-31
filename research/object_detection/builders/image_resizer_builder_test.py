# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for object_detection.builders.image_resizer_builder."""
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import image_resizer_builder
from object_detection.protos import image_resizer_pb2


class ImageResizerBuilderTest(tf.test.TestCase):

  def _shape_of_resized_random_image_given_text_proto(self, input_shape,
                                                      text_proto):
    image_resizer_config = image_resizer_pb2.ImageResizer()
    text_format.Merge(text_proto, image_resizer_config)
    image_resizer_fn = image_resizer_builder.build(image_resizer_config)
    images = tf.to_float(
        tf.random_uniform(input_shape, minval=0, maxval=255, dtype=tf.int32))
    resized_images, _ = image_resizer_fn(images)
    with self.test_session() as sess:
      return sess.run(resized_images).shape

  def test_build_keep_aspect_ratio_resizer_returns_expected_shape(self):
    image_resizer_text_proto = """
      keep_aspect_ratio_resizer {
        min_dimension: 10
        max_dimension: 20
      }
    """
    input_shape = (50, 25, 3)
    expected_output_shape = (20, 10, 3)
    output_shape = self._shape_of_resized_random_image_given_text_proto(
        input_shape, image_resizer_text_proto)
    self.assertEqual(output_shape, expected_output_shape)

  def test_build_keep_aspect_ratio_resizer_with_padding(self):
    image_resizer_text_proto = """
      keep_aspect_ratio_resizer {
        min_dimension: 10
        max_dimension: 20
        pad_to_max_dimension: true
      }
    """
    input_shape = (50, 25, 3)
    expected_output_shape = (20, 20, 3)
    output_shape = self._shape_of_resized_random_image_given_text_proto(
        input_shape, image_resizer_text_proto)
    self.assertEqual(output_shape, expected_output_shape)

  def test_built_fixed_shape_resizer_returns_expected_shape(self):
    image_resizer_text_proto = """
      fixed_shape_resizer {
        height: 10
        width: 20
      }
    """
    input_shape = (50, 25, 3)
    expected_output_shape = (10, 20, 3)
    output_shape = self._shape_of_resized_random_image_given_text_proto(
        input_shape, image_resizer_text_proto)
    self.assertEqual(output_shape, expected_output_shape)

  def test_raises_error_on_invalid_input(self):
    invalid_input = 'invalid_input'
    with self.assertRaises(ValueError):
      image_resizer_builder.build(invalid_input)

  def _resized_image_given_text_proto(self, image, text_proto):
    image_resizer_config = image_resizer_pb2.ImageResizer()
    text_format.Merge(text_proto, image_resizer_config)
    image_resizer_fn = image_resizer_builder.build(image_resizer_config)
    image_placeholder = tf.placeholder(tf.uint8, [1, None, None, 3])
    resized_image, _ = image_resizer_fn(image_placeholder)
    with self.test_session() as sess:
      return sess.run(resized_image, feed_dict={image_placeholder: image})

  def test_fixed_shape_resizer_nearest_neighbor_method(self):
    image_resizer_text_proto = """
      fixed_shape_resizer {
        height: 1
        width: 1
        resize_method: NEAREST_NEIGHBOR
      }
    """
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    image = np.expand_dims(image, axis=2)
    image = np.tile(image, (1, 1, 3))
    image = np.expand_dims(image, axis=0)
    resized_image = self._resized_image_given_text_proto(
        image, image_resizer_text_proto)
    vals = np.unique(resized_image).tolist()
    self.assertEqual(len(vals), 1)
    self.assertEqual(vals[0], 1)


if __name__ == '__main__':
  tf.test.main()
