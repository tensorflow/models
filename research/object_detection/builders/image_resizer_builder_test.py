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
from google.protobuf import text_format
import numpy as np
import tensorflow.compat.v1 as tf
from object_detection.builders import image_resizer_builder
from object_detection.core import preprocessor
from object_detection.protos import image_resizer_pb2
from object_detection.utils import shape_utils
from object_detection.utils import test_case


class ImageResizerBuilderTest(test_case.TestCase):

  def _shape_of_resized_random_image_given_text_proto(self, input_shape,
                                                      text_proto):
    image_resizer_config = text_format.Parse(
        text_proto, image_resizer_pb2.ImageResizer()
    )
    image_resizer_fn = image_resizer_builder.build(image_resizer_config)
    def graph_fn():
      images = tf.cast(
          tf.random_uniform(input_shape, minval=0, maxval=255, dtype=tf.int32),
          dtype=tf.float32)
      resized_images, _ = image_resizer_fn(images)
      return resized_images
    return self.execute_cpu(graph_fn, []).shape

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

  def test_build_keep_aspect_ratio_resizer_grayscale(self):
    image_resizer_text_proto = """
      keep_aspect_ratio_resizer {
        min_dimension: 10
        max_dimension: 20
        convert_to_grayscale: true
      }
    """
    input_shape = (50, 25, 3)
    expected_output_shape = (20, 10, 1)
    output_shape = self._shape_of_resized_random_image_given_text_proto(
        input_shape, image_resizer_text_proto)
    self.assertEqual(output_shape, expected_output_shape)

  def test_build_keep_aspect_ratio_resizer_with_padding(self):
    image_resizer_text_proto = """
      keep_aspect_ratio_resizer {
        min_dimension: 10
        max_dimension: 20
        pad_to_max_dimension: true
        per_channel_pad_value: 3
        per_channel_pad_value: 4
        per_channel_pad_value: 5
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

  def test_built_fixed_shape_resizer_grayscale(self):
    image_resizer_text_proto = """
      fixed_shape_resizer {
        height: 10
        width: 20
        convert_to_grayscale: true
      }
    """
    input_shape = (50, 25, 3)
    expected_output_shape = (10, 20, 1)
    output_shape = self._shape_of_resized_random_image_given_text_proto(
        input_shape, image_resizer_text_proto)
    self.assertEqual(output_shape, expected_output_shape)

  def test_identity_resizer_returns_expected_shape(self):
    image_resizer_text_proto = """
      identity_resizer {
      }
    """
    input_shape = (10, 20, 3)
    expected_output_shape = (10, 20, 3)
    output_shape = self._shape_of_resized_random_image_given_text_proto(
        input_shape, image_resizer_text_proto)
    self.assertEqual(output_shape, expected_output_shape)

  def test_raises_error_on_invalid_input(self):
    invalid_input = 'invalid_input'
    with self.assertRaises(ValueError):
      image_resizer_builder.build(invalid_input)

  def _resized_image_given_text_proto(self, image, text_proto):
    image_resizer_config = text_format.Parse(
        text_proto, image_resizer_pb2.ImageResizer()
    )
    image_resizer_fn = image_resizer_builder.build(image_resizer_config)
    def graph_fn(image):
      resized_image, _ = image_resizer_fn(image)
      return resized_image
    return self.execute_cpu(graph_fn, [image])

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

  def test_build_conditional_shape_resizer_greater_returns_expected_shape(self):
    image_resizer_text_proto = """
      conditional_shape_resizer {
        condition: GREATER
        size_threshold: 30
      }
    """
    input_shape = (60, 30, 3)
    expected_output_shape = (30, 15, 3)
    output_shape = self._shape_of_resized_random_image_given_text_proto(
        input_shape, image_resizer_text_proto)
    self.assertEqual(output_shape, expected_output_shape)

  def test_build_conditional_shape_resizer_same_shape_with_no_resize(self):
    image_resizer_text_proto = """
      conditional_shape_resizer {
        condition: GREATER
        size_threshold: 30
      }
    """
    input_shape = (15, 15, 3)
    expected_output_shape = (15, 15, 3)
    output_shape = self._shape_of_resized_random_image_given_text_proto(
        input_shape, image_resizer_text_proto)
    self.assertEqual(output_shape, expected_output_shape)

  def test_build_conditional_shape_resizer_smaller_returns_expected_shape(self):
    image_resizer_text_proto = """
      conditional_shape_resizer {
        condition: SMALLER
        size_threshold: 30
      }
    """
    input_shape = (30, 15, 3)
    expected_output_shape = (60, 30, 3)
    output_shape = self._shape_of_resized_random_image_given_text_proto(
        input_shape, image_resizer_text_proto)
    self.assertEqual(output_shape, expected_output_shape)

  def test_build_conditional_shape_resizer_grayscale(self):
    image_resizer_text_proto = """
      conditional_shape_resizer {
        condition: GREATER
        size_threshold: 30
        convert_to_grayscale: true
      }
    """
    input_shape = (60, 30, 3)
    expected_output_shape = (30, 15, 1)
    output_shape = self._shape_of_resized_random_image_given_text_proto(
        input_shape, image_resizer_text_proto)
    self.assertEqual(output_shape, expected_output_shape)

  def test_build_conditional_shape_resizer_error_on_invalid_condition(self):
    invalid_image_resizer_text_proto = """
      conditional_shape_resizer {
        condition: INVALID
        size_threshold: 30
      }
    """
    with self.assertRaises(ValueError):
      image_resizer_builder.build(invalid_image_resizer_text_proto)

  def test_build_pad_to_multiple_resizer(self):
    """Test building a pad_to_multiple_resizer from proto."""
    image_resizer_text_proto = """
      pad_to_multiple_resizer {
        multiple: 32
      }
    """
    input_shape = (60, 30, 3)
    expected_output_shape = (64, 32, 3)
    output_shape = self._shape_of_resized_random_image_given_text_proto(
        input_shape, image_resizer_text_proto)
    self.assertEqual(output_shape, expected_output_shape)

  def test_build_pad_to_multiple_resizer_invalid_multiple(self):
    """Test that building a pad_to_multiple_resizer errors with invalid multiple."""

    image_resizer_text_proto = """
      pad_to_multiple_resizer {
        multiple: -10
      }
    """

    with self.assertRaises(ValueError):
      image_resizer_builder.build(image_resizer_text_proto)

  def _assert_batched_matches_unrolled(
      self,
      resize_method_str: str,
      *,
      convert_to_grayscale: bool = False,
      batch_size: int = 4,
  ) -> None:
    text_proto_template = """
      fixed_shape_resizer {{
        height: 30
        width: 40
        resize_method: {resize_method}
        convert_to_grayscale: {convert_to_grayscale}
        resize_batched: {resize_batched}
      }}
    """
    config_off = text_format.Parse(
        text_proto_template.format(
            resize_method=resize_method_str,
            convert_to_grayscale=str(convert_to_grayscale).lower(),
            resize_batched='false',
        ),
        image_resizer_pb2.ImageResizer(),
    )
    resizer_off = image_resizer_builder.build(config_off)

    config_on = text_format.Parse(
        text_proto_template.format(
            resize_method=resize_method_str,
            convert_to_grayscale=str(convert_to_grayscale).lower(),
            resize_batched='true',
        ),
        image_resizer_pb2.ImageResizer(),
    )
    resizer_on = image_resizer_builder.build(config_on)

    num_elements = batch_size * 60 * 80 * 3
    images_np = (
        np.arange(num_elements, dtype=np.float32).reshape(
            (batch_size, 60, 80, 3)
        )
        % 255.0
    )

    def compute_fn(images):
      out_off, shapes_off = shape_utils.resize_images_and_return_shapes(
          images, resizer_off
      )
      out_on, shapes_on = shape_utils.resize_images_and_return_shapes(
          images, resizer_on
      )
      return out_off, shapes_off, out_on, shapes_on

    out_off, shapes_off, out_on, shapes_on = self.execute(
        compute_fn, [images_np]
    )

    expected_channels = 1 if convert_to_grayscale else 3
    expected_shapes = np.array(
        [[30, 40, expected_channels]] * batch_size, dtype=np.int32
    )

    self.assertAllEqual(shapes_off, shapes_on)
    self.assertAllEqual(shapes_on, expected_shapes)
    self.assertAllEqual(out_off, out_on)

  def test_fixed_shape_resizer_batched_nearest_neighbor(self):
    self._assert_batched_matches_unrolled('NEAREST_NEIGHBOR')

  def test_fixed_shape_resizer_batched_bilinear(self):
    self._assert_batched_matches_unrolled('BILINEAR')

  def test_fixed_shape_resizer_batched_grayscale(self):
    self._assert_batched_matches_unrolled('BILINEAR', convert_to_grayscale=True)

  def test_fixed_shape_resizer_batched_batch_size_2(self):
    self._assert_batched_matches_unrolled('NEAREST_NEIGHBOR', batch_size=2)

  def test_fixed_shape_resizer_batched_rank_3_and_masks_fallback(self):
    for convert_to_grayscale in (False, True):
      text_proto_off = f"""
        fixed_shape_resizer {{
          height: 30
          width: 40
          resize_method: NEAREST_NEIGHBOR
          convert_to_grayscale: {str(convert_to_grayscale).lower()}
          resize_batched: false
        }}
      """
      resizer_off = image_resizer_builder.build(
          text_format.Parse(text_proto_off, image_resizer_pb2.ImageResizer())
      )

      text_proto_on = f"""
        fixed_shape_resizer {{
          height: 30
          width: 40
          resize_method: NEAREST_NEIGHBOR
          convert_to_grayscale: {str(convert_to_grayscale).lower()}
          resize_batched: true
        }}
      """
      resizer_on = image_resizer_builder.build(
          text_format.Parse(text_proto_on, image_resizer_pb2.ImageResizer())
      )

      image_np = (
          np.arange(60 * 80 * 3, dtype=np.float32).reshape((60, 80, 3)) % 255.0
      )
      masks_np = (
          np.arange(2 * 60 * 80, dtype=np.float32).reshape((2, 60, 80)) % 2.0
      )

      # Test positional call
      def positional_fn(img, msk, r_off=resizer_off, r_on=resizer_on):
        ret_off = r_off(img, masks=msk)
        ret_on = r_on(img, masks=msk)
        return ret_off + ret_on

      (
          off_img,
          off_msk,
          off_shape,
          on_img,
          on_msk,
          on_shape,
      ) = self.execute(positional_fn, [image_np, masks_np])
      self.assertAllEqual(off_img, on_img)
      self.assertAllEqual(off_msk, on_msk)
      self.assertAllEqual(off_shape, on_shape)

      # Test keyword call (image=..., masks=...)
      def keyword_fn(img, msk, r_off=resizer_off, r_on=resizer_on):
        ret_off = r_off(image=img, masks=msk)
        ret_on = r_on(image=img, masks=msk)
        return ret_off + ret_on

      (
          kw_off_img,
          kw_off_msk,
          kw_off_shape,
          kw_on_img,
          kw_on_msk,
          kw_on_shape,
      ) = self.execute(keyword_fn, [image_np, masks_np])
      self.assertAllEqual(kw_off_img, kw_on_img)
      self.assertAllEqual(kw_off_msk, kw_on_msk)
      self.assertAllEqual(kw_off_shape, kw_on_shape)

  def test_fixed_shape_resizer_batched_rank_4_direct_call(self):
    text_proto = """
      fixed_shape_resizer {
        height: 30
        width: 40
        resize_method: NEAREST_NEIGHBOR
        resize_batched: true
      }
    """
    resizer = image_resizer_builder.build(
        text_format.Parse(text_proto, image_resizer_pb2.ImageResizer())
    )
    images_np = (
        np.arange(2 * 60 * 80 * 3, dtype=np.float32).reshape((2, 60, 80, 3))
        % 255.0
    )

    def direct_fn(imgs):
      out_pos, shapes_pos = resizer(imgs)
      out_kw, shapes_kw = resizer(image=imgs)
      return out_pos, shapes_pos, out_kw, shapes_kw

    out_pos, shapes_pos, out_kw, shapes_kw = self.execute(
        direct_fn, [images_np]
    )
    self.assertAllEqual(out_pos, out_kw)
    self.assertAllEqual(shapes_pos, shapes_kw)

  def test_fixed_shape_resizer_batched_masks_raises_error(self):
    text_proto = """
      fixed_shape_resizer {
        height: 10
        width: 20
        resize_batched: true
      }
    """
    config = text_format.Parse(text_proto, image_resizer_pb2.ImageResizer())
    resizer_fn = image_resizer_builder.build(config)
    images = tf.zeros([2, 50, 50, 3], dtype=tf.float32)
    masks = tf.zeros([2, 50, 50], dtype=tf.float32)
    with self.assertRaisesRegex(ValueError, 'does not support masks'):
      resizer_fn(images, masks=masks)

  def test_fixed_shape_resizer_batched_rank_unknown_raises_error(self):
    text_proto = """
      fixed_shape_resizer {
        height: 10
        width: 20
        resize_batched: true
      }
    """
    config = text_format.Parse(text_proto, image_resizer_pb2.ImageResizer())
    resizer_fn = image_resizer_builder.build(config)
    with tf.Graph().as_default():
      unknown_rank_tensor = tf.compat.v1.placeholder(tf.float32, shape=None)
      with self.assertRaisesRegex(ValueError, 'statically known rank'):
        resizer_fn(unknown_rank_tensor)
      with self.assertRaisesRegex(ValueError, 'statically known rank'):
        preprocessor.resize_images_batched(unknown_rank_tensor)

  def test_resize_images_batched_rank_not_4_raises_error(self):
    images_rank_3 = tf.zeros([50, 50, 3], dtype=tf.float32)
    with self.assertRaisesRegex(ValueError, 'rank-4'):
      preprocessor.resize_images_batched(images_rank_3)

  def test_resize_images_batched_tensor_dimensions(self):
    images_np = np.ones((2, 40, 50, 3), dtype=np.float32)

    def dynamic_dim_fn(imgs):
      new_h = tf.constant(20, dtype=tf.int32)
      new_w = tf.constant(25, dtype=tf.int32)
      return preprocessor.resize_images_batched(
          imgs, new_height=new_h, new_width=new_w
      )

    resized, shapes = self.execute(dynamic_dim_fn, [images_np])
    self.assertEqual(resized.shape, (2, 20, 25, 3))
    self.assertAllEqual(shapes, [[20, 25, 3], [20, 25, 3]])

  def test_fixed_shape_resizer_batched_invalid_dimensions(self):
    text_proto_invalid_height = """
      fixed_shape_resizer {
        height: -5
        width: 20
        resize_batched: true
      }
    """
    config1 = text_format.Parse(
        text_proto_invalid_height, image_resizer_pb2.ImageResizer()
    )
    with self.assertRaisesRegex(ValueError, 'must be > 0'):
      image_resizer_builder.build(config1)

    text_proto_invalid_width = """
      fixed_shape_resizer {
        height: 10
        width: 0
        resize_batched: true
      }
    """
    config2 = text_format.Parse(
        text_proto_invalid_width, image_resizer_pb2.ImageResizer()
    )
    with self.assertRaisesRegex(ValueError, 'must be > 0'):
      image_resizer_builder.build(config2)

  def test_fixed_shape_resizer_unbatched_matches_per_image_resize_graph(self):
    text_proto_off = """
      fixed_shape_resizer {
        height: 30
        width: 40
        resize_method: NEAREST_NEIGHBOR
        resize_batched: false
      }
    """
    config_off = text_format.Parse(
        text_proto_off, image_resizer_pb2.ImageResizer()
    )
    resizer_fn_off = image_resizer_builder.build(config_off)

    def baseline_resizer_fn(elem):
      return preprocessor.resize_image(
          elem,
          new_height=30,
          new_width=40,
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      )

    g1 = tf.Graph()
    with g1.as_default():
      inputs1 = tf.placeholder(tf.float32, shape=[4, 60, 80, 3], name='inputs')
      shape_utils.resize_images_and_return_shapes(inputs1, resizer_fn_off)

    g2 = tf.Graph()
    with g2.as_default():
      inputs2 = tf.placeholder(tf.float32, shape=[4, 60, 80, 3], name='inputs')
      shape_utils.resize_images_and_return_shapes(inputs2, baseline_resizer_fn)

    nodes1 = [(n.name, n.op) for n in g1.as_graph_def().node]
    nodes2 = [(n.name, n.op) for n in g2.as_graph_def().node]
    self.assertEqual(nodes1, nodes2)


if __name__ == '__main__':
  tf.test.main()
