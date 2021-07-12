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

"""Tests for object_detection.utils.shape_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.utils import shape_utils
from object_detection.utils import test_case


class UtilTest(test_case.TestCase):

  def test_pad_tensor_using_integer_input(self):

    print('........pad tensor using interger input.')
    def graph_fn():
      t1 = tf.constant([1], dtype=tf.int32)
      pad_t1 = shape_utils.pad_tensor(t1, 2)
      t2 = tf.constant([[0.1, 0.2]], dtype=tf.float32)
      pad_t2 = shape_utils.pad_tensor(t2, 2)

      return pad_t1, pad_t2

    pad_t1_result, pad_t2_result = self.execute(graph_fn, [])

    self.assertAllEqual([1, 0], pad_t1_result)
    self.assertAllClose([[0.1, 0.2], [0, 0]], pad_t2_result)

  def test_pad_tensor_using_tensor_input(self):

    def graph_fn():
      t1 = tf.constant([1], dtype=tf.int32)
      pad_t1 = shape_utils.pad_tensor(t1, tf.constant(2))
      t2 = tf.constant([[0.1, 0.2]], dtype=tf.float32)
      pad_t2 = shape_utils.pad_tensor(t2, tf.constant(2))

      return pad_t1, pad_t2

    pad_t1_result, pad_t2_result = self.execute(graph_fn, [])
    self.assertAllEqual([1, 0], pad_t1_result)
    self.assertAllClose([[0.1, 0.2], [0, 0]], pad_t2_result)

  def test_clip_tensor_using_integer_input(self):

    def graph_fn():
      t1 = tf.constant([1, 2, 3], dtype=tf.int32)
      clip_t1 = shape_utils.clip_tensor(t1, 2)
      t2 = tf.constant([[0.1, 0.2], [0.2, 0.4], [0.5, 0.8]], dtype=tf.float32)
      clip_t2 = shape_utils.clip_tensor(t2, 2)

      self.assertEqual(2, clip_t1.get_shape()[0])
      self.assertEqual(2, clip_t2.get_shape()[0])

      return clip_t1, clip_t2

    clip_t1_result, clip_t2_result = self.execute(graph_fn, [])
    self.assertAllEqual([1, 2], clip_t1_result)
    self.assertAllClose([[0.1, 0.2], [0.2, 0.4]], clip_t2_result)

  def test_clip_tensor_using_tensor_input(self):

    def graph_fn():
      t1 = tf.constant([1, 2, 3], dtype=tf.int32)
      clip_t1 = shape_utils.clip_tensor(t1, tf.constant(2))
      t2 = tf.constant([[0.1, 0.2], [0.2, 0.4], [0.5, 0.8]], dtype=tf.float32)
      clip_t2 = shape_utils.clip_tensor(t2, tf.constant(2))

      return clip_t1, clip_t2

    clip_t1_result, clip_t2_result = self.execute(graph_fn, [])
    self.assertAllEqual([1, 2], clip_t1_result)
    self.assertAllClose([[0.1, 0.2], [0.2, 0.4]], clip_t2_result)

  def test_pad_or_clip_tensor_using_integer_input(self):

    def graph_fn():
      t1 = tf.constant([1], dtype=tf.int32)
      tt1 = shape_utils.pad_or_clip_tensor(t1, 2)
      t2 = tf.constant([[0.1, 0.2]], dtype=tf.float32)
      tt2 = shape_utils.pad_or_clip_tensor(t2, 2)

      t3 = tf.constant([1, 2, 3], dtype=tf.int32)
      tt3 = shape_utils.clip_tensor(t3, 2)
      t4 = tf.constant([[0.1, 0.2], [0.2, 0.4], [0.5, 0.8]], dtype=tf.float32)
      tt4 = shape_utils.clip_tensor(t4, 2)

      self.assertEqual(2, tt1.get_shape()[0])
      self.assertEqual(2, tt2.get_shape()[0])
      self.assertEqual(2, tt3.get_shape()[0])
      self.assertEqual(2, tt4.get_shape()[0])

      return tt1, tt2, tt3, tt4

    tt1_result, tt2_result, tt3_result, tt4_result = self.execute(graph_fn, [])
    self.assertAllEqual([1, 0], tt1_result)
    self.assertAllClose([[0.1, 0.2], [0, 0]], tt2_result)
    self.assertAllEqual([1, 2], tt3_result)
    self.assertAllClose([[0.1, 0.2], [0.2, 0.4]], tt4_result)

  def test_pad_or_clip_tensor_using_tensor_input(self):

    def graph_fn():
      t1 = tf.constant([1], dtype=tf.int32)
      tt1 = shape_utils.pad_or_clip_tensor(t1, tf.constant(2))
      t2 = tf.constant([[0.1, 0.2]], dtype=tf.float32)
      tt2 = shape_utils.pad_or_clip_tensor(t2, tf.constant(2))

      t3 = tf.constant([1, 2, 3], dtype=tf.int32)
      tt3 = shape_utils.clip_tensor(t3, tf.constant(2))
      t4 = tf.constant([[0.1, 0.2], [0.2, 0.4], [0.5, 0.8]], dtype=tf.float32)
      tt4 = shape_utils.clip_tensor(t4, tf.constant(2))

      return tt1, tt2, tt3, tt4

    tt1_result, tt2_result, tt3_result, tt4_result = self.execute(graph_fn, [])
    self.assertAllEqual([1, 0], tt1_result)
    self.assertAllClose([[0.1, 0.2], [0, 0]], tt2_result)
    self.assertAllEqual([1, 2], tt3_result)
    self.assertAllClose([[0.1, 0.2], [0.2, 0.4]], tt4_result)

  def test_combined_static_dynamic_shape(self):

    for n in [2, 3, 4]:
      tensor = tf.zeros((n, 2, 3))
      combined_shape = shape_utils.combined_static_and_dynamic_shape(
          tensor)
      self.assertListEqual(combined_shape[1:], [2, 3])

  def test_pad_or_clip_nd_tensor(self):

    def graph_fn(input_tensor):
      output_tensor = shape_utils.pad_or_clip_nd(
          input_tensor, [None, 3, 5, tf.constant(6)])

      return output_tensor

    for n in [2, 3, 4, 5]:
      input_np = np.zeros((n, 5, 4, 7))
      output_tensor_np = self.execute(graph_fn, [input_np])
      self.assertAllEqual(output_tensor_np.shape[1:], [3, 5, 6])


class StaticOrDynamicMapFnTest(test_case.TestCase):

  def test_with_dynamic_shape(self):

    def fn(input_tensor):
      return tf.reduce_sum(input_tensor)

    def graph_fn(input_tensor):
      return shape_utils.static_or_dynamic_map_fn(fn, input_tensor)

    # The input has different shapes, but due to how self.execute()
    # works, the shape is known at graph compile time.
    result1 = self.execute(
        graph_fn, [np.array([[1, 2], [3, 1], [0, 4]]),])
    result2 = self.execute(
        graph_fn, [np.array([[-1, 1], [0, 9]]),])
    self.assertAllEqual(result1, [3, 4, 4])
    self.assertAllEqual(result2, [0, 9])

  def test_with_static_shape(self):
    def fn(input_tensor):
      return tf.reduce_sum(input_tensor)

    def graph_fn():
      input_tensor = tf.constant([[1, 2], [3, 1], [0, 4]], dtype=tf.float32)
      return shape_utils.static_or_dynamic_map_fn(fn, input_tensor)

    result = self.execute(graph_fn, [])
    self.assertAllEqual(result, [3, 4, 4])

  def test_with_multiple_dynamic_shapes(self):
    def fn(elems):
      input_tensor, scalar_index_tensor = elems
      return tf.reshape(tf.slice(input_tensor, scalar_index_tensor, [1]), [])

    def graph_fn(input_tensor, scalar_index_tensor):
      map_fn_output = shape_utils.static_or_dynamic_map_fn(
          fn, [input_tensor, scalar_index_tensor], dtype=tf.float32)
      return map_fn_output

    # The input has different shapes, but due to how self.execute()
    # works, the shape is known at graph compile time.

    result1 = self.execute(
        graph_fn, [
            np.array([[1, 2, 3], [4, 5, -1], [0, 6, 9]]),
            np.array([[0], [2], [1]]),
        ])
    result2 = self.execute(
        graph_fn, [
            np.array([[-1, 1, 0], [3, 9, 30]]),
            np.array([[1], [0]])
        ])
    self.assertAllEqual(result1, [1, -1, 6])
    self.assertAllEqual(result2, [1, 3])

  def test_with_multiple_static_shapes(self):
    def fn(elems):
      input_tensor, scalar_index_tensor = elems
      return tf.reshape(tf.slice(input_tensor, scalar_index_tensor, [1]), [])

    def graph_fn():
      input_tensor = tf.constant([[1, 2, 3], [4, 5, -1], [0, 6, 9]],
                                 dtype=tf.float32)
      scalar_index_tensor = tf.constant([[0], [2], [1]], dtype=tf.int32)
      map_fn_output = shape_utils.static_or_dynamic_map_fn(
          fn, [input_tensor, scalar_index_tensor], dtype=tf.float32)
      return map_fn_output

    result = self.execute(graph_fn, [])
    self.assertAllEqual(result, [1, -1, 6])

  def test_fails_with_nested_input(self):
    def fn(input_tensor):
      return input_tensor
    input_tensor1 = tf.constant([1])
    input_tensor2 = tf.constant([2])
    with self.assertRaisesRegexp(
        ValueError, '`elems` must be a Tensor or list of Tensors.'):
      shape_utils.static_or_dynamic_map_fn(
          fn, [input_tensor1, [input_tensor2]], dtype=tf.float32)


class CheckMinImageShapeTest(test_case.TestCase):

  def test_check_min_image_dim_static_shape(self):
    input_tensor = tf.constant(np.zeros([1, 42, 42, 3]))
    _ = shape_utils.check_min_image_dim(33, input_tensor)

    with self.assertRaisesRegexp(
        ValueError, 'image size must be >= 64 in both height and width.'):
      _ = shape_utils.check_min_image_dim(64, input_tensor)

  def test_check_min_image_dim_dynamic_shape(self):

    def graph_fn(input_tensor):
      return shape_utils.check_min_image_dim(33, input_tensor)

    self.execute(graph_fn,
                 [np.zeros([1, 42, 42, 3])])
    self.assertRaises(
        ValueError, self.execute,
        graph_fn, np.zeros([1, 32, 32, 3])
    )


class AssertShapeEqualTest(test_case.TestCase):

  def test_unequal_static_shape_raises_exception(self):
    shape_a = tf.constant(np.zeros([4, 2, 2, 1]))
    shape_b = tf.constant(np.zeros([4, 2, 3, 1]))
    self.assertRaisesRegex(
        ValueError, 'Unequal shapes',
        shape_utils.assert_shape_equal,
        shape_utils.combined_static_and_dynamic_shape(shape_a),
        shape_utils.combined_static_and_dynamic_shape(shape_b)
    )

  def test_equal_static_shape_succeeds(self):

    def graph_fn():
      shape_a = tf.constant(np.zeros([4, 2, 2, 1]))
      shape_b = tf.constant(np.zeros([4, 2, 2, 1]))

      shape_utils.assert_shape_equal(
          shape_utils.combined_static_and_dynamic_shape(shape_a),
          shape_utils.combined_static_and_dynamic_shape(shape_b))

      return tf.constant(0)

    self.execute(graph_fn, [])

  def test_unequal_dynamic_shape_raises_tf_assert(self):

    def graph_fn(tensor_a, tensor_b):
      shape_utils.assert_shape_equal(
          shape_utils.combined_static_and_dynamic_shape(tensor_a),
          shape_utils.combined_static_and_dynamic_shape(tensor_b))
      return tf.constant(0)

    self.assertRaises(ValueError,
                      self.execute, graph_fn,
                      [np.zeros([1, 2, 2, 3]), np.zeros([1, 4, 4, 3])])

  def test_equal_dynamic_shape_succeeds(self):

    def graph_fn(tensor_a, tensor_b):
      shape_utils.assert_shape_equal(
          shape_utils.combined_static_and_dynamic_shape(tensor_a),
          shape_utils.combined_static_and_dynamic_shape(tensor_b)
      )

      return tf.constant(0)

    self.execute(graph_fn, [np.zeros([1, 2, 2, 3]),
                            np.zeros([1, 2, 2, 3])])

  def test_unequal_static_shape_along_first_dim_raises_exception(self):
    shape_a = tf.constant(np.zeros([4, 2, 2, 1]))
    shape_b = tf.constant(np.zeros([6, 2, 3, 1]))

    self.assertRaisesRegexp(
        ValueError, 'Unequal first dimension',
        shape_utils.assert_shape_equal_along_first_dimension,
        shape_utils.combined_static_and_dynamic_shape(shape_a),
        shape_utils.combined_static_and_dynamic_shape(shape_b)
    )

  def test_equal_static_shape_along_first_dim_succeeds(self):

    def graph_fn():
      shape_a = tf.constant(np.zeros([4, 2, 2, 1]))
      shape_b = tf.constant(np.zeros([4, 7, 2]))
      shape_utils.assert_shape_equal_along_first_dimension(
          shape_utils.combined_static_and_dynamic_shape(shape_a),
          shape_utils.combined_static_and_dynamic_shape(shape_b))
      return tf.constant(0)

    self.execute(graph_fn, [])

  def test_unequal_dynamic_shape_along_first_dim_raises_tf_assert(self):

    def graph_fn(tensor_a, tensor_b):
      shape_utils.assert_shape_equal_along_first_dimension(
          shape_utils.combined_static_and_dynamic_shape(tensor_a),
          shape_utils.combined_static_and_dynamic_shape(tensor_b))

      return tf.constant(0)

    self.assertRaises(ValueError,
                      self.execute, graph_fn,
                      [np.zeros([1, 2, 2, 3]), np.zeros([2, 4, 3])])

  def test_equal_dynamic_shape_along_first_dim_succeeds(self):

    def graph_fn(tensor_a, tensor_b):
      shape_utils.assert_shape_equal_along_first_dimension(
          shape_utils.combined_static_and_dynamic_shape(tensor_a),
          shape_utils.combined_static_and_dynamic_shape(tensor_b))
      return tf.constant(0)

    self.execute(graph_fn, [np.zeros([5, 2, 2, 3]), np.zeros([5])])


class FlattenExpandDimensionTest(test_case.TestCase):

  def test_flatten_given_dims(self):

    def graph_fn():
      inputs = tf.random_uniform([5, 2, 10, 10, 3])
      actual_flattened = shape_utils.flatten_dimensions(inputs, first=1, last=3)
      expected_flattened = tf.reshape(inputs, [5, 20, 10, 3])

      return actual_flattened, expected_flattened

    (actual_flattened_np,
     expected_flattened_np) = self.execute(graph_fn, [])
    self.assertAllClose(expected_flattened_np, actual_flattened_np)

  def test_raises_value_error_incorrect_dimensions(self):
    inputs = tf.random_uniform([5, 2, 10, 10, 3])
    self.assertRaises(ValueError,
                      shape_utils.flatten_dimensions, inputs,
                      first=0, last=6)

  def test_flatten_first_two_dimensions(self):

    def graph_fn():
      inputs = tf.constant(
          [
              [[1, 2], [3, 4]],
              [[5, 6], [7, 8]],
              [[9, 10], [11, 12]]
          ], dtype=tf.int32)
      flattened_tensor = shape_utils.flatten_first_n_dimensions(
          inputs, 2)
      return flattened_tensor

    flattened_tensor_out = self.execute(graph_fn, [])

    expected_output = [[1, 2],
                       [3, 4],
                       [5, 6],
                       [7, 8],
                       [9, 10],
                       [11, 12]]
    self.assertAllEqual(expected_output, flattened_tensor_out)

  def test_expand_first_dimension(self):

    def graph_fn():
      inputs = tf.constant(
          [
              [1, 2],
              [3, 4],
              [5, 6],
              [7, 8],
              [9, 10],
              [11, 12]
          ], dtype=tf.int32)
      dims = [3, 2]
      expanded_tensor = shape_utils.expand_first_dimension(
          inputs, dims)
      return expanded_tensor

    expanded_tensor_out = self.execute(graph_fn, [])

    expected_output = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]]]
    self.assertAllEqual(expected_output, expanded_tensor_out)

  def test_expand_first_dimension_with_incompatible_dims(self):

    def graph_fn():
      inputs = tf.constant(
          [
              [[1, 2]],
              [[3, 4]],
              [[5, 6]],
          ], dtype=tf.int32)
      dims = [3, 2]
      expanded_tensor = shape_utils.expand_first_dimension(
          inputs, dims)
      return expanded_tensor

    self.assertRaises(ValueError, self.execute, graph_fn, [])


if __name__ == '__main__':
  tf.test.main()
