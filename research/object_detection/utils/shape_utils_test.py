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

import numpy as np
import tensorflow as tf

from object_detection.utils import shape_utils


class UtilTest(tf.test.TestCase):

  def test_pad_tensor_using_integer_input(self):
    t1 = tf.constant([1], dtype=tf.int32)
    pad_t1 = shape_utils.pad_tensor(t1, 2)
    t2 = tf.constant([[0.1, 0.2]], dtype=tf.float32)
    pad_t2 = shape_utils.pad_tensor(t2, 2)

    self.assertEqual(2, pad_t1.get_shape()[0])
    self.assertEqual(2, pad_t2.get_shape()[0])

    with self.test_session() as sess:
      pad_t1_result, pad_t2_result = sess.run([pad_t1, pad_t2])
      self.assertAllEqual([1, 0], pad_t1_result)
      self.assertAllClose([[0.1, 0.2], [0, 0]], pad_t2_result)

  def test_pad_tensor_using_tensor_input(self):
    t1 = tf.constant([1], dtype=tf.int32)
    pad_t1 = shape_utils.pad_tensor(t1, tf.constant(2))
    t2 = tf.constant([[0.1, 0.2]], dtype=tf.float32)
    pad_t2 = shape_utils.pad_tensor(t2, tf.constant(2))

    with self.test_session() as sess:
      pad_t1_result, pad_t2_result = sess.run([pad_t1, pad_t2])
      self.assertAllEqual([1, 0], pad_t1_result)
      self.assertAllClose([[0.1, 0.2], [0, 0]], pad_t2_result)

  def test_clip_tensor_using_integer_input(self):
    t1 = tf.constant([1, 2, 3], dtype=tf.int32)
    clip_t1 = shape_utils.clip_tensor(t1, 2)
    t2 = tf.constant([[0.1, 0.2], [0.2, 0.4], [0.5, 0.8]], dtype=tf.float32)
    clip_t2 = shape_utils.clip_tensor(t2, 2)

    self.assertEqual(2, clip_t1.get_shape()[0])
    self.assertEqual(2, clip_t2.get_shape()[0])

    with self.test_session() as sess:
      clip_t1_result, clip_t2_result = sess.run([clip_t1, clip_t2])
      self.assertAllEqual([1, 2], clip_t1_result)
      self.assertAllClose([[0.1, 0.2], [0.2, 0.4]], clip_t2_result)

  def test_clip_tensor_using_tensor_input(self):
    t1 = tf.constant([1, 2, 3], dtype=tf.int32)
    clip_t1 = shape_utils.clip_tensor(t1, tf.constant(2))
    t2 = tf.constant([[0.1, 0.2], [0.2, 0.4], [0.5, 0.8]], dtype=tf.float32)
    clip_t2 = shape_utils.clip_tensor(t2, tf.constant(2))

    with self.test_session() as sess:
      clip_t1_result, clip_t2_result = sess.run([clip_t1, clip_t2])
      self.assertAllEqual([1, 2], clip_t1_result)
      self.assertAllClose([[0.1, 0.2], [0.2, 0.4]], clip_t2_result)

  def test_pad_or_clip_tensor_using_integer_input(self):
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

    with self.test_session() as sess:
      tt1_result, tt2_result, tt3_result, tt4_result = sess.run(
          [tt1, tt2, tt3, tt4])
      self.assertAllEqual([1, 0], tt1_result)
      self.assertAllClose([[0.1, 0.2], [0, 0]], tt2_result)
      self.assertAllEqual([1, 2], tt3_result)
      self.assertAllClose([[0.1, 0.2], [0.2, 0.4]], tt4_result)

  def test_pad_or_clip_tensor_using_tensor_input(self):
    t1 = tf.constant([1], dtype=tf.int32)
    tt1 = shape_utils.pad_or_clip_tensor(t1, tf.constant(2))
    t2 = tf.constant([[0.1, 0.2]], dtype=tf.float32)
    tt2 = shape_utils.pad_or_clip_tensor(t2, tf.constant(2))

    t3 = tf.constant([1, 2, 3], dtype=tf.int32)
    tt3 = shape_utils.clip_tensor(t3, tf.constant(2))
    t4 = tf.constant([[0.1, 0.2], [0.2, 0.4], [0.5, 0.8]], dtype=tf.float32)
    tt4 = shape_utils.clip_tensor(t4, tf.constant(2))

    with self.test_session() as sess:
      tt1_result, tt2_result, tt3_result, tt4_result = sess.run(
          [tt1, tt2, tt3, tt4])
      self.assertAllEqual([1, 0], tt1_result)
      self.assertAllClose([[0.1, 0.2], [0, 0]], tt2_result)
      self.assertAllEqual([1, 2], tt3_result)
      self.assertAllClose([[0.1, 0.2], [0.2, 0.4]], tt4_result)

  def test_combines_static_dynamic_shape(self):
    tensor = tf.placeholder(tf.float32, shape=(None, 2, 3))
    combined_shape = shape_utils.combined_static_and_dynamic_shape(
        tensor)
    self.assertTrue(tf.contrib.framework.is_tensor(combined_shape[0]))
    self.assertListEqual(combined_shape[1:], [2, 3])

  def test_pad_or_clip_nd_tensor(self):
    tensor_placeholder = tf.placeholder(tf.float32, [None, 5, 4, 7])
    output_tensor = shape_utils.pad_or_clip_nd(
        tensor_placeholder, [None, 3, 5, tf.constant(6)])

    self.assertAllEqual(output_tensor.shape.as_list(), [None, 3, 5, None])

    with self.test_session() as sess:
      output_tensor_np = sess.run(
          output_tensor,
          feed_dict={
              tensor_placeholder: np.random.rand(2, 5, 4, 7),
          })

    self.assertAllEqual(output_tensor_np.shape, [2, 3, 5, 6])


class StaticOrDynamicMapFnTest(tf.test.TestCase):

  def test_with_dynamic_shape(self):
    def fn(input_tensor):
      return tf.reduce_sum(input_tensor)
    input_tensor = tf.placeholder(tf.float32, shape=(None, 2))
    map_fn_output = shape_utils.static_or_dynamic_map_fn(fn, input_tensor)

    op_names = [op.name for op in tf.get_default_graph().get_operations()]
    self.assertTrue(any(['map' == op_name[:3] for op_name in op_names]))

    with self.test_session() as sess:
      result1 = sess.run(
          map_fn_output, feed_dict={
              input_tensor: [[1, 2], [3, 1], [0, 4]]})
      result2 = sess.run(
          map_fn_output, feed_dict={
              input_tensor: [[-1, 1], [0, 9]]})
      self.assertAllEqual(result1, [3, 4, 4])
      self.assertAllEqual(result2, [0, 9])

  def test_with_static_shape(self):
    def fn(input_tensor):
      return tf.reduce_sum(input_tensor)
    input_tensor = tf.constant([[1, 2], [3, 1], [0, 4]], dtype=tf.float32)
    map_fn_output = shape_utils.static_or_dynamic_map_fn(fn, input_tensor)

    op_names = [op.name for op in tf.get_default_graph().get_operations()]
    self.assertTrue(all(['map' != op_name[:3] for op_name in op_names]))

    with self.test_session() as sess:
      result = sess.run(map_fn_output)
      self.assertAllEqual(result, [3, 4, 4])

  def test_with_multiple_dynamic_shapes(self):
    def fn(elems):
      input_tensor, scalar_index_tensor = elems
      return tf.reshape(tf.slice(input_tensor, scalar_index_tensor, [1]), [])

    input_tensor = tf.placeholder(tf.float32, shape=(None, 3))
    scalar_index_tensor = tf.placeholder(tf.int32, shape=(None, 1))
    map_fn_output = shape_utils.static_or_dynamic_map_fn(
        fn, [input_tensor, scalar_index_tensor], dtype=tf.float32)

    op_names = [op.name for op in tf.get_default_graph().get_operations()]
    self.assertTrue(any(['map' == op_name[:3] for op_name in op_names]))

    with self.test_session() as sess:
      result1 = sess.run(
          map_fn_output, feed_dict={
              input_tensor: [[1, 2, 3], [4, 5, -1], [0, 6, 9]],
              scalar_index_tensor: [[0], [2], [1]],
          })
      result2 = sess.run(
          map_fn_output, feed_dict={
              input_tensor: [[-1, 1, 0], [3, 9, 30]],
              scalar_index_tensor: [[1], [0]]
          })
      self.assertAllEqual(result1, [1, -1, 6])
      self.assertAllEqual(result2, [1, 3])

  def test_with_multiple_static_shapes(self):
    def fn(elems):
      input_tensor, scalar_index_tensor = elems
      return tf.reshape(tf.slice(input_tensor, scalar_index_tensor, [1]), [])

    input_tensor = tf.constant([[1, 2, 3], [4, 5, -1], [0, 6, 9]],
                               dtype=tf.float32)
    scalar_index_tensor = tf.constant([[0], [2], [1]], dtype=tf.int32)
    map_fn_output = shape_utils.static_or_dynamic_map_fn(
        fn, [input_tensor, scalar_index_tensor], dtype=tf.float32)

    op_names = [op.name for op in tf.get_default_graph().get_operations()]
    self.assertTrue(all(['map' != op_name[:3] for op_name in op_names]))

    with self.test_session() as sess:
      result = sess.run(map_fn_output)
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


class CheckMinImageShapeTest(tf.test.TestCase):

  def test_check_min_image_dim_static_shape(self):
    input_tensor = tf.constant(np.zeros([1, 42, 42, 3]))
    _ = shape_utils.check_min_image_dim(33, input_tensor)

    with self.assertRaisesRegexp(
        ValueError, 'image size must be >= 64 in both height and width.'):
      _ = shape_utils.check_min_image_dim(64, input_tensor)

  def test_check_min_image_dim_dynamic_shape(self):
    input_placeholder = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    image_tensor = shape_utils.check_min_image_dim(33, input_placeholder)

    with self.test_session() as sess:
      sess.run(image_tensor,
               feed_dict={input_placeholder: np.zeros([1, 42, 42, 3])})
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(image_tensor,
                 feed_dict={input_placeholder: np.zeros([1, 32, 32, 3])})


class AssertShapeEqualTest(tf.test.TestCase):

  def test_unequal_static_shape_raises_exception(self):
    shape_a = tf.constant(np.zeros([4, 2, 2, 1]))
    shape_b = tf.constant(np.zeros([4, 2, 3, 1]))
    with self.assertRaisesRegexp(
        ValueError, 'Unequal shapes'):
      shape_utils.assert_shape_equal(
          shape_utils.combined_static_and_dynamic_shape(shape_a),
          shape_utils.combined_static_and_dynamic_shape(shape_b))

  def test_equal_static_shape_succeeds(self):
    shape_a = tf.constant(np.zeros([4, 2, 2, 1]))
    shape_b = tf.constant(np.zeros([4, 2, 2, 1]))
    with self.test_session() as sess:
      op = shape_utils.assert_shape_equal(
          shape_utils.combined_static_and_dynamic_shape(shape_a),
          shape_utils.combined_static_and_dynamic_shape(shape_b))
      sess.run(op)

  def test_unequal_dynamic_shape_raises_tf_assert(self):
    tensor_a = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    tensor_b = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    op = shape_utils.assert_shape_equal(
        shape_utils.combined_static_and_dynamic_shape(tensor_a),
        shape_utils.combined_static_and_dynamic_shape(tensor_b))
    with self.test_session() as sess:
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(op, feed_dict={tensor_a: np.zeros([1, 2, 2, 3]),
                                tensor_b: np.zeros([1, 4, 4, 3])})

  def test_equal_dynamic_shape_succeeds(self):
    tensor_a = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    tensor_b = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    op = shape_utils.assert_shape_equal(
        shape_utils.combined_static_and_dynamic_shape(tensor_a),
        shape_utils.combined_static_and_dynamic_shape(tensor_b))
    with self.test_session() as sess:
      sess.run(op, feed_dict={tensor_a: np.zeros([1, 2, 2, 3]),
                              tensor_b: np.zeros([1, 2, 2, 3])})

  def test_unequal_static_shape_along_first_dim_raises_exception(self):
    shape_a = tf.constant(np.zeros([4, 2, 2, 1]))
    shape_b = tf.constant(np.zeros([6, 2, 3, 1]))
    with self.assertRaisesRegexp(
        ValueError, 'Unequal first dimension'):
      shape_utils.assert_shape_equal_along_first_dimension(
          shape_utils.combined_static_and_dynamic_shape(shape_a),
          shape_utils.combined_static_and_dynamic_shape(shape_b))

  def test_equal_static_shape_along_first_dim_succeeds(self):
    shape_a = tf.constant(np.zeros([4, 2, 2, 1]))
    shape_b = tf.constant(np.zeros([4, 7, 2]))
    with self.test_session() as sess:
      op = shape_utils.assert_shape_equal_along_first_dimension(
          shape_utils.combined_static_and_dynamic_shape(shape_a),
          shape_utils.combined_static_and_dynamic_shape(shape_b))
      sess.run(op)

  def test_unequal_dynamic_shape_along_first_dim_raises_tf_assert(self):
    tensor_a = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    tensor_b = tf.placeholder(tf.float32, shape=[None, None, 3])
    op = shape_utils.assert_shape_equal_along_first_dimension(
        shape_utils.combined_static_and_dynamic_shape(tensor_a),
        shape_utils.combined_static_and_dynamic_shape(tensor_b))
    with self.test_session() as sess:
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(op, feed_dict={tensor_a: np.zeros([1, 2, 2, 3]),
                                tensor_b: np.zeros([2, 4, 3])})

  def test_equal_dynamic_shape_along_first_dim_succeeds(self):
    tensor_a = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    tensor_b = tf.placeholder(tf.float32, shape=[None])
    op = shape_utils.assert_shape_equal_along_first_dimension(
        shape_utils.combined_static_and_dynamic_shape(tensor_a),
        shape_utils.combined_static_and_dynamic_shape(tensor_b))
    with self.test_session() as sess:
      sess.run(op, feed_dict={tensor_a: np.zeros([5, 2, 2, 3]),
                              tensor_b: np.zeros([5])})


if __name__ == '__main__':
  tf.test.main()
