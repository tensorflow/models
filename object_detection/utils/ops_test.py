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

"""Tests for object_detection.utils.ops."""
import numpy as np
import tensorflow as tf

from object_detection.core import standard_fields as fields
from object_detection.utils import ops


class NormalizedToImageCoordinatesTest(tf.test.TestCase):

  def test_normalized_to_image_coordinates(self):
    normalized_boxes = tf.placeholder(tf.float32, shape=(None, 1, 4))
    normalized_boxes_np = np.array([[[0.0, 0.0, 1.0, 1.0]],
                                    [[0.5, 0.5, 1.0, 1.0]]])
    image_shape = tf.convert_to_tensor([1, 4, 4, 3], dtype=tf.int32)
    absolute_boxes = ops.normalized_to_image_coordinates(normalized_boxes,
                                                         image_shape,
                                                         parallel_iterations=2)

    expected_boxes = np.array([[[0, 0, 4, 4]],
                               [[2, 2, 4, 4]]])
    with self.test_session() as sess:
      absolute_boxes = sess.run(absolute_boxes,
                                feed_dict={normalized_boxes:
                                           normalized_boxes_np})

    self.assertAllEqual(absolute_boxes, expected_boxes)


class MeshgridTest(tf.test.TestCase):

  def test_meshgrid_numpy_comparison(self):
    """Tests meshgrid op with vectors, for which it should match numpy."""
    x = np.arange(4)
    y = np.arange(6)
    exp_xgrid, exp_ygrid = np.meshgrid(x, y)
    xgrid, ygrid = ops.meshgrid(x, y)
    with self.test_session() as sess:
      xgrid_output, ygrid_output = sess.run([xgrid, ygrid])
      self.assertAllEqual(xgrid_output, exp_xgrid)
      self.assertAllEqual(ygrid_output, exp_ygrid)

  def test_meshgrid_multidimensional(self):
    np.random.seed(18)
    x = np.random.rand(4, 1, 2).astype(np.float32)
    y = np.random.rand(2, 3).astype(np.float32)

    xgrid, ygrid = ops.meshgrid(x, y)

    grid_shape = list(y.shape) + list(x.shape)
    self.assertEqual(xgrid.get_shape().as_list(), grid_shape)
    self.assertEqual(ygrid.get_shape().as_list(), grid_shape)
    with self.test_session() as sess:
      xgrid_output, ygrid_output = sess.run([xgrid, ygrid])

    # Check the shape of the output grids
    self.assertEqual(xgrid_output.shape, tuple(grid_shape))
    self.assertEqual(ygrid_output.shape, tuple(grid_shape))

    # Check a few elements
    test_elements = [((3, 0, 0), (1, 2)),
                     ((2, 0, 1), (0, 0)),
                     ((0, 0, 0), (1, 1))]
    for xind, yind in test_elements:
      # These are float equality tests, but the meshgrid op should not introduce
      # rounding.
      self.assertEqual(xgrid_output[yind + xind], x[xind])
      self.assertEqual(ygrid_output[yind + xind], y[yind])


class OpsTestPadToMultiple(tf.test.TestCase):

  def test_zero_padding(self):
    tensor = tf.constant([[[[0.], [0.]], [[0.], [0.]]]])
    padded_tensor = ops.pad_to_multiple(tensor, 1)
    with self.test_session() as sess:
      padded_tensor_out = sess.run(padded_tensor)
    self.assertEqual((1, 2, 2, 1), padded_tensor_out.shape)

  def test_no_padding(self):
    tensor = tf.constant([[[[0.], [0.]], [[0.], [0.]]]])
    padded_tensor = ops.pad_to_multiple(tensor, 2)
    with self.test_session() as sess:
      padded_tensor_out = sess.run(padded_tensor)
    self.assertEqual((1, 2, 2, 1), padded_tensor_out.shape)

  def test_padding(self):
    tensor = tf.constant([[[[0.], [0.]], [[0.], [0.]]]])
    padded_tensor = ops.pad_to_multiple(tensor, 4)
    with self.test_session() as sess:
      padded_tensor_out = sess.run(padded_tensor)
    self.assertEqual((1, 4, 4, 1), padded_tensor_out.shape)


class OpsTestPaddedOneHotEncoding(tf.test.TestCase):

  def test_correct_one_hot_tensor_with_no_pad(self):
    indices = tf.constant([1, 2, 3, 5])
    one_hot_tensor = ops.padded_one_hot_encoding(indices, depth=6, left_pad=0)
    expected_tensor = np.array([[0, 1, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 1]], np.float32)
    with self.test_session() as sess:
      out_one_hot_tensor = sess.run(one_hot_tensor)
      self.assertAllClose(out_one_hot_tensor, expected_tensor, rtol=1e-10,
                          atol=1e-10)

  def test_correct_one_hot_tensor_with_pad_one(self):
    indices = tf.constant([1, 2, 3, 5])
    one_hot_tensor = ops.padded_one_hot_encoding(indices, depth=6, left_pad=1)
    expected_tensor = np.array([[0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1]], np.float32)
    with self.test_session() as sess:
      out_one_hot_tensor = sess.run(one_hot_tensor)
      self.assertAllClose(out_one_hot_tensor, expected_tensor, rtol=1e-10,
                          atol=1e-10)

  def test_correct_one_hot_tensor_with_pad_three(self):
    indices = tf.constant([1, 2, 3, 5])
    one_hot_tensor = ops.padded_one_hot_encoding(indices, depth=6, left_pad=3)
    expected_tensor = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 1]], np.float32)
    with self.test_session() as sess:
      out_one_hot_tensor = sess.run(one_hot_tensor)
      self.assertAllClose(out_one_hot_tensor, expected_tensor, rtol=1e-10,
                          atol=1e-10)

  def test_correct_padded_one_hot_tensor_with_empty_indices(self):
    depth = 6
    pad = 2
    indices = tf.constant([])
    one_hot_tensor = ops.padded_one_hot_encoding(
        indices, depth=depth, left_pad=pad)
    expected_tensor = np.zeros((0, depth + pad))
    with self.test_session() as sess:
      out_one_hot_tensor = sess.run(one_hot_tensor)
      self.assertAllClose(out_one_hot_tensor, expected_tensor, rtol=1e-10,
                          atol=1e-10)

  def test_return_none_on_zero_depth(self):
    indices = tf.constant([1, 2, 3, 4, 5])
    one_hot_tensor = ops.padded_one_hot_encoding(indices, depth=0, left_pad=2)
    self.assertEqual(one_hot_tensor, None)

  def test_raise_value_error_on_rank_two_input(self):
    indices = tf.constant(1.0, shape=(2, 3))
    with self.assertRaises(ValueError):
      ops.padded_one_hot_encoding(indices, depth=6, left_pad=2)

  def test_raise_value_error_on_negative_pad(self):
    indices = tf.constant(1.0, shape=(2, 3))
    with self.assertRaises(ValueError):
      ops.padded_one_hot_encoding(indices, depth=6, left_pad=-1)

  def test_raise_value_error_on_float_pad(self):
    indices = tf.constant(1.0, shape=(2, 3))
    with self.assertRaises(ValueError):
      ops.padded_one_hot_encoding(indices, depth=6, left_pad=0.1)

  def test_raise_value_error_on_float_depth(self):
    indices = tf.constant(1.0, shape=(2, 3))
    with self.assertRaises(ValueError):
      ops.padded_one_hot_encoding(indices, depth=0.1, left_pad=2)


class OpsDenseToSparseBoxesTest(tf.test.TestCase):

  def test_return_all_boxes_when_all_input_boxes_are_valid(self):
    num_classes = 4
    num_valid_boxes = 3
    code_size = 4
    dense_location_placeholder = tf.placeholder(tf.float32,
                                                shape=(num_valid_boxes,
                                                       code_size))
    dense_num_boxes_placeholder = tf.placeholder(tf.int32, shape=(num_classes))
    box_locations, box_classes = ops.dense_to_sparse_boxes(
        dense_location_placeholder, dense_num_boxes_placeholder, num_classes)
    feed_dict = {dense_location_placeholder: np.random.uniform(
        size=[num_valid_boxes, code_size]),
                 dense_num_boxes_placeholder: np.array([1, 0, 0, 2],
                                                       dtype=np.int32)}

    expected_box_locations = feed_dict[dense_location_placeholder]
    expected_box_classses = np.array([0, 3, 3])
    with self.test_session() as sess:
      box_locations, box_classes = sess.run([box_locations, box_classes],
                                            feed_dict=feed_dict)

    self.assertAllClose(box_locations, expected_box_locations, rtol=1e-6,
                        atol=1e-6)
    self.assertAllEqual(box_classes, expected_box_classses)

  def test_return_only_valid_boxes_when_input_contains_invalid_boxes(self):
    num_classes = 4
    num_valid_boxes = 3
    num_boxes = 10
    code_size = 4

    dense_location_placeholder = tf.placeholder(tf.float32, shape=(num_boxes,
                                                                   code_size))
    dense_num_boxes_placeholder = tf.placeholder(tf.int32, shape=(num_classes))
    box_locations, box_classes = ops.dense_to_sparse_boxes(
        dense_location_placeholder, dense_num_boxes_placeholder, num_classes)
    feed_dict = {dense_location_placeholder: np.random.uniform(
        size=[num_boxes, code_size]),
                 dense_num_boxes_placeholder: np.array([1, 0, 0, 2],
                                                       dtype=np.int32)}

    expected_box_locations = (feed_dict[dense_location_placeholder]
                              [:num_valid_boxes])
    expected_box_classses = np.array([0, 3, 3])
    with self.test_session() as sess:
      box_locations, box_classes = sess.run([box_locations, box_classes],
                                            feed_dict=feed_dict)

    self.assertAllClose(box_locations, expected_box_locations, rtol=1e-6,
                        atol=1e-6)
    self.assertAllEqual(box_classes, expected_box_classses)


class OpsTestIndicesToDenseVector(tf.test.TestCase):

  def test_indices_to_dense_vector(self):
    size = 10000
    num_indices = np.random.randint(size)
    rand_indices = np.random.permutation(np.arange(size))[0:num_indices]

    expected_output = np.zeros(size, dtype=np.float32)
    expected_output[rand_indices] = 1.

    tf_rand_indices = tf.constant(rand_indices)
    indicator = ops.indices_to_dense_vector(tf_rand_indices, size)

    with self.test_session() as sess:
      output = sess.run(indicator)
      self.assertAllEqual(output, expected_output)
      self.assertEqual(output.dtype, expected_output.dtype)

  def test_indices_to_dense_vector_size_at_inference(self):
    size = 5000
    num_indices = 250
    all_indices = np.arange(size)
    rand_indices = np.random.permutation(all_indices)[0:num_indices]

    expected_output = np.zeros(size, dtype=np.float32)
    expected_output[rand_indices] = 1.

    tf_all_indices = tf.placeholder(tf.int32)
    tf_rand_indices = tf.constant(rand_indices)
    indicator = ops.indices_to_dense_vector(tf_rand_indices,
                                            tf.shape(tf_all_indices)[0])
    feed_dict = {tf_all_indices: all_indices}

    with self.test_session() as sess:
      output = sess.run(indicator, feed_dict=feed_dict)
      self.assertAllEqual(output, expected_output)
      self.assertEqual(output.dtype, expected_output.dtype)

  def test_indices_to_dense_vector_int(self):
    size = 500
    num_indices = 25
    rand_indices = np.random.permutation(np.arange(size))[0:num_indices]

    expected_output = np.zeros(size, dtype=np.int64)
    expected_output[rand_indices] = 1

    tf_rand_indices = tf.constant(rand_indices)
    indicator = ops.indices_to_dense_vector(
        tf_rand_indices, size, 1, dtype=tf.int64)

    with self.test_session() as sess:
      output = sess.run(indicator)
      self.assertAllEqual(output, expected_output)
      self.assertEqual(output.dtype, expected_output.dtype)

  def test_indices_to_dense_vector_custom_values(self):
    size = 100
    num_indices = 10
    rand_indices = np.random.permutation(np.arange(size))[0:num_indices]
    indices_value = np.random.rand(1)
    default_value = np.random.rand(1)

    expected_output = np.float32(np.ones(size) * default_value)
    expected_output[rand_indices] = indices_value

    tf_rand_indices = tf.constant(rand_indices)
    indicator = ops.indices_to_dense_vector(
        tf_rand_indices,
        size,
        indices_value=indices_value,
        default_value=default_value)

    with self.test_session() as sess:
      output = sess.run(indicator)
      self.assertAllClose(output, expected_output)
      self.assertEqual(output.dtype, expected_output.dtype)

  def test_indices_to_dense_vector_all_indices_as_input(self):
    size = 500
    num_indices = 500
    rand_indices = np.random.permutation(np.arange(size))[0:num_indices]

    expected_output = np.ones(size, dtype=np.float32)

    tf_rand_indices = tf.constant(rand_indices)
    indicator = ops.indices_to_dense_vector(tf_rand_indices, size)

    with self.test_session() as sess:
      output = sess.run(indicator)
      self.assertAllEqual(output, expected_output)
      self.assertEqual(output.dtype, expected_output.dtype)

  def test_indices_to_dense_vector_empty_indices_as_input(self):
    size = 500
    rand_indices = []

    expected_output = np.zeros(size, dtype=np.float32)

    tf_rand_indices = tf.constant(rand_indices)
    indicator = ops.indices_to_dense_vector(tf_rand_indices, size)

    with self.test_session() as sess:
      output = sess.run(indicator)
      self.assertAllEqual(output, expected_output)
      self.assertEqual(output.dtype, expected_output.dtype)


class GroundtruthFilterTest(tf.test.TestCase):

  def test_filter_groundtruth(self):
    input_image = tf.placeholder(tf.float32, shape=(None, None, 3))
    input_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    input_classes = tf.placeholder(tf.int32, shape=(None,))
    input_is_crowd = tf.placeholder(tf.bool, shape=(None,))
    input_area = tf.placeholder(tf.float32, shape=(None,))
    input_difficult = tf.placeholder(tf.float32, shape=(None,))
    input_label_types = tf.placeholder(tf.string, shape=(None,))
    valid_indices = tf.placeholder(tf.int32, shape=(None,))
    input_tensors = {
        fields.InputDataFields.image: input_image,
        fields.InputDataFields.groundtruth_boxes: input_boxes,
        fields.InputDataFields.groundtruth_classes: input_classes,
        fields.InputDataFields.groundtruth_is_crowd: input_is_crowd,
        fields.InputDataFields.groundtruth_area: input_area,
        fields.InputDataFields.groundtruth_difficult: input_difficult,
        fields.InputDataFields.groundtruth_label_types: input_label_types
    }
    output_tensors = ops.retain_groundtruth(input_tensors, valid_indices)

    image_tensor = np.random.rand(224, 224, 3)
    feed_dict = {
        input_image: image_tensor,
        input_boxes:
        np.array([[0.2, 0.4, 0.1, 0.8], [0.2, 0.4, 1.0, 0.8]], dtype=np.float),
        input_classes:
        np.array([1, 2], dtype=np.int32),
        input_is_crowd:
        np.array([False, True], dtype=np.bool),
        input_area:
        np.array([32, 48], dtype=np.float32),
        input_difficult:
        np.array([True, False], dtype=np.bool),
        input_label_types:
        np.array(['APPROPRIATE', 'INCORRECT'], dtype=np.string_),
        valid_indices:
        np.array([0], dtype=np.int32)
    }
    expected_tensors = {
        fields.InputDataFields.image:
        image_tensor,
        fields.InputDataFields.groundtruth_boxes:
        [[0.2, 0.4, 0.1, 0.8]],
        fields.InputDataFields.groundtruth_classes:
        [1],
        fields.InputDataFields.groundtruth_is_crowd:
        [False],
        fields.InputDataFields.groundtruth_area:
        [32],
        fields.InputDataFields.groundtruth_difficult:
        [True],
        fields.InputDataFields.groundtruth_label_types:
        ['APPROPRIATE']
    }
    with self.test_session() as sess:
      output_tensors = sess.run(output_tensors, feed_dict=feed_dict)
      for key in [fields.InputDataFields.image,
                  fields.InputDataFields.groundtruth_boxes,
                  fields.InputDataFields.groundtruth_area]:
        self.assertAllClose(expected_tensors[key], output_tensors[key])
      for key in [fields.InputDataFields.groundtruth_classes,
                  fields.InputDataFields.groundtruth_is_crowd,
                  fields.InputDataFields.groundtruth_label_types]:
        self.assertAllEqual(expected_tensors[key], output_tensors[key])

  def test_filter_with_missing_fields(self):
    input_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    input_classes = tf.placeholder(tf.int32, shape=(None,))
    input_tensors = {
        fields.InputDataFields.groundtruth_boxes: input_boxes,
        fields.InputDataFields.groundtruth_classes: input_classes
    }
    valid_indices = tf.placeholder(tf.int32, shape=(None,))

    feed_dict = {
        input_boxes:
        np.array([[0.2, 0.4, 0.1, 0.8], [0.2, 0.4, 1.0, 0.8]], dtype=np.float),
        input_classes:
        np.array([1, 2], dtype=np.int32),
        valid_indices:
        np.array([0], dtype=np.int32)
    }
    expected_tensors = {
        fields.InputDataFields.groundtruth_boxes:
        [[0.2, 0.4, 0.1, 0.8]],
        fields.InputDataFields.groundtruth_classes:
        [1]
    }

    output_tensors = ops.retain_groundtruth(input_tensors, valid_indices)
    with self.test_session() as sess:
      output_tensors = sess.run(output_tensors, feed_dict=feed_dict)
      for key in [fields.InputDataFields.groundtruth_boxes]:
        self.assertAllClose(expected_tensors[key], output_tensors[key])
      for key in [fields.InputDataFields.groundtruth_classes]:
        self.assertAllEqual(expected_tensors[key], output_tensors[key])

  def test_filter_with_empty_fields(self):
    input_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    input_classes = tf.placeholder(tf.int32, shape=(None,))
    input_is_crowd = tf.placeholder(tf.bool, shape=(None,))
    input_area = tf.placeholder(tf.float32, shape=(None,))
    input_difficult = tf.placeholder(tf.float32, shape=(None,))
    valid_indices = tf.placeholder(tf.int32, shape=(None,))
    input_tensors = {
        fields.InputDataFields.groundtruth_boxes: input_boxes,
        fields.InputDataFields.groundtruth_classes: input_classes,
        fields.InputDataFields.groundtruth_is_crowd: input_is_crowd,
        fields.InputDataFields.groundtruth_area: input_area,
        fields.InputDataFields.groundtruth_difficult: input_difficult
    }
    output_tensors = ops.retain_groundtruth(input_tensors, valid_indices)

    feed_dict = {
        input_boxes:
        np.array([[0.2, 0.4, 0.1, 0.8], [0.2, 0.4, 1.0, 0.8]], dtype=np.float),
        input_classes:
        np.array([1, 2], dtype=np.int32),
        input_is_crowd:
        np.array([False, True], dtype=np.bool),
        input_area:
        np.array([], dtype=np.float32),
        input_difficult:
        np.array([], dtype=np.float32),
        valid_indices:
        np.array([0], dtype=np.int32)
    }
    expected_tensors = {
        fields.InputDataFields.groundtruth_boxes:
        [[0.2, 0.4, 0.1, 0.8]],
        fields.InputDataFields.groundtruth_classes:
        [1],
        fields.InputDataFields.groundtruth_is_crowd:
        [False],
        fields.InputDataFields.groundtruth_area:
        [],
        fields.InputDataFields.groundtruth_difficult:
        []
    }
    with self.test_session() as sess:
      output_tensors = sess.run(output_tensors, feed_dict=feed_dict)
      for key in [fields.InputDataFields.groundtruth_boxes,
                  fields.InputDataFields.groundtruth_area]:
        self.assertAllClose(expected_tensors[key], output_tensors[key])
      for key in [fields.InputDataFields.groundtruth_classes,
                  fields.InputDataFields.groundtruth_is_crowd]:
        self.assertAllEqual(expected_tensors[key], output_tensors[key])

  def test_filter_with_empty_groundtruth_boxes(self):
    input_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    input_classes = tf.placeholder(tf.int32, shape=(None,))
    input_is_crowd = tf.placeholder(tf.bool, shape=(None,))
    input_area = tf.placeholder(tf.float32, shape=(None,))
    input_difficult = tf.placeholder(tf.float32, shape=(None,))
    valid_indices = tf.placeholder(tf.int32, shape=(None,))
    input_tensors = {
        fields.InputDataFields.groundtruth_boxes: input_boxes,
        fields.InputDataFields.groundtruth_classes: input_classes,
        fields.InputDataFields.groundtruth_is_crowd: input_is_crowd,
        fields.InputDataFields.groundtruth_area: input_area,
        fields.InputDataFields.groundtruth_difficult: input_difficult
    }
    output_tensors = ops.retain_groundtruth(input_tensors, valid_indices)

    feed_dict = {
        input_boxes:
        np.array([], dtype=np.float).reshape(0, 4),
        input_classes:
        np.array([], dtype=np.int32),
        input_is_crowd:
        np.array([], dtype=np.bool),
        input_area:
        np.array([], dtype=np.float32),
        input_difficult:
        np.array([], dtype=np.float32),
        valid_indices:
        np.array([], dtype=np.int32)
    }
    with self.test_session() as sess:
      output_tensors = sess.run(output_tensors, feed_dict=feed_dict)
      for key in input_tensors:
        if key == fields.InputDataFields.groundtruth_boxes:
          self.assertAllEqual([0, 4], output_tensors[key].shape)
        else:
          self.assertAllEqual([0], output_tensors[key].shape)


class RetainGroundTruthWithPositiveClasses(tf.test.TestCase):

  def test_filter_groundtruth_with_positive_classes(self):
    input_image = tf.placeholder(tf.float32, shape=(None, None, 3))
    input_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    input_classes = tf.placeholder(tf.int32, shape=(None,))
    input_is_crowd = tf.placeholder(tf.bool, shape=(None,))
    input_area = tf.placeholder(tf.float32, shape=(None,))
    input_difficult = tf.placeholder(tf.float32, shape=(None,))
    input_label_types = tf.placeholder(tf.string, shape=(None,))
    valid_indices = tf.placeholder(tf.int32, shape=(None,))
    input_tensors = {
        fields.InputDataFields.image: input_image,
        fields.InputDataFields.groundtruth_boxes: input_boxes,
        fields.InputDataFields.groundtruth_classes: input_classes,
        fields.InputDataFields.groundtruth_is_crowd: input_is_crowd,
        fields.InputDataFields.groundtruth_area: input_area,
        fields.InputDataFields.groundtruth_difficult: input_difficult,
        fields.InputDataFields.groundtruth_label_types: input_label_types
    }
    output_tensors = ops.retain_groundtruth_with_positive_classes(input_tensors)

    image_tensor = np.random.rand(224, 224, 3)
    feed_dict = {
        input_image: image_tensor,
        input_boxes:
        np.array([[0.2, 0.4, 0.1, 0.8], [0.2, 0.4, 1.0, 0.8]], dtype=np.float),
        input_classes:
        np.array([1, 0], dtype=np.int32),
        input_is_crowd:
        np.array([False, True], dtype=np.bool),
        input_area:
        np.array([32, 48], dtype=np.float32),
        input_difficult:
        np.array([True, False], dtype=np.bool),
        input_label_types:
        np.array(['APPROPRIATE', 'INCORRECT'], dtype=np.string_),
        valid_indices:
        np.array([0], dtype=np.int32)
    }
    expected_tensors = {
        fields.InputDataFields.image:
        image_tensor,
        fields.InputDataFields.groundtruth_boxes:
        [[0.2, 0.4, 0.1, 0.8]],
        fields.InputDataFields.groundtruth_classes:
        [1],
        fields.InputDataFields.groundtruth_is_crowd:
        [False],
        fields.InputDataFields.groundtruth_area:
        [32],
        fields.InputDataFields.groundtruth_difficult:
        [True],
        fields.InputDataFields.groundtruth_label_types:
        ['APPROPRIATE']
    }
    with self.test_session() as sess:
      output_tensors = sess.run(output_tensors, feed_dict=feed_dict)
      for key in [fields.InputDataFields.image,
                  fields.InputDataFields.groundtruth_boxes,
                  fields.InputDataFields.groundtruth_area]:
        self.assertAllClose(expected_tensors[key], output_tensors[key])
      for key in [fields.InputDataFields.groundtruth_classes,
                  fields.InputDataFields.groundtruth_is_crowd,
                  fields.InputDataFields.groundtruth_label_types]:
        self.assertAllEqual(expected_tensors[key], output_tensors[key])


class GroundtruthFilterWithNanBoxTest(tf.test.TestCase):

  def test_filter_groundtruth_with_nan_box_coordinates(self):
    input_tensors = {
        fields.InputDataFields.groundtruth_boxes:
        [[np.nan, np.nan, np.nan, np.nan], [0.2, 0.4, 0.1, 0.8]],
        fields.InputDataFields.groundtruth_classes:
        [1, 2],
        fields.InputDataFields.groundtruth_is_crowd:
        [False, True],
        fields.InputDataFields.groundtruth_area:
        [100.0, 238.7]
    }

    expected_tensors = {
        fields.InputDataFields.groundtruth_boxes:
        [[0.2, 0.4, 0.1, 0.8]],
        fields.InputDataFields.groundtruth_classes:
        [2],
        fields.InputDataFields.groundtruth_is_crowd:
        [True],
        fields.InputDataFields.groundtruth_area:
        [238.7]
    }

    output_tensors = ops.filter_groundtruth_with_nan_box_coordinates(
        input_tensors)
    with self.test_session() as sess:
      output_tensors = sess.run(output_tensors)
      for key in [fields.InputDataFields.groundtruth_boxes,
                  fields.InputDataFields.groundtruth_area]:
        self.assertAllClose(expected_tensors[key], output_tensors[key])
      for key in [fields.InputDataFields.groundtruth_classes,
                  fields.InputDataFields.groundtruth_is_crowd]:
        self.assertAllEqual(expected_tensors[key], output_tensors[key])


class OpsTestNormalizeToTarget(tf.test.TestCase):

  def test_create_normalize_to_target(self):
    inputs = tf.random_uniform([5, 10, 12, 3])
    target_norm_value = 4.0
    dim = 3
    with self.test_session():
      output = ops.normalize_to_target(inputs, target_norm_value, dim)
      self.assertEqual(output.op.name, 'NormalizeToTarget/mul')
      var_name = tf.contrib.framework.get_variables()[0].name
      self.assertEqual(var_name, 'NormalizeToTarget/weights:0')

  def test_invalid_dim(self):
    inputs = tf.random_uniform([5, 10, 12, 3])
    target_norm_value = 4.0
    dim = 10
    with self.assertRaisesRegexp(
        ValueError,
        'dim must be non-negative but smaller than the input rank.'):
      ops.normalize_to_target(inputs, target_norm_value, dim)

  def test_invalid_target_norm_values(self):
    inputs = tf.random_uniform([5, 10, 12, 3])
    target_norm_value = [4.0, 4.0]
    dim = 3
    with self.assertRaisesRegexp(
        ValueError, 'target_norm_value must be a float or a list of floats'):
      ops.normalize_to_target(inputs, target_norm_value, dim)

  def test_correct_output_shape(self):
    inputs = tf.random_uniform([5, 10, 12, 3])
    target_norm_value = 4.0
    dim = 3
    with self.test_session():
      output = ops.normalize_to_target(inputs, target_norm_value, dim)
      self.assertEqual(output.get_shape().as_list(),
                       inputs.get_shape().as_list())

  def test_correct_initial_output_values(self):
    inputs = tf.constant([[[[3, 4], [7, 24]],
                           [[5, -12], [-1, 0]]]], tf.float32)
    target_norm_value = 10.0
    dim = 3
    expected_output = [[[[30/5.0, 40/5.0], [70/25.0, 240/25.0]],
                        [[50/13.0, -120/13.0], [-10, 0]]]]
    with self.test_session() as sess:
      normalized_inputs = ops.normalize_to_target(inputs, target_norm_value,
                                                  dim)
      sess.run(tf.global_variables_initializer())
      output = normalized_inputs.eval()
      self.assertAllClose(output, expected_output)

  def test_multiple_target_norm_values(self):
    inputs = tf.constant([[[[3, 4], [7, 24]],
                           [[5, -12], [-1, 0]]]], tf.float32)
    target_norm_value = [10.0, 20.0]
    dim = 3
    expected_output = [[[[30/5.0, 80/5.0], [70/25.0, 480/25.0]],
                        [[50/13.0, -240/13.0], [-10, 0]]]]
    with self.test_session() as sess:
      normalized_inputs = ops.normalize_to_target(inputs, target_norm_value,
                                                  dim)
      sess.run(tf.global_variables_initializer())
      output = normalized_inputs.eval()
      self.assertAllClose(output, expected_output)


class OpsTestPositionSensitiveCropRegions(tf.test.TestCase):

  def test_position_sensitive(self):
    num_spatial_bins = [3, 2]
    image_shape = [1, 3, 2, 6]

    # First channel is 1's, second channel is 2's, etc.
    image = tf.constant(range(1, 3 * 2 + 1) * 6, dtype=tf.float32,
                        shape=image_shape)
    boxes = tf.random_uniform((2, 4))
    box_ind = tf.constant([0, 0], dtype=tf.int32)

    # The result for both boxes should be [[1, 2], [3, 4], [5, 6]]
    # before averaging.
    expected_output = np.array([3.5, 3.5]).reshape([2, 1, 1, 1])

    for crop_size_mult in range(1, 3):
      crop_size = [3 * crop_size_mult, 2 * crop_size_mult]
      ps_crop_and_pool = ops.position_sensitive_crop_regions(
          image, boxes, box_ind, crop_size, num_spatial_bins, global_pool=True)

      with self.test_session() as sess:
        output = sess.run(ps_crop_and_pool)
        self.assertAllClose(output, expected_output)

  def test_position_sensitive_with_equal_channels(self):
    num_spatial_bins = [2, 2]
    image_shape = [1, 3, 3, 4]
    crop_size = [2, 2]

    image = tf.constant(range(1, 3 * 3 + 1), dtype=tf.float32,
                        shape=[1, 3, 3, 1])
    tiled_image = tf.tile(image, [1, 1, 1, image_shape[3]])
    boxes = tf.random_uniform((3, 4))
    box_ind = tf.constant([0, 0, 0], dtype=tf.int32)

    # All channels are equal so position-sensitive crop and resize should
    # work as the usual crop and resize for just one channel.
    crop = tf.image.crop_and_resize(image, boxes, box_ind, crop_size)
    crop_and_pool = tf.reduce_mean(crop, [1, 2], keep_dims=True)

    ps_crop_and_pool = ops.position_sensitive_crop_regions(
        tiled_image,
        boxes,
        box_ind,
        crop_size,
        num_spatial_bins,
        global_pool=True)

    with self.test_session() as sess:
      expected_output, output = sess.run((crop_and_pool, ps_crop_and_pool))
      self.assertAllClose(output, expected_output)

  def test_position_sensitive_with_single_bin(self):
    num_spatial_bins = [1, 1]
    image_shape = [2, 3, 3, 4]
    crop_size = [2, 2]

    image = tf.random_uniform(image_shape)
    boxes = tf.random_uniform((6, 4))
    box_ind = tf.constant([0, 0, 0, 1, 1, 1], dtype=tf.int32)

    # When a single bin is used, position-sensitive crop and pool should be
    # the same as non-position sensitive crop and pool.
    crop = tf.image.crop_and_resize(image, boxes, box_ind, crop_size)
    crop_and_pool = tf.reduce_mean(crop, [1, 2], keep_dims=True)

    ps_crop_and_pool = ops.position_sensitive_crop_regions(
        image, boxes, box_ind, crop_size, num_spatial_bins, global_pool=True)

    with self.test_session() as sess:
      expected_output, output = sess.run((crop_and_pool, ps_crop_and_pool))
      self.assertAllClose(output, expected_output)

  def test_raise_value_error_on_num_bins_less_than_one(self):
    num_spatial_bins = [1, -1]
    image_shape = [1, 1, 1, 2]
    crop_size = [2, 2]

    image = tf.constant(1, dtype=tf.float32, shape=image_shape)
    boxes = tf.constant([[0, 0, 1, 1]], dtype=tf.float32)
    box_ind = tf.constant([0], dtype=tf.int32)

    with self.assertRaisesRegexp(ValueError, 'num_spatial_bins should be >= 1'):
      ops.position_sensitive_crop_regions(
          image, boxes, box_ind, crop_size, num_spatial_bins, global_pool=True)

  def test_raise_value_error_on_non_divisible_crop_size(self):
    num_spatial_bins = [2, 3]
    image_shape = [1, 1, 1, 6]
    crop_size = [3, 2]

    image = tf.constant(1, dtype=tf.float32, shape=image_shape)
    boxes = tf.constant([[0, 0, 1, 1]], dtype=tf.float32)
    box_ind = tf.constant([0], dtype=tf.int32)

    with self.assertRaisesRegexp(
        ValueError, 'crop_size should be divisible by num_spatial_bins'):
      ops.position_sensitive_crop_regions(
          image, boxes, box_ind, crop_size, num_spatial_bins, global_pool=True)

  def test_raise_value_error_on_non_divisible_num_channels(self):
    num_spatial_bins = [2, 2]
    image_shape = [1, 1, 1, 5]
    crop_size = [2, 2]

    image = tf.constant(1, dtype=tf.float32, shape=image_shape)
    boxes = tf.constant([[0, 0, 1, 1]], dtype=tf.float32)
    box_ind = tf.constant([0], dtype=tf.int32)

    with self.assertRaisesRegexp(
        ValueError, 'Dimension size must be evenly divisible by 4 but is 5'):
      ops.position_sensitive_crop_regions(
          image, boxes, box_ind, crop_size, num_spatial_bins, global_pool=True)

  def test_position_sensitive_with_global_pool_false(self):
    num_spatial_bins = [3, 2]
    image_shape = [1, 3, 2, 6]
    num_boxes = 2

    # First channel is 1's, second channel is 2's, etc.
    image = tf.constant(range(1, 3 * 2 + 1) * 6, dtype=tf.float32,
                        shape=image_shape)
    boxes = tf.random_uniform((num_boxes, 4))
    box_ind = tf.constant([0, 0], dtype=tf.int32)

    expected_output = []

    # Expected output, when crop_size = [3, 2].
    expected_output.append(np.expand_dims(
        np.tile(np.array([[1, 2],
                          [3, 4],
                          [5, 6]]), (num_boxes, 1, 1)),
        axis=-1))

    # Expected output, when crop_size = [6, 4].
    expected_output.append(np.expand_dims(
        np.tile(np.array([[1, 1, 2, 2],
                          [1, 1, 2, 2],
                          [3, 3, 4, 4],
                          [3, 3, 4, 4],
                          [5, 5, 6, 6],
                          [5, 5, 6, 6]]), (num_boxes, 1, 1)),
        axis=-1))

    for crop_size_mult in range(1, 3):
      crop_size = [3 * crop_size_mult, 2 * crop_size_mult]
      ps_crop = ops.position_sensitive_crop_regions(
          image, boxes, box_ind, crop_size, num_spatial_bins, global_pool=False)
      with self.test_session() as sess:
        output = sess.run(ps_crop)

      self.assertAllEqual(output, expected_output[crop_size_mult - 1])

  def test_position_sensitive_with_global_pool_false_and_known_boxes(self):
    num_spatial_bins = [2, 2]
    image_shape = [2, 2, 2, 4]
    crop_size = [2, 2]

    image = tf.constant(range(1, 2 * 2 * 4  + 1) * 2, dtype=tf.float32,
                        shape=image_shape)

    # First box contains whole image, and second box contains only first row.
    boxes = tf.constant(np.array([[0., 0., 1., 1.],
                                  [0., 0., 0.5, 1.]]), dtype=tf.float32)
    box_ind = tf.constant([0, 1], dtype=tf.int32)

    expected_output = []

    # Expected output, when the box containing whole image.
    expected_output.append(
        np.reshape(np.array([[4, 7],
                             [10, 13]]),
                   (1, 2, 2, 1))
    )

    # Expected output, when the box containing only first row.
    expected_output.append(
        np.reshape(np.array([[3, 6],
                             [7, 10]]),
                   (1, 2, 2, 1))
    )
    expected_output = np.concatenate(expected_output, axis=0)

    ps_crop = ops.position_sensitive_crop_regions(
        image, boxes, box_ind, crop_size, num_spatial_bins, global_pool=False)

    with self.test_session() as sess:
      output = sess.run(ps_crop)
      self.assertAllEqual(output, expected_output)

  def test_position_sensitive_with_global_pool_false_and_single_bin(self):
    num_spatial_bins = [1, 1]
    image_shape = [2, 3, 3, 4]
    crop_size = [1, 1]

    image = tf.random_uniform(image_shape)
    boxes = tf.random_uniform((6, 4))
    box_ind = tf.constant([0, 0, 0, 1, 1, 1], dtype=tf.int32)

    # Since single_bin is used and crop_size = [1, 1] (i.e., no crop resize),
    # the outputs are the same whatever the global_pool value is.
    ps_crop_and_pool = ops.position_sensitive_crop_regions(
        image, boxes, box_ind, crop_size, num_spatial_bins, global_pool=True)
    ps_crop = ops.position_sensitive_crop_regions(
        image, boxes, box_ind, crop_size, num_spatial_bins, global_pool=False)

    with self.test_session() as sess:
      pooled_output, unpooled_output = sess.run((ps_crop_and_pool, ps_crop))
      self.assertAllClose(pooled_output, unpooled_output)

  def test_position_sensitive_with_global_pool_false_and_do_global_pool(self):
    num_spatial_bins = [3, 2]
    image_shape = [1, 3, 2, 6]
    num_boxes = 2

    # First channel is 1's, second channel is 2's, etc.
    image = tf.constant(range(1, 3 * 2 + 1) * 6, dtype=tf.float32,
                        shape=image_shape)
    boxes = tf.random_uniform((num_boxes, 4))
    box_ind = tf.constant([0, 0], dtype=tf.int32)

    expected_output = []

    # Expected output, when crop_size = [3, 2].
    expected_output.append(np.mean(
        np.expand_dims(
            np.tile(np.array([[1, 2],
                              [3, 4],
                              [5, 6]]), (num_boxes, 1, 1)),
            axis=-1),
        axis=(1, 2), keepdims=True))

    # Expected output, when crop_size = [6, 4].
    expected_output.append(np.mean(
        np.expand_dims(
            np.tile(np.array([[1, 1, 2, 2],
                              [1, 1, 2, 2],
                              [3, 3, 4, 4],
                              [3, 3, 4, 4],
                              [5, 5, 6, 6],
                              [5, 5, 6, 6]]), (num_boxes, 1, 1)),
            axis=-1),
        axis=(1, 2), keepdims=True))

    for crop_size_mult in range(1, 3):
      crop_size = [3 * crop_size_mult, 2 * crop_size_mult]

      # Perform global_pooling after running the function with
      # global_pool=False.
      ps_crop = ops.position_sensitive_crop_regions(
          image, boxes, box_ind, crop_size, num_spatial_bins, global_pool=False)
      ps_crop_and_pool = tf.reduce_mean(
          ps_crop, reduction_indices=(1, 2), keep_dims=True)

      with self.test_session() as sess:
        output = sess.run(ps_crop_and_pool)

      self.assertAllEqual(output, expected_output[crop_size_mult - 1])

  def test_raise_value_error_on_non_square_block_size(self):
    num_spatial_bins = [3, 2]
    image_shape = [1, 3, 2, 6]
    crop_size = [6, 2]

    image = tf.constant(1, dtype=tf.float32, shape=image_shape)
    boxes = tf.constant([[0, 0, 1, 1]], dtype=tf.float32)
    box_ind = tf.constant([0], dtype=tf.int32)

    with self.assertRaisesRegexp(
        ValueError, 'Only support square bin crop size for now.'):
      ops.position_sensitive_crop_regions(
          image, boxes, box_ind, crop_size, num_spatial_bins, global_pool=False)


class ReframeBoxMasksToImageMasksTest(tf.test.TestCase):

  def testZeroImageOnEmptyMask(self):
    box_masks = tf.constant([[[0, 0],
                              [0, 0]]], dtype=tf.float32)
    boxes = tf.constant([[0.0, 0.0, 1.0, 1.0]], dtype=tf.float32)
    image_masks = ops.reframe_box_masks_to_image_masks(box_masks, boxes,
                                                       image_height=4,
                                                       image_width=4)
    np_expected_image_masks = np.array([[[0, 0, 0, 0],
                                         [0, 0, 0, 0],
                                         [0, 0, 0, 0],
                                         [0, 0, 0, 0]]], dtype=np.float32)
    with self.test_session() as sess:
      np_image_masks = sess.run(image_masks)
      self.assertAllClose(np_image_masks, np_expected_image_masks)

  def testMaskIsCenteredInImageWhenBoxIsCentered(self):
    box_masks = tf.constant([[[1, 1],
                              [1, 1]]], dtype=tf.float32)
    boxes = tf.constant([[0.25, 0.25, 0.75, 0.75]], dtype=tf.float32)
    image_masks = ops.reframe_box_masks_to_image_masks(box_masks, boxes,
                                                       image_height=4,
                                                       image_width=4)
    np_expected_image_masks = np.array([[[0, 0, 0, 0],
                                         [0, 1, 1, 0],
                                         [0, 1, 1, 0],
                                         [0, 0, 0, 0]]], dtype=np.float32)
    with self.test_session() as sess:
      np_image_masks = sess.run(image_masks)
      self.assertAllClose(np_image_masks, np_expected_image_masks)

  def testMaskOffCenterRemainsOffCenterInImage(self):
    box_masks = tf.constant([[[1, 0],
                              [0, 1]]], dtype=tf.float32)
    boxes = tf.constant([[0.25, 0.5, 0.75, 1.0]], dtype=tf.float32)
    image_masks = ops.reframe_box_masks_to_image_masks(box_masks, boxes,
                                                       image_height=4,
                                                       image_width=4)
    np_expected_image_masks = np.array([[[0, 0, 0, 0],
                                         [0, 0, 0.6111111, 0.16666669],
                                         [0, 0, 0.3888889, 0.83333337],
                                         [0, 0, 0, 0]]], dtype=np.float32)
    with self.test_session() as sess:
      np_image_masks = sess.run(image_masks)
      self.assertAllClose(np_image_masks, np_expected_image_masks)


if __name__ == '__main__':
  tf.test.main()
