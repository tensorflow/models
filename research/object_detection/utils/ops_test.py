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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl.testing import parameterized
import numpy as np
import six
from six.moves import range
import tensorflow.compat.v1 as tf
import tf_slim as slim
from object_detection.core import standard_fields as fields
from object_detection.utils import ops
from object_detection.utils import test_case


class NormalizedToImageCoordinatesTest(test_case.TestCase):

  def test_normalized_to_image_coordinates(self):
    normalized_boxes_np = np.array([[[0.0, 0.0, 1.0, 1.0]],
                                    [[0.5, 0.5, 1.0, 1.0]]])

    def graph_fn(normalized_boxes):
      image_shape = tf.convert_to_tensor([1, 4, 4, 3], dtype=tf.int32)
      absolute_boxes = ops.normalized_to_image_coordinates(
          normalized_boxes, image_shape, parallel_iterations=2)
      return absolute_boxes

    expected_boxes = np.array([[[0, 0, 4, 4]],
                               [[2, 2, 4, 4]]])

    absolute_boxes = self.execute(graph_fn, [normalized_boxes_np])
    self.assertAllEqual(absolute_boxes, expected_boxes)


class ReduceSumTrailingDimensions(test_case.TestCase):

  def test_reduce_sum_trailing_dimensions(self):

    def graph_fn(input_tensor):
      reduced_tensor = ops.reduce_sum_trailing_dimensions(input_tensor, ndims=2)
      return reduced_tensor

    reduced_np = self.execute(graph_fn, [np.ones((2, 2, 2), np.float32)])
    self.assertAllClose(reduced_np, 2 * np.ones((2, 2), np.float32))


class MeshgridTest(test_case.TestCase):

  def test_meshgrid_numpy_comparison(self):
    """Tests meshgrid op with vectors, for which it should match numpy."""

    x = np.arange(4)
    y = np.arange(6)

    def graph_fn():
      xgrid, ygrid = ops.meshgrid(x, y)
      return xgrid, ygrid

    exp_xgrid, exp_ygrid = np.meshgrid(x, y)
    xgrid_output, ygrid_output = self.execute(graph_fn, [])
    self.assertAllEqual(xgrid_output, exp_xgrid)
    self.assertAllEqual(ygrid_output, exp_ygrid)

  def test_meshgrid_multidimensional(self):
    np.random.seed(18)
    x = np.random.rand(4, 1, 2).astype(np.float32)
    y = np.random.rand(2, 3).astype(np.float32)

    grid_shape = list(y.shape) + list(x.shape)

    def graph_fn():
      xgrid, ygrid = ops.meshgrid(x, y)
      self.assertEqual(xgrid.get_shape().as_list(), grid_shape)
      self.assertEqual(ygrid.get_shape().as_list(), grid_shape)
      return xgrid, ygrid

    xgrid_output, ygrid_output = self.execute(graph_fn, [])

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


class OpsTestFixedPadding(test_case.TestCase):

  def test_3x3_kernel(self):

    def graph_fn():
      tensor = tf.constant([[[[0.], [0.]], [[0.], [0.]]]])
      padded_tensor = ops.fixed_padding(tensor, 3)
      return padded_tensor

    padded_tensor_out = self.execute(graph_fn, [])
    self.assertEqual((1, 4, 4, 1), padded_tensor_out.shape)

  def test_5x5_kernel(self):

    def graph_fn():
      tensor = tf.constant([[[[0.], [0.]], [[0.], [0.]]]])
      padded_tensor = ops.fixed_padding(tensor, 5)
      return padded_tensor

    padded_tensor_out = self.execute(graph_fn, [])
    self.assertEqual((1, 6, 6, 1), padded_tensor_out.shape)

  def test_3x3_atrous_kernel(self):

    def graph_fn():
      tensor = tf.constant([[[[0.], [0.]], [[0.], [0.]]]])
      padded_tensor = ops.fixed_padding(tensor, 3, 2)
      return padded_tensor

    padded_tensor_out = self.execute(graph_fn, [])
    self.assertEqual((1, 6, 6, 1), padded_tensor_out.shape)


class OpsTestPadToMultiple(test_case.TestCase):

  def test_zero_padding(self):

    def graph_fn():
      tensor = tf.constant([[[[0.], [0.]], [[0.], [0.]]]])
      padded_tensor = ops.pad_to_multiple(tensor, 1)
      return padded_tensor

    padded_tensor_out = self.execute(graph_fn, [])
    self.assertEqual((1, 2, 2, 1), padded_tensor_out.shape)

  def test_no_padding(self):

    def graph_fn():
      tensor = tf.constant([[[[0.], [0.]], [[0.], [0.]]]])
      padded_tensor = ops.pad_to_multiple(tensor, 2)
      return padded_tensor

    padded_tensor_out = self.execute(graph_fn, [])
    self.assertEqual((1, 2, 2, 1), padded_tensor_out.shape)

  def test_non_square_padding(self):

    def graph_fn():
      tensor = tf.constant([[[[0.], [0.]]]])
      padded_tensor = ops.pad_to_multiple(tensor, 2)
      return padded_tensor

    padded_tensor_out = self.execute(graph_fn, [])
    self.assertEqual((1, 2, 2, 1), padded_tensor_out.shape)

  def test_padding(self):

    def graph_fn():
      tensor = tf.constant([[[[0.], [0.]], [[0.], [0.]]]])
      padded_tensor = ops.pad_to_multiple(tensor, 4)
      return padded_tensor

    padded_tensor_out = self.execute(graph_fn, [])
    self.assertEqual((1, 4, 4, 1), padded_tensor_out.shape)


class OpsTestPaddedOneHotEncoding(test_case.TestCase):

  def test_correct_one_hot_tensor_with_no_pad(self):

    def graph_fn():
      indices = tf.constant([1, 2, 3, 5])
      one_hot_tensor = ops.padded_one_hot_encoding(indices, depth=6, left_pad=0)
      return one_hot_tensor

    expected_tensor = np.array([[0, 1, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 1]], np.float32)

    out_one_hot_tensor = self.execute(graph_fn, [])
    self.assertAllClose(out_one_hot_tensor, expected_tensor, rtol=1e-10,
                        atol=1e-10)

  def test_correct_one_hot_tensor_with_pad_one(self):

    def graph_fn():
      indices = tf.constant([1, 2, 3, 5])
      one_hot_tensor = ops.padded_one_hot_encoding(indices, depth=6, left_pad=1)
      return one_hot_tensor

    expected_tensor = np.array([[0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1]], np.float32)
    out_one_hot_tensor = self.execute(graph_fn, [])
    self.assertAllClose(out_one_hot_tensor, expected_tensor, rtol=1e-10,
                        atol=1e-10)

  def test_correct_one_hot_tensor_with_pad_three(self):

    def graph_fn():
      indices = tf.constant([1, 2, 3, 5])
      one_hot_tensor = ops.padded_one_hot_encoding(indices, depth=6, left_pad=3)
      return one_hot_tensor

    expected_tensor = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 1]], np.float32)

    out_one_hot_tensor = self.execute(graph_fn, [])
    self.assertAllClose(out_one_hot_tensor, expected_tensor, rtol=1e-10,
                        atol=1e-10)

  def test_correct_padded_one_hot_tensor_with_empty_indices(self):

    depth = 6
    pad = 2

    def graph_fn():
      indices = tf.constant([])
      one_hot_tensor = ops.padded_one_hot_encoding(
          indices, depth=depth, left_pad=pad)
      return one_hot_tensor

    expected_tensor = np.zeros((0, depth + pad))
    out_one_hot_tensor = self.execute(graph_fn, [])
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


class OpsDenseToSparseBoxesTest(test_case.TestCase):

  def test_return_all_boxes_when_all_input_boxes_are_valid(self):
    num_classes = 4
    num_valid_boxes = 3
    code_size = 4

    def graph_fn(dense_location, dense_num_boxes):
      box_locations, box_classes = ops.dense_to_sparse_boxes(
          dense_location, dense_num_boxes, num_classes)
      return box_locations, box_classes

    dense_location_np = np.random.uniform(size=[num_valid_boxes, code_size])
    dense_num_boxes_np = np.array([1, 0, 0, 2], dtype=np.int32)

    expected_box_locations = dense_location_np
    expected_box_classses = np.array([0, 3, 3])

    # Executing on CPU only since output shape is not constant.
    box_locations, box_classes = self.execute_cpu(
        graph_fn, [dense_location_np, dense_num_boxes_np])

    self.assertAllClose(box_locations, expected_box_locations, rtol=1e-6,
                        atol=1e-6)
    self.assertAllEqual(box_classes, expected_box_classses)

  def test_return_only_valid_boxes_when_input_contains_invalid_boxes(self):
    num_classes = 4
    num_valid_boxes = 3
    num_boxes = 10
    code_size = 4

    def graph_fn(dense_location, dense_num_boxes):
      box_locations, box_classes = ops.dense_to_sparse_boxes(
          dense_location, dense_num_boxes, num_classes)
      return box_locations, box_classes

    dense_location_np = np.random.uniform(size=[num_boxes, code_size])
    dense_num_boxes_np = np.array([1, 0, 0, 2], dtype=np.int32)

    expected_box_locations = dense_location_np[:num_valid_boxes]
    expected_box_classses = np.array([0, 3, 3])

    # Executing on CPU only since output shape is not constant.
    box_locations, box_classes = self.execute_cpu(
        graph_fn, [dense_location_np, dense_num_boxes_np])

    self.assertAllClose(box_locations, expected_box_locations, rtol=1e-6,
                        atol=1e-6)
    self.assertAllEqual(box_classes, expected_box_classses)


class OpsTestIndicesToDenseVector(test_case.TestCase):

  def test_indices_to_dense_vector(self):
    size = 10000
    num_indices = np.random.randint(size)
    rand_indices = np.random.permutation(np.arange(size))[0:num_indices]

    expected_output = np.zeros(size, dtype=np.float32)
    expected_output[rand_indices] = 1.

    def graph_fn():
      tf_rand_indices = tf.constant(rand_indices)
      indicator = ops.indices_to_dense_vector(tf_rand_indices, size)
      return indicator

    output = self.execute(graph_fn, [])
    self.assertAllEqual(output, expected_output)
    self.assertEqual(output.dtype, expected_output.dtype)

  def test_indices_to_dense_vector_size_at_inference(self):
    size = 5000
    num_indices = 250
    all_indices = np.arange(size)
    rand_indices = np.random.permutation(all_indices)[0:num_indices]

    expected_output = np.zeros(size, dtype=np.float32)
    expected_output[rand_indices] = 1.

    def graph_fn(tf_all_indices):
      tf_rand_indices = tf.constant(rand_indices)
      indicator = ops.indices_to_dense_vector(tf_rand_indices,
                                              tf.shape(tf_all_indices)[0])
      return indicator

    output = self.execute(graph_fn, [all_indices])
    self.assertAllEqual(output, expected_output)
    self.assertEqual(output.dtype, expected_output.dtype)

  def test_indices_to_dense_vector_int(self):
    size = 500
    num_indices = 25
    rand_indices = np.random.permutation(np.arange(size))[0:num_indices]

    expected_output = np.zeros(size, dtype=np.int64)
    expected_output[rand_indices] = 1

    def graph_fn():
      tf_rand_indices = tf.constant(rand_indices)
      indicator = ops.indices_to_dense_vector(
          tf_rand_indices, size, 1, dtype=tf.int64)
      return indicator

    output = self.execute(graph_fn, [])
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

    def graph_fn():
      tf_rand_indices = tf.constant(rand_indices)
      indicator = ops.indices_to_dense_vector(
          tf_rand_indices,
          size,
          indices_value=indices_value,
          default_value=default_value)
      return indicator

    output = self.execute(graph_fn, [])
    self.assertAllClose(output, expected_output)
    self.assertEqual(output.dtype, expected_output.dtype)

  def test_indices_to_dense_vector_all_indices_as_input(self):
    size = 500
    num_indices = 500
    rand_indices = np.random.permutation(np.arange(size))[0:num_indices]

    expected_output = np.ones(size, dtype=np.float32)

    def graph_fn():
      tf_rand_indices = tf.constant(rand_indices)
      indicator = ops.indices_to_dense_vector(tf_rand_indices, size)
      return indicator

    output = self.execute(graph_fn, [])
    self.assertAllEqual(output, expected_output)
    self.assertEqual(output.dtype, expected_output.dtype)

  def test_indices_to_dense_vector_empty_indices_as_input(self):
    size = 500
    rand_indices = []

    expected_output = np.zeros(size, dtype=np.float32)

    def graph_fn():
      tf_rand_indices = tf.constant(rand_indices)
      indicator = ops.indices_to_dense_vector(tf_rand_indices, size)
      return indicator

    output = self.execute(graph_fn, [])
    self.assertAllEqual(output, expected_output)
    self.assertEqual(output.dtype, expected_output.dtype)


class GroundtruthFilterTest(test_case.TestCase):

  def test_filter_groundtruth(self):

    def graph_fn(input_image, input_boxes, input_classes, input_is_crowd,
                 input_area, input_difficult, input_label_types,
                 input_confidences, valid_indices):
      input_tensors = {
          fields.InputDataFields.image: input_image,
          fields.InputDataFields.groundtruth_boxes: input_boxes,
          fields.InputDataFields.groundtruth_classes: input_classes,
          fields.InputDataFields.groundtruth_is_crowd: input_is_crowd,
          fields.InputDataFields.groundtruth_area: input_area,
          fields.InputDataFields.groundtruth_difficult: input_difficult,
          fields.InputDataFields.groundtruth_label_types: input_label_types,
          fields.InputDataFields.groundtruth_confidences: input_confidences,
      }

      output_tensors = ops.retain_groundtruth(input_tensors, valid_indices)
      return output_tensors

    input_image = np.random.rand(224, 224, 3)
    input_boxes = np.array([[0.2, 0.4, 0.1, 0.8], [0.2, 0.4, 1.0, 0.8]],
                           dtype=np.float32)
    input_classes = np.array([1, 2], dtype=np.int32)
    input_is_crowd = np.array([False, True], dtype=np.bool)
    input_area = np.array([32, 48], dtype=np.float32)
    input_difficult = np.array([True, False], dtype=np.bool)
    input_label_types = np.array(['APPROPRIATE', 'INCORRECT'],
                                 dtype=np.string_)
    input_confidences = np.array([0.99, 0.5], dtype=np.float32)
    valid_indices = np.array([0], dtype=np.int32)

    # Strings are not supported on TPU.
    output_tensors = self.execute_cpu(
        graph_fn,
        [input_image, input_boxes, input_classes, input_is_crowd, input_area,
         input_difficult, input_label_types, input_confidences, valid_indices]
    )

    expected_tensors = {
        fields.InputDataFields.image: input_image,
        fields.InputDataFields.groundtruth_boxes: [[0.2, 0.4, 0.1, 0.8]],
        fields.InputDataFields.groundtruth_classes: [1],
        fields.InputDataFields.groundtruth_is_crowd: [False],
        fields.InputDataFields.groundtruth_area: [32],
        fields.InputDataFields.groundtruth_difficult: [True],
        fields.InputDataFields.groundtruth_label_types: [six.b('APPROPRIATE')],
        fields.InputDataFields.groundtruth_confidences: [0.99],
    }
    for key in [fields.InputDataFields.image,
                fields.InputDataFields.groundtruth_boxes,
                fields.InputDataFields.groundtruth_area,
                fields.InputDataFields.groundtruth_confidences]:
      self.assertAllClose(expected_tensors[key], output_tensors[key])

    for key in [fields.InputDataFields.groundtruth_classes,
                fields.InputDataFields.groundtruth_is_crowd,
                fields.InputDataFields.groundtruth_label_types]:
      self.assertAllEqual(expected_tensors[key], output_tensors[key])

  def test_filter_with_missing_fields(self):

    input_boxes = np.array([[0.2, 0.4, 0.1, 0.8], [0.2, 0.4, 1.0, 0.8]],
                           dtype=np.float)
    input_classes = np.array([1, 2], dtype=np.int32)
    valid_indices = np.array([0], dtype=np.int32)

    expected_tensors = {
        fields.InputDataFields.groundtruth_boxes:
        [[0.2, 0.4, 0.1, 0.8]],
        fields.InputDataFields.groundtruth_classes:
        [1]
    }

    def graph_fn(input_boxes, input_classes, valid_indices):
      input_tensors = {
          fields.InputDataFields.groundtruth_boxes: input_boxes,
          fields.InputDataFields.groundtruth_classes: input_classes
      }
      output_tensors = ops.retain_groundtruth(input_tensors, valid_indices)
      return output_tensors

    output_tensors = self.execute(graph_fn, [input_boxes, input_classes,
                                             valid_indices])

    for key in [fields.InputDataFields.groundtruth_boxes]:
      self.assertAllClose(expected_tensors[key], output_tensors[key])
    for key in [fields.InputDataFields.groundtruth_classes]:
      self.assertAllEqual(expected_tensors[key], output_tensors[key])

  def test_filter_with_empty_fields(self):

    def graph_fn(input_boxes, input_classes, input_is_crowd, input_area,
                 input_difficult, input_confidences, valid_indices):
      input_tensors = {
          fields.InputDataFields.groundtruth_boxes: input_boxes,
          fields.InputDataFields.groundtruth_classes: input_classes,
          fields.InputDataFields.groundtruth_is_crowd: input_is_crowd,
          fields.InputDataFields.groundtruth_area: input_area,
          fields.InputDataFields.groundtruth_difficult: input_difficult,
          fields.InputDataFields.groundtruth_confidences: input_confidences,
      }
      output_tensors = ops.retain_groundtruth(input_tensors, valid_indices)
      return output_tensors

    input_boxes = np.array([[0.2, 0.4, 0.1, 0.8], [0.2, 0.4, 1.0, 0.8]],
                           dtype=np.float)
    input_classes = np.array([1, 2], dtype=np.int32)
    input_is_crowd = np.array([False, True], dtype=np.bool)
    input_area = np.array([], dtype=np.float32)
    input_difficult = np.array([], dtype=np.float32)
    input_confidences = np.array([0.99, 0.5], dtype=np.float32)
    valid_indices = np.array([0], dtype=np.int32)

    expected_tensors = {
        fields.InputDataFields.groundtruth_boxes: [[0.2, 0.4, 0.1, 0.8]],
        fields.InputDataFields.groundtruth_classes: [1],
        fields.InputDataFields.groundtruth_is_crowd: [False],
        fields.InputDataFields.groundtruth_area: [],
        fields.InputDataFields.groundtruth_difficult: [],
        fields.InputDataFields.groundtruth_confidences: [0.99],
    }
    output_tensors = self.execute(graph_fn, [
        input_boxes, input_classes, input_is_crowd, input_area,
        input_difficult, input_confidences, valid_indices])

    for key in [fields.InputDataFields.groundtruth_boxes,
                fields.InputDataFields.groundtruth_area,
                fields.InputDataFields.groundtruth_confidences]:
      self.assertAllClose(expected_tensors[key], output_tensors[key])
    for key in [fields.InputDataFields.groundtruth_classes,
                fields.InputDataFields.groundtruth_is_crowd]:
      self.assertAllEqual(expected_tensors[key], output_tensors[key])

  def test_filter_with_empty_groundtruth_boxes(self):

    def graph_fn(input_boxes, input_classes, input_is_crowd, input_area,
                 input_difficult, input_confidences, valid_indices):
      input_tensors = {
          fields.InputDataFields.groundtruth_boxes: input_boxes,
          fields.InputDataFields.groundtruth_classes: input_classes,
          fields.InputDataFields.groundtruth_is_crowd: input_is_crowd,
          fields.InputDataFields.groundtruth_area: input_area,
          fields.InputDataFields.groundtruth_difficult: input_difficult,
          fields.InputDataFields.groundtruth_confidences: input_confidences,
      }
      output_tensors = ops.retain_groundtruth(input_tensors, valid_indices)
      return output_tensors

    input_boxes = np.array([], dtype=np.float).reshape(0, 4)
    input_classes = np.array([], dtype=np.int32)
    input_is_crowd = np.array([], dtype=np.bool)
    input_area = np.array([], dtype=np.float32)
    input_difficult = np.array([], dtype=np.float32)
    input_confidences = np.array([], dtype=np.float32)
    valid_indices = np.array([], dtype=np.int32)

    output_tensors = self.execute(graph_fn, [input_boxes, input_classes,
                                             input_is_crowd, input_area,
                                             input_difficult,
                                             input_confidences,
                                             valid_indices])
    for key in output_tensors:
      if key == fields.InputDataFields.groundtruth_boxes:
        self.assertAllEqual([0, 4], output_tensors[key].shape)
      else:
        self.assertAllEqual([0], output_tensors[key].shape)


class RetainGroundTruthWithPositiveClasses(test_case.TestCase):

  def test_filter_groundtruth_with_positive_classes(self):

    def graph_fn(input_image, input_boxes, input_classes, input_is_crowd,
                 input_area, input_difficult, input_label_types,
                 input_confidences):
      input_tensors = {
          fields.InputDataFields.image: input_image,
          fields.InputDataFields.groundtruth_boxes: input_boxes,
          fields.InputDataFields.groundtruth_classes: input_classes,
          fields.InputDataFields.groundtruth_is_crowd: input_is_crowd,
          fields.InputDataFields.groundtruth_area: input_area,
          fields.InputDataFields.groundtruth_difficult: input_difficult,
          fields.InputDataFields.groundtruth_label_types: input_label_types,
          fields.InputDataFields.groundtruth_confidences: input_confidences,
      }
      output_tensors = ops.retain_groundtruth_with_positive_classes(
          input_tensors)
      return output_tensors

    input_image = np.random.rand(224, 224, 3)
    input_boxes = np.array([[0.2, 0.4, 0.1, 0.8], [0.2, 0.4, 1.0, 0.8]],
                           dtype=np.float)
    input_classes = np.array([1, 0], dtype=np.int32)
    input_is_crowd = np.array([False, True], dtype=np.bool)
    input_area = np.array([32, 48], dtype=np.float32)
    input_difficult = np.array([True, False], dtype=np.bool)
    input_label_types = np.array(['APPROPRIATE', 'INCORRECT'],
                                 dtype=np.string_)
    input_confidences = np.array([0.99, 0.5], dtype=np.float32)

    expected_tensors = {
        fields.InputDataFields.image: input_image,
        fields.InputDataFields.groundtruth_boxes: [[0.2, 0.4, 0.1, 0.8]],
        fields.InputDataFields.groundtruth_classes: [1],
        fields.InputDataFields.groundtruth_is_crowd: [False],
        fields.InputDataFields.groundtruth_area: [32],
        fields.InputDataFields.groundtruth_difficult: [True],
        fields.InputDataFields.groundtruth_label_types: [six.b('APPROPRIATE')],
        fields.InputDataFields.groundtruth_confidences: [0.99],
    }

    # Executing on CPU because string types are not supported on TPU.
    output_tensors = self.execute_cpu(graph_fn,
                                      [input_image, input_boxes,
                                       input_classes, input_is_crowd,
                                       input_area,
                                       input_difficult, input_label_types,
                                       input_confidences])

    for key in [fields.InputDataFields.image,
                fields.InputDataFields.groundtruth_boxes,
                fields.InputDataFields.groundtruth_area,
                fields.InputDataFields.groundtruth_confidences]:
      self.assertAllClose(expected_tensors[key], output_tensors[key])
    for key in [fields.InputDataFields.groundtruth_classes,
                fields.InputDataFields.groundtruth_is_crowd,
                fields.InputDataFields.groundtruth_label_types]:
      self.assertAllEqual(expected_tensors[key], output_tensors[key])


class ReplaceNaNGroundtruthLabelScoresWithOnes(test_case.TestCase):

  def test_replace_nan_groundtruth_label_scores_with_ones(self):

    def graph_fn():
      label_scores = tf.constant([np.nan, 1.0, np.nan])
      output_tensor = ops.replace_nan_groundtruth_label_scores_with_ones(
          label_scores)
      return output_tensor

    expected_tensor = [1.0, 1.0, 1.0]
    output_tensor = self.execute(graph_fn, [])
    self.assertAllClose(expected_tensor, output_tensor)

  def test_input_equals_output_when_no_nans(self):

    input_label_scores = [0.5, 1.0, 1.0]
    def graph_fn():
      label_scores_tensor = tf.constant(input_label_scores)
      output_label_scores = ops.replace_nan_groundtruth_label_scores_with_ones(
          label_scores_tensor)
      return output_label_scores

    output_label_scores = self.execute(graph_fn, [])

    self.assertAllClose(input_label_scores, output_label_scores)


class GroundtruthFilterWithCrowdBoxesTest(test_case.TestCase):

  def test_filter_groundtruth_with_crowd_boxes(self):

    def graph_fn():
      input_tensors = {
          fields.InputDataFields.groundtruth_boxes:
          [[0.1, 0.2, 0.6, 0.8], [0.2, 0.4, 0.1, 0.8]],
          fields.InputDataFields.groundtruth_classes: [1, 2],
          fields.InputDataFields.groundtruth_is_crowd: [True, False],
          fields.InputDataFields.groundtruth_area: [100.0, 238.7],
          fields.InputDataFields.groundtruth_confidences: [0.5, 0.99],
      }
      output_tensors = ops.filter_groundtruth_with_crowd_boxes(
          input_tensors)
      return output_tensors

    expected_tensors = {
        fields.InputDataFields.groundtruth_boxes: [[0.2, 0.4, 0.1, 0.8]],
        fields.InputDataFields.groundtruth_classes: [2],
        fields.InputDataFields.groundtruth_is_crowd: [False],
        fields.InputDataFields.groundtruth_area: [238.7],
        fields.InputDataFields.groundtruth_confidences: [0.99],
    }

    output_tensors = self.execute(graph_fn, [])
    for key in [fields.InputDataFields.groundtruth_boxes,
                fields.InputDataFields.groundtruth_area,
                fields.InputDataFields.groundtruth_confidences]:
      self.assertAllClose(expected_tensors[key], output_tensors[key])
    for key in [fields.InputDataFields.groundtruth_classes,
                fields.InputDataFields.groundtruth_is_crowd]:
      self.assertAllEqual(expected_tensors[key], output_tensors[key])


class GroundtruthFilterWithNanBoxTest(test_case.TestCase):

  def test_filter_groundtruth_with_nan_box_coordinates(self):

    def graph_fn():
      input_tensors = {
          fields.InputDataFields.groundtruth_boxes:
          [[np.nan, np.nan, np.nan, np.nan], [0.2, 0.4, 0.1, 0.8]],
          fields.InputDataFields.groundtruth_classes: [1, 2],
          fields.InputDataFields.groundtruth_is_crowd: [False, True],
          fields.InputDataFields.groundtruth_area: [100.0, 238.7],
          fields.InputDataFields.groundtruth_confidences: [0.5, 0.99],
      }
      output_tensors = ops.filter_groundtruth_with_nan_box_coordinates(
          input_tensors)
      return output_tensors

    expected_tensors = {
        fields.InputDataFields.groundtruth_boxes: [[0.2, 0.4, 0.1, 0.8]],
        fields.InputDataFields.groundtruth_classes: [2],
        fields.InputDataFields.groundtruth_is_crowd: [True],
        fields.InputDataFields.groundtruth_area: [238.7],
        fields.InputDataFields.groundtruth_confidences: [0.99],
    }

    output_tensors = self.execute(graph_fn, [])
    for key in [fields.InputDataFields.groundtruth_boxes,
                fields.InputDataFields.groundtruth_area,
                fields.InputDataFields.groundtruth_confidences]:
      self.assertAllClose(expected_tensors[key], output_tensors[key])
    for key in [fields.InputDataFields.groundtruth_classes,
                fields.InputDataFields.groundtruth_is_crowd]:
      self.assertAllEqual(expected_tensors[key], output_tensors[key])


class GroundtruthFilterWithUnrecognizedClassesTest(test_case.TestCase):

  def test_filter_unrecognized_classes(self):
    def graph_fn():
      input_tensors = {
          fields.InputDataFields.groundtruth_boxes:
          [[.3, .3, .5, .7], [0.2, 0.4, 0.1, 0.8]],
          fields.InputDataFields.groundtruth_classes: [-1, 2],
          fields.InputDataFields.groundtruth_is_crowd: [False, True],
          fields.InputDataFields.groundtruth_area: [100.0, 238.7],
          fields.InputDataFields.groundtruth_confidences: [0.5, 0.99],
      }
      output_tensors = ops.filter_unrecognized_classes(input_tensors)
      return output_tensors

    expected_tensors = {
        fields.InputDataFields.groundtruth_boxes: [[0.2, 0.4, 0.1, 0.8]],
        fields.InputDataFields.groundtruth_classes: [2],
        fields.InputDataFields.groundtruth_is_crowd: [True],
        fields.InputDataFields.groundtruth_area: [238.7],
        fields.InputDataFields.groundtruth_confidences: [0.99],
    }

    output_tensors = self.execute(graph_fn, [])
    for key in [fields.InputDataFields.groundtruth_boxes,
                fields.InputDataFields.groundtruth_area,
                fields.InputDataFields.groundtruth_confidences]:
      self.assertAllClose(expected_tensors[key], output_tensors[key])
    for key in [fields.InputDataFields.groundtruth_classes,
                fields.InputDataFields.groundtruth_is_crowd]:
      self.assertAllEqual(expected_tensors[key], output_tensors[key])


class OpsTestNormalizeToTarget(test_case.TestCase):

  def test_create_normalize_to_target(self):

    if self.is_tf2():
      self.skipTest('Skipping as variable names not supported in eager mode.')

    inputs = tf.random_uniform([5, 10, 12, 3])
    target_norm_value = 4.0
    dim = 3
    with self.test_session():
      output = ops.normalize_to_target(inputs, target_norm_value, dim)
      self.assertEqual(output.op.name, 'NormalizeToTarget/mul')
      var_name = slim.get_variables()[0].name
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

    if self.is_tf2():
      self.skipTest('normalize_to_target not supported in eager mode because,'
                    ' it requires creating variables.')

    inputs = np.random.uniform(size=(5, 10, 12, 3)).astype(np.float32)
    def graph_fn(inputs):
      target_norm_value = 4.0
      dim = 3
      output = ops.normalize_to_target(inputs, target_norm_value, dim)
      return output

    # Executing on CPU since creating a variable inside a conditional is not
    # supported.
    outputs = self.execute_cpu(graph_fn, [inputs])
    self.assertEqual(outputs.shape, inputs.shape)

  def test_correct_initial_output_values(self):

    if self.is_tf2():
      self.skipTest('normalize_to_target not supported in eager mode because,'
                    ' it requires creating variables.')
    def graph_fn():
      inputs = tf.constant([[[[3, 4], [7, 24]],
                             [[5, -12], [-1, 0]]]], tf.float32)
      target_norm_value = 10.0
      dim = 3
      normalized_inputs = ops.normalize_to_target(inputs, target_norm_value,
                                                  dim)
      return normalized_inputs

    expected_output = [[[[30/5.0, 40/5.0], [70/25.0, 240/25.0]],
                        [[50/13.0, -120/13.0], [-10, 0]]]]
    # Executing on CPU since creating a variable inside a conditional is not
    # supported.
    output = self.execute_cpu(graph_fn, [])
    self.assertAllClose(output, expected_output)

  def test_multiple_target_norm_values(self):

    if self.is_tf2():
      self.skipTest('normalize_to_target not supported in eager mode because,'
                    ' it requires creating variables.')

    def graph_fn():
      inputs = tf.constant([[[[3, 4], [7, 24]],
                             [[5, -12], [-1, 0]]]], tf.float32)
      target_norm_value = [10.0, 20.0]
      dim = 3
      normalized_inputs = ops.normalize_to_target(inputs, target_norm_value,
                                                  dim)
      return normalized_inputs

    expected_output = [[[[30/5.0, 80/5.0], [70/25.0, 480/25.0]],
                        [[50/13.0, -240/13.0], [-10, 0]]]]

    # Executing on CPU since creating a variable inside a conditional is not
    # supported.
    output = self.execute_cpu(graph_fn, [])
    self.assertAllClose(output, expected_output)


class OpsTestPositionSensitiveCropRegions(test_case.TestCase):

  def test_position_sensitive(self):
    num_spatial_bins = [3, 2]
    image_shape = [3, 2, 6]

    # The result for both boxes should be [[1, 2], [3, 4], [5, 6]]
    # before averaging.
    expected_output = np.array([3.5, 3.5]).reshape([2, 1, 1, 1])

    for crop_size_mult in range(1, 3):
      crop_size = [3 * crop_size_mult, 2 * crop_size_mult]

      def graph_fn():
        # First channel is 1's, second channel is 2's, etc.
        image = tf.constant(
            list(range(1, 3 * 2 + 1)) * 6, dtype=tf.float32, shape=image_shape)
        boxes = tf.random_uniform((2, 4))

        # pylint:disable=cell-var-from-loop
        ps_crop_and_pool = ops.position_sensitive_crop_regions(
            image, boxes, crop_size, num_spatial_bins, global_pool=True)
        return ps_crop_and_pool

      output = self.execute(graph_fn, [])
      self.assertAllClose(output, expected_output)

  def test_position_sensitive_with_equal_channels(self):
    num_spatial_bins = [2, 2]
    image_shape = [3, 3, 4]
    crop_size = [2, 2]

    def graph_fn():
      image = tf.constant(
          list(range(1, 3 * 3 + 1)), dtype=tf.float32, shape=[3, 3, 1])
      tiled_image = tf.tile(image, [1, 1, image_shape[2]])
      boxes = tf.random_uniform((3, 4))
      box_ind = tf.constant([0, 0, 0], dtype=tf.int32)

      # All channels are equal so position-sensitive crop and resize should
      # work as the usual crop and resize for just one channel.
      crop = tf.image.crop_and_resize(tf.expand_dims(image, axis=0), boxes,
                                      box_ind, crop_size)
      crop_and_pool = tf.reduce_mean(crop, [1, 2], keepdims=True)

      ps_crop_and_pool = ops.position_sensitive_crop_regions(
          tiled_image,
          boxes,
          crop_size,
          num_spatial_bins,
          global_pool=True)

      return crop_and_pool, ps_crop_and_pool

    # Crop and resize op is not supported in TPUs.
    expected_output, output = self.execute_cpu(graph_fn, [])
    self.assertAllClose(output, expected_output)

  def test_raise_value_error_on_num_bins_less_than_one(self):
    num_spatial_bins = [1, -1]
    image_shape = [1, 1, 2]
    crop_size = [2, 2]

    image = tf.constant(1, dtype=tf.float32, shape=image_shape)
    boxes = tf.constant([[0, 0, 1, 1]], dtype=tf.float32)

    with self.assertRaisesRegexp(ValueError, 'num_spatial_bins should be >= 1'):
      ops.position_sensitive_crop_regions(
          image, boxes, crop_size, num_spatial_bins, global_pool=True)

  def test_raise_value_error_on_non_divisible_crop_size(self):
    num_spatial_bins = [2, 3]
    image_shape = [1, 1, 6]
    crop_size = [3, 2]

    image = tf.constant(1, dtype=tf.float32, shape=image_shape)
    boxes = tf.constant([[0, 0, 1, 1]], dtype=tf.float32)

    with self.assertRaisesRegexp(
        ValueError, 'crop_size should be divisible by num_spatial_bins'):
      ops.position_sensitive_crop_regions(
          image, boxes, crop_size, num_spatial_bins, global_pool=True)

  def test_raise_value_error_on_non_divisible_num_channels(self):
    num_spatial_bins = [2, 2]
    image_shape = [1, 1, 5]
    crop_size = [2, 2]

    def graph_fn():
      image = tf.constant(1, dtype=tf.float32, shape=image_shape)
      boxes = tf.constant([[0, 0, 1, 1]], dtype=tf.float32)

      return ops.position_sensitive_crop_regions(
          image, boxes, crop_size, num_spatial_bins, global_pool=True)

    with self.assertRaisesRegexp(
        ValueError, 'Dimension size must be evenly divisible by 4 but is 5'):
      self.execute(graph_fn, [])

  def test_position_sensitive_with_global_pool_false(self):
    num_spatial_bins = [3, 2]
    image_shape = [3, 2, 6]
    num_boxes = 2

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
      # First channel is 1's, second channel is 2's, etc.

      def graph_fn():
        # pylint:disable=cell-var-from-loop
        image = tf.constant(
            list(range(1, 3 * 2 + 1)) * 6, dtype=tf.float32, shape=image_shape)
        boxes = tf.random_uniform((num_boxes, 4))

        ps_crop = ops.position_sensitive_crop_regions(
            image, boxes, crop_size, num_spatial_bins, global_pool=False)
        return ps_crop

      output = self.execute(graph_fn, [])
      self.assertAllClose(output, expected_output[crop_size_mult - 1])

  def test_position_sensitive_with_global_pool_false_and_do_global_pool(self):
    num_spatial_bins = [3, 2]
    image_shape = [3, 2, 6]
    num_boxes = 2

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

      def graph_fn():
        # pylint:disable=cell-var-from-loop
        # First channel is 1's, second channel is 2's, etc.
        image = tf.constant(
            list(range(1, 3 * 2 + 1)) * 6, dtype=tf.float32, shape=image_shape)
        boxes = tf.random_uniform((num_boxes, 4))

        # Perform global_pooling after running the function with
        # global_pool=False.
        ps_crop = ops.position_sensitive_crop_regions(
            image, boxes, crop_size, num_spatial_bins, global_pool=False)
        ps_crop_and_pool = tf.reduce_mean(
            ps_crop, reduction_indices=(1, 2), keepdims=True)
        return ps_crop_and_pool

      output = self.execute(graph_fn, [])
      self.assertAllClose(output, expected_output[crop_size_mult - 1])

  def test_raise_value_error_on_non_square_block_size(self):
    num_spatial_bins = [3, 2]
    image_shape = [3, 2, 6]
    crop_size = [6, 2]

    image = tf.constant(1, dtype=tf.float32, shape=image_shape)
    boxes = tf.constant([[0, 0, 1, 1]], dtype=tf.float32)

    with self.assertRaisesRegexp(
        ValueError, 'Only support square bin crop size for now.'):
      ops.position_sensitive_crop_regions(
          image, boxes, crop_size, num_spatial_bins, global_pool=False)


class OpsTestBatchPositionSensitiveCropRegions(test_case.TestCase):

  def test_position_sensitive_with_single_bin(self):
    num_spatial_bins = [1, 1]
    image_shape = [2, 3, 3, 4]
    crop_size = [2, 2]

    def graph_fn():
      image = tf.random_uniform(image_shape)
      boxes = tf.random_uniform((2, 3, 4))
      box_ind = tf.constant([0, 0, 0, 1, 1, 1], dtype=tf.int32)

      # When a single bin is used, position-sensitive crop and pool should be
      # the same as non-position sensitive crop and pool.
      crop = tf.image.crop_and_resize(image,
                                      tf.reshape(boxes, [-1, 4]), box_ind,
                                      crop_size)
      crop_and_pool = tf.reduce_mean(crop, [1, 2], keepdims=True)
      crop_and_pool = tf.reshape(crop_and_pool, [2, 3, 1, 1, 4])

      ps_crop_and_pool = ops.batch_position_sensitive_crop_regions(
          image, boxes, crop_size, num_spatial_bins, global_pool=True)
      return crop_and_pool, ps_crop_and_pool

    # Crop and resize is not supported on TPUs.
    expected_output, output = self.execute_cpu(graph_fn, [])
    self.assertAllClose(output, expected_output)

  def test_position_sensitive_with_global_pool_false_and_known_boxes(self):
    num_spatial_bins = [2, 2]
    image_shape = [2, 2, 2, 4]
    crop_size = [2, 2]

    # box_ind = tf.constant([0, 1], dtype=tf.int32)

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
    expected_output = np.stack(expected_output, axis=0)

    def graph_fn():
      images = tf.constant(
          list(range(1, 2 * 2 * 4 + 1)) * 2, dtype=tf.float32,
          shape=image_shape)

      # First box contains whole image, and second box contains only first row.
      boxes = tf.constant(np.array([[[0., 0., 1., 1.]],
                                    [[0., 0., 0.5, 1.]]]), dtype=tf.float32)

      ps_crop = ops.batch_position_sensitive_crop_regions(
          images, boxes, crop_size, num_spatial_bins, global_pool=False)
      return ps_crop

    output = self.execute(graph_fn, [])
    self.assertAllEqual(output, expected_output)

  def test_position_sensitive_with_global_pool_false_and_single_bin(self):
    num_spatial_bins = [1, 1]
    image_shape = [2, 3, 3, 4]
    crop_size = [1, 1]

    def graph_fn():
      images = tf.random_uniform(image_shape)
      boxes = tf.random_uniform((2, 3, 4))
      # box_ind = tf.constant([0, 0, 0, 1, 1, 1], dtype=tf.int32)

      # Since single_bin is used and crop_size = [1, 1] (i.e., no crop resize),
      # the outputs are the same whatever the global_pool value is.
      ps_crop_and_pool = ops.batch_position_sensitive_crop_regions(
          images, boxes, crop_size, num_spatial_bins, global_pool=True)
      ps_crop = ops.batch_position_sensitive_crop_regions(
          images, boxes, crop_size, num_spatial_bins, global_pool=False)
      return ps_crop_and_pool, ps_crop

    pooled_output, unpooled_output = self.execute(graph_fn, [])
    self.assertAllClose(pooled_output, unpooled_output)


# The following tests are only executed on CPU because the output
# shape is not constant.
class ReframeBoxMasksToImageMasksTest(test_case.TestCase,
                                      parameterized.TestCase):

  def test_reframe_image_corners_relative_to_boxes(self):

    def graph_fn():
      return ops.reframe_image_corners_relative_to_boxes(
          tf.constant([[0.1, 0.2, 0.3, 0.4]]))
    np_boxes = self.execute_cpu(graph_fn, [])
    self.assertAllClose(np_boxes, [[-0.5, -1, 4.5, 4.]])

  @parameterized.parameters(
      {'mask_dtype': tf.float32, 'mask_dtype_np': np.float32,
       'resize_method': 'bilinear'},
      {'mask_dtype': tf.float32, 'mask_dtype_np': np.float32,
       'resize_method': 'nearest'},
      {'mask_dtype': tf.uint8, 'mask_dtype_np': np.uint8,
       'resize_method': 'bilinear'},
      {'mask_dtype': tf.uint8, 'mask_dtype_np': np.uint8,
       'resize_method': 'nearest'},
  )
  def testZeroImageOnEmptyMask(self, mask_dtype, mask_dtype_np, resize_method):
    np_expected_image_masks = np.array([[[0, 0, 0, 0],
                                         [0, 0, 0, 0],
                                         [0, 0, 0, 0],
                                         [0, 0, 0, 0]]])
    def graph_fn():
      box_masks = tf.constant([[[0, 0],
                                [0, 0]]], dtype=mask_dtype)
      boxes = tf.constant([[0.0, 0.0, 1.0, 1.0]], dtype=tf.float32)
      image_masks = ops.reframe_box_masks_to_image_masks(
          box_masks, boxes, image_height=4, image_width=4,
          resize_method=resize_method)
      return image_masks

    np_image_masks = self.execute_cpu(graph_fn, [])
    self.assertEqual(np_image_masks.dtype, mask_dtype_np)
    self.assertAllClose(np_image_masks, np_expected_image_masks)

  @parameterized.parameters(
      {'mask_dtype': tf.float32, 'mask_dtype_np': np.float32,
       'resize_method': 'bilinear'},
      {'mask_dtype': tf.float32, 'mask_dtype_np': np.float32,
       'resize_method': 'nearest'},
      {'mask_dtype': tf.uint8, 'mask_dtype_np': np.uint8,
       'resize_method': 'bilinear'},
      {'mask_dtype': tf.uint8, 'mask_dtype_np': np.uint8,
       'resize_method': 'nearest'},
  )
  def testZeroBoxMasks(self, mask_dtype, mask_dtype_np, resize_method):

    def graph_fn():
      box_masks = tf.zeros([0, 3, 3], dtype=mask_dtype)
      boxes = tf.zeros([0, 4], dtype=tf.float32)
      image_masks = ops.reframe_box_masks_to_image_masks(
          box_masks, boxes, image_height=4, image_width=4,
          resize_method=resize_method)
      return image_masks

    np_image_masks = self.execute_cpu(graph_fn, [])
    self.assertEqual(np_image_masks.dtype, mask_dtype_np)
    self.assertAllEqual(np_image_masks.shape, np.array([0, 4, 4]))

  def testBoxWithZeroArea(self):

    def graph_fn():
      box_masks = tf.zeros([1, 3, 3], dtype=tf.float32)
      boxes = tf.constant([[0.1, 0.2, 0.1, 0.7]], dtype=tf.float32)
      image_masks = ops.reframe_box_masks_to_image_masks(box_masks, boxes,
                                                         image_height=4,
                                                         image_width=4)
      return image_masks

    np_image_masks = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(np_image_masks.shape, np.array([1, 4, 4]))

  @parameterized.parameters(
      {'mask_dtype': tf.float32, 'mask_dtype_np': np.float32,
       'resize_method': 'bilinear'},
      {'mask_dtype': tf.float32, 'mask_dtype_np': np.float32,
       'resize_method': 'nearest'},
      {'mask_dtype': tf.uint8, 'mask_dtype_np': np.uint8,
       'resize_method': 'bilinear'},
      {'mask_dtype': tf.uint8, 'mask_dtype_np': np.uint8,
       'resize_method': 'nearest'},
  )
  def testMaskIsCenteredInImageWhenBoxIsCentered(self, mask_dtype,
                                                 mask_dtype_np, resize_method):

    def graph_fn():
      box_masks = tf.constant([[[4, 4],
                                [4, 4]]], dtype=mask_dtype)
      boxes = tf.constant([[0.25, 0.25, 0.75, 0.75]], dtype=tf.float32)
      image_masks = ops.reframe_box_masks_to_image_masks(
          box_masks, boxes, image_height=4, image_width=4,
          resize_method=resize_method)
      return image_masks

    np_expected_image_masks = np.array([[[0, 0, 0, 0],
                                         [0, 4, 4, 0],
                                         [0, 4, 4, 0],
                                         [0, 0, 0, 0]]], dtype=mask_dtype_np)
    np_image_masks = self.execute_cpu(graph_fn, [])
    self.assertEqual(np_image_masks.dtype, mask_dtype_np)
    self.assertAllClose(np_image_masks, np_expected_image_masks)

  @parameterized.parameters(
      {'mask_dtype': tf.float32, 'mask_dtype_np': np.float32,
       'resize_method': 'bilinear'},
      {'mask_dtype': tf.float32, 'mask_dtype_np': np.float32,
       'resize_method': 'nearest'},
      {'mask_dtype': tf.uint8, 'mask_dtype_np': np.uint8,
       'resize_method': 'bilinear'},
      {'mask_dtype': tf.uint8, 'mask_dtype_np': np.uint8,
       'resize_method': 'nearest'},
  )
  def testMaskOffCenterRemainsOffCenterInImage(self, mask_dtype,
                                               mask_dtype_np, resize_method):

    def graph_fn():
      box_masks = tf.constant([[[1, 0],
                                [0, 1]]], dtype=mask_dtype)
      boxes = tf.constant([[0.25, 0.5, 0.75, 1.0]], dtype=tf.float32)
      image_masks = ops.reframe_box_masks_to_image_masks(
          box_masks, boxes, image_height=4, image_width=4,
          resize_method=resize_method)
      return image_masks

    if mask_dtype == tf.float32 and resize_method == 'bilinear':
      np_expected_image_masks = np.array([[[0, 0, 0, 0],
                                           [0, 0, 0.6111111, 0.16666669],
                                           [0, 0, 0.3888889, 0.83333337],
                                           [0, 0, 0, 0]]], dtype=np.float32)
    else:
      np_expected_image_masks = np.array([[[0, 0, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1],
                                           [0, 0, 0, 0]]], dtype=mask_dtype_np)
    np_image_masks = self.execute_cpu(graph_fn, [])
    self.assertEqual(np_image_masks.dtype, mask_dtype_np)
    self.assertAllClose(np_image_masks, np_expected_image_masks)


class MergeBoxesWithMultipleLabelsTest(test_case.TestCase):

  def testMergeBoxesWithMultipleLabels(self):

    def graph_fn():
      boxes = tf.constant(
          [[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.5, 0.75],
           [0.25, 0.25, 0.75, 0.75]],
          dtype=tf.float32)
      class_indices = tf.constant([0, 4, 2], dtype=tf.int32)
      class_confidences = tf.constant([0.8, 0.2, 0.1], dtype=tf.float32)
      num_classes = 5
      merged_boxes, merged_classes, merged_confidences, merged_box_indices = (
          ops.merge_boxes_with_multiple_labels(
              boxes, class_indices, class_confidences, num_classes))

      return (merged_boxes, merged_classes, merged_confidences,
              merged_box_indices)

    expected_merged_boxes = np.array(
        [[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.5, 0.75]], dtype=np.float32)
    expected_merged_classes = np.array(
        [[1, 0, 1, 0, 0], [0, 0, 0, 0, 1]], dtype=np.int32)
    expected_merged_confidences = np.array(
        [[0.8, 0, 0.1, 0, 0], [0, 0, 0, 0, 0.2]], dtype=np.float32)
    expected_merged_box_indices = np.array([0, 1], dtype=np.int32)

    # Running on CPU only as tf.unique is not supported on TPU.
    (np_merged_boxes, np_merged_classes, np_merged_confidences,
     np_merged_box_indices) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(np_merged_boxes, expected_merged_boxes)
    self.assertAllClose(np_merged_classes, expected_merged_classes)
    self.assertAllClose(np_merged_confidences, expected_merged_confidences)
    self.assertAllClose(np_merged_box_indices, expected_merged_box_indices)

  def testMergeBoxesWithMultipleLabelsCornerCase(self):

    def graph_fn():
      boxes = tf.constant(
          [[0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1],
           [1, 1, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]],
          dtype=tf.float32)
      class_indices = tf.constant([0, 1, 2, 3, 2, 1, 0, 3], dtype=tf.int32)
      class_confidences = tf.constant([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
                                      dtype=tf.float32)
      num_classes = 4
      merged_boxes, merged_classes, merged_confidences, merged_box_indices = (
          ops.merge_boxes_with_multiple_labels(
              boxes, class_indices, class_confidences, num_classes))
      return (merged_boxes, merged_classes, merged_confidences,
              merged_box_indices)

    expected_merged_boxes = np.array(
        [[0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]],
        dtype=np.float32)
    expected_merged_classes = np.array(
        [[1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
        dtype=np.int32)
    expected_merged_confidences = np.array(
        [[0.1, 0, 0, 0.6], [0.4, 0.9, 0, 0],
         [0, 0.7, 0.2, 0], [0, 0, 0.3, 0.8]], dtype=np.float32)
    expected_merged_box_indices = np.array([0, 1, 2, 3], dtype=np.int32)

    # Running on CPU only as tf.unique is not supported on TPU.
    (np_merged_boxes, np_merged_classes, np_merged_confidences,
     np_merged_box_indices) = self.execute_cpu(graph_fn, [])

    self.assertAllClose(np_merged_boxes, expected_merged_boxes)
    self.assertAllClose(np_merged_classes, expected_merged_classes)
    self.assertAllClose(np_merged_confidences, expected_merged_confidences)
    self.assertAllClose(np_merged_box_indices, expected_merged_box_indices)

  def testMergeBoxesWithEmptyInputs(self):

    def graph_fn():
      boxes = tf.zeros([0, 4], dtype=tf.float32)
      class_indices = tf.constant([], dtype=tf.int32)
      class_confidences = tf.constant([], dtype=tf.float32)
      num_classes = 5
      merged_boxes, merged_classes, merged_confidences, merged_box_indices = (
          ops.merge_boxes_with_multiple_labels(
              boxes, class_indices, class_confidences, num_classes))
      return (merged_boxes, merged_classes, merged_confidences,
              merged_box_indices)

    # Running on CPU only as tf.unique is not supported on TPU.
    (np_merged_boxes, np_merged_classes, np_merged_confidences,
     np_merged_box_indices) = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(np_merged_boxes.shape, [0, 4])
    self.assertAllEqual(np_merged_classes.shape, [0, 5])
    self.assertAllEqual(np_merged_confidences.shape, [0, 5])
    self.assertAllEqual(np_merged_box_indices.shape, [0])

  def testMergeBoxesWithMultipleLabelsUsesInt64(self):

    if self.is_tf2():
      self.skipTest('Getting op names is not supported in eager mode.')

    boxes = tf.constant(
        [[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.5, 0.75],
         [0.25, 0.25, 0.75, 0.75]],
        dtype=tf.float32)
    class_indices = tf.constant([0, 4, 2], dtype=tf.int32)
    class_confidences = tf.constant([0.8, 0.2, 0.1], dtype=tf.float32)
    num_classes = 5
    ops.merge_boxes_with_multiple_labels(
        boxes, class_indices, class_confidences, num_classes)

    graph = tf.get_default_graph()

    def assert_dtype_is_int64(op_name):
      op = graph.get_operation_by_name(op_name)
      self.assertEqual(op.get_attr('dtype'), tf.int64)

    def assert_t_is_int64(op_name):
      op = graph.get_operation_by_name(op_name)
      self.assertEqual(op.get_attr('T'), tf.int64)

    assert_dtype_is_int64('map/TensorArray')
    assert_dtype_is_int64('map/TensorArray_1')
    assert_dtype_is_int64('map/while/TensorArrayReadV3')
    assert_t_is_int64('map/while/TensorArrayWrite/TensorArrayWriteV3')
    assert_t_is_int64(
        'map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3')
    assert_dtype_is_int64('map/TensorArrayStack/TensorArrayGatherV3')


class NearestNeighborUpsamplingTest(test_case.TestCase):

  def test_upsampling_with_single_scale(self):

    def graph_fn(inputs):
      custom_op_output = ops.nearest_neighbor_upsampling(inputs, scale=2)
      return custom_op_output
    inputs = np.reshape(np.arange(4).astype(np.float32), [1, 2, 2, 1])
    custom_op_output = self.execute(graph_fn, [inputs])

    expected_output = [[[[0], [0], [1], [1]],
                        [[0], [0], [1], [1]],
                        [[2], [2], [3], [3]],
                        [[2], [2], [3], [3]]]]
    self.assertAllClose(custom_op_output, expected_output)

  def test_upsampling_with_separate_height_width_scales(self):

    def graph_fn(inputs):
      custom_op_output = ops.nearest_neighbor_upsampling(inputs,
                                                         height_scale=2,
                                                         width_scale=3)
      return custom_op_output
    inputs = np.reshape(np.arange(4).astype(np.float32), [1, 2, 2, 1])
    custom_op_output = self.execute(graph_fn, [inputs])

    expected_output = [[[[0], [0], [0], [1], [1], [1]],
                        [[0], [0], [0], [1], [1], [1]],
                        [[2], [2], [2], [3], [3], [3]],
                        [[2], [2], [2], [3], [3], [3]]]]
    self.assertAllClose(custom_op_output, expected_output)


class MatmulGatherOnZerothAxis(test_case.TestCase):

  def test_gather_2d(self):

    def graph_fn(params, indices):
      return ops.matmul_gather_on_zeroth_axis(params, indices)

    params = np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [0, 1, 0, 0]], dtype=np.float32)
    indices = np.array([2, 2, 1], dtype=np.int32)
    expected_output = np.array([[9, 10, 11, 12], [9, 10, 11, 12], [5, 6, 7, 8]])
    gather_output = self.execute(graph_fn, [params, indices])
    self.assertAllClose(gather_output, expected_output)

  def test_gather_3d(self):

    def graph_fn(params, indices):
      return ops.matmul_gather_on_zeroth_axis(params, indices)

    params = np.array([[[1, 2], [3, 4]],
                       [[5, 6], [7, 8]],
                       [[9, 10], [11, 12]],
                       [[0, 1], [0, 0]]], dtype=np.float32)
    indices = np.array([0, 3, 1], dtype=np.int32)
    expected_output = np.array([[[1, 2], [3, 4]],
                                [[0, 1], [0, 0]],
                                [[5, 6], [7, 8]]])
    gather_output = self.execute(graph_fn, [params, indices])
    self.assertAllClose(gather_output, expected_output)

  def test_gather_with_many_indices(self):

    def graph_fn(params, indices):
      return ops.matmul_gather_on_zeroth_axis(params, indices)

    params = np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [0, 1, 0, 0]], dtype=np.float32)
    indices = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
    expected_output = np.array(6*[[1, 2, 3, 4]])
    gather_output = self.execute(graph_fn, [params, indices])
    self.assertAllClose(gather_output, expected_output)

  def test_gather_with_dynamic_shape_input(self):

    def graph_fn(params, indices):
      return ops.matmul_gather_on_zeroth_axis(params, indices)

    params = np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [0, 1, 0, 0]], dtype=np.float32)
    indices = np.array([0, 0, 0, 0, 0, 0])
    expected_output = np.array(6*[[1, 2, 3, 4]])
    gather_output = self.execute(graph_fn, [params, indices])
    self.assertAllClose(gather_output, expected_output)


class FpnFeatureLevelsTest(test_case.TestCase):

  def test_correct_fpn_levels(self):
    image_size = 640
    pretraininig_image_size = 224
    image_ratio = image_size * 1.0 / pretraininig_image_size
    boxes = np.array(
        [
            [
                [0, 0, 111, 111],  # Level 0.
                [0, 0, 113, 113],  # Level 1.
                [0, 0, 223, 223],  # Level 1.
                [0, 0, 225, 225],  # Level 2.
                [0, 0, 449, 449]   # Level 3.
            ],
        ],
        dtype=np.float32) / image_size

    def graph_fn(boxes):
      return ops.fpn_feature_levels(
          num_levels=5, unit_scale_index=2, image_ratio=image_ratio,
          boxes=boxes)

    levels = self.execute(graph_fn, [boxes])
    self.assertAllEqual([[0, 1, 1, 2, 3]], levels)


class TestBfloat16ToFloat32(test_case.TestCase):

  def test_convert_list(self):
    var_list = [
        tf.constant([1.], dtype=tf.bfloat16),
        tf.constant([2], dtype=tf.int32)
    ]
    casted_var_list = ops.bfloat16_to_float32_nested(var_list)
    self.assertEqual(casted_var_list[0].dtype, tf.float32)
    self.assertEqual(casted_var_list[1].dtype, tf.int32)

  def test_convert_tensor_dict(self):
    tensor_dict = {
        'key1': tf.constant([1.], dtype=tf.bfloat16),
        'key2': [
            tf.constant([0.5], dtype=tf.bfloat16),
            tf.constant([7], dtype=tf.int32),
        ],
        'key3': tf.constant([2], dtype=tf.uint8),
    }
    tensor_dict = ops.bfloat16_to_float32_nested(tensor_dict)

    self.assertEqual(tensor_dict['key1'].dtype, tf.float32)
    self.assertEqual(tensor_dict['key2'][0].dtype, tf.float32)
    self.assertEqual(tensor_dict['key2'][1].dtype, tf.int32)
    self.assertEqual(tensor_dict['key3'].dtype, tf.uint8)


class TestGatherWithPaddingValues(test_case.TestCase):

  def test_gather_with_padding_values(self):
    expected_gathered_tensor = [
        [0, 0, 0.2, 0.2],
        [0, 0, 0, 0],
        [0, 0, 0.1, 0.1],
        [0, 0, 0, 0],
    ]

    def graph_fn():
      indices = tf.constant([1, -1, 0, -1])
      input_tensor = tf.constant([[0, 0, 0.1, 0.1], [0, 0, 0.2, 0.2]],
                                 dtype=tf.float32)

      gathered_tensor = ops.gather_with_padding_values(
          input_tensor,
          indices=indices,
          padding_value=tf.zeros_like(input_tensor[0]))
      self.assertEqual(gathered_tensor.dtype, tf.float32)

      return gathered_tensor

    gathered_tensor_np = self.execute(graph_fn, [])
    self.assertAllClose(expected_gathered_tensor, gathered_tensor_np)








class TestGIoU(test_case.TestCase):

  def test_giou_with_no_overlap(self):
    expected_giou_tensor = [
        0, -1/3, -3/4, 0, -98/100
    ]

    def graph_fn():
      boxes1 = tf.constant([[3, 4, 5, 6], [3, 3, 5, 5],
                            [0, 0, 0, 0], [3, 3, 5, 5],
                            [9, 9, 10, 10]],
                           dtype=tf.float32)
      boxes2 = tf.constant([[3, 2, 5, 4], [3, 7, 5, 9],
                            [5, 5, 10, 10], [3, 5, 5, 7],
                            [0, 0, 1, 1]], dtype=tf.float32)

      giou = ops.giou(boxes1, boxes2)
      self.assertEqual(giou.dtype, tf.float32)

      return giou

    giou = self.execute(graph_fn, [])
    self.assertAllClose(expected_giou_tensor, giou)

  def test_giou_with_overlaps(self):
    expected_giou_tensor = [
        1/25, 1/4, 1/3, 1/7 - 2/9
    ]

    def graph_fn():
      boxes1 = tf.constant([[2, 1, 7, 6], [2, 2, 4, 4],
                            [2, 2, 4, 4], [2, 2, 4, 4]],
                           dtype=tf.float32)
      boxes2 = tf.constant([[4, 3, 5, 4], [3, 3, 4, 4],
                            [2, 3, 4, 5], [3, 3, 5, 5]], dtype=tf.float32)

      giou = ops.giou(boxes1, boxes2)
      self.assertEqual(giou.dtype, tf.float32)

      return giou

    giou = self.execute(graph_fn, [])
    self.assertAllClose(expected_giou_tensor, giou)

  def test_giou_with_perfect_overlap(self):
    expected_giou_tensor = [1]

    def graph_fn():
      boxes1 = tf.constant([[3, 3, 5, 5]], dtype=tf.float32)
      boxes2 = tf.constant([[3, 3, 5, 5]], dtype=tf.float32)

      giou = ops.giou(boxes1, boxes2)
      self.assertEqual(giou.dtype, tf.float32)

      return giou

    giou = self.execute(graph_fn, [])
    self.assertAllClose(expected_giou_tensor, giou)

  def test_giou_with_zero_area_boxes(self):
    expected_giou_tensor = [0]

    def graph_fn():
      boxes1 = tf.constant([[1, 1, 1, 1]], dtype=tf.float32)
      boxes2 = tf.constant([[1, 1, 1, 1]], dtype=tf.float32)

      giou = ops.giou(boxes1, boxes2)
      self.assertEqual(giou.dtype, tf.float32)

      return giou

    giou = self.execute(graph_fn, [])
    self.assertAllClose(expected_giou_tensor, giou)

  def test_giou_different_with_l1_same(self):
    expected_giou_tensor = [
        2/3, 3/5
    ]

    def graph_fn():
      boxes1 = tf.constant([[3, 3, 5, 5], [3, 3, 5, 5]], dtype=tf.float32)
      boxes2 = tf.constant([[3, 2.5, 5, 5.5], [3, 2.5, 5, 4.5]],
                           dtype=tf.float32)

      giou = ops.giou(boxes1, boxes2)
      self.assertEqual(giou.dtype, tf.float32)

      return giou

    giou = self.execute(graph_fn, [])
    self.assertAllClose(expected_giou_tensor, giou)


class TestCoordinateConversion(test_case.TestCase):

  def test_coord_conv(self):
    expected_box_tensor = [
        [0.5, 0.5, 5.5, 5.5], [2, 1, 4, 7], [0, 0, 0, 0]
    ]

    def graph_fn():
      boxes = tf.constant([[3, 3, 5, 5], [3, 4, 2, 6], [0, 0, 0, 0]],
                          dtype=tf.float32)

      converted = ops.center_to_corner_coordinate(boxes)
      self.assertEqual(converted.dtype, tf.float32)

      return converted

    converted = self.execute(graph_fn, [])
    self.assertAllClose(expected_box_tensor, converted)


if __name__ == '__main__':
  tf.test.main()
