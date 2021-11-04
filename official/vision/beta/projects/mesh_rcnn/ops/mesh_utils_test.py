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

"""Test for mesh utility methods."""

from typing import List

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from official.vision.beta.projects.mesh_rcnn.ops import mesh_utils


class MeshUtilsTest(parameterized.TestCase, tf.test.TestCase):
  """Mesh Utils Tests"""

  @parameterized.named_parameters(
      {'testcase_name': 'single-tensor',
       'num_tensors': [1], 'tensor_length': 3},
      {'testcase_name': 'multiple-tensors-same-shapes',
       'num_tensors': [4, 4, 4], 'tensor_length': 3},
      {'testcase_name': 'multiple-tensors-varied-shapes',
       'num_tensors': [3, 1, 7, 5], 'tensor_length': 3}
  )
  def test_list_to_packed(self, num_tensors: List[int], tensor_length: int):
    """Test list of tensors to packed tensor conversion.

    Args:
      num_tensors: `List` of integers for 0th dimension of the test tensors.
      tensor_length: `int` for the 1st dimension of the test tensors.
    """
    tensor_list = []

    for size in num_tensors:
      tensor_list.append(tf.random.uniform([size, tensor_length], -1, 1))

    packed_output = mesh_utils.list_to_packed_tensor(tensor_list)

    packed_tensor = packed_output[0]
    num_items = packed_output[1]
    items_packed_first_idx = packed_output[2]
    items_packed_to_list_idx = packed_output[3]

    # Ensure shapes are correct
    self.assertAllEqual(
        tf.shape(packed_tensor),
        [sum(num_tensors), tensor_length])
    self.assertAllEqual(tf.shape(num_items), [len(num_tensors)])
    self.assertAllEqual(tf.shape(items_packed_first_idx), [len(num_tensors)])
    self.assertAllEqual(tf.shape(items_packed_to_list_idx), [sum(num_tensors)])

    # Ensure that output values are correct
    np_output = packed_tensor.numpy()
    np_input = np.concatenate([x.numpy() for x in tensor_list], axis=0)
    self.assertAllEqual(np_output, np_input)

    self.assertAllEqual(num_items, tf.convert_to_tensor([num_tensors]))

    self.assertAllEqual(
        items_packed_first_idx,
        tf.convert_to_tensor(
            [sum(num_tensors[:i]) for i in range(len(num_tensors))]
        )
    )

    items_packed_to_list_idx_as_list = []
    for i in range(len(num_tensors)):
      items_packed_to_list_idx_as_list += [i] * num_tensors[i]
    self.assertAllEqual(
        items_packed_to_list_idx,
        tf.convert_to_tensor(items_packed_to_list_idx_as_list)
    )

  @parameterized.named_parameters(
      {'testcase_name': 'single-tensor',
       'num_tensors': [1], 'tensor_length': 3, 'pad_value': 0},
      {'testcase_name': 'multiple-tensors-same-shapes',
       'num_tensors': [4, 4, 4], 'tensor_length': 3, 'pad_value': 0},
      {'testcase_name': 'multiple-tensors-varied-shapes',
       'num_tensors': [3, 1, 7, 5], 'tensor_length': 3, 'pad_value': 0},
      {'testcase_name': 'multiple-tensors-varied-shapes-non-zero-padding',
       'num_tensors': [3, 1, 7, 5], 'tensor_length': 3, 'pad_value': -1}
  )
  def test_list_to_padded(self, num_tensors: List[int], tensor_length: int,
                          pad_value: int):
    """Test list of tensors to padded tensor conversion.

    Args:
      num_tensors: `List` of integers for 0th dimension of the test tensors.
      tensor_length: `int` for the 1st dimension of the test tensors.
      pad_value: `int` denoting the desired padding value.
    """
    tensor_list = []

    for size in num_tensors:
      tensor_list.append(tf.random.uniform([size, tensor_length], -1, 1))

    padded_tensor = mesh_utils.list_to_padded_tensor(tensor_list, pad_value)

    # Ensure shaps are correct
    self.assertAllEqual(
        tf.shape(padded_tensor),
        [len(num_tensors), max(num_tensors), tensor_length])

    # Ensure output is correct
    padded_tensors_unstacked = tf.unstack(padded_tensor)

    for padded_tensor, tensor in zip(padded_tensors_unstacked, tensor_list):
      output_sum = tf.math.reduce_sum(padded_tensor)
      input_sum = tf.math.reduce_sum(tensor)

      delta = tf.shape(padded_tensor)[0] - tf.shape(tensor)[0]

      output_sum -= tf.cast(
          pad_value * delta * tensor_length, padded_tensor.dtype)

      self.assertAllClose(output_sum, input_sum)

if __name__ == "__main__":
  tf.test.main()
