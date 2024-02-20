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

"""Tests for utils."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras
from official.recommendation.uplift import utils


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "same_rank",
          "a": tf.zeros((3, 1)),
          "b": tf.zeros((3, 2)),
          "expected_output": tf.zeros((3, 1)),
      },
      {
          "testcase_name": "higher_rank",
          "a": tf.zeros((3, 1)),
          "b": tf.zeros((3,)),
          "expected_output": tf.zeros((3, 1)),
      },
      {
          "testcase_name": "one_less_rank",
          "a": tf.zeros((3,)),
          "b": tf.zeros((3, 2)),
          "expected_output": tf.zeros((3, 1)),
      },
      {
          "testcase_name": "multiple_rank_difference",
          "a": tf.zeros((3, 4)),
          "b": tf.zeros((3, 1, 2, 4)),
          "expected_output": tf.zeros((3, 4, 1, 1)),
      },
  )
  def test_expand_to_match_rank(self, a, b, expected_output):
    self.assertAllEqual(expected_output, utils.expand_to_match_rank(a, b))

  @parameterized.named_parameters(
      {
          "testcase_name": "treatment_only",
          "values": tf.constant([1.5, 2, 3]),
          "is_treatment": tf.ones((3, 1)),
          "expected_control_values": tf.zeros((0,)),
          "expected_treatment_values": tf.constant([1.5, 2, 3]),
      },
      {
          "testcase_name": "control_only",
          "values": tf.constant([1.5, 2, 3]),
          "is_treatment": tf.zeros((3, 1)),
          "expected_control_values": tf.constant([1.5, 2, 3]),
          "expected_treatment_values": tf.zeros((0,)),
      },
      {
          "testcase_name": "control_and_treatment",
          "values": tf.concat(
              values=[tf.ones((2, 2, 3)), 2.0 * tf.ones((1, 2, 3))], axis=0
          ),
          "is_treatment": tf.constant([[0], [0], [1]]),
          "expected_control_values": tf.ones((2, 2, 3)),
          "expected_treatment_values": 2.0 * tf.ones((1, 2, 3)),
      },
      {
          "testcase_name": "one_dimensional_is_treatment",
          "values": tf.concat(
              values=[tf.ones((2, 2, 3)), 2.0 * tf.ones((1, 2, 3))], axis=0
          ),
          "is_treatment": tf.constant([0, 0, 1]),
          "expected_control_values": tf.ones((2, 2, 3)),
          "expected_treatment_values": 2.0 * tf.ones((1, 2, 3)),
      },
      {
          "testcase_name": "empty_values",
          "values": tf.raw_ops.Empty(shape=(0,), dtype=tf.float32, init=True),
          "is_treatment": tf.raw_ops.Empty(
              shape=(0,), dtype=tf.float32, init=True
          ),
          "expected_control_values": tf.zeros((0,)),
          "expected_treatment_values": tf.zeros((0,)),
      },
  )
  def test_split_by_treatment(
      self,
      values,
      is_treatment,
      expected_control_values,
      expected_treatment_values,
  ):
    control_values, treatment_values = utils.split_by_treatment(
        values=values, is_treatment=is_treatment
    )
    self.assertAllEqual(expected_control_values, control_values)
    self.assertAllEqual(expected_treatment_values, treatment_values)

  @parameterized.named_parameters(
      {
          "testcase_name": "decimal_values",
          "is_treatment": tf.constant([1.0, 0.3, 0.0]),
          "expected_error": tf.errors.InvalidArgumentError,
      },
      {
          "testcase_name": "string_values",
          "is_treatment": tf.constant(["a", "b", "c"]),
          "expected_error": ValueError,
      },
  )
  def test_invalid_treatment_indicator_tensor(
      self, is_treatment, expected_error
  ):
    values = tf.ones((3, 1))
    with self.assertRaises(expected_error):
      utils.split_by_treatment(values, is_treatment)

  def test_shape_mismatch(self):
    values = tf.ones((4, 1))
    is_treatment = tf.constant([0, 0, 1])
    with self.assertRaises(ValueError):
      utils.split_by_treatment(values, is_treatment)


if __name__ == "__main__":
  tf.test.main()
