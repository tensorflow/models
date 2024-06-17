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

"""Tests for variance metric."""

from typing import Optional

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.recommendation.uplift import keras_test_case
from official.recommendation.uplift.metrics import variance


class VarianceTest(keras_test_case.KerasTestCase, parameterized.TestCase):

  def _compute_variance(
      self, values: tf.Tensor, weights: Optional[tf.Tensor] = None
  ) -> float:
    values = values.numpy()

    if weights is None:
      return values.var()

    weights = weights.numpy()
    weights = np.broadcast_to(weights, shape=values.shape)
    weighted_mean = np.average(values, weights=weights)
    return np.average((values - weighted_mean) ** 2, weights=weights)

  @parameterized.named_parameters(
      {
          "testcase_name": "unweighted",
          "values": tf.constant([-2, 0, 3, 5]),
          "sample_weight": None,
      },
      {
          "testcase_name": "weighted",
          "values": tf.constant([-2, 0, 3, 5]),
          "sample_weight": tf.constant([1, 0.3, 0.0, 1.5]),
      },
      {
          "testcase_name": "negative_weights",
          "values": tf.constant([-2, 0, 3, 5]),
          "sample_weight": tf.constant([1, 0.3, 0.0, -1.5]),
      },
  )
  def test_single_batch_correctness(self, values, sample_weight):
    metric = variance.Variance()
    metric(values=values, sample_weight=sample_weight)

    expected_variance = self._compute_variance(values, sample_weight)
    self.assertAllClose(expected_variance, metric.result())

  @parameterized.named_parameters(
      {
          "testcase_name": "unweighted",
          "values_batches": [tf.constant([-2, 0, 3, 5]), tf.constant([10])],
          "sample_weight_batches": [None, None],
          "all_values": tf.constant([-2, 0, 3, 5, 10]),
          "all_weights": tf.ones((5,)),
      },
      {
          "testcase_name": "weighted",
          "values_batches": [tf.constant([-2, 0, 3, 5]), tf.constant([10, -4])],
          "sample_weight_batches": [
              tf.constant([1, 0.3, 0.0, -1.5]),
              tf.constant([-4.0]),
          ],
          "all_values": tf.constant([-2, 0, 3, 5, 10, -4]),
          "all_weights": tf.constant([1, 0.3, 0.0, -1.5, -4.0, -4.0]),
      },
      {
          "testcase_name": "mix_weighted_and_unweighted",
          "values_batches": [
              tf.constant([-2.2, 0, 3, 5]),
              tf.constant([10.5, -4]),
              tf.ones((3,), dtype=tf.float32),
          ],
          "sample_weight_batches": [
              tf.constant([1, 0.3, 0.0, -1.5]),
              None,
              None,
          ],
          "all_values": tf.constant([-2.2, 0, 3, 5, 10.5, -4, 1, 1, 1]),
          "all_weights": tf.constant([1, 0.3, 0.0, -1.5, 1, 1, 1, 1, 1]),
      },
  )
  def test_multi_batch_correctness(
      self, values_batches, sample_weight_batches, all_values, all_weights
  ):
    metric = variance.Variance()

    for values, sample_weight in zip(values_batches, sample_weight_batches):
      metric(values=values, sample_weight=sample_weight)

    expected_variance = self._compute_variance(all_values, all_weights)
    self.assertAllClose(expected_variance, metric.result())
    self.assertAllGreaterEqual(metric.result(), 0.0)

  @parameterized.named_parameters(
      {
          "testcase_name": "unit_weight",
          "values": tf.constant([0, 1, 2, 3]),
          "sample_weight": tf.constant([1.0]),
          "expected_variance": 1.25,
      },
      {
          "testcase_name": "zero_weight",
          "values": tf.constant([0, 1, 2, 3]),
          "sample_weight": tf.constant([0.0]),
          "expected_variance": 0.0,
      },
      {
          "testcase_name": "decimal_weight",
          "values": tf.constant([0, 1, 2, 3]),
          "sample_weight": tf.constant([0.2]),
          "expected_variance": 1.25,
      },
      {
          "testcase_name": "negative_weight",
          "values": tf.constant([0, 1, 2, 3]),
          "sample_weight": tf.constant([-0.2]),
          "expected_variance": 1.25,
      },
  )
  def test_float_sample_weight(self, values, sample_weight, expected_variance):
    metric = variance.Variance()
    metric(values, sample_weight=sample_weight)
    self.assertAllClose(expected_variance, metric.result())

  def test_empty_input(self):
    metric = variance.Variance()
    values = tf.constant([0, 1, 2, 3])
    metric(values)
    self.assertAllClose(1.25, metric.result())
    metric(tf.ones(shape=(0,)), sample_weight=None)
    self.assertAllClose(1.25, metric.result())

  def test_initial_state(self):
    metric = variance.Variance()
    self.assertAllClose(0.0, metric.result())

  def test_dtype_correctness(self):
    # 1 << 128 overflows for float32 but fits in float64.
    value = tf.constant([1 << 128], dtype=tf.float64)

    metric = variance.Variance(dtype=tf.float32)
    metric(value)
    self.assertAllEqual(np.nan, metric.result().numpy())

    metric = variance.Variance(dtype=tf.float64)
    metric(value)
    self.assertAllEqual(0.0, metric.result().numpy())

  def test_invalid_dtype(self):
    with self.assertRaises(ValueError):
      metric = variance.Variance(dtype=tf.string)
      metric(tf.constant(["hello, world!"], tf.string))

  @parameterized.named_parameters(
      {
          "testcase_name": "squeeze_dimension_invalid",
          "values": tf.ones((10, 10)),
          "weights": tf.ones((10, 10, 10)),
      },
      {
          "testcase_name": "dimension_mismatch",
          "values": tf.ones((10, 10)),
          "weights": tf.ones((10, 7)),
      },
  )
  def test_invalid_weight_shape(self, values, weights):
    metric = variance.Variance()
    with self.assertRaises(tf.errors.InvalidArgumentError):
      metric(values, weights)

  def test_name(self):
    metric = variance.Variance(name="test_name")
    self.assertEqual("test_name", metric.name)

  def test_multiple_result_calls(self):
    metric = variance.Variance()

    values = tf.constant([1, 2, 1, 4])
    metric.update_state(values)

    self.assertAllClose(values.numpy().var(), metric.result())
    self.assertAllClose(values.numpy().var(), metric.result())

    metric.update_state(tf.constant([-1, -2, 0]))

    self.assertAllClose(
        np.array([1, 2, 1, 4, -1, -2, 0]).var(), metric.result()
    )

  def test_reset_state(self):
    metric = variance.Variance()
    values = tf.constant([1, 2, 1, 4])

    metric.update_state(values)
    self.assertAllClose(1.5, metric.result())

    metric.reset_state()

    metric.update_state(values, sample_weight=tf.constant([1, 0, 1, 0]))
    self.assertAllClose(0.0, metric.result())

  def test_numpy_correctness(self):
    metric = variance.Variance()

    values = np.array([-1.3, 2.4, 1, 4])
    weights = np.array([0.7, 0, 1.3, 1.0])

    metric.update_state(values, weights)

    expected_variance = self._compute_variance(
        tf.convert_to_tensor(values), tf.convert_to_tensor(weights)
    )
    self.assertAllClose(expected_variance, metric.result())

  def test_metric_config(self):
    metric = variance.Variance()
    self.assertLayerConfigurable(layer=metric)


if __name__ == "__main__":
  tf.test.main()
