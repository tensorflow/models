# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for treatment_sliced_metric."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras
from official.recommendation.uplift import keras_test_case
from official.recommendation.uplift.metrics import treatment_sliced_metric


class MeanSquared(tf_keras.metrics.Mean):

  def result(self):
    mean = super().result()
    return {
        self.name + "/mean": mean,
        self.name + "/squared": tf.math.square(mean),
    }


# TODO(b/271487910): Add test case to ensure the right inputs are passed to the
# sliced metrics.
class TreatmentSlicedMetricTest(
    keras_test_case.KerasTestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      {
          "testcase_name": "unweighted",
          "values": tf.constant([0, 1, 5, 6]),
          "is_treatment": tf.constant([1, 0, 1, 0]),
          "sample_weight": None,
          "expected_result": {
              "test/mean": 3.0,
              "test/mean/control": 3.5,
              "test/mean/treatment": 2.5,
          },
      },
      {
          "testcase_name": "weighted",
          "values": tf.constant([0, 1, 5, 6, -7]),
          "is_treatment": tf.constant([1, 0, 1, 1, 0]),
          "sample_weight": tf.constant([0.5, 0.5, 0, 0.7, 1.8]),
          "expected_result": {
              "test/mean": np.average(
                  np.array([0, 1, 5, 6, -7]),
                  weights=np.array([0.5, 0.5, 0, 0.7, 1.8]),
              ),
              "test/mean/control": np.average(
                  np.array([1, -7]), weights=np.array([0.5, 1.8])
              ),
              "test/mean/treatment": np.average(
                  np.array([0, 5, 6]), weights=np.array([0.5, 0, 0.7])
              ),
          },
      },
      {
          "testcase_name": "only_control",
          "values": tf.constant([[0], [1], [5]]),
          "is_treatment": tf.constant([[0], [0], [0]]),
          "sample_weight": tf.constant([1, 0, 1]),
          "expected_result": {
              "test/mean": 2.5,
              "test/mean/control": 2.5,
              "test/mean/treatment": 0.0,
          },
      },
      {
          "testcase_name": "only_treatment",
          "values": tf.constant([[0], [1], [5]]),
          "is_treatment": tf.constant([[1], [1], [1]]),
          "sample_weight": tf.constant([0, 1, 1]),
          "expected_result": {
              "test/mean": 3.0,
              "test/mean/control": 0.0,
              "test/mean/treatment": 3.0,
          },
      },
  )
  def test_treatment_sliced_metric(
      self, values, is_treatment, sample_weight, expected_result
  ):
    sliced_metric = treatment_sliced_metric.TreatmentSlicedMetric(
        metric=tf_keras.metrics.Mean(name="test/mean")
    )
    sliced_metric(values, is_treatment, sample_weight=sample_weight)
    self.assertDictEqual(expected_result, sliced_metric.result())

  def test_multiple_batches(self):
    sliced_metric = treatment_sliced_metric.TreatmentSlicedMetric(
        metric=tf_keras.metrics.Mean(name="test/mean")
    )

    sliced_metric(
        values=tf.constant([[1], [2], [4]]),
        is_treatment=tf.ones((3, 1)),
        sample_weight=None,
    )
    sliced_metric(
        values=tf.constant([[-3], [0], [5]]),
        is_treatment=tf.zeros((3, 1)),
        sample_weight=None,
    )
    sliced_metric(
        values=tf.constant([[0], [1], [-5]]),
        is_treatment=tf.constant([1, 0, 1]),
        sample_weight=tf.constant([0.3, 0.25, 0.7]),
    )

    expected_results = {
        "test/mean": np.average(
            np.array([1, 2, 4, -3, 0, 5, 0, 1, -5]),
            weights=np.array([1, 1, 1, 1, 1, 1, 0.3, 0.25, 0.7]),
        ),
        "test/mean/control": np.average(
            np.array([-3, 0, 5, 1]), weights=np.array([1, 1, 1, 0.25])
        ),
        "test/mean/treatment": np.average(
            np.array([1, 2, 4, 0, -5]), weights=np.array([1, 1, 1, 0.3, 0.7])
        ),
    }
    self.assertDictEqual(expected_results, sliced_metric.result())

  def test_metric_states(self):
    sliced_metric = treatment_sliced_metric.TreatmentSlicedMetric(
        metric=tf_keras.metrics.Mean(name="test/mean")
    )

    expected_initial_result = {
        "test/mean": 0.0,
        "test/mean/control": 0.0,
        "test/mean/treatment": 0.0,
    }
    self.assertDictEqual(expected_initial_result, sliced_metric.result())

    sliced_metric(tf.constant([1, 2, 6]), tf.constant([1, 0, 1]))
    self.assertDictEqual(
        {
            "test/mean": 3.0,
            "test/mean/control": 2.0,
            "test/mean/treatment": 3.5,
        },
        sliced_metric.result(),
    )

    sliced_metric.reset_state()
    self.assertDictEqual(expected_initial_result, sliced_metric.result())

  def test_metric_config(self):
    sliced_metric = treatment_sliced_metric.TreatmentSlicedMetric(
        metric=tf_keras.metrics.BinaryCrossentropy(
            name="loss/bc", from_logits=True
        )
    )
    self.assertLayerConfigurable(layer=sliced_metric)

  def test_multi_output_result(self):
    sliced_metric = treatment_sliced_metric.TreatmentSlicedMetric(
        metric=MeanSquared(name="test_metric")
    )

    x1 = np.array([1, 2, 3])
    x2 = np.array([-1, 4, -6])
    x = np.concatenate([x1, x2], axis=0)

    sliced_metric(tf.convert_to_tensor(x1), tf.zeros((3, 1)))
    sliced_metric(tf.convert_to_tensor(x2), tf.ones((3, 1)))

    expected_result = {
        "test_metric/mean": x.mean(),
        "test_metric/squared": x.mean() ** 2,
        "test_metric/mean/control": x1.mean(),
        "test_metric/squared/control": x1.mean() ** 2,
        "test_metric/mean/treatment": x2.mean(),
        "test_metric/squared/treatment": x2.mean() ** 2,
    }
    self.assertDictEqual(expected_result, sliced_metric.result())


if __name__ == "__main__":
  tf.test.main()
