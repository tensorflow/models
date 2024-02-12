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

"""Tests for treatment_fraction."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras
from official.recommendation.uplift import keras_test_case
from official.recommendation.uplift import types
from official.recommendation.uplift.metrics import treatment_fraction


class TreatmentFractionTest(
    keras_test_case.KerasTestCase, parameterized.TestCase
):

  def _get_y_pred(
      self, is_treatment: tf.Tensor
  ) -> types.TwoTowerTrainingOutputs:
    # Only the is_treatment tensor is required for testing.
    return types.TwoTowerTrainingOutputs(
        shared_embedding=tf.ones_like(is_treatment),
        control_predictions=tf.ones_like(is_treatment),
        treatment_predictions=tf.ones_like(is_treatment),
        uplift=tf.ones_like(is_treatment),
        control_logits=tf.ones_like(is_treatment),
        treatment_logits=tf.ones_like(is_treatment),
        true_logits=tf.ones_like(is_treatment),
        true_predictions=tf.ones_like(is_treatment),
        is_treatment=is_treatment,
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "unweighted",
          "is_treatment": tf.constant([[True], [False], [True], [False]]),
          "sample_weight": None,
          "expected_result": 0.5,
      },
      {
          "testcase_name": "weighted",
          "is_treatment": tf.constant(
              [[True], [False], [True], [True], [False]]
          ),
          "sample_weight": tf.constant([0.5, 0.5, 0, 0.7, 1.8]),
          "expected_result": np.average(
              [1, 0, 1, 1, 0], weights=[0.5, 0.5, 0, 0.7, 1.8]
          ),
      },
      {
          "testcase_name": "only_control",
          "is_treatment": tf.constant([[False], [False], [False]]),
          "sample_weight": tf.constant([1, 0, 1]),
          "expected_result": 0.0,
      },
      {
          "testcase_name": "only_treatment",
          "is_treatment": tf.constant([[True], [True], [True]]),
          "sample_weight": tf.constant([0, 1, 1]),
          "expected_result": 1.0,
      },
      {
          "testcase_name": "one_entry",
          "is_treatment": tf.constant([True]),
          "sample_weight": None,
          "expected_result": 1.0,
      },
      {
          "testcase_name": "no_entry",
          "is_treatment": tf.constant([], dtype=tf.bool),
          "sample_weight": tf.constant([]),
          "expected_result": 0.0,
      },
  )
  def test_treatment_fraction_computes_weighted_mean_of_is_treatment_tensor(
      self, is_treatment, sample_weight, expected_result
  ):
    metric = treatment_fraction.TreatmentFraction()
    y_true = tf.zeros_like(is_treatment)
    y_pred = self._get_y_pred(is_treatment)
    metric.update_state(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
    )
    self.assertEqual(expected_result, metric.result())

  def test_multiple_update_batches_returns_aggregated_treatment_fractions(self):
    metric = treatment_fraction.TreatmentFraction()

    metric.update_state(
        y_true=tf.zeros(3),
        y_pred=self._get_y_pred(tf.constant([[True], [True], [True]])),
        sample_weight=None,
    )
    metric.update_state(
        y_true=tf.zeros(3),
        y_pred=self._get_y_pred(tf.constant([[False], [False], [False]])),
        sample_weight=None,
    )
    metric.update_state(
        y_true=tf.zeros(3),
        y_pred=self._get_y_pred(tf.constant([[True], [False], [True]])),
        sample_weight=tf.constant([0.3, 0.25, 0.7]),
    )

    expected_treatment_fraction = np.average(
        [1, 1, 1, 0, 0, 0, 1, 0, 1], weights=[1, 1, 1, 1, 1, 1, 0.3, 0.25, 0.7]
    )
    self.assertEqual(expected_treatment_fraction, metric.result())

  def test_initial_and_reset_state_return_zero_treatment_fraction(self):
    metric = treatment_fraction.TreatmentFraction()
    self.assertEqual(0.0, metric.result())

    metric(
        y_true=tf.zeros(3),
        y_pred=self._get_y_pred(tf.constant([[True], [False], [True]])),
    )
    self.assertEqual(2 / 3, metric.result())

    metric.reset_states()
    self.assertEqual(0.0, metric.result())

  def test_metric_config_is_serializable(self):
    metric = treatment_fraction.TreatmentFraction(
        name="test_name", dtype=tf.float16
    )
    y_pred = self._get_y_pred(
        is_treatment=tf.constant([[True], [False], [True], [False]]),
    )
    self.assertLayerConfigurable(
        layer=metric, y_true=tf.zeros(4), y_pred=y_pred, serializable=True
    )

  def test_invalid_prediction_tensor_type_raises_type_error(self):
    metric = treatment_fraction.TreatmentFraction()

    with self.assertRaisesRegex(
        TypeError, "y_pred must be of type `TwoTowerTrainingOutputs`"
    ):
      metric.update_state(y_true=tf.ones((3, 1)), y_pred=tf.ones((3, 1)))


if __name__ == "__main__":
  tf.test.main()
