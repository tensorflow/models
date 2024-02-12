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

"""Tests for uplift_mean."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras
from official.recommendation.uplift import keras_test_case
from official.recommendation.uplift import types
from official.recommendation.uplift.metrics import uplift_mean


class UpliftMeanTest(keras_test_case.KerasTestCase, parameterized.TestCase):

  def _get_y_pred(
      self, uplift: tf.Tensor, is_treatment: tf.Tensor
  ) -> types.TwoTowerTrainingOutputs:
    # Only the uplift and is_treatment tensors are required for testing.
    return types.TwoTowerTrainingOutputs(
        shared_embedding=tf.ones_like(is_treatment),
        control_predictions=tf.ones_like(is_treatment),
        treatment_predictions=tf.ones_like(is_treatment),
        uplift=uplift,
        control_logits=tf.ones_like(is_treatment),
        treatment_logits=tf.ones_like(is_treatment),
        true_logits=tf.ones_like(is_treatment),
        true_predictions=tf.ones_like(is_treatment),
        is_treatment=is_treatment,
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "unweighted",
          "uplift": tf.constant([0, 1, 5, 6]),
          "is_treatment": tf.constant([[True], [False], [True], [False]]),
          "sample_weight": None,
          "expected_result": {
              "uplift/mean": 3.0,
              "uplift/mean/control": 3.5,
              "uplift/mean/treatment": 2.5,
          },
      },
      {
          "testcase_name": "weighted",
          "uplift": tf.constant([0, 1, 5, 6, -7]),
          "is_treatment": tf.constant(
              [[True], [False], [True], [True], [False]]
          ),
          "sample_weight": tf.constant([0.5, 0.5, 0, 0.7, 1.8]),
          "expected_result": {
              "uplift/mean": np.average(
                  [0, 1, 5, 6, -7], weights=[0.5, 0.5, 0, 0.7, 1.8]
              ),
              "uplift/mean/control": np.average([1, -7], weights=[0.5, 1.8]),
              "uplift/mean/treatment": np.average(
                  [0, 5, 6], weights=[0.5, 0, 0.7]
              ),
          },
      },
      {
          "testcase_name": "only_control",
          "uplift": tf.constant([[0], [1], [5]]),
          "is_treatment": tf.constant([[False], [False], [False]]),
          "sample_weight": tf.constant([1, 0, 1]),
          "expected_result": {
              "uplift/mean": 2.5,
              "uplift/mean/control": 2.5,
              "uplift/mean/treatment": 0.0,
          },
      },
      {
          "testcase_name": "only_treatment",
          "uplift": tf.constant([[0], [1], [5]]),
          "is_treatment": tf.constant([[True], [True], [True]]),
          "sample_weight": tf.constant([0, 1, 1]),
          "expected_result": {
              "uplift/mean": 3.0,
              "uplift/mean/control": 0.0,
              "uplift/mean/treatment": 3.0,
          },
      },
      {
          "testcase_name": "one_entry",
          "uplift": tf.constant([2.5]),
          "is_treatment": tf.constant([True]),
          "sample_weight": tf.constant([1]),
          "expected_result": {
              "uplift/mean": 2.5,
              "uplift/mean/control": 0.0,
              "uplift/mean/treatment": 2.5,
          },
      },
      {
          "testcase_name": "no_entry",
          "uplift": tf.constant([]),
          "is_treatment": tf.constant([], dtype=tf.bool),
          "sample_weight": tf.constant([]),
          "expected_result": {
              "uplift/mean": 0.0,
              "uplift/mean/control": 0.0,
              "uplift/mean/treatment": 0.0,
          },
      },
  )
  def test_metric_computes_sliced_uplift_means(
      self, uplift, is_treatment, sample_weight, expected_result
  ):
    metric = uplift_mean.UpliftMean()
    y_pred = self._get_y_pred(uplift=uplift, is_treatment=is_treatment)
    metric(
        y_true=tf.zeros_like(uplift), y_pred=y_pred, sample_weight=sample_weight
    )
    self.assertEqual(expected_result, metric.result())

  def test_multiple_update_batches_returns_aggregated_uplift_means(self):
    metric = uplift_mean.UpliftMean(name="uplift")

    metric.update_state(
        y_true=tf.zeros(3),
        y_pred=self._get_y_pred(
            uplift=tf.constant([[1], [2], [4]]),
            is_treatment=tf.constant([[True], [True], [True]]),
        ),
        sample_weight=None,
    )
    metric.update_state(
        y_true=tf.zeros(3),
        y_pred=self._get_y_pred(
            uplift=tf.constant([[-3], [0], [5]]),
            is_treatment=tf.constant([[False], [False], [False]]),
        ),
        sample_weight=None,
    )
    metric.update_state(
        y_true=tf.zeros(3),
        y_pred=self._get_y_pred(
            uplift=tf.constant([[0], [1], [-5]]),
            is_treatment=tf.constant([[True], [False], [True]]),
        ),
        sample_weight=tf.constant([0.3, 0.25, 0.7]),
    )

    expected_results = {
        "uplift": np.average(
            [1, 2, 4, -3, 0, 5, 0, 1, -5],
            weights=[1, 1, 1, 1, 1, 1, 0.3, 0.25, 0.7],
        ),
        "uplift/control": np.average([-3, 0, 5, 1], weights=[1, 1, 1, 0.25]),
        "uplift/treatment": np.average(
            [1, 2, 4, 0, -5], weights=[1, 1, 1, 0.3, 0.7]
        ),
    }
    self.assertEqual(expected_results, metric.result())

  def test_initial_and_reset_state_return_zero_uplift_means(self):
    metric = uplift_mean.UpliftMean()

    expected_initial_result = {
        "uplift/mean": 0.0,
        "uplift/mean/control": 0.0,
        "uplift/mean/treatment": 0.0,
    }
    self.assertEqual(expected_initial_result, metric.result())

    metric(
        y_true=tf.zeros(3),
        y_pred=self._get_y_pred(
            uplift=tf.constant([1, 2, 6]),
            is_treatment=tf.constant([[True], [False], [True]]),
        ),
    )
    self.assertEqual(
        {
            "uplift/mean": 3.0,
            "uplift/mean/control": 2.0,
            "uplift/mean/treatment": 3.5,
        },
        metric.result(),
    )

    metric.reset_states()
    self.assertEqual(expected_initial_result, metric.result())

  def test_metric_config_is_serializable(self):
    metric = uplift_mean.UpliftMean(name="test_name", dtype=tf.float16)
    y_pred = self._get_y_pred(
        uplift=tf.constant([[1], [2], [3], [4]]),
        is_treatment=tf.constant([[True], [False], [True], [False]]),
    )
    self.assertLayerConfigurable(
        layer=metric, y_true=tf.zeros(4), y_pred=y_pred, serializable=True
    )

  def test_invalid_prediction_tensor_type_raises_type_error(self):
    metric = uplift_mean.UpliftMean()

    with self.assertRaisesRegex(
        TypeError, "y_pred must be of type `TwoTowerTrainingOutputs`"
    ):
      metric.update_state(y_true=tf.ones((3, 1)), y_pred=tf.ones((3, 1)))


if __name__ == "__main__":
  tf.test.main()
