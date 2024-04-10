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

"""Tests for loss_metric."""

from typing import Callable

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.recommendation.uplift import keras_test_case
from official.recommendation.uplift import types
from official.recommendation.uplift.metrics import loss_metric


class LossMetricTest(keras_test_case.KerasTestCase, parameterized.TestCase):

  def _get_outputs(
      self,
      true_logits: tf.Tensor,
      true_predictions: tf.Tensor,
      is_treatment: tf.Tensor,
  ) -> types.TwoTowerTrainingOutputs:
    # Only the true_logits, true_predictions and is_treatment tensors are used
    # for testing.
    return types.TwoTowerTrainingOutputs(
        shared_embedding=tf.ones_like(is_treatment),
        control_predictions=tf.ones_like(is_treatment),
        treatment_predictions=tf.ones_like(is_treatment),
        uplift=tf.ones_like(is_treatment),
        control_logits=tf.ones_like(is_treatment),
        treatment_logits=tf.ones_like(is_treatment),
        true_logits=true_logits,
        true_predictions=true_predictions,
        is_treatment=is_treatment,
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "unweighted",
          "loss_fn": tf_keras.losses.mean_squared_error,
          "from_logits": False,
          "y_true": tf.constant([[0], [0], [2], [2]]),
          "y_pred": tf.constant([[1], [2], [3], [4]]),
          "is_treatment": tf.constant([[True], [False], [True], [False]]),
          "sample_weight": None,
          "expected_losses": {
              "loss": 2.5,
              "loss/control": 4.0,
              "loss/treatment": 1.0,
          },
      },
      {
          "testcase_name": "unweighted_metric",
          "loss_fn": tf_keras.metrics.MeanSquaredError(name="loss"),
          "from_logits": False,
          "y_true": tf.constant([[0], [0], [2], [2]]),
          "y_pred": tf.constant([[1], [2], [3], [4]]),
          "is_treatment": tf.constant([[True], [False], [True], [False]]),
          "sample_weight": None,
          "expected_losses": {
              "loss": 2.5,
              "loss/control": 4.0,
              "loss/treatment": 1.0,
          },
      },
      {
          "testcase_name": "weighted",
          "loss_fn": tf_keras.losses.mean_absolute_error,
          "from_logits": False,
          "y_true": tf.constant([[0], [0], [2], [7]]),
          "y_pred": tf.constant([[1], [2], [3], [4]]),
          "is_treatment": tf.constant([[True], [False], [True], [False]]),
          "sample_weight": tf.constant([0.5, 0.5, 0.7, 1.8]),
          "expected_losses": {
              "loss": np.average([1, 2, 1, 3], weights=[0.5, 0.5, 0.7, 1.8]),
              "loss/control": np.average([2, 3], weights=[0.5, 1.8]),
              "loss/treatment": 1.0,
          },
      },
      {
          "testcase_name": "weighted_keras_metric",
          "loss_fn": tf_keras.metrics.MeanAbsoluteError(name="loss"),
          "from_logits": False,
          "y_true": tf.constant([[0], [0], [2], [7]]),
          "y_pred": tf.constant([[1], [2], [3], [4]]),
          "is_treatment": tf.constant([[True], [False], [True], [False]]),
          "sample_weight": tf.constant([[0.5], [0.5], [0.7], [1.8]]),
          "expected_losses": {
              "loss": np.average([1, 2, 1, 3], weights=[0.5, 0.5, 0.7, 1.8]),
              "loss/control": np.average([2, 3], weights=[0.5, 1.8]),
              "loss/treatment": 1.0,
          },
      },
      {
          "testcase_name": "only_control",
          "loss_fn": tf_keras.metrics.mean_squared_error,
          "from_logits": False,
          "y_true": tf.constant([[0], [1], [5]]),
          "y_pred": tf.constant([[1], [2], [5]]),
          "is_treatment": tf.constant([[False], [False], [False]]),
          "sample_weight": tf.constant([1, 0, 1]),
          "expected_losses": {
              "loss": 0.5,
              "loss/control": 0.5,
              "loss/treatment": 0.0,
          },
      },
      {
          "testcase_name": "only_control_metric",
          "loss_fn": tf_keras.metrics.MeanSquaredError(name="loss"),
          "from_logits": False,
          "y_true": tf.constant([[0], [1], [5]]),
          "y_pred": tf.constant([[1], [2], [5]]),
          "is_treatment": tf.constant([[False], [False], [False]]),
          "sample_weight": tf.constant([1, 0, 1]),
          "expected_losses": {
              "loss": 0.5,
              "loss/control": 0.5,
              "loss/treatment": 0.0,
          },
      },
      {
          "testcase_name": "only_treatment",
          "loss_fn": tf_keras.metrics.mean_absolute_error,
          "from_logits": False,
          "y_true": tf.constant([[0], [1], [5]]),
          "y_pred": tf.constant([[1], [2], [5]]),
          "is_treatment": tf.constant([[True], [True], [True]]),
          "sample_weight": tf.constant([1, 0, 1]),
          "expected_losses": {
              "loss": 0.5,
              "loss/control": 0.0,
              "loss/treatment": 0.5,
          },
      },
      {
          "testcase_name": "only_treatment_metric",
          "loss_fn": tf_keras.metrics.MeanAbsoluteError(name="loss"),
          "from_logits": False,
          "y_true": tf.constant([[0], [1], [5]]),
          "y_pred": tf.constant([[1], [2], [5]]),
          "is_treatment": tf.constant([[True], [True], [True]]),
          "sample_weight": tf.constant([1, 0, 1]),
          "expected_losses": {
              "loss": 0.5,
              "loss/control": 0.0,
              "loss/treatment": 0.5,
          },
      },
      {
          "testcase_name": "one_entry",
          "loss_fn": tf.nn.log_poisson_loss,
          "from_logits": True,
          "y_true": tf.constant([0.2]),
          "y_pred": tf.constant([2.5]),
          "is_treatment": tf.constant([True]),
          "sample_weight": tf.constant([1]),
          "expected_losses": {
              "loss": tf.nn.log_poisson_loss(
                  tf.constant([0.2]), tf.constant([2.5])
              ),
              "loss/control": 0.0,
              "loss/treatment": tf.nn.log_poisson_loss(
                  tf.constant([0.2]), tf.constant([2.5])
              ),
          },
      },
      {
          "testcase_name": "no_entry",
          "loss_fn": tf_keras.losses.binary_crossentropy,
          "from_logits": True,
          "y_true": tf.constant([[]]),
          "y_pred": tf.constant([[]]),
          "is_treatment": tf.constant([[]]),
          "sample_weight": tf.constant([[]]),
          "expected_losses": {
              "loss": 0.0,
              "loss/control": 0.0,
              "loss/treatment": 0.0,
          },
      },
      {
          "testcase_name": "no_entry_metric",
          "loss_fn": tf_keras.metrics.BinaryCrossentropy(name="loss"),
          "from_logits": False,
          "y_true": tf.constant([[]]),
          "y_pred": tf.constant([[]]),
          "is_treatment": tf.constant([[]]),
          "sample_weight": tf.constant([[]]),
          "expected_losses": {
              "loss": 0.0,
              "loss/control": 0.0,
              "loss/treatment": 0.0,
          },
      },
      {
          "testcase_name": "auc_metric",
          "loss_fn": tf_keras.metrics.AUC(from_logits=True, name="loss"),
          "from_logits": True,
          "y_true": tf.constant([[0], [0], [1], [1]]),
          "y_pred": tf.constant([[0], [0.5], [0.3], [0.9]]),
          "is_treatment": tf.constant([[1], [1], [1], [1]]),
          "sample_weight": None,
          "expected_losses": {
              "loss": 0.75,
              "loss/control": 0.0,
              "loss/treatment": 0.75,
          },
      },
      {
          "testcase_name": "loss_fn_with_from_logits",
          "loss_fn": tf_keras.losses.binary_crossentropy,
          "from_logits": True,
          "y_true": tf.constant([[0.0, 1.0]]),
          "y_pred": tf.constant([[0.0, 1.0]]),
          "is_treatment": tf.constant([[0], [0]]),
          "sample_weight": None,
          "expected_losses": {
              "loss": 0.50320446,
              "loss/control": 0.50320446,
              "loss/treatment": 0.0,
          },
      },
      {
          "testcase_name": "no_treatment_slice",
          "loss_fn": tf_keras.losses.binary_crossentropy,
          "from_logits": True,
          "y_true": tf.constant([[0.0, 1.0]]),
          "y_pred": tf.constant([[0.0, 1.0]]),
          "is_treatment": tf.constant([[0], [0]]),
          "sample_weight": None,
          "expected_losses": 0.50320446,
          "slice_by_treatment": False,
      },
      {
          "testcase_name": "no_treatment_slice_metric",
          "loss_fn": tf_keras.metrics.BinaryCrossentropy(from_logits=False),
          "from_logits": False,
          "y_true": tf.constant([[0.0, 1.0]]),
          "y_pred": tf.constant([[0.0, 1.0]]),
          "is_treatment": tf.constant([[0], [0]]),
          "sample_weight": None,
          "expected_losses": 0,
          "slice_by_treatment": False,
      },
  )
  def test_metric_computes_sliced_losses(
      self,
      loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
      from_logits: bool,
      y_true: tf.Tensor,
      y_pred: tf.Tensor,
      is_treatment: tf.Tensor,
      sample_weight: tf.Tensor | None,
      expected_losses: float | dict[str, float],
      slice_by_treatment: bool = True,
  ):
    if from_logits:
      true_logits = y_pred
      true_predictions = tf.zeros_like(y_pred)  # Irrelevant for testing.
    else:
      true_logits = tf.zeros_like(y_pred)  # Irrelevant for testing.
      true_predictions = y_pred

    metric = loss_metric.LossMetric(
        loss_fn=loss_fn,
        from_logits=from_logits,
        slice_by_treatment=slice_by_treatment,
    )
    outputs = self._get_outputs(
        true_logits=true_logits,
        true_predictions=true_predictions,
        is_treatment=is_treatment,
    )
    metric(y_true, outputs, sample_weight=sample_weight)
    self.assertEqual(expected_losses, metric.result())

  def test_metric_with_y_pred_tensor(self):
    y_true = tf.constant([[0], [0], [2], [7]])
    y_pred = tf.constant([[1], [2], [3], [4]])
    sample_weight = tf.constant([[0.5], [0.5], [0.7], [1.8]])

    metric = loss_metric.LossMetric(
        loss_fn=tf_keras.metrics.mae, slice_by_treatment=False
    )
    metric(y_true, y_pred, sample_weight)

    expected_loss = np.average([1, 2, 1, 3], weights=[0.5, 0.5, 0.7, 1.8])
    self.assertAllClose(expected_loss, metric.result())

  def test_multiple_update_batches_returns_aggregated_sliced_losses(self):
    metric = loss_metric.LossMetric(
        loss_fn=tf_keras.losses.mean_absolute_error,
        from_logits=False,
        name="mean_absolute_error",
    )

    metric.update_state(
        y_true=tf.constant([[0], [0], [2]]),
        y_pred=self._get_outputs(
            true_logits=tf.constant([[1], [2], [4]]),
            true_predictions=tf.constant([[1], [2], [4]]),
            is_treatment=tf.constant([[True], [True], [True]]),
        ),
        sample_weight=None,
    )
    metric.update_state(
        y_true=tf.constant([[0], [1], [5]]),
        y_pred=self._get_outputs(
            true_logits=tf.constant([[-3], [0], [5]]),
            true_predictions=tf.constant([[-3], [0], [5]]),
            is_treatment=tf.constant([[False], [False], [False]]),
        ),
        sample_weight=None,
    )
    metric.update_state(
        y_true=tf.constant([[2], [3], [-4]]),
        y_pred=self._get_outputs(
            true_logits=tf.constant([[0], [1], [-5]]),
            true_predictions=tf.constant([[0], [1], [-5]]),
            is_treatment=tf.constant([[True], [False], [True]]),
        ),
        sample_weight=tf.constant([0.3, 0.25, 0.7]),
    )

    expected_results = {
        "mean_absolute_error": np.average(
            [1, 2, 2, 3, 1, 0, 2, 2, 1],
            weights=[1, 1, 1, 1, 1, 1, 0.3, 0.25, 0.7],
        ),
        "mean_absolute_error/control": np.average(
            [3, 1, 0, 2], weights=[1, 1, 1, 0.25]
        ),
        "mean_absolute_error/treatment": np.average(
            [1, 2, 2, 2, 1], weights=[1, 1, 1, 0.3, 0.7]
        ),
    }
    self.assertEqual(expected_results, metric.result())

  def test_initial_and_reset_state_return_zero_losses(self):
    metric = loss_metric.LossMetric(
        tf_keras.losses.binary_crossentropy, from_logits=True
    )

    expected_initial_result = {
        "loss": 0.0,
        "loss/control": 0.0,
        "loss/treatment": 0.0,
    }
    self.assertEqual(expected_initial_result, metric.result())

    metric.update_state(
        y_true=tf.constant([[1], [1], [0]]),
        y_pred=self._get_outputs(
            true_logits=tf.constant([[2.3], [0.5], [-3.3]]),
            true_predictions=tf.random.normal((3, 1)),  # Will not be used.
            is_treatment=tf.constant([[True], [False], [True]]),
        ),
    )
    metric.reset_states()
    self.assertEqual(expected_initial_result, metric.result())

  @parameterized.product(
      loss_fn=(
          tf_keras.losses.binary_crossentropy,
          tf_keras.metrics.BinaryCrossentropy(
              from_logits=True, name="bce_loss"
          ),
      ),
      slice_by_treatment=(True, False),
  )
  def test_metric_is_configurable(
      self,
      loss_fn: (
          Callable[[tf.Tensor, tf.Tensor], tf.Tensor] | tf_keras.metrics.Metric
      ),
      slice_by_treatment: bool,
  ):
    metric = loss_metric.LossMetric(
        loss_fn,
        from_logits=True,
        slice_by_treatment=slice_by_treatment,
        name="bce_loss",
    )
    self.assertLayerConfigurable(
        layer=metric,
        y_true=tf.constant([[1], [1], [0]]),
        y_pred=self._get_outputs(
            true_logits=tf.constant([[2.3], [0.5], [-3.3]]),
            true_predictions=tf.constant([[2.3], [0.5], [-3.3]]),
            is_treatment=tf.constant([[True], [False], [True]]),
        ),
        serializable=True,
    )

  def test_invalid_prediction_tensor_type_raises_type_error(self):
    metric = loss_metric.LossMetric(
        tf_keras.metrics.mean_absolute_percentage_error
    )
    y_true = tf.ones((3, 1))
    y_pred = types.TwoTowerNetworkOutputs(
        shared_embedding=tf.ones((3, 5)),
        control_logits=tf.ones((3, 1)),
        treatment_logits=tf.ones((3, 1)),
    )
    with self.assertRaisesRegex(
        TypeError,
        "y_pred must be of type `TwoTowerTrainingOutputs`, `tf.Tensor` or"
        " `np.ndarray`",
    ):
      metric.update_state(y_true=y_true, y_pred=y_pred)

  def test_slice_by_treatment_with_y_pred_tensor_raises_error(self):
    metric = loss_metric.LossMetric(
        tf_keras.metrics.mae, slice_by_treatment=True
    )
    with self.assertRaisesRegex(
        ValueError,
        "`slice_by_treatment` must be False when y_pred is a `tf.Tensor`.",
    ):
      metric.update_state(y_true=tf.ones((3, 1)), y_pred=tf.ones((3, 1)))

  def test_passing_loss_object_raises_error(self):
    with self.assertRaisesRegex(
        TypeError, "`loss_fn` cannot be a Keras `Loss` object"
    ):
      loss_metric.LossMetric(loss_fn=tf_keras.losses.MeanAbsoluteError())

  def test_conflicting_from_logits_values_raises_error(self):
    with self.assertRaises(ValueError):
      loss_metric.LossMetric(
          loss_fn=tf_keras.metrics.BinaryCrossentropy(from_logits=True),
          from_logits=False,
      )


if __name__ == "__main__":
  tf.test.main()
