# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for poisson regression metrics."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.recommendation.uplift import keras_test_case
from official.recommendation.uplift import types
from official.recommendation.uplift.metrics import poisson_metrics


def _get_two_tower_outputs(
    is_treatment: tf.Tensor,
    true_logits: tf.Tensor | None = None,
) -> types.TwoTowerTrainingOutputs:
  # Only the true_logits and is_treatment tensors are needed for testing.
  return types.TwoTowerTrainingOutputs(
      shared_embedding=tf.ones_like(is_treatment),
      control_predictions=tf.ones_like(is_treatment),
      treatment_predictions=tf.ones_like(is_treatment),
      uplift=tf.ones_like(is_treatment),
      control_logits=tf.ones_like(is_treatment),
      treatment_logits=tf.ones_like(is_treatment),
      true_logits=(
          true_logits if true_logits is not None else tf.ones_like(is_treatment)
      ),
      true_predictions=tf.ones_like(is_treatment),
      is_treatment=is_treatment,
  )


class LogLossTest(keras_test_case.KerasTestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "two_tower_outputs_not_sliced",
          "from_logits": True,
          "compute_full_loss": False,
          "slice_by_treatment": False,
          "y_true": tf.constant([[0], [0], [2], [7]], dtype=tf.float32),
          "y_pred": _get_two_tower_outputs(
              true_logits=tf.constant([[1], [2], [3], [4]], dtype=tf.float32),
              is_treatment=tf.constant([[1], [0], [1], [0]]),
          ),
          "expected_loss": tf.reduce_mean(
              tf.nn.log_poisson_loss(
                  tf.constant([[0], [0], [2], [7]], dtype=tf.float32),
                  tf.constant([[1], [2], [3], [4]], dtype=tf.float32),
              )
          ),
      },
      {
          "testcase_name": "two_tower_outputs_sliced",
          "from_logits": True,
          "compute_full_loss": False,
          "slice_by_treatment": True,
          "y_true": tf.constant([[1], [0]], dtype=tf.float32),
          "y_pred": _get_two_tower_outputs(
              true_logits=tf.constant([[1], [0]], dtype=tf.float32),
              is_treatment=tf.constant([[1], [0]]),
          ),
          "expected_loss": {
              "poisson_log_loss/treatment": (
                  tf.math.exp(1.0) - 1  # exp(1) - 1 * 1
              ),
              "poisson_log_loss/control": 1.0,  # exp(0) - 0 * 0
              "poisson_log_loss": ((tf.math.exp(1.0) - 1) + 1) / 2,
          },
      },
      {
          "testcase_name": "tensor_outputs_from_logits",
          "from_logits": True,
          "compute_full_loss": False,
          "slice_by_treatment": False,
          "y_true": tf.constant([[0], [0], [2], [7]], dtype=tf.float32),
          "y_pred": tf.constant([[1], [2], [3], [4]], dtype=tf.float32),
          "expected_loss": tf.reduce_mean(
              tf.nn.log_poisson_loss(
                  tf.constant([[0], [0], [2], [7]], dtype=tf.float32),
                  tf.constant([[1], [2], [3], [4]], dtype=tf.float32),
              )
          ),
      },
      {
          "testcase_name": "tensor_outputs_from_logits_full_loss",
          "from_logits": True,
          "compute_full_loss": True,
          "slice_by_treatment": False,
          "y_true": tf.constant([[0], [0], [2], [7]], dtype=tf.float32),
          "y_pred": tf.constant([[1], [2], [3], [4]], dtype=tf.float32),
          "expected_loss": tf.reduce_mean(
              tf.nn.log_poisson_loss(
                  tf.constant([[0], [0], [2], [7]], dtype=tf.float32),
                  tf.constant([[1], [2], [3], [4]], dtype=tf.float32),
                  compute_full_loss=True,
              )
          ),
      },
      {
          "testcase_name": "tensor_outputs_from_predictions",
          "from_logits": False,
          "compute_full_loss": False,
          "slice_by_treatment": False,
          "y_true": tf.constant([[0], [0], [2], [7]], dtype=tf.float32),
          "y_pred": tf.constant([[1], [2], [3], [4]], dtype=tf.float32),
          "expected_loss": tf.reduce_mean(
              tf.nn.log_poisson_loss(
                  tf.constant([[0], [0], [2], [7]], dtype=tf.float32),
                  tf.math.log(
                      tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
                  ),
              )
          ),
      },
      {
          "testcase_name": "tensor_outputs_from_predictions_full_loss",
          "from_logits": False,
          "compute_full_loss": True,
          "slice_by_treatment": False,
          "y_true": tf.constant([[0], [0], [2], [7]], dtype=tf.float32),
          "y_pred": tf.constant([[1], [2], [3], [4]], dtype=tf.float32),
          "expected_loss": tf.reduce_mean(
              tf.nn.log_poisson_loss(
                  tf.constant([[0], [0], [2], [7]], dtype=tf.float32),
                  tf.math.log(
                      tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
                  ),
                  compute_full_loss=True,
              )
          ),
      },
  )
  def test_metric_computes_correct_loss(
      self,
      from_logits: bool,
      compute_full_loss: bool,
      slice_by_treatment: bool,
      y_true: tf.Tensor,
      y_pred: tf.Tensor,
      expected_loss: tf.Tensor,
  ):
    metric = poisson_metrics.LogLoss(
        from_logits=from_logits,
        compute_full_loss=compute_full_loss,
        slice_by_treatment=slice_by_treatment,
    )
    metric.update_state(y_true, y_pred)
    self.assertAllClose(expected_loss, metric.result())

  @parameterized.product(
      from_logits=(True, False),
      compute_full_loss=(True, False),
  )
  def test_metric_is_configurable(
      self, from_logits: bool, compute_full_loss: bool
  ):
    metric = poisson_metrics.LogLoss(
        from_logits=from_logits,
        compute_full_loss=compute_full_loss,
        slice_by_treatment=False,
    )
    self.assertLayerConfigurable(
        layer=metric,
        y_true=tf.constant([[0], [0], [2], [7]], dtype=tf.float32),
        y_pred=tf.constant([[1], [2], [3], [4]], dtype=tf.float32),
        serializable=True,
    )


class LogLossMeanBaselineTest(
    keras_test_case.KerasTestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      {
          "testcase_name": "label_zero",
          "expected_loss": 0.0,
          "y_true": tf.constant([0], dtype=tf.float32),
      },
      {
          "testcase_name": "small_positive_label",
          "expected_loss": 0.0,
          "y_true": tf.constant([1e-10], dtype=tf.float32),
      },
      {
          "testcase_name": "label_one",
          "expected_loss": 1.0,
          "y_true": tf.constant([1], dtype=tf.float32),
      },
      {
          "testcase_name": "weighted_loss",
          "expected_loss": 1.0,
          "y_true": tf.constant([[0], [1]], dtype=tf.float32),
          "sample_weight": tf.constant([[0], [1]], dtype=tf.float32),
      },
      {
          "testcase_name": "two_tower_outputs",
          "expected_loss": 0.5 - 0.5 * tf.math.log(0.5),
          "y_true": tf.constant([[0], [1]], dtype=tf.float32),
          "y_pred": _get_two_tower_outputs(
              is_treatment=tf.constant([[0], [1]], dtype=tf.float32),
          ),
      },
      {
          "testcase_name": "two_tower_outputs_sliced_loss",
          "expected_loss": {
              "loss": 0.5 - 0.5 * tf.math.log(0.5),
              "loss/control": 0.0,
              "loss/treatment": 1.0,
          },
          "y_true": tf.constant([[0], [1]], dtype=tf.float32),
          "y_pred": _get_two_tower_outputs(
              is_treatment=tf.constant([[0], [1]], dtype=tf.float32),
          ),
          "slice_by_treatment": True,
      },
  )
  def test_metric_computes_correct_loss(
      self,
      expected_loss: tf.Tensor,
      y_true: tf.Tensor,
      y_pred: types.TwoTowerTrainingOutputs | tf.Tensor | None = None,
      sample_weight: tf.Tensor | None = None,
      slice_by_treatment: bool = False,
  ):
    metric = poisson_metrics.LogLossMeanBaseline(
        slice_by_treatment=slice_by_treatment, name="loss"
    )
    metric.update_state(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(expected_loss, metric.result())

  def test_negative_label_returns_nan_loss(self):
    metric = poisson_metrics.LogLossMeanBaseline(slice_by_treatment=False)
    metric.update_state(tf.constant([-1.0]))
    self.assertTrue(tf.math.is_nan(metric.result()).numpy().item())

  def test_metric_is_configurable(self):
    metric = poisson_metrics.LogLossMeanBaseline(slice_by_treatment=False)
    self.assertLayerConfigurable(
        layer=metric,
        y_true=tf.constant([[0], [0], [2], [7]], dtype=tf.float32),
        serializable=True,
    )


class LogLossMinimumTest(keras_test_case.KerasTestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "label_zero",
          "expected_loss": 0.0,
          "y_true": tf.constant([0], dtype=tf.float32),
      },
      {
          "testcase_name": "small_positive_label",
          "expected_loss": 0.0,
          "y_true": tf.constant([1e-10], dtype=tf.float32),
      },
      {
          "testcase_name": "label_one",
          "expected_loss": 1.0,
          "y_true": tf.constant([1], dtype=tf.float32),
      },
      {
          "testcase_name": "weighted_loss",
          "expected_loss": 1.0,
          "y_true": tf.constant([[0], [1]], dtype=tf.float32),
          "sample_weight": tf.constant([[0], [1]], dtype=tf.float32),
      },
      {
          "testcase_name": "two_tower_outputs",
          "expected_loss": 0.5,
          "y_true": tf.constant([[0], [1]], dtype=tf.float32),
          "y_pred": _get_two_tower_outputs(
              is_treatment=tf.constant([[0], [1]], dtype=tf.float32),
          ),
      },
      {
          "testcase_name": "two_tower_outputs_sliced_loss",
          "expected_loss": {
              "loss": 0.5,
              "loss/control": 0.0,
              "loss/treatment": 1.0,
          },
          "y_true": tf.constant([[0], [1]], dtype=tf.float32),
          "y_pred": _get_two_tower_outputs(
              is_treatment=tf.constant([[0], [1]], dtype=tf.float32),
          ),
          "slice_by_treatment": True,
      },
  )
  def test_metric_computes_correct_loss(
      self,
      expected_loss: tf.Tensor,
      y_true: tf.Tensor,
      y_pred: types.TwoTowerTrainingOutputs | tf.Tensor | None = None,
      sample_weight: tf.Tensor | None = None,
      slice_by_treatment: bool = False,
  ):
    metric = poisson_metrics.LogLossMinimum(
        slice_by_treatment=slice_by_treatment, name="loss"
    )
    metric.update_state(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(expected_loss, metric.result())

  def test_negative_label_returns_nan_loss(self):
    metric = poisson_metrics.LogLossMinimum(slice_by_treatment=False)
    metric.update_state(tf.constant([-1.0]))
    self.assertTrue(tf.math.is_nan(metric.result()).numpy().item())

  def test_metric_is_configurable(self):
    metric = poisson_metrics.LogLossMinimum(slice_by_treatment=False)
    self.assertLayerConfigurable(
        layer=metric,
        y_true=tf.constant([[0], [0], [2], [7]], dtype=tf.float32),
        serializable=True,
    )


class PseudoRSquaredTest(keras_test_case.KerasTestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "no_data",
          "expected_loss": 0.0,
          "y_true": tf.constant([], dtype=tf.float32),
          "y_pred": tf.constant([], dtype=tf.float32),
      },
      {
          "testcase_name": "one_correct_prediction",
          "expected_loss": 0.0,
          "y_true": tf.constant([1], dtype=tf.float32),
          "y_pred": tf.constant([1], dtype=tf.float32),
      },
      {
          "testcase_name": "one_wrong_prediction",
          "expected_loss": 0.0,  # LLmax and LLbaseline are equal.
          "y_true": tf.constant([0], dtype=tf.float32),
          "y_pred": tf.constant([1], dtype=tf.float32),
      },
      {
          "testcase_name": "all_correct_predictions",
          "expected_loss": 1.0,
          "y_true": tf.constant([[1], [2], [3]], dtype=tf.float32),
          "y_pred": tf.constant([[1], [2], [3]], dtype=tf.float32),
          "from_logits": False,
      },
      {
          "testcase_name": "almost_correct_predictions",
          "expected_loss": 1.0,
          "y_true": tf.constant([[1], [2], [3]], dtype=tf.float32),
          "y_pred": tf.constant([[1], [1.9999], [3.0001]], dtype=tf.float32),
      },
      {
          "testcase_name": "from_logits",
          "expected_loss": (
              (tf.math.exp(1.0) / 2) - (0.5 - 0.5 * tf.math.log(0.5))
          ) / (0.5 - (0.5 - 0.5 * tf.math.log(0.5))),
          "y_true": tf.constant([[0], [1]], dtype=tf.float32),
          "y_pred": tf.constant([[0], [1]], dtype=tf.float32),
          "from_logits": True,
      },
      {
          "testcase_name": "two_tower_outputs",
          "expected_loss": (
              ((tf.math.exp(1.0) - 1) + 1) / 2 - (0.5 - 0.5 * tf.math.log(0.5))
          ) / (0.5 - (0.5 - 0.5 * tf.math.log(0.5))),
          "y_true": tf.constant([[0], [1]], dtype=tf.float32),
          "y_pred": _get_two_tower_outputs(
              true_logits=tf.constant([[0], [1]], dtype=tf.float32),
              is_treatment=tf.constant([[0], [1]], dtype=tf.float32),
          ),
          "from_logits": True,
      },
      {
          "testcase_name": "two_tower_outputs_sliced_loss",
          "expected_loss": {
              "r2": (
                  ((tf.math.exp(1.0) - 1) + 1) / 2  # LLfit
                  - (0.5 - 0.5 * tf.math.log(0.5))  # LLbaseline
              ) / (0.5 - (0.5 - 0.5 * tf.math.log(0.5))),
              "r2/control": 0.0,
              "r2/treatment": 0.0,
          },
          "y_true": tf.constant([[0], [1]], dtype=tf.float32),
          "y_pred": _get_two_tower_outputs(
              true_logits=tf.constant([[0], [1]], dtype=tf.float32),
              is_treatment=tf.constant([[0], [1]], dtype=tf.float32),
          ),
          "from_logits": True,
          "slice_by_treatment": True,
      },
  )
  def test_metric_computation_is_correct(
      self,
      expected_loss: tf.Tensor,
      y_true: tf.Tensor,
      y_pred: types.TwoTowerTrainingOutputs | tf.Tensor,
      sample_weight: tf.Tensor | None = None,
      from_logits: bool = False,
      slice_by_treatment: bool = False,
  ):
    metric = poisson_metrics.PseudoRSquared(
        from_logits=from_logits,
        slice_by_treatment=slice_by_treatment,
        name="r2",
    )
    metric.update_state(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(expected_loss, metric.result())

  def test_slicing_raises_error_when_input_is_tensor(self):
    metric = poisson_metrics.PseudoRSquared()
    y_true = tf.constant([[0], [0], [2], [7]], dtype=tf.float32)
    y_pred = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
    with self.assertRaisesRegex(
        ValueError,
        "`slice_by_treatment` must be set to `False` when `y_pred` is not of"
        " type `TwoTowerTrainingOutputs`.",
    ):
      metric(y_true, y_pred)

  @parameterized.parameters(True, False)
  def test_metric_is_configurable(self, from_logits: bool):
    metric = poisson_metrics.PseudoRSquared(
        from_logits=from_logits, slice_by_treatment=False
    )
    self.assertLayerConfigurable(
        layer=metric,
        y_true=tf.constant([[0], [0], [2], [7]], dtype=tf.float32),
        y_pred=tf.constant([[1], [2], [3], [4]], dtype=tf.float32),
        serializable=True,
    )


if __name__ == "__main__":
  tf.test.main()
