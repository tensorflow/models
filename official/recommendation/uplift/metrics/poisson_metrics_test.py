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

"""Tests for poisson regression metrics."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.recommendation.uplift import keras_test_case
from official.recommendation.uplift import types
from official.recommendation.uplift.metrics import poisson_metrics


def _get_two_tower_outputs(
    true_logits: tf.Tensor, is_treatment: tf.Tensor
) -> types.TwoTowerTrainingOutputs:
  # Only the true_logits and is_treatment tensors are needed for testing.
  return types.TwoTowerTrainingOutputs(
      shared_embedding=tf.ones_like(is_treatment),
      control_predictions=tf.ones_like(is_treatment),
      treatment_predictions=tf.ones_like(is_treatment),
      uplift=tf.ones_like(is_treatment),
      control_logits=tf.ones_like(is_treatment),
      treatment_logits=tf.ones_like(is_treatment),
      true_logits=true_logits,
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


if __name__ == "__main__":
  tf.test.main()
