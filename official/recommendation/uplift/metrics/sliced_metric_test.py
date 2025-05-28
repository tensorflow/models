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

"""Tests for sliced metrics."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras
from official.recommendation.uplift import keras_test_case
from official.recommendation.uplift.metrics import sliced_metric


class MeanSquared(tf_keras.metrics.Mean):

  def result(self):
    mean = super().result()
    return {
        self.name + "/mean": mean,
        self.name + "/squared": tf.math.square(mean),
    }


class SlicedMetricTest(keras_test_case.KerasTestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "bool_slicing",
          "labels": tf.constant([0, 1, 0, 1], dtype=tf.int32),
          "predictions": tf.constant([1, 0, 1, 1], dtype=tf.int32),
          "slicing_spec": {"treatment": True, "control": False},
          "slicing_feature": tf.constant([True, False, True, False]),
          "expected_result": {
              "accuracy": 0.25,
              "accuracy/treatment": 0,
              "accuracy/control": 0.5,
          },
      },
      {
          "testcase_name": "int_slicing",
          "labels": tf.constant([0, 1, 0, 1], dtype=tf.int32),
          "predictions": tf.constant([1, 0, 1, 1], dtype=tf.int32),
          "slicing_spec": {"app_usage": 3, "install": 4, "purchase": 5},
          "slicing_feature": tf.constant([4, 5, 4, 3], dtype=tf.int32),
          "expected_result": {
              "accuracy": 0.25,
              "accuracy/install": 0,
              "accuracy/purchase": 0,
              "accuracy/app_usage": 1,
          },
      },
      {
          "testcase_name": "str_slicing",
          "labels": tf.constant([0, 1, 0, 1], dtype=tf.int32),
          "predictions": tf.constant([1, 0, 1, 1], dtype=tf.int32),
          "slicing_spec": {"install": "install", "purchase": "purchase"},
          "slicing_feature": tf.constant(
              ["install", "purchase", "install", "app_usage"]
          ),
          "expected_result": {
              "accuracy": 0.25,
              "accuracy/install": 0,
              "accuracy/purchase": 0,
          },
      },
      {
          "testcase_name": "int_slicing_weighted",
          "labels": tf.constant([0, 1, 0, 1], dtype=tf.int32),
          "predictions": tf.constant([1, 0, 1, 1], dtype=tf.int32),
          "slicing_spec": {"app_usage": 3, "install": 4, "purchase": 5},
          "slicing_feature": tf.constant([4, 5, 4, 3], dtype=tf.int32),
          "weights": tf.constant([0, 0, 1, 1], dtype=tf.float32),
          "expected_result": {
              "accuracy": 0.5,
              "accuracy/install": 0,
              "accuracy/purchase": 0,
              "accuracy/app_usage": 1,
          },
      },
  )
  def test_binary_sliced_metric(
      self,
      labels: tf.Tensor,
      predictions: tf.Tensor,
      slicing_spec: dict[str, int | str | bool],
      slicing_feature: tf.Tensor,
      expected_result: dict[str, float],
      weights: tf.Tensor | None = None,
  ):
    metric = sliced_metric.SlicedMetric(
        tf_keras.metrics.Accuracy("accuracy"),
        slicing_spec=slicing_spec,
    )
    metric.update_state(
        labels,
        predictions,
        sample_weight=weights,
        slicing_feature=slicing_feature,
    )
    self.assertDictEqual(expected_result, metric.result())

  @parameterized.named_parameters(
      {
          "testcase_name": "bool_slicing",
          "labels": tf.constant([0, 1, 0, 1], dtype=tf.int32),
          "slicing_spec": {"treatment": True, "control": False},
          "slicing_feature": tf.constant([True, False, True, False]),
          "expected_result": {
              "mean": 0.5,
              "mean/treatment": 0,
              "mean/control": 1,
          },
      },
      {
          "testcase_name": "int_slicing",
          "labels": tf.constant([0, 1, 0, 1], dtype=tf.int32),
          "slicing_spec": {"app_usage": 3, "install": 4, "purchase": 5},
          "slicing_feature": tf.constant([4, 5, 4, 3], dtype=tf.int32),
          "expected_result": {
              "mean": 0.5,
              "mean/install": 0,
              "mean/purchase": 1,
              "mean/app_usage": 1,
          },
      },
  )
  def test_unary_sliced_metrics(
      self,
      labels: tf.Tensor,
      slicing_spec: dict[str, int | str | bool],
      slicing_feature: tf.Tensor,
      expected_result: dict[str, float],
  ):
    metric = sliced_metric.SlicedMetric(
        tf_keras.metrics.Mean("mean"),
        slicing_spec=slicing_spec,
    )
    metric.update_state(labels, slicing_feature=slicing_feature)
    self.assertDictEqual(expected_result, metric.result())

  @parameterized.named_parameters(
      {
          "testcase_name": "int_slicing",
          "labels": tf.constant([0, 1, 0, 1], dtype=tf.int32),
          "slicing_spec": {"app_usage": 3, "install": 4, "purchase": 5},
          "slicing_feature": tf.constant([4, 5, 4, 3], dtype=tf.int32),
          "expected_result": {
              "msq/mean": 0.5,
              "msq/mean/install": 0,
              "msq/mean/purchase": 1,
              "msq/mean/app_usage": 1,
              "msq/squared": 0.25,
              "msq/squared/install": 0,
              "msq/squared/purchase": 1,
              "msq/squared/app_usage": 1,
          },
      },
      {
          "testcase_name": "str_slicing",
          "labels": tf.constant([0, 1, 0, 1], dtype=tf.int32),
          "slicing_spec": {"install": "install", "purchase": "purchase"},
          "slicing_feature": tf.constant(
              ["install", "purchase", "install", "app_usage"]
          ),
          "expected_result": {
              "msq/mean": 0.5,
              "msq/mean/install": 0,
              "msq/mean/purchase": 1,
              "msq/squared": 0.25,
              "msq/squared/install": 0,
              "msq/squared/purchase": 1,
          },
      },
  )
  def test_metric_with_dict_result(
      self,
      labels: tf.Tensor,
      slicing_spec: dict[str, int | str | bool],
      slicing_feature: tf.Tensor,
      expected_result: dict[str, float],
  ):
    metric = sliced_metric.SlicedMetric(
        MeanSquared("msq"), slicing_spec=slicing_spec
    )
    metric.update_state(labels, slicing_feature=slicing_feature)
    self.assertDictEqual(expected_result, metric.result())

  @parameterized.named_parameters(
      {
          "testcase_name": "empty_slicing_spec",
          "slicing_spec": {},
      },
      {
          "testcase_name": "incompatible_types",
          "slicing_spec": {"a": True, "b": 1, "c": 0},
      },
      {
          "testcase_name": "duplicate_slicing_values",
          "slicing_spec": {"a": True, "b": False, "c": True},
      },
  )
  def test_invalid_inputs(self, slicing_spec):
    self.assertRaises(
        ValueError,
        sliced_metric.SlicedMetric,
        tf_keras.metrics.Mean(),
        slicing_spec=slicing_spec,
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "float_to_int",
          "slicing_spec": {"a": 1, "b": 2},
          "slicing_feature": tf.constant([1, 2, 1, 0], dtype=tf.float32),
      },
      {
          "testcase_name": "int_to_bool",
          "slicing_spec": {"a": False, "b": True},
          "slicing_feature": tf.constant([1, 0, 0, 1], dtype=tf.int32),
      },
  )
  def test_invalid_update(self, slicing_spec, slicing_feature):
    # Invalid cast
    metric = sliced_metric.SlicedMetric(
        tf_keras.metrics.Mean(), slicing_spec=slicing_spec
    )
    self.assertRaises(
        ValueError,
        metric.update_state,
        values=tf.constant([1, 0, 1, 0], dtype=tf.int32),
        slicing_feature=slicing_feature,
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "inputs_2x2_slicing_feature_2",
          "slicing_feature": tf.constant([False, True]),
          "expected_result": {
              "accuracy": 0.5,
              "accuracy/control": 0.5,
              "accuracy/treatment": 0.5,
          },
      },
      {
          "testcase_name": "inputs_2x2_slicing_feature_1x2",
          "slicing_feature": tf.constant([[False, True]]),
          "expected_result": {
              "accuracy": 0.5,
              "accuracy/control": 0,
              "accuracy/treatment": 1.0,
          },
      },
      {
          "testcase_name": "inputs_2x2_slicing_feature_1x2_weight_1",
          "slicing_feature": tf.constant([[False, True]]),
          "sample_weight": tf.constant([1.0]),
          "expected_result": {
              "accuracy": 0.5,
              "accuracy/control": 0,
              "accuracy/treatment": 1.0,
          },
      },
      {
          "testcase_name": "inputs_2x2_slicing_feature_2_weight_2",
          "slicing_feature": tf.constant([False, True]),
          "sample_weight": tf.constant([0.5, 0.5]),
          "expected_result": {
              "accuracy": 0.5,
              "accuracy/control": 0.5,
              "accuracy/treatment": 0.5,
          },
      },
  )
  def test_broadcastable_weights_and_slicing_feature(
      self,
      slicing_feature: tf.Tensor,
      expected_result: dict[str, float],
      sample_weight: tf.Tensor | None = None,
  ):
    metric = sliced_metric.SlicedMetric(
        tf_keras.metrics.Accuracy("accuracy"),
        slicing_spec={"control": False, "treatment": True},
    )
    metric.update_state(
        tf.constant([[0, 1], [0, 1]], dtype=tf.int32),
        tf.constant([[1, 1], [1, 1]], dtype=tf.int32),
        sample_weight=sample_weight,
        slicing_feature=slicing_feature,
    )
    self.assertDictEqual(expected_result, metric.result())

  def test_batched_inputs(self):
    metric = sliced_metric.SlicedMetric(
        tf_keras.metrics.Accuracy("accuracy"),
        slicing_spec={"install": 4, "purchase": 5},
    )
    metric.update_state(
        tf.constant([[0, 1], [1, 0]], dtype=tf.int32),
        tf.constant([[1, 1], [1, 1]], dtype=tf.int32),
        slicing_feature=tf.constant([[4, 5], [4, 3]], dtype=tf.int32),
    )
    expected_result = {
        "accuracy": 0.5,
        "accuracy/install": 0.5,
        "accuracy/purchase": 1.0,
    }
    self.assertDictEqual(expected_result, metric.result())

  def test_reset_state(self):
    metric = sliced_metric.SlicedMetric(
        metric=tf_keras.metrics.AUC(curve="PR", from_logits=False, name="auc"),
        slicing_spec={"control": False, "treatment": True},
    )

    expected_initial_result = {
        "auc": 0.0,
        "auc/control": 0.0,
        "auc/treatment": 0.0,
    }
    self.assertAllClose(expected_initial_result, metric.result())

    metric.update_state(
        tf.constant([[0], [0], [1], [1]]),  # y_true
        tf.constant([[0.2], [0.6], [0.3], [0.7]]),  # y_pred
        slicing_feature=tf.constant([[True], [False], [True], [False]]),
    )

    result = metric.result()
    self.assertGreater(result["auc"], 0.0)
    self.assertGreater(result["auc/control"], 0.0)
    self.assertGreater(result["auc/treatment"], 0.0)

    metric.reset_state()
    self.assertAllClose(expected_initial_result, metric.result())

  def test_metric_config(self):
    metric = sliced_metric.SlicedMetric(
        tf_keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="accuracy@2"),
        slicing_spec={"a": False, "b": True},
        slicing_feature_dtype=tf.bool,
        name="sliced_accuracy",
    )
    y_true = tf.constant([1, 0, 1, 0])
    y_pred = tf.constant([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    slicing_feature = tf.constant([True, False, False, True])
    self.assertLayerConfigurable(
        metric, y_true=y_true, y_pred=y_pred, slicing_feature=slicing_feature
    )


if __name__ == "__main__":
  tf.test.main()
