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

"""Tests for two_tower_output_head."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.recommendation.uplift import keras_test_case
from official.recommendation.uplift import types
from official.recommendation.uplift.layers.uplift_networks import two_tower_output_head


_DEFAULT_SHARED_EMBEDDING = tf.ones((3, 4, 5), dtype=tf.float32)
_DEFAULT_CONTROL_LOGITS = tf.constant([-1, 0, 3], dtype=tf.float32)
_DEFAULT_TREATMENT_LOGITS = tf.constant([2, -1, 1], dtype=tf.float32)
_DEFAULT_IS_TREATMENT = tf.constant([1, 1, 0], dtype=tf.float32)


@tf_keras.utils.register_keras_serializable()
class TwoTowerUpliftNetworkMock(tf_keras.layers.Layer):

  def call(self, inputs, training=None, mask=None):
    return types.TwoTowerNetworkOutputs(
        shared_embedding=inputs["shared_embedding"],
        control_logits=inputs["control_logits"],
        treatment_logits=inputs["treatment_logits"],
    )


class TwoTowerOutputHeadTest(
    keras_test_case.KerasTestCase, parameterized.TestCase
):

  def _get_layer(self, inverse_link_fn=None):
    layer = two_tower_output_head.TwoTowerOutputHead(
        treatment_indicator_feature_name="is_treatment",
        uplift_network=TwoTowerUpliftNetworkMock(),
        inverse_link_fn=inverse_link_fn,
    )
    return layer

  def _get_inputs(
      self,
      shared_embedding=_DEFAULT_SHARED_EMBEDDING,
      control_logits=_DEFAULT_CONTROL_LOGITS,
      treatment_logits=_DEFAULT_TREATMENT_LOGITS,
      is_treatment=None,
  ):
    inputs = {
        "shared_embedding": shared_embedding,
        "control_logits": control_logits,
        "treatment_logits": treatment_logits,
    }
    if is_treatment is not None:
      inputs["is_treatment"] = is_treatment
    return inputs

  @parameterized.named_parameters(
      {
          "testcase_name": "without_inverse_link",
          "inverse_link_fn": None,
      },
      {
          "testcase_name": "with_inverse_link",
          "inverse_link_fn": tf.math.sigmoid,
      },
  )
  def test_training_outputs_are_returned_when_treatment_indicator_in_inputs(
      self, inverse_link_fn
  ):
    layer = self._get_layer(inverse_link_fn=inverse_link_fn)
    inputs = self._get_inputs(is_treatment=_DEFAULT_IS_TREATMENT)
    outputs = layer(inputs)
    self.assertIsInstance(outputs, types.TwoTowerTrainingOutputs)

  @parameterized.named_parameters(
      {
          "testcase_name": "without_inverse_link",
          "inverse_link_fn": None,
      },
      {
          "testcase_name": "with_inverse_link",
          "inverse_link_fn": tf.math.sigmoid,
      },
  )
  def test_prediction_outputs_are_returned_when_treatment_indicator_not_in_inputs(
      self, inverse_link_fn
  ):
    layer = self._get_layer(inverse_link_fn=inverse_link_fn)
    inputs = self._get_inputs(is_treatment=None)
    outputs = layer(inputs)
    self.assertIsInstance(outputs, types.TwoTowerPredictionOutputs)

  @parameterized.named_parameters(
      {
          "testcase_name": "treatment_indicator_without_inverse_link",
          "is_treatment": _DEFAULT_IS_TREATMENT,
          "inverse_link_fn": None,
      },
      {
          "testcase_name": "treatment_indicator_with_inverse_link",
          "is_treatment": _DEFAULT_IS_TREATMENT,
          "inverse_link_fn": tf.math.exp,
      },
      {
          "testcase_name": "without_treatment_indicator_and_inverse_link",
          "is_treatment": None,
          "inverse_link_fn": None,
      },
      {
          "testcase_name": "without_treatment_indicator_but_with_inverse_link",
          "is_treatment": None,
          "inverse_link_fn": tf.math.exp,
      },
  )
  def test_uplift_network_outputs_remain_unchanged(
      self, is_treatment, inverse_link_fn
  ):
    layer = self._get_layer(inverse_link_fn=inverse_link_fn)
    inputs = self._get_inputs(is_treatment=is_treatment)
    outputs = layer(inputs)

    self.assertAllEqual(_DEFAULT_SHARED_EMBEDDING, outputs.shared_embedding)
    self.assertAllEqual(_DEFAULT_CONTROL_LOGITS, outputs.control_logits)
    self.assertAllEqual(_DEFAULT_TREATMENT_LOGITS, outputs.treatment_logits)

  @parameterized.product(
      (dict(is_treatment=None), dict(is_treatment=_DEFAULT_IS_TREATMENT)),
      (
          dict(
              inverse_link_fn=None,
              expected_control_predictions=_DEFAULT_CONTROL_LOGITS,
              expected_treatment_predictions=_DEFAULT_TREATMENT_LOGITS,
              expected_uplift=_DEFAULT_TREATMENT_LOGITS
              - _DEFAULT_CONTROL_LOGITS,
          ),
          dict(
              inverse_link_fn=tf.math.exp,
              expected_control_predictions=tf.math.exp(_DEFAULT_CONTROL_LOGITS),
              expected_treatment_predictions=tf.math.exp(
                  _DEFAULT_TREATMENT_LOGITS
              ),
              expected_uplift=tf.math.exp(_DEFAULT_TREATMENT_LOGITS)
              - tf.math.exp(_DEFAULT_CONTROL_LOGITS),
          ),
      ),
  )
  def test_uplift_predictions_are_computed_from_logits(
      self,
      is_treatment,
      inverse_link_fn,
      expected_control_predictions,
      expected_treatment_predictions,
      expected_uplift,
  ):
    layer = self._get_layer(inverse_link_fn=inverse_link_fn)
    inputs = self._get_inputs(
        control_logits=_DEFAULT_CONTROL_LOGITS,
        treatment_logits=_DEFAULT_TREATMENT_LOGITS,
        is_treatment=is_treatment,
    )
    outputs = layer(inputs)

    self.assertAllClose(
        expected_control_predictions, outputs.control_predictions
    )
    self.assertAllClose(
        expected_treatment_predictions, outputs.treatment_predictions
    )
    self.assertAllClose(expected_uplift, outputs.uplift)

  def test_true_logits_correspond_to_control_and_treatment_logits(self):
    layer = self._get_layer()
    inputs = self._get_inputs(
        control_logits=tf.constant([-1, 0, 3]),
        treatment_logits=tf.constant([2, -1, 1]),
        is_treatment=tf.constant([1, 1, 0]),
    )
    outputs = layer(inputs)
    self.assertAllEqual(tf.constant([2, -1, 3]), outputs.true_logits)

  def test_true_preds_correspond_to_control_and_treatment_preds(self):
    layer = self._get_layer(inverse_link_fn=tf.nn.relu)
    inputs = self._get_inputs(
        control_logits=tf.constant([2, 0, 3]),
        treatment_logits=tf.constant([-1, 2, 1]),
        is_treatment=tf.constant([1, 1, 0]),
    )
    outputs = layer(inputs)
    self.assertAllEqual(tf.constant([0, 2, 3]), outputs.true_predictions)

  def test_is_treatment_tensor_gets_converted_to_boolean_tensor(self):
    layer = self._get_layer()
    inputs = self._get_inputs(is_treatment=tf.constant([1, 1, 0]))
    outputs = layer(inputs)
    self.assertAllEqual(tf.constant([True, True, False]), outputs.is_treatment)

  def test_true_logits_correctness_with_logits_rank_mismatch(self):
    layer = self._get_layer()
    inputs = self._get_inputs(
        control_logits=tf.constant([[2], [0], [3]]),
        treatment_logits=tf.constant([[-1], [2], [1]]),
        is_treatment=tf.constant([1, 1, 0]),
    )
    outputs = layer(inputs)
    self.assertAllEqual(tf.constant([[-1], [2], [3]]), outputs.true_predictions)

  @parameterized.product(
      (dict(is_treatment=None), dict(is_treatment=_DEFAULT_IS_TREATMENT)),
      (dict(inverse_link_fn=None), dict(inverse_link_fn=tf.math.sigmoid)),
  )
  def test_multiple_layer_calls_with_same_input_returns_same_output(
      self, is_treatment, inverse_link_fn
  ):
    layer = self._get_layer(inverse_link_fn=inverse_link_fn)
    inputs = self._get_inputs(is_treatment=is_treatment)
    self.assertLayerStable(layer=layer, inputs=inputs)

  @parameterized.product(
      (dict(is_treatment=None), dict(is_treatment=_DEFAULT_IS_TREATMENT)),
      (dict(inverse_link_fn=None), dict(inverse_link_fn=tf.math.sigmoid)),
  )
  def test_layer_saving_succeeds(self, is_treatment, inverse_link_fn):
    layer = self._get_layer(inverse_link_fn=inverse_link_fn)
    inputs = self._get_inputs(is_treatment=is_treatment)
    self.assertLayerSavable(layer=layer, inputs=inputs)

  @parameterized.product(
      (dict(is_treatment=None), dict(is_treatment=_DEFAULT_IS_TREATMENT)),
      (dict(inverse_link_fn=None), dict(inverse_link_fn=tf.math.sigmoid)),
  )
  def test_from_config_layer_returns_same_output_as_original_layer(
      self, is_treatment, inverse_link_fn
  ):
    layer = self._get_layer(inverse_link_fn=inverse_link_fn)
    inputs = self._get_inputs(is_treatment=is_treatment)
    self.assertLayerConfigurable(layer=layer, inputs=inputs)


if __name__ == "__main__":
  tf.test.main()
