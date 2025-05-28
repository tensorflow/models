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

"""Tests for two_tower_uplift_network."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras
from official.recommendation.uplift import keras_test_case
from official.recommendation.uplift.layers.uplift_networks import two_tower_uplift_network


class TwoTowerUpliftNetworkTest(
    keras_test_case.KerasTestCase, parameterized.TestCase
):

  def _get_full_layer(self, **kwargs):
    layer = two_tower_uplift_network.TwoTowerUpliftNetwork(
        backbone=tf_keras.layers.Lambda(lambda inputs: inputs["shared_inputs"]),
        control_tower=tf_keras.layers.Dense(
            1, kernel_initializer=tf_keras.initializers.Constant(-1.0)
        ),
        treatment_tower=tf_keras.layers.Dense(
            1, kernel_initializer=tf_keras.initializers.Constant(2.0)
        ),
        logits_head=tf_keras.layers.Identity(),
        control_feature_encoder=kwargs.get(
            "control_feature_encoder",
            tf_keras.layers.Lambda(lambda inputs: inputs["control_inputs"]),
        ),
        control_input_combiner=kwargs.get(
            "control_input_combiner", tf_keras.layers.Concatenate()
        ),
        treatment_feature_encoder=kwargs.get(
            "treatment_feature_encoder",
            tf_keras.layers.Lambda(lambda inputs: inputs["treatment_inputs"]),
        ),
        treatment_input_combiner=kwargs.get(
            "treatment_input_combiner", tf_keras.layers.Concatenate()
        ),
    )
    return tf_keras.models.clone_model(layer)

  def _get_full_layer_inputs(self):
    return {
        "shared_inputs": tf.ones((3, 3)),
        "control_inputs": tf.ones((3, 10)),
        "treatment_inputs": tf.ones((3, 10)),
    }

  def test_forward_pass_no_control_or_treatment_encoders(self):
    layer = two_tower_uplift_network.TwoTowerUpliftNetwork(
        backbone=tf_keras.layers.Lambda(lambda inputs: inputs["shared_inputs"]),
        control_tower=tf_keras.layers.Dense(
            1, kernel_initializer=tf_keras.initializers.Constant(-1.0)
        ),
        treatment_tower=tf_keras.layers.Dense(
            1, kernel_initializer=tf_keras.initializers.Constant(2.0)
        ),
        logits_head=tf_keras.layers.Identity(),
    )
    inputs = {"shared_inputs": tf.ones((3, 3))}
    outputs = layer(inputs)

    self.assertAllClose(tf.ones((3, 3)), outputs.shared_embedding)
    self.assertAllClose(3 * -1 * tf.ones((3, 1)), outputs.control_logits)
    self.assertAllClose(3 * 2 * tf.ones((3, 1)), outputs.treatment_logits)

  def test_forward_pass_only_control_encoder(self):
    layer = two_tower_uplift_network.TwoTowerUpliftNetwork(
        backbone=tf_keras.layers.Lambda(lambda inputs: inputs["shared_inputs"]),
        control_tower=tf_keras.layers.Dense(
            1, kernel_initializer=tf_keras.initializers.Constant(-1.0)
        ),
        treatment_tower=tf_keras.layers.Dense(
            1, kernel_initializer=tf_keras.initializers.Constant(2.0)
        ),
        logits_head=tf_keras.layers.Identity(),
        control_feature_encoder=tf_keras.layers.Lambda(
            lambda inputs: inputs["control_inputs"]
        ),
        control_input_combiner=tf_keras.layers.Concatenate(),
    )
    inputs = {
        "shared_inputs": tf.ones((3, 3)),
        "control_inputs": tf.ones((3, 10)),
    }
    outputs = layer(inputs)

    self.assertAllClose(tf.ones((3, 3)), outputs.shared_embedding)
    self.assertAllClose((3 + 10) * -1 * tf.ones((3, 1)), outputs.control_logits)
    self.assertAllClose(3 * 2 * tf.ones((3, 1)), outputs.treatment_logits)

  def test_forward_pass_only_treatment_encoder(self):
    layer = two_tower_uplift_network.TwoTowerUpliftNetwork(
        backbone=tf_keras.layers.Lambda(lambda inputs: inputs["shared_inputs"]),
        control_tower=tf_keras.layers.Dense(
            1, kernel_initializer=tf_keras.initializers.Constant(-1.0)
        ),
        treatment_tower=tf_keras.layers.Dense(
            1, kernel_initializer=tf_keras.initializers.Constant(2.0)
        ),
        logits_head=tf_keras.layers.Identity(),
        treatment_feature_encoder=tf_keras.layers.Lambda(
            lambda inputs: inputs["treatment_inputs"]
        ),
        treatment_input_combiner=tf_keras.layers.Concatenate(),
    )
    inputs = {
        "shared_inputs": tf.ones((3, 3)),
        "treatment_inputs": tf.ones((3, 10)),
    }
    outputs = layer(inputs)

    self.assertAllClose(tf.ones((3, 3)), outputs.shared_embedding)
    self.assertAllClose(3 * -1 * tf.ones((3, 1)), outputs.control_logits)
    self.assertAllClose(
        (3 + 10) * 2 * tf.ones((3, 1)), outputs.treatment_logits
    )

  def test_forward_pass_both_control_and_treatment_encoders(self):
    layer = two_tower_uplift_network.TwoTowerUpliftNetwork(
        backbone=tf_keras.layers.Lambda(lambda inputs: inputs["shared_inputs"]),
        control_tower=tf_keras.layers.Dense(
            1, kernel_initializer=tf_keras.initializers.Constant(-1.0)
        ),
        treatment_tower=tf_keras.layers.Dense(
            1, kernel_initializer=tf_keras.initializers.Constant(2.0)
        ),
        logits_head=tf_keras.layers.Identity(),
        control_feature_encoder=tf_keras.layers.Lambda(
            lambda inputs: inputs["control_inputs"]
        ),
        control_input_combiner=tf_keras.layers.Concatenate(),
        treatment_feature_encoder=tf_keras.layers.Lambda(
            lambda inputs: inputs["treatment_inputs"]
        ),
        treatment_input_combiner=tf_keras.layers.Concatenate(),
    )
    inputs = {
        "shared_inputs": tf.ones((3, 3)),
        "control_inputs": tf.ones((3, 10)),
        "treatment_inputs": tf.ones((3, 10)),
    }
    outputs = layer(inputs)

    self.assertAllClose(tf.ones((3, 3)), outputs.shared_embedding)
    self.assertAllClose((3 + 10) * -1 * tf.ones((3, 1)), outputs.control_logits)
    self.assertAllClose(
        (3 + 10) * 2 * tf.ones((3, 1)), outputs.treatment_logits
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "encoder_without_combiner",
          "control_feature_encoder": tf_keras.layers.Lambda(
              lambda inputs: inputs["control_inputs"]
          ),
          "control_input_combiner": None,
      },
      {
          "testcase_name": "combiner_without_encoder",
          "control_feature_encoder": None,
          "control_input_combiner": tf_keras.layers.Concatenate(),
      },
  )
  def test_invalid_control_encoder_combiner_combination_raises_error(
      self, control_feature_encoder, control_input_combiner
  ):
    with self.assertRaises(ValueError):
      self._get_full_layer(
          control_feature_encoder=control_feature_encoder,
          control_input_combiner=control_input_combiner,
      )

  @parameterized.named_parameters(
      {
          "testcase_name": "encoder_without_combiner",
          "treatment_feature_encoder": tf_keras.layers.Lambda(
              lambda inputs: inputs["treatment_inputs"]
          ),
          "treatment_input_combiner": None,
      },
      {
          "testcase_name": "combiner_without_encoder",
          "treatment_feature_encoder": None,
          "treatment_input_combiner": tf_keras.layers.Concatenate(),
      },
  )
  def test_invalid_treatment_encoder_combiner_combination_raises_error(
      self, treatment_feature_encoder, treatment_input_combiner
  ):
    with self.assertRaises(ValueError):
      self._get_full_layer(
          treatment_feature_encoder=treatment_feature_encoder,
          treatment_input_combiner=treatment_input_combiner,
      )

  def test_multiple_layer_calls_with_same_input_returns_same_output(self):
    layer = self._get_full_layer()
    inputs = self._get_full_layer_inputs()
    self.assertLayerStable(layer=layer, inputs=inputs)

  def test_layer_saving_succeeds(self):
    layer = self._get_full_layer()
    inputs = self._get_full_layer_inputs()
    self.assertLayerSavable(layer=layer, inputs=inputs)

  def test_from_config_layer_returns_same_output_as_original_layer(self):
    layer = self._get_full_layer()
    inputs = self._get_full_layer_inputs()
    self.assertLayerConfigurable(layer=layer, inputs=inputs)


if __name__ == "__main__":
  tf_keras.__internal__.enable_unsafe_deserialization()
  tf.test.main()
