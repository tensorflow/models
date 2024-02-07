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

"""Tests for two_tower_uplift_model."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras
from official.recommendation.uplift import keras_test_case
from official.recommendation.uplift import keys
from official.recommendation.uplift.layers.uplift_networks import two_tower_uplift_network
from official.recommendation.uplift.losses import true_logits_loss
from official.recommendation.uplift.models import two_tower_uplift_model


class TwoTowerUpliftModelTest(
    keras_test_case.KerasTestCase, parameterized.TestCase
):

  def _get_uplift_network(self, **kwargs):
    network = two_tower_uplift_network.TwoTowerUpliftNetwork(
        backbone=kwargs.get(
            "backbone",
            tf_keras.layers.Lambda(lambda inputs: inputs["shared_feature"]),
        ),
        control_tower=kwargs.get("control_tower", tf_keras.layers.Dense(1)),
        treatment_tower=kwargs.get("treatment_tower", tf_keras.layers.Dense(1)),
        logits_head=tf_keras.layers.Identity(),
        control_feature_encoder=kwargs.get(
            "control_feature_encoder",
            tf_keras.layers.Lambda(lambda inputs: inputs["control_feature"]),
        ),
        control_input_combiner=kwargs.get(
            "control_input_combiner", tf_keras.layers.Concatenate()
        ),
        treatment_feature_encoder=kwargs.get(
            "treatment_feature_encoder",
            tf_keras.layers.Lambda(lambda inputs: inputs["treatment_feature"]),
        ),
        treatment_input_combiner=kwargs.get(
            "treatment_input_combiner", tf_keras.layers.Concatenate()
        ),
    )
    return network

  def _get_compiled_model(self, **kwargs):
    model = two_tower_uplift_model.TwoTowerUpliftModel(
        treatment_indicator_feature_name="is_treatment",
        uplift_network=self._get_uplift_network(**kwargs),
    )
    model.compile(
        optimizer=tf_keras.optimizers.SGD(0.1),
        loss=true_logits_loss.TrueLogitsLoss(
            tf_keras.losses.mean_squared_error
        ),
    )
    return model

  def _get_inputs(self):
    return {
        "shared_feature": tf.ones((3, 1)),
        "control_feature": tf.ones((3, 1)) * -2.0,
        "treatment_feature": tf.ones((3, 1)) * 3.0,
    }

  def test_model_training_and_inference(self):
    tf_keras.utils.set_random_seed(1)

    # Create MSE uplift model.
    uplift_network = self._get_uplift_network(
        control_feature_encoder=None, control_input_combiner=None
    )
    model = two_tower_uplift_model.TwoTowerUpliftModel(
        treatment_indicator_feature_name="is_treatment",
        uplift_network=uplift_network,
    )
    model.compile(
        optimizer=tf_keras.optimizers.SGD(0.1),
        loss=true_logits_loss.TrueLogitsLoss(
            tf_keras.losses.mean_squared_error
        ),
    )

    # Create toy regression dataset.
    shared_feature, treatment_feature = np.ones((10, 1)), 2 * np.ones((10, 1))
    treatment = tf.constant([[1], [1], [0], [1], [1], [1], [0], [1], [0], [1]])
    y = (shared_feature + treatment_feature) * treatment
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            "shared_feature": shared_feature,
            "treatment_feature": treatment_feature,
            "is_treatment": treatment,
        },
        y,
    )).batch(5)

    # Test model training.
    history = model.fit(dataset, epochs=100)
    self.assertIn("loss", history.history)
    self.assertLen(history.history["loss"], 100)
    self.assertTrue(
        history.history["loss"][0] >= history.history["loss"][-1] >= 0.0
    )

    # Test model evaluation.
    loss = model.evaluate(dataset)
    self.assertLessEqual(loss, 1e-10)
    self.assertGreaterEqual(loss, 0.0)
    self.assertAllClose(history.history["loss"][-1], loss)

    # Test model inference predictions.
    expected_predictions = {
        keys.TwoTowerOutputKeys.CONTROL_PREDICTIONS: tf.zeros((10, 1)),
        keys.TwoTowerOutputKeys.TREATMENT_PREDICTIONS: 3 * tf.ones((10, 1)),
        keys.TwoTowerOutputKeys.UPLIFT_PREDICTIONS: 3 * tf.ones((10, 1)),
    }
    self.assertAllClose(expected_predictions, model.predict(dataset))

  @parameterized.named_parameters(
      {
          "testcase_name": "identity",
          "inverse_link_fn": tf.identity,
          "expected_predictions": {
              keys.TwoTowerOutputKeys.CONTROL_PREDICTIONS: (
                  tf.ones((3, 1)) * -1.0
              ),  # 1 - 2 = -1
              keys.TwoTowerOutputKeys.TREATMENT_PREDICTIONS: (
                  tf.ones((3, 1)) * 4.0
              ),  # 1 + 3 = 4
              keys.TwoTowerOutputKeys.UPLIFT_PREDICTIONS: tf.ones((3, 1)) * 5.0,
          },
      },
      {
          "testcase_name": "abs",
          "inverse_link_fn": tf.math.abs,
          "expected_predictions": {
              keys.TwoTowerOutputKeys.CONTROL_PREDICTIONS: (
                  tf.ones((3, 1)) * 1.0
              ),
              keys.TwoTowerOutputKeys.TREATMENT_PREDICTIONS: (
                  tf.ones((3, 1)) * 4.0
              ),
              keys.TwoTowerOutputKeys.UPLIFT_PREDICTIONS: tf.ones((3, 1)) * 3.0,
          },
      },
      {
          "testcase_name": "relu",
          "inverse_link_fn": tf_keras.activations.relu,
          "expected_predictions": {
              keys.TwoTowerOutputKeys.CONTROL_PREDICTIONS: (
                  tf.ones((3, 1)) * 0.0
              ),
              keys.TwoTowerOutputKeys.TREATMENT_PREDICTIONS: (
                  tf.ones((3, 1)) * 4.0
              ),
              keys.TwoTowerOutputKeys.UPLIFT_PREDICTIONS: tf.ones((3, 1)) * 4.0,
          },
      },
  )
  def test_predict_step(self, inverse_link_fn, expected_predictions):
    uplift_network = self._get_uplift_network(
        control_tower=tf_keras.layers.Dense(1, kernel_initializer="ones"),
        treatment_tower=tf_keras.layers.Dense(1, kernel_initializer="ones"),
    )
    model = two_tower_uplift_model.TwoTowerUpliftModel(
        treatment_indicator_feature_name="is_treatment",
        uplift_network=uplift_network,
        inverse_link_fn=inverse_link_fn,
    )
    inputs = {
        "shared_feature": tf.ones((3, 1)),
        "control_feature": tf.ones((3, 1)) * -2.0,
        "treatment_feature": tf.ones((3, 1)) * 3.0,
    }
    self.assertAllClose(expected_predictions, model.predict_step(inputs))

  def test_missing_treatment_indicator_from_inputs_during_training_raises_value_error(
      self,
  ):
    model = self._get_compiled_model()
    inputs = {"x": tf.ones((3, 1))}
    dataset = tf.data.Dataset.from_tensor_slices((inputs, tf.ones((3, 1))))

    with self.assertRaises(ValueError):
      model.fit(dataset)

  def test_missing_treatment_indicator_from_inputs_during_evaluation_raises_value_error(
      self,
  ):
    model = self._get_compiled_model()
    inputs = {"x": tf.ones((3, 1))}
    dataset = tf.data.Dataset.from_tensor_slices((inputs, tf.ones((3, 1))))

    with self.assertRaises(ValueError):
      model.evaluate(dataset)

  def test_model_is_stable(self):
    model = self._get_compiled_model()
    inputs = self._get_inputs()
    self.assertLayerStable(layer=model, inputs=inputs)

  def test_model_is_savable(self):
    model = self._get_compiled_model()
    inputs = self._get_inputs()
    self.assertModelSavable(model=model, inputs=inputs)

  def test_layer_configurable(self):
    # Cannot use lambda layers since they are not serializable.
    model = self._get_compiled_model(
        backbone=tf_keras.layers.Identity(),
        control_feature_encoder=tf_keras.layers.Identity(),
        treatment_feature_encoder=tf_keras.layers.Identity(),
        inverse_link_fn=tf.math.sigmoid,
    )
    self.assertLayerConfigurable(layer=model)


if __name__ == "__main__":
  tf.test.main()
