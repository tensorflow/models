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

"""Defines a Keras model for the `TwoTowerUpliftNetwork` layer."""

from __future__ import annotations

from typing import Any, Callable, Mapping, MutableMapping

import tensorflow as tf, tf_keras

from official.recommendation.uplift import keys
from official.recommendation.uplift import types
from official.recommendation.uplift.layers.uplift_networks import base_uplift_networks
from official.recommendation.uplift.layers.uplift_networks import two_tower_output_head


@tf_keras.utils.register_keras_serializable(package="Uplift")
class TwoTowerUpliftModel(tf_keras.Model):
  """Training and inference model for a `BaseTwoTowerUpliftNetwork` layer."""

  def __init__(
      self,
      treatment_indicator_feature_name: str,
      uplift_network: base_uplift_networks.BaseTwoTowerUpliftNetwork,
      inverse_link_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
      **kwargs,
  ):
    """Initializes the instance.

    Args:
      treatment_indicator_feature_name: the name of the feature representing the
        treatment_indicator tensor, which should be castable to a boolean tensor
        (False for control and True for treatment). This tensor is required
        during training and evaluation to compute the true logits needed for
        loss computation.
      uplift_network: a layer for computing control and treatment logits. Its
        input is expected to be a dictionary of feature tensors and its output
        is exptected to be a `TwoTowerNetworkOutputs` instance.
      inverse_link_fn: a function for computing the control and treatment
        predictions from their respective logits. If left as `None` it is
        functionally equivalent to the identity function.
      **kwargs: base model keyword arguments.
    """
    super().__init__(**kwargs)

    self._treatment_indicator_feature_name = treatment_indicator_feature_name
    self._uplift_network = uplift_network
    self._inverse_link_fn = inverse_link_fn

    self._output_head = two_tower_output_head.TwoTowerOutputHead(
        treatment_indicator_feature_name=treatment_indicator_feature_name,
        uplift_network=uplift_network,
        inverse_link_fn=inverse_link_fn,
    )

  def call(
      self,
      inputs: types.DictOfTensors,
      training: bool | None = None,
      mask: tf.Tensor | None = None,
  ) -> types.TwoTowerPredictionOutputs | types.TwoTowerTrainingOutputs:
    return self._output_head(inputs=inputs, training=training, mask=mask)

  def _assert_treatment_indicator_in_data(self, data):
    inputs, _, _ = tf_keras.utils.unpack_x_y_sample_weight(data)

    if self._treatment_indicator_feature_name not in inputs:
      raise ValueError(
          "The treatment_indicator feature (specified as"
          f" '{self._treatment_indicator_feature_name}') must be part of the"
          " inputs during training and evaluation, but got input features"
          f" {set(inputs.keys())} instead."
      )

  def train_step(self, data) -> types.TwoTowerTrainingOutputs:
    self._assert_treatment_indicator_in_data(data)
    return super().train_step(data)

  def test_step(self, data) -> types.TwoTowerTrainingOutputs:
    self._assert_treatment_indicator_in_data(data)
    return super().test_step(data)

  def predict_step(self, data) -> dict[str, tf.Tensor]:
    outputs = super().predict_step(data)

    return {
        keys.TwoTowerPredictionKeys.CONTROL: outputs.control_predictions,
        keys.TwoTowerPredictionKeys.TREATMENT: outputs.treatment_predictions,
        keys.TwoTowerPredictionKeys.UPLIFT: outputs.uplift,
    }

  def get_config(self) -> Mapping[str, Any]:
    config = super().get_config()
    config.update({
        "treatment_indicator_feature_name": (
            self._treatment_indicator_feature_name
        ),
        "uplift_network": tf_keras.utils.serialize_keras_object(
            self._uplift_network
        ),
        "inverse_link_fn": tf_keras.utils.serialize_keras_object(
            self._inverse_link_fn
        ),
    })
    return config

  @classmethod
  def from_config(cls, config: MutableMapping[str, Any]) -> TwoTowerUpliftModel:
    config["uplift_network"] = tf_keras.layers.deserialize(
        config["uplift_network"]
    )
    config["inverse_link_fn"] = tf_keras.utils.deserialize_keras_object(
        config["inverse_link_fn"]
    )
    return cls(**config)
