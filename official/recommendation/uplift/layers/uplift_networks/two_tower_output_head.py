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

"""Builds a TwoTowerOutputHead layer."""

from __future__ import annotations

from typing import Any, Callable, Mapping, MutableMapping

import tensorflow as tf, tf_keras

from official.recommendation.uplift import types
from official.recommendation.uplift import utils
from official.recommendation.uplift.layers.uplift_networks import base_uplift_networks


@tf_keras.utils.register_keras_serializable(package="Uplift")
class TwoTowerOutputHead(tf_keras.layers.Layer):
  """Two tower training and inference output computation.

  This layer is intended to be used in conjunction with a two tower uplift
  network layer to compute training and inference outputs. It passes the input
  dictionary of feature tensors to the uplift network and computes training and
  inference tensors from the uplift network's outputs.

  The control, treatment, and uplift predictions are computed from the uplift
  network's control and treatment logits. Additionally, if the treatment
  indicator tensor is part of the inputs, the true logits are also computed to
  be used for loss computation. The treatment indicator tensor should therefore
  be present in the inputs dictionary during training and evaluation.
  """

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
        treatment indicator tensor, which should be castable to a boolean tensor
        (False for control and True for treatment). When this tensor is part of
        the layer inputs, additional true logits are returned for the loss
        computation. In this case the layer returns a `TwoTowerTrainingOutputs`
        instance, otherwise a `TwoTowerPredictionOutputs` instance is returned.
      uplift_network: a layer for computing control and treatment logits. Its
        input is expected to be a dictionary of feature tensors and its output
        is exptected to be a `TwoTowerNetworkOutputs` instance.
      inverse_link_fn: a function for computing the control and treatment
        predictions from their respective logits. If left as `None` it is
        functionally equivalent to the idendity function.
      **kwargs: base layer keyword arguments.
    """
    super().__init__(**kwargs)

    self._treatment_indicator_feature_name = treatment_indicator_feature_name
    self._uplift_network = uplift_network
    self._inverse_link_fn = inverse_link_fn

  def call(
      self,
      inputs: types.DictOfTensors,
      training: bool | None = None,
      mask: tf.Tensor | None = None,
  ) -> types.TwoTowerPredictionOutputs | types.TwoTowerTrainingOutputs:
    """Computes two tower inference and training outputs.

    Args:
      inputs: feature tensors to be passed to the uplift network. For training
        and evaluation, the treatment indicator tensor must be given as part of
        the inputs.
      training: optional boolean training flag to pass to the uplift network.
      mask: optional tensor mask to pass to the uplift network.

    Returns:
      `TwoTowerTrainingOutputs` when the treatment indicator tensor is part of
      the input features, otherwise `TwoTowerPredictionOutputs`.
    """
    outputs: types.TwoTowerNetworkOutputs = self._uplift_network(
        inputs=inputs, training=training, mask=mask
    )

    if self._inverse_link_fn is None:
      control_predictions = outputs.control_logits
      treatment_predictions = outputs.treatment_logits
    else:
      control_predictions = self._inverse_link_fn(outputs.control_logits)
      treatment_predictions = self._inverse_link_fn(outputs.treatment_logits)

    uplift = treatment_predictions - control_predictions

    # If the treatment indicator tensor is not in the layer inputs return
    # inference output.
    if self._treatment_indicator_feature_name not in inputs:
      return types.TwoTowerPredictionOutputs(
          shared_embedding=outputs.shared_embedding,
          control_logits=outputs.control_logits,
          treatment_logits=outputs.treatment_logits,
          control_predictions=control_predictions,
          treatment_predictions=treatment_predictions,
          uplift=uplift,
      )

    # Compute the true logits only when the treatment_indicator tensor is given.
    is_treatment = tf.cast(
        inputs[self._treatment_indicator_feature_name], tf.bool
    )

    # Expand treatment_indicator tensor to match the logits and predictions.
    # This is done to prevent tf.where from broadcasting the condition. For
    # example, if is_treatment is of shape (3,) and the predictions are of shape
    # (3, 1) then tf.where will return a tensor of shape (3, 3). However,
    # tf.where will return a tensor of shape (3, 1) if the is_treatment tensor
    # is also of that shape.
    # Also, note that for generalizability the logits are allowed to have a
    # different shape than the predictions, however the control/treatment logits
    # shapes must be equal to each other and likewise for the control/treatment
    # prediction shapes.
    true_logits = tf.where(
        utils.expand_to_match_rank(is_treatment, outputs.control_logits),
        outputs.treatment_logits,
        outputs.control_logits,
    )
    true_predictions = tf.where(
        utils.expand_to_match_rank(is_treatment, control_predictions),
        treatment_predictions,
        control_predictions,
    )

    # Create a new tensor since ExtensionTypes are immutable.
    return types.TwoTowerTrainingOutputs(
        shared_embedding=outputs.shared_embedding,
        control_logits=outputs.control_logits,
        treatment_logits=outputs.treatment_logits,
        control_predictions=control_predictions,
        treatment_predictions=treatment_predictions,
        uplift=uplift,
        true_logits=true_logits,
        true_predictions=true_predictions,
        is_treatment=is_treatment,
    )

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
  def from_config(cls, config: MutableMapping[str, Any]) -> TwoTowerOutputHead:
    config["uplift_network"] = tf_keras.layers.deserialize(
        config["uplift_network"]
    )
    config["inverse_link_fn"] = tf_keras.utils.deserialize_keras_object(
        config["inverse_link_fn"]
    )
    return cls(**config)
