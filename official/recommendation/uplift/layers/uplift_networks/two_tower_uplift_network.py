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

"""Builds a TwoTowerUpliftNetwork layer."""

from __future__ import annotations

from typing import Any, Dict

import tensorflow as tf, tf_keras

from official.recommendation.uplift import types
from official.recommendation.uplift.layers.uplift_networks import base_uplift_networks


@tf_keras.utils.register_keras_serializable(package="Uplift")
class TwoTowerUpliftNetwork(base_uplift_networks.BaseTwoTowerUpliftNetwork):
  """Computes control and treatment logits in two separate towers.

  Takes as input a dictionary of feature tensors and computes logits for the
  control and treatment groups. The layer returns a `TwoTowerNetworkOutputs`
  which contains the main tensors computed during the feed forward pass of the
  network.

  See ../../README.md for more details on the network's architecture.
  """

  def __init__(
      self,
      backbone: tf_keras.layers.Layer,
      control_tower: tf_keras.layers.Layer,
      treatment_tower: tf_keras.layers.Layer,
      logits_head: tf_keras.layers.Layer,
      control_feature_encoder: tf_keras.layers.Layer | None = None,
      control_input_combiner: tf_keras.layers.Layer | None = None,
      treatment_feature_encoder: tf_keras.layers.Layer | None = None,
      treatment_input_combiner: tf_keras.layers.Layer | None = None,
      **kwargs,
  ):
    """Initializes a TwoTowerUpliftNetwork layer.

    Args:
      backbone: encodes input features into a shared embedding for the control
        and treatment towers. Its input is a dictionary with the input features
        and its output is expected to be a single dense embedding.
      control_tower: computes an embedding for the control group. Its input and
        output is a single dense tensor.
      treatment_tower: computes an embedding for the treatment group. Its input
        and output is a single dense tensor.
      logits_head: computes control and treatment logits. Its inputs is a tuple
        of (control_embedding, treatment_embedding) and its output is expected
        to be a tuple of (control_logits, treatment_logits).
      control_feature_encoder: encodes control specific input features into a
        dense tensor. Its input is the entire inputs dictionary. If this layer
        is provided, a control_input_combiner layer must also be given.
      control_input_combiner: combines the shared embedding from the backbone
        with the control_feature_encoder output embedding. Its input is a list
        with the shared and control specific embedding, and its output is a
        single dense embedding to be fed into the control_tower layer. This
        layer should only be provided if a control_feature_encoder is also
        provided.
      treatment_feature_encoder: encodes treatment specific input features into
        a dense tensor. Its input is the entire inputs dictionary. If this layer
        is provided, a treatment_input_combiner layer must also be given.
      treatment_input_combiner: combines the shared embedding from the backbone
        with the treatment_feature_encoder output embedding. Its input is a list
        with the shared and treatment specific embedding, and its output is a
        single dense embedding to be fed into the treatment_tower layer. This
        layer should only be provided if a treatment_feature_encoder is also
        provided.
      **kwargs: base layer keyword arguments.

    Raises:
      ValueError if only one of the control encoder or combiner layers is set.
        Both must be a Keras layer or None.
      ValueError if only one of the treatment encoder or combiner layers is set.
        Both must be a Keras layer or None.
    """
    super().__init__(**kwargs)

    self._backbone = backbone
    self._control_tower = control_tower
    self._treatment_tower = treatment_tower
    self._logits_head = logits_head
    self._control_feature_encoder = control_feature_encoder
    self._control_input_combiner = control_input_combiner
    self._treatment_feature_encoder = treatment_feature_encoder
    self._treatment_input_combiner = treatment_input_combiner

    self._validate_encoder_combiner_layers(
        control_feature_encoder, control_input_combiner, "control"
    )
    self._validate_encoder_combiner_layers(
        treatment_feature_encoder, treatment_input_combiner, "treatment"
    )

  def _validate_encoder_combiner_layers(
      self,
      encoder: tf_keras.layer.Layer,
      combiner: tf_keras.layer.Layer,
      name: str,
  ) -> None:
    if encoder is not None and combiner is None:
      raise ValueError(
          f"The {name}_input_combiner layer must be specified if the"
          f" {name}_feature_encoder is not None. Consider using"
          " tf_keras.layers.Concatenate() as a combiner layer."
      )
    if encoder is None and combiner is not None:
      raise ValueError(
          f"The {name}_feature_encoder layer must be specified if the"
          f" {name}_input_combiner is not None."
      )

  def call(
      self,
      inputs: types.DictOfTensors,
      training: bool | None = None,
      mask: tf.Tensor | None = None,
  ) -> types.TwoTowerNetworkOutputs:
    """Computes control and treatment logits.

    Args:
      inputs: dictionary of feature tensors to be passed to the backbone and
        control/treatment feature encoder layers when set.
      training: unused optional training flag.
      mask: unused optional mask.

    Returns:
      A `TwoTowerNetworkOutputs` ExtensionType which should be used as a
      dataclass to access the main tensors computed during the feed forward pass
      of the network.
    """
    # Compute shared embedding for control and treatment towers.
    shared_embedding = self._backbone(inputs)

    # Compute control embedding.
    if self._control_feature_encoder is not None:
      control_feature_encoding = self._control_feature_encoder(inputs)
      control_tower_input = self._control_input_combiner(
          [shared_embedding, control_feature_encoding]
      )
    else:
      control_tower_input = shared_embedding
    control_embedding = self._control_tower(control_tower_input)

    # Compute treatment embedding.
    if self._treatment_feature_encoder is not None:
      treatment_feature_encoding = self._treatment_feature_encoder(inputs)
      treatment_tower_input = self._treatment_input_combiner(
          [shared_embedding, treatment_feature_encoding]
      )
    else:
      treatment_tower_input = shared_embedding
    treatment_embedding = self._treatment_tower(treatment_tower_input)

    # Compute control and treatment logits.
    control_logits, treatment_logits = self._logits_head(
        (control_embedding, treatment_embedding)
    )

    return types.TwoTowerNetworkOutputs(
        shared_embedding=shared_embedding,
        control_logits=control_logits,
        treatment_logits=treatment_logits,
    )

  def get_config(self) -> Dict[str, Any]:
    config = super().get_config()

    for layer_name, layer in (
        ("backbone", self._backbone),
        ("control_tower", self._control_tower),
        ("treatment_tower", self._treatment_tower),
        ("logits_head", self._logits_head),
        ("control_feature_encoder", self._control_feature_encoder),
        ("control_input_combiner", self._control_input_combiner),
        ("treatment_feature_encoder", self._treatment_feature_encoder),
        ("treatment_input_combiner", self._treatment_input_combiner),
    ):
      config[layer_name] = tf_keras.utils.serialize_keras_object(layer)

    return config

  @classmethod
  def from_config(cls, config: Dict[str, Any]) -> TwoTowerUpliftNetwork:
    for layer_name in (
        "backbone",
        "control_tower",
        "treatment_tower",
        "logits_head",
        "control_feature_encoder",
        "control_input_combiner",
        "treatment_feature_encoder",
        "treatment_input_combiner",
    ):
      # layers.deserialize does not accept empty config
      if config.get(layer_name):
        config[layer_name] = tf_keras.layers.deserialize(config[layer_name])

    return cls(**config)
