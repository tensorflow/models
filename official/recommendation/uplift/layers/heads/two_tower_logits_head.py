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

"""Builds a TwoTowerLogitsHead layer."""

from __future__ import annotations

import dataclasses
import enum
from typing import Any

import tensorflow as tf, tf_keras


@enum.unique
class LayeringMethod(str, enum.Enum):
  """Layering method between the control and treatment towers."""

  # No layering.
  NONE = "none"

  # The treatment logits are adjusted by the following function:
  #   treatment_logits += tf.stop_gradient(control_logits)
  LOGIT_SUM = "logit_sum"

  # The treatment embedding is adjusted by the following function:
  #   treatment_embedding += stop_gradient(control_embedding) * W
  # Where "W" is a learnable weight matrix of shape CxT where C is the control
  # embedding dimension and T is the treatment embedding dimension.
  LINEAR_LAYERING = "linear_layering"


@dataclasses.dataclass(frozen=True)
class LinearLayeringConfig:
  """Configuration for the linear layering method.

  Attributes:
    kernel_initializer: kernel initializer for the learnable weight matrix.
    kernel_regularizer: kernel regularizer for the learnable weight matrix.
  """

  kernel_initializer: str = "glorot_uniform"
  kernel_regularizer: str | None = None


@dataclasses.dataclass(frozen=True)
class LayeringConfig:
  """Configuration for all layering methods.

  Attributes:
    layering_method: specifies what layering method to apply. Defaults to
      `LayeringMethod.NONE`.
    linear_layering_config: configuration for the linear layering method. Will
      only be used if the `layering_method` is set to
      `LayeringMethod.LINEAR_LAYERING`.
  """

  layering_method: LayeringMethod = LayeringMethod.NONE
  linear_layering_config: LinearLayeringConfig | None = None

  @classmethod
  def from_dict(cls, config: dict[str, Any]) -> LayeringConfig:
    linear_layering_config = (
        LinearLayeringConfig(**config["linear_layering_config"])
        if config["linear_layering_config"] is not None
        else None
    )
    return cls(
        layering_method=LayeringMethod(config["layering_method"]),
        linear_layering_config=linear_layering_config,
    )


@tf_keras.utils.register_keras_serializable(package="Uplift")
class TwoTowerLogitsHead(tf_keras.layers.Layer):
  """Computes control and treatment logits from their respective embeddings.

  Takes as input a tuple of control and treatment embeddings and computes
  control and treatment logits.
  """

  def __init__(
      self,
      control_head: tf_keras.layers.Layer,
      treatment_head: tf_keras.layers.Layer,
      layering_config: LayeringConfig = LayeringConfig(),
      **kwargs,
  ):
    """Initializes the instance.

    The control and treatment heads must compute logits of the same shape.

    Args:
      control_head: computes control logits from the control embedding. Its
        input and output is expected to be a dense tensor.
      treatment_head: computes treatment logits from the treatment embedding.
        Its input and output is expected to be a dense tensor.
      layering_config: configuration for the layering method. Defaults to no
        layering.
      **kwargs: base layer keyword arguments.
    """
    super().__init__(**kwargs)

    self._control_head = control_head
    self._treatment_head = treatment_head
    self._layering_config = layering_config

  def build(self, input_shapes: tuple[tf.TensorShape, tf.TensorShape]):
    if self._layering_config.layering_method == LayeringMethod.LINEAR_LAYERING:
      if self._layering_config.linear_layering_config is None:
        raise ValueError(
            "The linear layering config cannot be `None` when using the linear"
            " layering method."
        )
      # Build a learnable weight matrix that projects from the control embedding
      # space to the treatment embedding space.
      _, treatment_embedding_shape = input_shapes
      self._linear_layering = tf_keras.layers.Dense(
          units=treatment_embedding_shape[-1],
          activation=None,
          use_bias=True,
          kernel_initializer=(
              self._layering_config.linear_layering_config.kernel_initializer
          ),
          kernel_regularizer=(
              self._layering_config.linear_layering_config.kernel_regularizer
          ),
      )
    super().build(input_shapes)

  def call(
      self, inputs: tuple[tf.Tensor, tf.Tensor]
  ) -> tuple[tf.Tensor, tf.Tensor]:
    control_embedding, treatment_embedding = inputs

    if self._layering_config.layering_method == LayeringMethod.LINEAR_LAYERING:
      treatment_embedding += self._linear_layering(
          tf.stop_gradient(control_embedding)
      )

    control_logits = self._control_head(control_embedding)
    treatment_logits = self._treatment_head(treatment_embedding)

    if control_logits.shape != treatment_logits.shape:
      raise ValueError(
          "The control logits and treatment logits computed by the control and"
          " treatment heads must be tensors of the same shape, but got shape"
          f" {control_logits.shape} for the control logits and shape"
          f" {treatment_logits.shape} for the treatment logits."
      )

    if self._layering_config.layering_method == LayeringMethod.LOGIT_SUM:
      treatment_logits += tf.stop_gradient(control_logits)

    return control_logits, treatment_logits

  def get_config(self) -> dict[str, Any]:
    config = super().get_config()
    config["layering_config"] = dataclasses.asdict(self._layering_config)

    for layer_name, layer in (
        ("control_head", self._control_head),
        ("treatment_head", self._treatment_head),
    ):
      config[layer_name] = tf_keras.utils.serialize_keras_object(layer)

    return config

  @classmethod
  def from_config(cls, config: dict[str, Any]) -> TwoTowerLogitsHead:
    config["layering_config"] = LayeringConfig.from_dict(
        config["layering_config"]
    )
    for layer_name in ("control_head", "treatment_head"):
      config[layer_name] = tf_keras.layers.deserialize(config[layer_name])

    return cls(**config)
