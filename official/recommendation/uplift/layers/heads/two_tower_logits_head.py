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

"""Builds a TwoTowerLogitsHead layer."""

from __future__ import annotations

from typing import Any

import tensorflow as tf, tf_keras


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
      **kwargs,
  ):
    """Initializes the instance.

    The control and treatment heads must compute logits of the same shape.

    Args:
      control_head: computes control logits from the control embedding. Its
        input and output is expected to be a dense tensor.
      treatment_head: computes treatment logits from the treatment embedding.
        Its input and output is expected to be a dense tensor.
      **kwargs: base layer keyword arguments.
    """
    super().__init__(**kwargs)

    self._control_head = control_head
    self._treatment_head = treatment_head

  def call(
      self, inputs: tuple[tf.Tensor, tf.Tensor]
  ) -> tuple[tf.Tensor, tf.Tensor]:
    control_embedding, treatment_embedding = inputs

    control_logits = self._control_head(control_embedding)
    treatment_logits = self._treatment_head(treatment_embedding)

    if control_logits.shape != treatment_logits.shape:
      raise ValueError(
          "The control logits and treatment logits computed by the control and"
          " treatment heads must be tensors of the same shape, but got shape"
          f" {control_logits.shape} for the control logits and shape"
          f" {treatment_logits.shape} for the treatment logits."
      )

    return control_logits, treatment_logits

  def get_config(self) -> dict[str, Any]:
    config = super().get_config()

    for layer_name, layer in (
        ("control_head", self._control_head),
        ("treatment_head", self._treatment_head),
    ):
      config[layer_name] = tf_keras.utils.serialize_keras_object(layer)

    return config

  @classmethod
  def from_config(cls, config: dict[str, Any]) -> TwoTowerLogitsHead:
    for layer_name in ("control_head", "treatment_head"):
      config[layer_name] = tf_keras.layers.deserialize(config[layer_name])

    return cls(**config)
