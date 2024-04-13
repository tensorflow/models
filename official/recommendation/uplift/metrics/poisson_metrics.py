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

"""Poisson regression metrics."""

from __future__ import annotations

from typing import Any

import tensorflow as tf, tf_keras

from official.recommendation.uplift import types
from official.recommendation.uplift.metrics import loss_metric


@tf_keras.utils.register_keras_serializable(package="Uplift")
class LogLoss(loss_metric.LossMetric):
  """Computes the (weighted) poisson log loss sliced by treatment group.

  Given labels `y` and the model's predictions `x`, the loss is computed as:
  `loss = x - y * log(x) + [y * log(y) - y + 0.5 * log(2 * pi * y)]`

  Note that from a numerical perspective it is preferred to compute the loss
  from the model's logits as opposed to directly using its predictions, where
  `logits = log(x) = log(prediction)`. In this case the loss is computed as:
  `loss = exp(logits) - y * logits + [y * log(y) - y + 0.5 * log(2 * pi * y)]`

  Example standalone usage:

  >>> poisson_loss = poisson_metrics.LogLoss()
  >>> y_true = tf.constant([[1.0], [0.0]])
  >>> y_pred = types.TwoTowerTrainingOutputs(
      true_logits=tf.constant([[1.0], [0.0]]),
      is_treatment=tf.constant([[1], [0]]),
  )
  >>> poisson_loss(y_true=y_true, y_pred=y_pred)
  {
      "poisson_log_loss/treatment": 1.7182817  # exp(1) - 1 * 1
      "poisson_log_loss/control": 1.0  # exp(0) - 0 * 0
      "poisson_log_loss": 1.3591409  # (1.7182817 + 1.0) / 2
  }

  Example usage with the `model.compile()` API:

  >>> model.compile(
  ...     optimizer="sgd",
  ...     loss=TrueLogitsLoss(tf.nn.log_poisson_loss),
  ...     metrics=[poisson_metrics.LogLoss()]
  ... )
  """

  def __init__(
      self,
      from_logits: bool = True,
      compute_full_loss: bool = False,
      slice_by_treatment: bool = True,
      name: str = "poisson_log_loss",
      dtype: tf.DType = tf.float32,
  ):
    """Initializes the instance.

    Args:
      from_logits: When `y_pred` is of type `tf.Tensor`, specifies whether
        `y_pred` represents the model's logits or predictions. Otherwise, when
        `y_pred` is of type `TwoTowerTrainingOutputs`, set this to `True` in
        order to compute the loss using the true logits.
      compute_full_loss: Specifies whether the full log loss will be computed.
        If `True`, the expression `[y_true * log(y_true) - y_true + 0.5 * log(2
        * pi * y_true)]` will be added to the loss, otherwise the loss will be
        computed solely by the expression `[y_pred - y_true * log(y_pred)]`.
      slice_by_treatment: Specifies whether the loss should be sliced by the
        treatment indicator tensor. If `True`, the metric's result will return
        the loss values sliced by the treatment group. Note that this can only
        be set to `True` when `y_pred` is of type `TwoTowerTrainingOutputs`.
      name: Optional name for the instance.
      dtype: Optional data type for the instance.
    """
    super().__init__(
        loss_fn=tf.nn.log_poisson_loss,
        from_logits=from_logits,
        compute_full_loss=compute_full_loss,
        slice_by_treatment=slice_by_treatment,
        name=name,
        dtype=dtype,
    )

  def update_state(
      self,
      y_true: tf.Tensor,
      y_pred: types.TwoTowerTrainingOutputs | tf.Tensor,
      sample_weight: tf.Tensor | None = None,
  ):
    if not self._from_logits:
      if isinstance(y_pred, types.TwoTowerTrainingOutputs):
        raise ValueError(
            "`from_logits` must be set to `True` when `y_pred` is of type"
            " TwoTowerTrainingOutputs. Note that the true logits and true"
            " predictions are assumed to be linked to each other through the"
            " log link function: `true_logits = tf.math.log(true_predictions)."
        )
      y_pred = tf.math.log(y_pred)

    super().update_state(y_true, y_pred, sample_weight)

  def get_config(self) -> dict[str, Any]:
    config = super().get_config()
    del config["loss_fn"]
    return config

  @classmethod
  def from_config(cls, config: dict[str, Any]) -> LogLoss:
    return cls(**config)
