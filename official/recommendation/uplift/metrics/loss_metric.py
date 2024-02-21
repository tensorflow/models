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

"""Keras metric for computing a loss sliced by treatment group."""

from __future__ import annotations

import inspect
from typing import Any, Callable

import tensorflow as tf, tf_keras

from official.recommendation.uplift import types
from official.recommendation.uplift.metrics import treatment_sliced_metric


@tf_keras.utils.register_keras_serializable(package="Uplift")
class LossMetric(tf_keras.metrics.Metric):
  """Computes a loss sliced by treatment group.

  Note that the prediction tensor is expected to be of type
  `TwoTowerTrainingOutputs`.

  Example standalone usage:

  >>> sliced_loss = LossMetric(tf_keras.losses.mean_squared_error)
  >>> y_true = tf.constant([0, 0, 2, 2])
  >>> y_pred = types.TwoTowerTrainingOutputs(
  ...     true_logits=tf.constant([1, 2, 3, 4])
  ...     is_treatment=tf.constant([True, False, True, False]),
  ... )
  >>> sliced_loss(y_true=tf.zeros(4), y_pred=y_pred)
  {
      "loss": 2.5
      "loss/control": 4.0
      "loss/treatment": 1.0
  }

  Example usage with the `model.compile()` API:

  >>> model.compile(
  ...     optimizer="sgd",
  ...     loss=TrueLogitsLoss(tf_keras.losses.mean_squared_error),
  ...     metrics=[LossMetric(tf_keras.losses.mean_squared_error)]
  ... )
  """

  def __init__(
      self,
      loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
      from_logits: bool = True,
      name: str = "loss",
      dtype: tf.DType = tf.float32,
      **loss_fn_kwargs,
  ):
    """Initializes the instance.

    Args:
      loss_fn: The loss function to apply between the targets and model outputs,
        with signature `loss_fn(y_true, y_pred, **loss_fn_kwargs)`.
      from_logits: Specifies whether the true logits or true predictions should
        be used from the model outputs to compute the loss. Defaults to using
        the true logits.
      name: Optional name for the instance.
      dtype: Optional data type for the instance.
      **loss_fn_kwargs: The keyword arguments that are passed on to `loss_fn`.
    """
    super().__init__(name=name, dtype=dtype)

    self._loss_fn = loss_fn
    self._from_logits = from_logits
    self._loss_fn_kwargs = loss_fn_kwargs

    if "from_logits" in inspect.signature(loss_fn).parameters:
      self._loss_fn_kwargs.update({"from_logits": from_logits})

    self._treatment_sliced_loss = treatment_sliced_metric.TreatmentSlicedMetric(
        tf_keras.metrics.Mean(name=name, dtype=dtype)
    )

  def update_state(
      self,
      y_true: tf.Tensor,
      y_pred: types.TwoTowerTrainingOutputs,
      sample_weight: tf.Tensor | None = None,
  ):
    """Updates the overall, control and treatment losses.

    Args:
      y_true: A `tf.Tensor` with the targets.
      y_pred: Two tower training outputs. The treatment indicator tensor is used
        to slice the true logits or true predictions into control and treatment
        losses.
      sample_weight: Optional sample weight to compute weighted losses. If
        given, the sample weight will also be sliced by the treatment indicator
        tensor to compute the weighted control and treatment losses.

    Raises:
      TypeError: if `y_pred` is not of type `TwoTowerTrainingOutputs`.
    """
    if not isinstance(y_pred, types.TwoTowerTrainingOutputs):
      raise TypeError(
          "y_pred must be of type `TwoTowerTrainingOutputs` but got type"
          f" {type(y_pred)} instead."
      )

    if self._from_logits:
      loss = self._loss_fn(y_true, y_pred.true_logits)
    else:
      loss = self._loss_fn(y_true, y_pred.true_predictions)

    self._treatment_sliced_loss.update_state(
        values=loss,
        is_treatment=y_pred.is_treatment,
        sample_weight=sample_weight,
    )

  def result(self) -> dict[str, tf.Tensor]:
    return self._treatment_sliced_loss.result()

  def get_config(self) -> dict[str, Any]:
    config = super().get_config()
    config["loss_fn"] = tf_keras.utils.serialize_keras_object(self._loss_fn)
    config["from_logits"] = self._from_logits
    config.update(self._loss_fn_kwargs)
    return config

  @classmethod
  def from_config(cls, config: dict[str, Any]) -> LossMetric:
    config["loss_fn"] = tf_keras.utils.deserialize_keras_object(
        config["loss_fn"]
    )
    return LossMetric(**config)
