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
from official.recommendation.uplift.metrics import treatment_sliced_metric


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


def _safe_x_minus_xlogx(x: tf.Tensor) -> tf.Tensor:
  """Computes x - x * log(x) with 0 as its continuity point when x equals 0."""
  values = x * (1.0 - tf.math.log(x))
  return tf.where(tf.equal(x, 0.0), tf.zeros_like(x), values)


@tf_keras.utils.register_keras_serializable(package="Uplift")
class LogLossMeanBaseline(tf_keras.metrics.Metric):
  """Computes the (weighted) poisson log loss for a mean predictor."""

  def __init__(
      self,
      compute_full_loss: bool = False,
      slice_by_treatment: bool = True,
      name: str = "poisson_log_loss_mean_baseline",
      dtype: tf.DType = tf.float32,
  ):
    """Initializes the instance.

    Args:
      compute_full_loss: Specifies whether to compute the full poisson log loss
        for the mean predictor or not. Defaults to `False`.
      slice_by_treatment: Specifies whether the loss should be sliced by the
        treatment indicator tensor. If `True`, the metric's result will return
        the loss values sliced by the treatment group. Note that this can only
        be set to `True` when `y_pred` is of type `TwoTowerTrainingOutputs`.
      name: Optional name for the instance.
      dtype: Optional data type for the instance.
    """
    super().__init__(name=name, dtype=dtype)

    if compute_full_loss:
      raise NotImplementedError("Full loss computation is not yet supported.")

    self._compute_full_loss = compute_full_loss
    self._slice_by_treatment = slice_by_treatment

    if slice_by_treatment:
      self._mean_label = treatment_sliced_metric.TreatmentSlicedMetric(
          metric=tf_keras.metrics.Mean(name=name, dtype=dtype)
      )
    else:
      self._mean_label = tf_keras.metrics.Mean(name=name, dtype=dtype)

  def update_state(
      self,
      y_true: tf.Tensor,
      y_pred: types.TwoTowerTrainingOutputs | tf.Tensor | None = None,
      sample_weight: tf.Tensor | None = None,
  ):
    is_treatment = {}
    if self._slice_by_treatment:
      if not isinstance(y_pred, types.TwoTowerTrainingOutputs):
        raise ValueError(
            "`slice_by_treatment` must be set to `False` when `y_pred` is not"
            " of type `TwoTowerTrainingOutputs`."
        )
      is_treatment["is_treatment"] = y_pred.is_treatment

    self._mean_label.update_state(
        y_true, sample_weight=sample_weight, **is_treatment
    )

  def result(self) -> tf.Tensor | dict[str, tf.Tensor]:
    return tf.nest.map_structure(_safe_x_minus_xlogx, self._mean_label.result())

  def get_config(self) -> dict[str, Any]:
    config = super().get_config()
    config["compute_full_loss"] = self._compute_full_loss
    config["slice_by_treatment"] = self._slice_by_treatment
    return config

  @classmethod
  def from_config(cls, config: dict[str, Any]) -> LogLossMeanBaseline:
    return cls(**config)


@tf_keras.utils.register_keras_serializable(package="Uplift")
class LogLossMinimum(tf_keras.metrics.Metric):
  """Computes the minimum achievable (weighted) poisson log loss.

  Given labels `y` and the model's predictions `x`, the minimum loss is obtained
  when `x` equals `y`. In this case the loss is computed as:
  `loss = y - y * log(y) + [y * log(y) - y + 0.5 * log(2 * pi * y)]`

  Note that `[y * log(y) - y + 0.5 * log(2 * pi * y)]` is only computed if
  `compute_full_loss` is set to `True`.
  """

  def __init__(
      self,
      compute_full_loss: bool = False,
      slice_by_treatment: bool = True,
      name: str = "poisson_log_loss_minimum",
      dtype: tf.DType = tf.float32,
  ):
    """Initializes the instance.

    Args:
      compute_full_loss: Specifies whether to compute the full minimum log loss
        or not. Defaults to `False`.
      slice_by_treatment: Specifies whether the loss should be sliced by the
        treatment indicator tensor. If `True`, the metric's result will return
        the loss values sliced by the treatment group. Note that this can only
        be set to `True` when `y_pred` is of type `TwoTowerTrainingOutputs`.
      name: Optional name for the instance.
      dtype: Optional data type for the instance.
    """
    super().__init__(name=name, dtype=dtype)

    if compute_full_loss:
      raise NotImplementedError("Full loss computation is not yet supported.")

    self._compute_full_loss = compute_full_loss
    self._slice_by_treatment = slice_by_treatment

    if slice_by_treatment:
      self._loss = treatment_sliced_metric.TreatmentSlicedMetric(
          metric=tf_keras.metrics.Mean(name=name, dtype=dtype)
      )
    else:
      self._loss = tf_keras.metrics.Mean(name=name, dtype=dtype)

  def update_state(
      self,
      y_true: tf.Tensor,
      y_pred: types.TwoTowerTrainingOutputs | tf.Tensor | None = None,
      sample_weight: tf.Tensor | None = None,
  ):
    is_treatment = {}
    if self._slice_by_treatment:
      if not isinstance(y_pred, types.TwoTowerTrainingOutputs):
        raise ValueError(
            "`slice_by_treatment` must be set to `False` when `y_pred` is not"
            " of type `TwoTowerTrainingOutputs`."
        )
      is_treatment["is_treatment"] = y_pred.is_treatment

    self._loss.update_state(
        _safe_x_minus_xlogx(y_true), sample_weight=sample_weight, **is_treatment
    )

  def result(self) -> tf.Tensor | dict[str, tf.Tensor]:
    return self._loss.result()

  def get_config(self) -> dict[str, Any]:
    config = super().get_config()
    config["compute_full_loss"] = self._compute_full_loss
    config["slice_by_treatment"] = self._slice_by_treatment
    return config

  @classmethod
  def from_config(cls, config: dict[str, Any]) -> LogLossMinimum:
    return cls(**config)


@tf_keras.utils.register_keras_serializable(package="Uplift")
class PseudoRSquared(tf_keras.metrics.Metric):
  """Computes the pseudo R-squared metric for poisson regression.

  The pseudo R-squared is computed from log likelihoods of three models:
  1) LLbaseline: log likelihood of a mean baseline predictor.
  2) LLfit: log likelihood of the fitted model.
  3) LLmax: maximum achievable log likelihood, which occurs when the predictions
  equal to the labels.

  The equation that computes the pseudo R-squared is:
  >>> R_squared = (LLfit - LLbaseline) / (LLmax - LLbaseline)
  """

  def __init__(
      self,
      from_logits: bool = True,
      slice_by_treatment: bool = True,
      name: str = "pseudo_r_squared",
      dtype: tf.DType = tf.float32,
  ):
    """Initializes the instance.

    Args:
      from_logits: When `y_pred` is of type `tf.Tensor`, specifies whether
        `y_pred` represents the model's logits or predictions. Otherwise, when
        `y_pred` is of type `TwoTowerTrainingOutputs`, set this to `True` in
        order to compute the loss using the true logits.
      slice_by_treatment: Specifies whether the loss should be sliced by the
        treatment indicator tensor. If `True`, the metric's result will return
        the loss values sliced by the treatment group. Note that this can only
        be set to `True` when `y_pred` is of type `TwoTowerTrainingOutputs`.
      name: Optional name for the instance.
      dtype: Optional data type for the instance.
    """
    super().__init__(name=name, dtype=dtype)

    self._from_logits = from_logits
    self._slice_by_treatment = slice_by_treatment

    # Since log_loss = -1 * log_likelihood we can just accumulate the losses.
    loss = LogLoss(
        from_logits=from_logits,
        compute_full_loss=False,
        slice_by_treatment=False,
        name=name,
        dtype=dtype,
    )
    minimum_loss = LogLossMinimum(
        compute_full_loss=False,
        slice_by_treatment=False,
        name=name,
        dtype=dtype,
    )
    mean_baseline_loss = LogLossMeanBaseline(
        compute_full_loss=False,
        slice_by_treatment=False,
        name=name,
        dtype=dtype,
    )

    if slice_by_treatment:
      self._model_loss = treatment_sliced_metric.TreatmentSlicedMetric(
          metric=loss
      )
      self._minimum_loss = treatment_sliced_metric.TreatmentSlicedMetric(
          metric=minimum_loss
      )
      self._mean_baseline_loss = treatment_sliced_metric.TreatmentSlicedMetric(
          metric=mean_baseline_loss
      )
    else:
      self._model_loss = loss
      self._minimum_loss = minimum_loss
      self._mean_baseline_loss = mean_baseline_loss

  def update_state(
      self,
      y_true: tf.Tensor,
      y_pred: types.TwoTowerTrainingOutputs | tf.Tensor,
      sample_weight: tf.Tensor | None = None,
  ):
    is_treatment = {}
    if self._slice_by_treatment:
      if not isinstance(y_pred, types.TwoTowerTrainingOutputs):
        raise ValueError(
            "`slice_by_treatment` must be set to `False` when `y_pred` is not"
            " of type `TwoTowerTrainingOutputs`."
        )
      is_treatment["is_treatment"] = y_pred.is_treatment

    self._model_loss.update_state(
        y_true, y_pred=y_pred, sample_weight=sample_weight, **is_treatment
    )
    self._minimum_loss.update_state(
        y_true, y_pred=y_pred, sample_weight=sample_weight, **is_treatment
    )
    self._mean_baseline_loss.update_state(
        y_true, y_pred=y_pred, sample_weight=sample_weight, **is_treatment
    )

  def result(self) -> tf.Tensor | dict[str, tf.Tensor]:
    def _pseudo_r_squared(
        loss_model: tf.Tensor, loss_baseline: tf.Tensor, loss_min: tf.Tensor
    ) -> tf.Tensor:
      return tf.math.divide_no_nan(
          loss_model - loss_baseline, loss_min - loss_baseline
      )

    return tf.nest.map_structure(
        _pseudo_r_squared,
        self._model_loss.result(),
        self._mean_baseline_loss.result(),
        self._minimum_loss.result(),
    )

  def get_config(self) -> dict[str, Any]:
    config = super().get_config()
    config["from_logits"] = self._from_logits
    config["slice_by_treatment"] = self._slice_by_treatment
    return config

  @classmethod
  def from_config(cls, config: dict[str, Any]) -> PseudoRSquared:
    return cls(**config)
