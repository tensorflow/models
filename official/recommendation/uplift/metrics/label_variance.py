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

"""Keras metric for computing the label variance sliced by treatment group."""

import tensorflow as tf, tf_keras

from official.recommendation.uplift import types
from official.recommendation.uplift.metrics import treatment_sliced_metric
from official.recommendation.uplift.metrics import variance


@tf_keras.utils.register_keras_serializable(package="Uplift")
class LabelVariance(tf_keras.metrics.Metric):
  """Computes the overall and treatment sliced label variance.

  Note that the prediction tensor is expected to be of type
  `TwoTowerTrainingOutputs`.

  Example standalone usage:

  >>> label_variance = LabelVariance()
  >>> y_true = tf.constant([1, 2, 3, 10])
  >>> y_pred = types.TwoTowerTrainingOutputs(
  ...     control_logits=tf.zeros(4),
  ...     treatment_logits=tf.zeros(4),
  ...     true_logits=tf.zeros(4),
  ...     is_treatment=tf.constant([True, False, True, False]),
  ... )
  >>> label_variance(y_true=y_true, y_pred=y_pred)
  {
      "label/variance": 12.5
      "label/variance/control": 16.0
      "label/variance/treatment": 1.0
  }

  Example usage with the `model.compile()` API:

  >>> model.compile(
  ...     optimizer="sgd",
  ...     loss=TrueLogitsLoss(tf_keras.losses.mean_squared_error),
  ...     metrics=[LabelVariance()]
  ... )
  """

  def __init__(self, name: str = "label/variance", **kwargs):
    """Initializes the instance.

    Args:
      name: name for the overall label variance metric result. The control and
        treatment label variances will have "/control" and "/treatment" appended
        to the result name.
      **kwargs: other base metric keyword arguments.
    """
    super().__init__(name=name, **kwargs)
    self._sliced_variance = treatment_sliced_metric.TreatmentSlicedMetric(
        metric=variance.Variance(name=name, **kwargs)
    )

  def update_state(
      self,
      y_true: tf.Tensor,
      y_pred: types.TwoTowerTrainingOutputs,
      sample_weight: tf.Tensor | None = None,
  ):
    """Updates the overall, control and treatment label variances.

    Args:
      y_true: tensor labels.
      y_pred: two tower training outputs. The treatment indicator tensor is used
        to slice the labels into control and treatment groups.
      sample_weight: optional sample weight to compute weighted label variances.
        If given, the sample weight will also be sliced by the treatment
        indicator tensor to compute the weighted control and treatment label
        variances.

    Raises:
      TypeError: if y_pred is not of type `TwoTowerTrainingOutputs`.
    """
    if not isinstance(y_pred, types.TwoTowerTrainingOutputs):
      raise TypeError(
          "y_pred must be of type `TwoTowerTrainingOutputs` but got type"
          f" {type(y_pred)} instead."
      )

    self._sliced_variance.update_state(
        values=y_true,
        is_treatment=y_pred.is_treatment,
        sample_weight=sample_weight,
    )

  def result(self) -> dict[str, tf.Tensor]:
    return self._sliced_variance.result()
