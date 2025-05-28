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

"""Keras metric for computing fraction of treated examples."""

import tensorflow as tf, tf_keras

from official.recommendation.uplift import types


@tf_keras.utils.register_keras_serializable(package="Uplift")
class TreatmentFraction(tf_keras.metrics.Metric):
  """Computes the fraction of treated examples.

  Note that the prediction tensor is expected to be of type
  `TwoTowerTrainingOutputs`.

  Example standalone usage:

  >>> treatment_fraction = TreatmentFraction()
  >>> y_pred = types.TwoTowerTrainingOutputs(
  ...     is_treatment=tf.constant([True, False, True, True]),
  ... )
  >>> treatment_fraction(y_true=tf.zeros(4), y_pred=y_pred)
  0.75

  Example usage with the `model.compile()` API:

  >>> model.compile(
  ...     optimizer="sgd",
  ...     loss=TrueLogitsLoss(tf_keras.losses.mean_squared_error),
  ...     metrics=[TreatmentFraction()]
  ... )
  """

  def __init__(self, **kwargs):
    """Initializes the instance.

    Args:
      **kwargs: base metric keyword arguments.
    """
    super().__init__(**kwargs)
    self._treatment_fraction = tf_keras.metrics.Mean(**kwargs)

  def update_state(
      self,
      y_true: tf.Tensor,
      y_pred: types.TwoTowerTrainingOutputs,
      sample_weight: tf.Tensor | None = None,
  ) -> None:
    """Updates the treatment fraction.

    Args:
      y_true: tensor labels.
      y_pred: two tower training outputs. The treatment indicator tensor is used
        update the treatment fraction.
      sample_weight: optional sample weight tensor for computing the weighted
        treatment fraction. The unweighted treatment fraction is computed
        instead if it is left as `None`.

    Raises:
      TypeError: if y_pred is not of type `TwoTowerTrainingOutputs`.
    """
    if not isinstance(y_pred, types.TwoTowerTrainingOutputs):
      raise TypeError(
          "y_pred must be of type `TwoTowerTrainingOutputs` but got type"
          f" {type(y_pred)} instead."
      )

    self._treatment_fraction.update_state(
        values=y_pred.is_treatment, sample_weight=sample_weight
    )

  def result(self) -> tf.Tensor:
    return self._treatment_fraction.result()
