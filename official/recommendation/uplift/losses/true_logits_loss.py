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

"""Wrapper to apply any loss function on the true logits tensor."""

from __future__ import annotations

from typing import Any, Callable, Mapping, MutableMapping

import tensorflow as tf, tf_keras

from official.recommendation.uplift import types


@tf_keras.utils.register_keras_serializable(package="Uplift")
class TrueLogitsLoss(tf_keras.__internal__.losses.LossFunctionWrapper):
  """Computes any arbitrary loss between the labels and the true logits tensor.

  Note that the prediction tensor is expected to be of a tensor of type
  `TwoTowerTrainingOutputs`.

  Example standalone usage:

  >>> y_true = tf.ones((3, 1))
  >>> y_pred = types.TwoTowerTrainingOutputs(
  ...     control_logits=tf.constant([[0], [1], [0]]),
  ...     treatment_logits=tf.constant([[1], [0], [1]]),
  ...     true_logits=tf.ones((3, 1)),
  ...     is_treatment=tf.constant([[True], [False], [True]])
  ... )
  >>> loss = TrueLogitsLoss(
  ...     loss_fn=tf_keras.losses.mean_squared_error,
  ...     name="mean_squared_error",
  ...     reduction=tf_keras.losses.Reduction.SUM,
  ... )
  >>> loss(y_true, y_pred)
  0.0

  Example usage with the `compile()` API:

  ```python
  model.compile(
      optimizer='sgd'.
      loss=TrueLogitsLoss(
          loss_fn=tf_keras.losses.categorical_crossentropy,
          name="categorical_crossentropy",
          from_logits=True
      )
  )
  ```
  """

  def __init__(
      self,
      loss_fn: Callable[[Any, tf.Tensor], tf.Tensor],
      name: str = "true_logits_loss",
      reduction=tf_keras.losses.Reduction.AUTO,
      **loss_fn_kwargs,
  ):
    """Initialize `TrueLogitsLoss` instance.

    Args:
      loss_fn: The loss function to apply between the labels and true logits
        tensor, with signature `loss_fn(y_true, y_pred, **loss_fn_kwargs)`.
      name: Optional name for the instance.
      reduction: Type of `tf_keras.losses.Reduction` to apply to loss. Default
        value is `AUTO`. `AUTO` indicates that the reduction option will be
        determined by the usage context. For almost all cases this defaults to
        `SUM_OVER_BATCH_SIZE`. When used under a `tf.distribute.Strategy`,
        except via `Model.compile()` and `Model.fit()`, using `AUTO` or
        `SUM_OVER_BATCH_SIZE` will raise an error.
      **loss_fn_kwargs: The keyword arguments that are passed on to `loss_fn`.
    """
    super().__init__(
        fn=loss_fn, name=name, reduction=reduction, **loss_fn_kwargs
    )

  def call(
      self, y_true: Any, y_pred: types.TwoTowerTrainingOutputs
  ) -> tf.Tensor:
    return super().call(y_true, y_pred.true_logits)

  def get_config(self) -> Mapping[str, Any]:
    config = super().get_config()
    config["loss_fn"] = config.pop("fn")
    return config

  @classmethod
  def from_config(cls, config: MutableMapping[str, Any]) -> TrueLogitsLoss:
    config["loss_fn"] = tf_keras.losses.get(config["loss_fn"])
    return cls(**config)
