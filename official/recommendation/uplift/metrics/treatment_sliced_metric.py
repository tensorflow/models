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

"""Keras metric for computing sliced metric values based on treatment group."""

from typing import Optional

import tensorflow as tf, tf_keras

from official.recommendation.uplift.metrics import sliced_metric


@tf_keras.utils.register_keras_serializable(package="Uplift")
class TreatmentSlicedMetric(sliced_metric.SlicedMetric):
  """Computes (weighted) sliced metric values based on treatment group.

  Two copies are made of the given metric to compute the same metric for the
  control and treatment groups. The metric name is used for the output results,
  with a "/control" and "/treatment" suffix for the control and treatment
  groups. If the given metric outputs a dictionary of tensors the result will be
  a flattened dictionary with all the sliced metric results.

  Example usage:

  >>> sliced_metric = TreatmentSlicedMetric(
  ...     metric=tf_keras.metrics.Mean(name="example/mean")
  ... )
  >>> sliced_metric.update_state(
  ...     values=tf.constant([[0], [1], [5]]),
  ...     is_treatment=tf.constant([[1], [0], [1]]),
  ... )
  >>> sliced_metric.result()
  {
      "example/mean": 2.0,
      "example/mean/control": 1.0,
      "example/mean/treatment": 2.5
  }
  """

  def __init__(self, metric: tf_keras.metrics.Metric):
    """Initializes a `TreatmentSlicedMetric` instance.

    Args:
      metric: A `keras.metrics.Metric` instance with implemented get_config and
        from_config methods. Its update_state method is expected have a
        `update_state(self, values, sample_weight=None)` signature. The control
        and treatment metrics will have the same name as the given metric, with
        "/control" and "/treatment" suffixes.
    """
    super().__init__(
        metric=metric,
        slicing_spec={"control": False, "treatment": True},
        name=f"treatment_sliced_{metric.name}",
    )

  def update_state(
      self,
      values: tf.Tensor,
      is_treatment: tf.Tensor,
      sample_weight: Optional[tf.Tensor] = None,
      **kwargs,
  ):
    """Computes aggregate, control and treatment metric updates.

    Args:
      values: A `tf.Tensor` instance of shape (D0, ..., DN) passed to the
        `update_state` method of the treatment, control, and overall metrics.
      is_treatment: a `tf.Tensor` of shape (D0,) or (D0, 1) castable to boolean
        indicating if the example belongs to the treatment group (True) or
        control group (False).
      sample_weight: optional `tf.Tensor` for the sample weight. If given it
        will also be sliced by the is_treatment tensor.
      **kwargs: Keyword arguments that will be passed to the `update_state`
        method of each metric.
    """
    slicing_feature = tf.reshape(tf.cast(is_treatment, tf.bool), [-1])
    if sample_weight is not None:
      sample_weight = tf.reshape(sample_weight, [-1])
    super().update_state(
        values,
        sample_weight=sample_weight,
        slicing_feature=slicing_feature,
        **kwargs,
    )

  def get_config(self):
    return {"metric": tf_keras.metrics.serialize(self._metric)}

  @classmethod
  def from_config(cls, config):
    config["metric"] = tf_keras.metrics.deserialize(config["metric"])
    return cls(**config)
