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

"""Keras metric for computing the (weighted) variance of a tensor."""

from typing import Optional
import tensorflow as tf, tf_keras


class Variance(tf_keras.metrics.Metric):
  """Computes the (weighted) variance of the given values.

  For example, if values is [1, 2, 1, 4] then the variance is 1.5.
  If the weights were specified as [1, 0, 1, 0] then the variance would be 0.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Standalone usage:

  >>> m = Variance()
  >>> m.update_state([1, 2, 1, 4])
  >>> m.result().numpy()
  1.5
  >>> m.reset_state()
  >>> m.update_state([1, 2, 1, 4], sample_weight=[1, 0, 1, 0])
  >>> m.result().numpy()
  0.0

  Usage within a Keras layer:

  ```python
  layer.add_metric(Variance(name="variance")(values))
  ```
  """

  def __init__(self, name: str = "variance", dtype: Optional[tf.DType] = None):
    """Initializes a Variance metric instance.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    super().__init__(name=name, dtype=dtype)
    self._first_moment = tf_keras.metrics.Mean(name="first_moment", dtype=dtype)
    self._second_moment = tf_keras.metrics.Mean(
        name="second_moment", dtype=dtype
    )

  def update_state(
      self, values: tf.Tensor, sample_weight: Optional[tf.Tensor] = None
  ):
    self._first_moment.update_state(values=values, sample_weight=sample_weight)
    self._second_moment.update_state(
        values=tf.math.square(values), sample_weight=sample_weight
    )

  def result(self) -> tf.Tensor:
    return self._second_moment.result() - tf.math.square(
        self._first_moment.result()
    )
