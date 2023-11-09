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

"""Common utilities for the Keras uplift library."""

from typing import Tuple
import tensorflow as tf, tf_keras


def split_by_treatment(
    values: tf.Tensor, is_treatment: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Splits a tensor into control and treatment tensors.

  Args:
    values: a `tf.Tensor` of shape (D0, D1, ..., DN).
    is_treatment: a `tf.Tensor` of shape (D0,) or (D0, 1) castable to boolean
      indicating if the example belongs to the treatment group (True) or control
      group (False).

  Returns:
    A tuple with control and treatment values sliced by the is_treatment tensor.
  """
  if is_treatment.shape.rank > 2 or (
      is_treatment.shape == 2 and is_treatment.shape[1] != 1
  ):
    raise ValueError(
        "is_treatment tensor must be a tensor of shape (D0,) (D0, 1) but got a"
        f" tensor of shape {is_treatment.shape} instead."
    )

  if values.shape[0] != is_treatment.shape[0]:
    raise ValueError(
        "values and is_treatment must be tensors of shapes (D0, D1, ..., DN)"
        f" and (D0, 1) (or (D0,)), but got tensors of shapes {values.shape} and"
        f" {is_treatment.shape} respectively."
    )

  if is_treatment.dtype == tf.string:
    raise ValueError(
        "is_treatment must be a tensor castable to boolean but got tensor"
        f" {is_treatment} of dtype {is_treatment.dtype} instead."
    )

  # Assert is_treatment tensor containss only 0 or 1 values.
  if is_treatment.dtype != tf.bool:
    is_treatment_float = tf.cast(is_treatment, tf.float32)
    tf.debugging.assert_equal(
        tf.reduce_all(
            tf.logical_or(is_treatment_float == 1.0, is_treatment_float == 0.0)
        ),
        tf.convert_to_tensor(True),
        message=(
            "When is_treatment is not a boolean tensor all of its values must"
            f" either be 0 or 1, but got tensor {is_treatment} instead."
        ),
    )

  if is_treatment.shape.rank == 1:
    is_treatment = tf.expand_dims(is_treatment, axis=1)

  is_treatment = tf.cast(is_treatment, tf.bool)

  control_indices = tf.cast(tf.where(~is_treatment)[:, 0], dtype=tf.int32)
  treatment_indices = tf.cast(tf.where(is_treatment)[:, 0], dtype=tf.int32)

  control_values = tf.gather(values, control_indices)
  treatment_values = tf.gather(values, treatment_indices)

  return control_values, treatment_values
