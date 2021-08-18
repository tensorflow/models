# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""A set of private math operations used to safely implement the YOLO loss."""
import tensorflow as tf


def rm_nan_inf(x, val=0.0):
  """Remove nan and infinity.

  Args:
    x: any `Tensor` of any type.
    val: value to replace nan and infinity with.

  Returns:
    a `Tensor` with nan and infinity removed.
  """
  cond = tf.math.logical_or(tf.math.is_nan(x), tf.math.is_inf(x))
  val = tf.cast(val, dtype=x.dtype)
  x = tf.where(cond, val, x)
  return x


def rm_nan(x, val=0.0):
  """Remove nan and infinity.

  Args:
    x: any `Tensor` of any type.
    val: value to replace nan.

  Returns:
    a `Tensor` with nan removed.
  """
  cond = tf.math.is_nan(x)
  val = tf.cast(val, dtype=x.dtype)
  x = tf.where(cond, val, x)
  return x


def divide_no_nan(a, b):
  """Nan safe divide operation built to allow model compilation in tflite.

  Args:
    a: any `Tensor` of any type.
    b: any `Tensor` of any type with the same shape as tensor a.

  Returns:
    a `Tensor` representing a divided by b, with all nan values removed.
  """
  zero = tf.cast(0.0, b.dtype)
  return tf.where(b == zero, zero, a / b)


def mul_no_nan(x, y):
  """Nan safe multiply operation.

  Built to allow model compilation in tflite and
  to allow one tensor to mask another. Where ever x is zero the
  multiplication is not computed and the value is replaced with a zero. This is
  required because 0 * nan = nan. This can make computation unstable in some
  cases where the intended behavior is for zero to mean ignore.

  Args:
    x: any `Tensor` of any type.
    y: any `Tensor` of any type with the same shape as tensor x.

  Returns:
    a `Tensor` representing x times y, where x is used to safely mask the
    tensor y.
  """
  return tf.where(x == 0, tf.cast(0, x.dtype), x * y)
