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

"""A replacement for tf.nn.dropout that uses stateless random ops."""

import numbers
from typing import Optional, Sequence, Text, Union

from absl import logging
import tensorflow as tf


def _as_shape(shape: Union[Sequence[int], tf.TensorShape]) -> tf.TensorShape:
  """Converts the given object to a TensorShape."""
  return shape if isinstance(shape, tf.TensorShape) else tf.TensorShape(shape)


def _get_noise_shape(
    x: tf.Tensor, noise_shape: Union[Sequence[int], tf.TensorShape]
) -> Union[tf.Tensor, tf.TensorShape, Sequence[int]]:
  """Computes the shape of the binary mask for dropout."""
  # If noise_shape is none return immediately.
  if noise_shape is None:
    return tf.shape(x)

  try:
    # Best effort to figure out the intended shape.
    # If not possible, let the op to handle it.
    # In eager mode exception will show up.
    noise_shape_ = _as_shape(noise_shape)
  except (TypeError, ValueError):
    return noise_shape

  if x.shape.dims is not None and len(x.shape.dims) == len(noise_shape_.dims):
    new_dims = []
    for i, dim in enumerate(x.shape.dims):
      if noise_shape_.dims[i].value is None and dim.value is not None:
        new_dims.append(dim.value)
      else:
        new_dims.append(noise_shape_.dims[i].value)
    return tf.TensorShape(new_dims)

  return noise_shape


def stateless_dropout(x: tf.Tensor,
                      rate: float,
                      seed: tf.Tensor,
                      noise_shape: Optional[Union[Sequence[int],
                                                  tf.TensorShape]] = None,
                      name: Optional[Text] = None) -> tf.Tensor:
  """Computes dropout: randomly sets elements to zero to prevent overfitting.

  See https://www.tensorflow.org/api_docs/python/tf/nn/dropout.
  This version differs in that the seed is required if the rate is nonzero.

  Args:
    x: A floating point tensor.
    rate: A scalar `Tensor` with the same type as x. The probability that each
      element is dropped. For example, setting rate=0.1 would drop 10% of input
      elements.
    seed: A shape [2] integer Tensor of seeds to the random number generator.
      Must have dtype `tf.int32` when compiling to XLA.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the shape for
      randomly generated keep/drop flags.
    name: A name for this operation (optional).

  Returns:
    A `Tensor` of the same shape of `x`.

  Raises:
    ValueError: If `rate` is not in `[0, 1)` or if `x` is not a floating point
      tensor. `rate=1` is disallowed, because the output would be all zeros,
      which is likely not what was intended.
  """
  with tf.name_scope(name or 'stateless_dropout') as name:
    x = tf.convert_to_tensor(x, name='x')
    if not x.dtype.is_floating:
      raise ValueError('x has to be a floating point tensor since it\'s going '
                       ' to be scaled. Got a %s tensor instead.' % x.dtype)
    if isinstance(rate, numbers.Real):
      if not (rate >= 0 and rate < 1):
        raise ValueError('rate must be a scalar tensor or a float in the '
                         'range [0, 1), got %g' % rate)
      if rate > 0.5:
        logging.log_first_n(
            logging.WARN, 'Large dropout rate: %g (>0.5). In TensorFlow '
            '.x, dropout() uses dropout rate instead of keep_prob. '
            'Please ensure that this is intended.', 5, rate)

    # Early return if nothing needs to be dropped.
    if tf.get_static_value(rate) == 0:
      return x

    rate = tf.convert_to_tensor(rate, dtype=x.dtype, name='rate')
    rate.shape.assert_has_rank(0)
    noise_shape = _get_noise_shape(x, noise_shape)
    # Sample a uniform distribution on [0.0, 1.0) and select values larger than
    # rate.
    #
    # NOTE: Random uniform actually can only generate 2^23 floats on [1.0, 2.0)
    # and subtract 1.0.
    random_tensor = tf.random.stateless_uniform(
        noise_shape, seed=seed, dtype=x.dtype)
    keep_prob = 1 - rate
    scale = 1 / keep_prob
    # NOTE: if (1.0 + rate) - 1 is equal to rate, then we want to consider that
    # float to be selected, hence we use a >= comparison.
    keep_mask = random_tensor >= rate
    ret = x * scale * tf.cast(keep_mask, x.dtype)
    if not tf.executing_eagerly():
      ret.set_shape(x.get_shape())
    return ret
