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

"""Common operations."""

import functools
import math
from typing import Optional

from absl import logging
import numpy as np
import tensorflow as tf, tf_keras


def activation_fn(features: tf.Tensor, act_fn: str):
  """Customized non-linear activation type."""
  if act_fn in ('silu', 'swish'):
    return tf.nn.swish(features)
  elif act_fn == 'silu_native':
    return features * tf.sigmoid(features)
  elif act_fn == 'hswish':
    return features * tf.nn.relu6(features + 3) / 6
  elif act_fn == 'relu':
    return tf.nn.relu(features)
  elif act_fn == 'relu6':
    return tf.nn.relu6(features)
  elif act_fn == 'elu':
    return tf.nn.elu(features)
  elif act_fn == 'leaky_relu':
    return tf.nn.leaky_relu(features)
  elif act_fn == 'selu':
    return tf.nn.selu(features)
  elif act_fn == 'mish':
    return features * tf.math.tanh(tf.math.softplus(features))
  elif act_fn == 'gelu':
    return (
        0.5
        * features
        * (
            1
            + tf.tanh(
                np.sqrt(2 / np.pi) * (features + 0.044715 * tf.pow(features, 3))
            )
        )
    )
  else:
    raise ValueError('Unsupported act_fn {}'.format(act_fn))


def get_act_fn(act_fn):
  if act_fn is None:
    act_fn = 'gelu'
  if isinstance(act_fn, str):
    return functools.partial(activation_fn, act_fn=act_fn)
  elif callable(act_fn):
    return act_fn
  else:
    raise ValueError('Unsupported act_fn %s.' % act_fn)


def pooling_2d(inputs, pool_type, stride, **kwargs):
  """Perform 2D pooling."""
  if stride > 1:
    if pool_type == 'max':
      pool_op = tf_keras.layers.MaxPool2D
    elif pool_type == 'avg':
      pool_op = tf_keras.layers.AveragePooling2D
    else:
      raise ValueError('Unsurpported pool_type %s' % pool_type)
    output = pool_op(
        pool_size=(stride, stride), strides=(stride, stride), **kwargs
    )(inputs)
  else:
    output = inputs
  return output


def drop_connect(inputs, training, survival_prob):
  """Drop the entire conv with given survival probability."""
  # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
  if not training:
    return inputs

  # Compute tensor.
  batch_size = tf.shape(inputs)[0]
  random_tensor = survival_prob
  random_tensor += tf.random.uniform([batch_size], dtype=inputs.dtype)
  for _ in range(inputs.shape.rank - 1):
    random_tensor = tf.expand_dims(random_tensor, axis=-1)
  binary_tensor = tf.floor(random_tensor)
  # Unlike conventional way that multiply survival_prob at test time, here we
  # divide survival_prob at training time, such that no addition compute is
  # needed at test time.
  output = inputs / survival_prob * binary_tensor
  return output


def residual_add(residual, shortcut, survival_prob, training):
  """Combine residual and shortcut."""
  if survival_prob is not None and 0 < survival_prob < 1:
    residual = drop_connect(residual, training, survival_prob)
  return shortcut + residual


def maybe_reshape_to_2d(x, height=None):
  """Reshape tensor to 2d if not already 2d."""
  if x.shape.rank == 3:
    _, length, num_channel = x.shape.as_list()
    if height is None:
      height = int(np.sqrt(length))
    else:
      assert length % height == 0
    width = length // height
    logging.debug(
        'Reshape %s -> %s', [length, num_channel], [height, width, num_channel]
    )
    return tf.reshape(x, [-1, height, width, num_channel])
  elif x.shape.rank == 4:
    return x
  else:
    raise ValueError('Unsupport shape {}'.format(x.shape))


def maybe_reshape_to_1d(x):
  """Reshape tensor to 1d if not already 1d."""
  if x.shape.rank == 4:
    _, h, w, num_channel = x.shape.as_list()
    logging.debug('Reshape %s -> %s', [h, w, num_channel], [h * w, num_channel])
    return tf.reshape(x, [-1, h * w, num_channel])
  elif x.shape.rank == 3:
    return x
  else:
    raise ValueError('Unsupport shape {}'.format(x.shape))


def generate_lookup_tensor(
    length: int,
    max_relative_position: Optional[int] = None,
    clamp_out_of_range: bool = False,
    dtype: tf.DType = tf.float32) -> tf.Tensor:
  """Generate a one_hot lookup tensor to reindex embeddings along one dimension.

  Args:
    length: the length to reindex to.
    max_relative_position: the maximum relative position to consider.
      Relative position embeddings for distances above this threshold
      are zeroed out.
    clamp_out_of_range: bool. Whether to clamp out of range locations to the
      maximum relative distance. If False, the out of range locations will be
      filled with all-zero vectors.
    dtype: dtype for the returned lookup tensor.
  Returns:
    ret: [length, length, vocab_size] lookup tensor that satisfies
      ret[n,m,v] = 1{m - n + max_relative_position = v}.
  """
  if max_relative_position is None:
    max_relative_position = length - 1
  vocab_size = 2 * max_relative_position + 1
  ret = np.zeros((length, length, vocab_size))
  for i in range(length):
    for x in range(length):
      v = x - i + max_relative_position
      if abs(x - i) > max_relative_position:
        if clamp_out_of_range:
          v = np.clip(v, 0, vocab_size - 1)
        else:
          continue
      ret[i, x, v] = 1
  return tf.constant(ret, dtype)


def reindex_2d_einsum_lookup(
    relative_position_tensor: tf.Tensor,
    height: int,
    width: int,
    max_relative_height: Optional[int] = None,
    max_relative_width: Optional[int] = None,
    h_axis=None) -> tf.Tensor:
  """Reindex 2d relative position bias with 2 independent einsum lookups.

  Args:
    relative_position_tensor: tensor of shape
      [..., vocab_height, vocab_width, ...].
    height: height to reindex to.
    width: width to reindex to.
    max_relative_height: maximum relative height.
      Position embeddings corresponding to vertical distances larger
      than max_relative_height are zeroed out. None to disable.
    max_relative_width: maximum relative width.
      Position embeddings corresponding to horizontal distances larger
      than max_relative_width are zeroed out. None to disable.
    h_axis: Axis corresponding to vocab_height. Default to 0 if None.

  Returns:
    reindexed_bias: a Tensor of shape
      [..., height * width, height * width, ...]
  """
  height_lookup = generate_lookup_tensor(
      height, max_relative_position=max_relative_height,
      dtype=relative_position_tensor.dtype)
  width_lookup = generate_lookup_tensor(
      width, max_relative_position=max_relative_width,
      dtype=relative_position_tensor.dtype)

  if h_axis is None:
    h_axis = 0

  non_spatial_rank = relative_position_tensor.shape.rank - 2
  non_spatial_expr = ''.join(chr(ord('n') + i) for i in range(non_spatial_rank))
  prefix = non_spatial_expr[:h_axis]
  suffix = non_spatial_expr[h_axis:]

  reindexed_tensor = tf.einsum(
      '{0}hw{1},ixh->{0}ixw{1}'.format(prefix, suffix),
      relative_position_tensor, height_lookup, name='height_lookup')
  reindexed_tensor = tf.einsum(
      '{0}ixw{1},jyw->{0}ijxy{1}'.format(prefix, suffix),
      reindexed_tensor, width_lookup, name='width_lookup')

  ret_shape = relative_position_tensor.shape.as_list()
  ret_shape[h_axis] = height * width
  ret_shape[h_axis + 1] = height * width
  reindexed_tensor = tf.reshape(reindexed_tensor, ret_shape)

  return reindexed_tensor


def float32_softmax(x: tf.Tensor, *args, **kwargs) -> tf.Tensor:
  y = tf.cast(tf.nn.softmax(tf.cast(x, tf.float32), *args, **kwargs), x.dtype)
  return y


def get_shape_from_length(length: int, height: int = 1, width: int = 1):
  """Gets input 2D shape from 1D sequence length."""
  input_height = int(math.sqrt(length * height // width))
  input_width = input_height * width // height
  if input_height * input_width != length:
    raise ValueError(
        f'Invalid sequence length: {length} or shape: ({height, width}).'
    )
  return (input_height, input_width)
