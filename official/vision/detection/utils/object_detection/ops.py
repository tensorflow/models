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

"""A module for helper tensorflow ops.

This is originally implemented in TensorFlow Object Detection API.
"""

import tensorflow as tf

from official.vision.detection.utils.object_detection import shape_utils


def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0,
                            dtype=tf.float32):
  """Creates dense vector with indices set to specific value and rest to zeros.

  This function exists because it is unclear if it is safe to use
    tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
  with indices which are not ordered.
  This function accepts a dynamic size (e.g. tf.shape(tensor)[0])

  Args:
    indices: 1d Tensor with integer indices which are to be set to
      indices_values.
    size: scalar with size (integer) of output Tensor.
    indices_value: values of elements specified by indices in the output vector
    default_value: values of other elements in the output vector.
    dtype: data type.

  Returns:
    dense 1D Tensor of shape [size] with indices set to indices_values and the
        rest set to default_value.
  """
  size = tf.cast(size, dtype=tf.int32)
  zeros = tf.ones([size], dtype=dtype) * default_value
  values = tf.ones_like(indices, dtype=dtype) * indices_value

  return tf.dynamic_stitch(
      [tf.range(size), tf.cast(indices, dtype=tf.int32)], [zeros, values])


def matmul_gather_on_zeroth_axis(params, indices, scope=None):
  """Matrix multiplication based implementation of tf.gather on zeroth axis.

  TODO(rathodv, jonathanhuang): enable sparse matmul option.

  Args:
    params: A float32 Tensor. The tensor from which to gather values. Must be at
      least rank 1.
    indices: A Tensor. Must be one of the following types: int32, int64. Must be
      in range [0, params.shape[0])
    scope: A name for the operation (optional).

  Returns:
    A Tensor. Has the same type as params. Values from params gathered
    from indices given by indices, with shape indices.shape + params.shape[1:].
  """
  scope = scope or 'MatMulGather'
  with tf.name_scope(scope):
    params_shape = shape_utils.combined_static_and_dynamic_shape(params)
    indices_shape = shape_utils.combined_static_and_dynamic_shape(indices)
    params2d = tf.reshape(params, [params_shape[0], -1])
    indicator_matrix = tf.one_hot(indices, params_shape[0])
    gathered_result_flattened = tf.matmul(indicator_matrix, params2d)
    return tf.reshape(gathered_result_flattened,
                      tf.stack(indices_shape + params_shape[1:]))
