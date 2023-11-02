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

"""A module for helper tensorflow ops.

This is originally implemented in TensorFlow Object Detection API.
"""

import tensorflow as tf, tf_keras

from official.vision.utils.object_detection import shape_utils


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


def merge_boxes_with_multiple_labels(
    boxes, classes, confidences, num_classes, quantization_bins=10000
):
  """Merges boxes with same coordinates and returns K-hot encoded classes.

  Args:
    boxes: A tf.float32 tensor with shape [N, 4] holding N boxes. Only
      normalized coordinates are allowed.
    classes: A tf.int32 tensor with shape [N] holding class indices. The class
      index starts at 0.
    confidences: A tf.float32 tensor with shape [N] holding class confidences.
    num_classes: total number of classes to use for K-hot encoding.
    quantization_bins: the number of bins used to quantize the box coordinate.

  Returns:
    merged_boxes: A tf.float32 tensor with shape [N', 4] holding boxes,
      where N' <= N.
    class_encodings: A tf.int32 tensor with shape [N', num_classes] holding
      K-hot encodings for the merged boxes.
    confidence_encodings: A tf.float32 tensor with shape [N', num_classes]
      holding encodings of confidences for the merged boxes.
    merged_box_indices: A tf.int32 tensor with shape [N'] holding original
      indices of the boxes.
  """
  quantized_boxes = tf.cast(boxes * (quantization_bins - 1), dtype=tf.int64)
  ymin, xmin, ymax, xmax = tf.unstack(quantized_boxes, axis=1)
  hashcodes = (
      ymin
      + xmin * quantization_bins
      + ymax * quantization_bins * quantization_bins
      + xmax * quantization_bins * quantization_bins * quantization_bins
  )
  unique_hashcodes, unique_indices = tf.unique(hashcodes)
  num_boxes = tf.shape(boxes)[0]
  num_unique_boxes = tf.shape(unique_hashcodes)[0]
  merged_box_indices = tf.math.unsorted_segment_min(
      tf.range(num_boxes), unique_indices, num_unique_boxes
  )
  merged_boxes = tf.gather(boxes, merged_box_indices)
  unique_indices = tf.cast(unique_indices, dtype=tf.int64)
  classes = tf.cast(classes, dtype=tf.int64)

  def map_box_encodings(i):
    """Produces box K-hot and score encodings for each class index."""
    box_mask = tf.equal(unique_indices, i * tf.ones(num_boxes, dtype=tf.int64))
    box_mask = tf.reshape(box_mask, [-1])
    box_indices = tf.boolean_mask(classes, box_mask)
    box_confidences = tf.boolean_mask(confidences, box_mask)
    box_indices = tf.cast(box_indices, dtype=tf.int64)

    if tf.rank(box_indices) == 1:
      box_indices = tf.expand_dims(box_indices, axis=-1)

    box_class_encodings = tf.SparseTensor(
        box_indices,
        tf.squeeze(tf.ones_like(box_indices, dtype=tf.int64), axis=-1),
        [num_classes],
    )
    box_class_encodings = tf.sparse.reorder(box_class_encodings)
    box_class_encodings = tf.sparse.to_dense(box_class_encodings)

    if tf.rank(box_confidences) > 1:
      box_confidences = tf.squeeze(box_confidences, axis=-1)

    box_confidence_encodings = tf.SparseTensor(
        box_indices,
        box_confidences,
        [num_classes],
    )
    box_confidence_encodings = tf.sparse.reorder(box_confidence_encodings)
    box_confidence_encodings = tf.sparse.to_dense(box_confidence_encodings)

    return box_class_encodings, box_confidence_encodings

  # Important to avoid int32 here since there is no GPU kernel for int32.
  # int64 and float32 are fine.
  class_encodings, confidence_encodings = tf.nest.map_structure(
      tf.stop_gradient,
      tf.map_fn(
          map_box_encodings,
          tf.range(tf.cast(num_unique_boxes, dtype=tf.int64)),
          dtype=(tf.int64, tf.float32),
      ),
  )

  merged_boxes = tf.reshape(merged_boxes, [-1, 4])
  class_encodings = tf.cast(class_encodings, dtype=tf.int32)
  class_encodings = tf.reshape(class_encodings, [-1, num_classes])
  confidence_encodings = tf.reshape(confidence_encodings, [-1, num_classes])
  merged_box_indices = tf.reshape(merged_box_indices, [-1])
  return (
      merged_boxes,
      class_encodings,
      confidence_encodings,
      merged_box_indices,
  )
