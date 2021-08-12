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

import tensorflow as tf


def to_float32(x):
  return tf.cast(x, tf.float32)


def get_shape(tensor, num_dims):
  assert len(tensor.shape.as_list()) == num_dims
  return combined_static_and_dynamic_shape(tensor)


def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.
  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.
  Args:
    tensor: A tensor of any type.
  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape


def flatten_spatial_dimensions(batch_images):
  batch_size, height, width, channels = get_shape(batch_images, 4)
  return tf.reshape(batch_images, [batch_size, height * width,
                                   channels])


def get_num_instances_from_weights(groundtruth_weights_list):
  """Computes the number of instances/boxes from the weights in a batch.
  Args:
    groundtruth_weights_list: A list of float tensors with shape
      [max_num_instances] representing whether there is an actual instance in
      the image (with non-zero value) or is padded to match the
      max_num_instances (with value 0.0). The list represents the batch
      dimension.
  Returns:
    A scalar integer tensor incidating how many instances/boxes are in the
    images in the batch. Note that this function is usually used to normalize
    the loss so the minimum return value is 1 to avoid weird behavior.
  """
  # num_instances = tf.reduce_sum(
  #     [tf.math.count_nonzero(w) for w in groundtruth_weights_list])
  
  # This can execute in graph mode
  groundtruth_weights_list = tf.convert_to_tensor(
      groundtruth_weights_list, dtype=groundtruth_weights_list[0].dtype)
  num_instances = tf.map_fn(
      fn=lambda x: tf.math.count_nonzero(x, dtype=groundtruth_weights_list[
        0].dtype),
      elems=groundtruth_weights_list)
  
  num_instances = tf.reduce_sum(num_instances)
  num_instances = tf.maximum(num_instances, 1)
  return num_instances


def multi_range(limit,
                value_repetitions=1,
                range_repetitions=1,
                dtype=tf.int32):
  """ Creates a sequence with optional value duplication and range repetition.

  As an example (see the Args section for more details),
  _multi_range(limit=2, value_repetitions=3, range_repetitions=4) returns:
  [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
  NOTE: Repurposed from Google OD API.

  Args:
    limit: A 0-D Tensor (scalar). Upper limit of sequence, exclusive.
    value_repetitions: Integer. The number of times a value in the sequence is
      repeated. With value_repetitions=3, the result is [0, 0, 0, 1, 1, 1, ..].
    range_repetitions: Integer. The number of times the range is repeated. With
      range_repetitions=3, the result is [0, 1, 2, .., 0, 1, 2, ..].
    dtype: The type of the elements of the resulting tensor.
  
  Returns:
    A 1-D tensor of type `dtype` and size
      [`limit` * `value_repetitions` * `range_repetitions`] that contains the
      specified range with given repetitions.
  """
  return tf.reshape(
      tf.tile(
          tf.expand_dims(tf.range(limit, dtype=dtype), axis=-1),
          multiples=[range_repetitions, value_repetitions]), [-1])


def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Args:
    tensor: A tensor of any type.

  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape


def get_batch_predictions_from_indices(batch_predictions, indices):
  """Gets the values of predictions in a batch at the given indices.

  The indices are expected to come from the offset targets generation functions
  in this library. The returned value is intended to be used inside a loss
  function.

  Args:
    batch_predictions: A tensor of shape [batch_size, height, width, channels]
      or [batch_size, height, width, class, channels] for class-specific
      features (e.g. keypoint joint offsets).
    indices: A tensor of shape [num_instances, 3] for single class features or
      [num_instances, 4] for multiple classes features.

  Returns:
    values: A tensor of shape [num_instances, channels] holding the predicted
      values at the given indices.
  """
  # Note, gather_nd (and its gradient scatter_nd) runs significantly slower (on
  # TPU) than gather with flattened inputs, so reshape the tensor, flatten the
  # indices, and run gather.
  shape = combined_static_and_dynamic_shape(batch_predictions)

  # [B, H, W, C] -> [H*W, W, 1] or [B, H, W, N, C] -> [H*W*N, W*N, N, 1]
  rev_cum_interior_indices = tf.reverse(tf.math.cumprod(shape[-2:0:-1]), [0])
  rev_cum_interior_indices = tf.concat([rev_cum_interior_indices, [1]], axis=0)

  # Compute flattened indices and gather.
  flattened_inds = tf.linalg.matmul(
      indices, rev_cum_interior_indices[:, tf.newaxis])[:, 0]
  batch_predictions_2d = tf.reshape(batch_predictions, [-1, shape[-1]])
  return tf.gather(batch_predictions_2d, flattened_inds, axis=0)


def add_batch_to_indices(indices):
  shape = tf.shape(indices)
  batch_size = shape[0]
  num_instances = shape[1]
  batch_range = multi_range(limit=batch_size, value_repetitions=num_instances)
  batch_range = tf.reshape(batch_range, shape=(batch_size, num_instances, 1))
  
  return tf.concat([indices, batch_range], axis=-1)


def get_valid_anchor_weights_in_flattened_image(true_image_shapes, height,
                                                width):
  """Computes valid anchor weights for an image assuming pixels will be flattened.

  This function is useful when we only want to penalize valid areas in the
  image in the case when padding is used. The function assumes that the loss
  function will be applied after flattening the spatial dimensions and returns
  anchor weights accordingly.

  Args:
    true_image_shapes: An integer tensor of shape [batch_size, 3] representing
      the true image shape (without padding) for each sample in the batch.
    height: height of the prediction from the network.
    width: width of the prediction from the network.

  Returns:
    valid_anchor_weights: a float tensor of shape [batch_size, height * width]
    with 1s in locations where the spatial coordinates fall within the height
    and width in true_image_shapes.
  """
  
  indices = tf.reshape(tf.range(height * width), [1, -1])
  batch_size = tf.shape(true_image_shapes)[0]
  batch_indices = tf.ones((batch_size, 1), dtype=tf.int32) * indices
  
  y_coords, x_coords, _ = get_row_col_channel_indices_from_flattened_indices(
      batch_indices, width, 1)
  
  max_y, max_x = true_image_shapes[:, 0], true_image_shapes[:, 1]
  max_x = to_float32(tf.expand_dims(max_x, 1))
  max_y = to_float32(tf.expand_dims(max_y, 1))
  
  x_coords = to_float32(x_coords)
  y_coords = to_float32(y_coords)
  
  valid_mask = tf.math.logical_and(x_coords < max_x, y_coords < max_y)
  
  return to_float32(valid_mask)


def get_row_col_channel_indices_from_flattened_indices(indices: int,
                                                       num_cols: int,
                                                       num_channels: int):
  """ Computes row, column and channel indices from flattened indices.

  NOTE: Repurposed from Google OD API.

  Args:
    indices: An `int` tensor of any shape holding the indices in the flattened
      space.
    num_cols: `int`, number of columns in the image (width).
    num_channels: `int`, number of channels in the image.

  Returns:
    row_indices: The row indices corresponding to each of the input indices.
      Same shape as indices.
    col_indices: The column indices corresponding to each of the input indices.
      Same shape as indices.
    channel_indices. The channel indices corresponding to each of the input
      indices.
  """
  # Avoid using mod operator to make the ops more easy to be compatible with
  # different environments, e.g. WASM.
  
  # all inputs and outputs are dtype int32
  row_indices = (indices // num_channels) // num_cols
  col_indices = (indices // num_channels) - row_indices * num_cols
  channel_indices_temp = indices // num_channels
  channel_indices = indices - channel_indices_temp * num_channels
  
  return row_indices, col_indices, channel_indices
