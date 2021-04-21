# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""A module for helper tensorflow ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import six

from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf
import tf_slim as slim
from object_detection.core import standard_fields as fields
from object_detection.utils import shape_utils
from object_detection.utils import spatial_transform_ops as spatial_ops
from object_detection.utils import static_shape


matmul_crop_and_resize = spatial_ops.matmul_crop_and_resize
multilevel_roi_align = spatial_ops.multilevel_roi_align
native_crop_and_resize = spatial_ops.native_crop_and_resize


def expanded_shape(orig_shape, start_dim, num_dims):
  """Inserts multiple ones into a shape vector.

  Inserts an all-1 vector of length num_dims at position start_dim into a shape.
  Can be combined with tf.reshape to generalize tf.expand_dims.

  Args:
    orig_shape: the shape into which the all-1 vector is added (int32 vector)
    start_dim: insertion position (int scalar)
    num_dims: length of the inserted all-1 vector (int scalar)
  Returns:
    An int32 vector of length tf.size(orig_shape) + num_dims.
  """
  with tf.name_scope('ExpandedShape'):
    start_dim = tf.expand_dims(start_dim, 0)  # scalar to rank-1
    before = tf.slice(orig_shape, [0], start_dim)
    add_shape = tf.ones(tf.reshape(num_dims, [1]), dtype=tf.int32)
    after = tf.slice(orig_shape, start_dim, [-1])
    new_shape = tf.concat([before, add_shape, after], 0)
    return new_shape


def normalized_to_image_coordinates(normalized_boxes, image_shape,
                                    parallel_iterations=32):
  """Converts a batch of boxes from normal to image coordinates.

  Args:
    normalized_boxes: a tensor of shape [None, num_boxes, 4] in
      normalized coordinates. The dtype of this tensor must support tf.mul.
    image_shape: a tensor of shape [4] containing the image shape, with same
      dtype as `normalized_boxes`.
    parallel_iterations: parallelism for the map_fn op.

  Returns:
    absolute_boxes: a tensor of shape [None, num_boxes, 4] containing
      the boxes in image coordinates, with same
      dtype as `normalized_boxes`.
  """
  x_scale = tf.cast(image_shape[2], normalized_boxes.dtype)
  y_scale = tf.cast(image_shape[1], normalized_boxes.dtype)
  def _to_absolute_coordinates(normalized_boxes):
    y_min, x_min, y_max, x_max = tf.split(
        value=normalized_boxes, num_or_size_splits=4, axis=1)
    y_min = y_scale * y_min
    y_max = y_scale * y_max
    x_min = x_scale * x_min
    x_max = x_scale * x_max
    scaled_boxes = tf.concat([y_min, x_min, y_max, x_max], 1)
    return scaled_boxes

  absolute_boxes = shape_utils.static_or_dynamic_map_fn(
      _to_absolute_coordinates,
      elems=(normalized_boxes),
      dtype=normalized_boxes.dtype,
      parallel_iterations=parallel_iterations,
      back_prop=True)
  return absolute_boxes


def meshgrid(x, y):
  """Tiles the contents of x and y into a pair of grids.

  Multidimensional analog of numpy.meshgrid, giving the same behavior if x and y
  are vectors. Generally, this will give:

  xgrid(i1, ..., i_m, j_1, ..., j_n) = x(j_1, ..., j_n)
  ygrid(i1, ..., i_m, j_1, ..., j_n) = y(i_1, ..., i_m)

  Keep in mind that the order of the arguments and outputs is reverse relative
  to the order of the indices they go into, done for compatibility with numpy.
  The output tensors have the same shapes.  Specifically:

  xgrid.get_shape() = y.get_shape().concatenate(x.get_shape())
  ygrid.get_shape() = y.get_shape().concatenate(x.get_shape())

  Args:
    x: A tensor of arbitrary shape and rank. xgrid will contain these values
       varying in its last dimensions.
    y: A tensor of arbitrary shape and rank. ygrid will contain these values
       varying in its first dimensions.
  Returns:
    A tuple of tensors (xgrid, ygrid).
  """
  with tf.name_scope('Meshgrid'):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x_exp_shape = expanded_shape(tf.shape(x), 0, tf.rank(y))
    y_exp_shape = expanded_shape(tf.shape(y), tf.rank(y), tf.rank(x))

    xgrid = tf.tile(tf.reshape(x, x_exp_shape), y_exp_shape)
    ygrid = tf.tile(tf.reshape(y, y_exp_shape), x_exp_shape)
    new_shape = y.get_shape().concatenate(x.get_shape())
    xgrid.set_shape(new_shape)
    ygrid.set_shape(new_shape)

    return xgrid, ygrid


def fixed_padding(inputs, kernel_size, rate=1):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    rate: An integer, rate for atrous convolution.

  Returns:
    output: A tensor of size [batch, height_out, width_out, channels] with the
      input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
  pad_total = kernel_size_effective - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                  [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def pad_to_multiple(tensor, multiple):
  """Returns the tensor zero padded to the specified multiple.

  Appends 0s to the end of the first and second dimension (height and width) of
  the tensor until both dimensions are a multiple of the input argument
  'multiple'. E.g. given an input tensor of shape [1, 3, 5, 1] and an input
  multiple of 4, PadToMultiple will append 0s so that the resulting tensor will
  be of shape [1, 4, 8, 1].

  Args:
    tensor: rank 4 float32 tensor, where
            tensor -> [batch_size, height, width, channels].
    multiple: the multiple to pad to.

  Returns:
    padded_tensor: the tensor zero padded to the specified multiple.
  """
  if multiple == 1:
    return tensor

  tensor_shape = tensor.get_shape()
  batch_size = static_shape.get_batch_size(tensor_shape)
  tensor_height = static_shape.get_height(tensor_shape)
  tensor_width = static_shape.get_width(tensor_shape)
  tensor_depth = static_shape.get_depth(tensor_shape)

  if batch_size is None:
    batch_size = tf.shape(tensor)[0]

  if tensor_height is None:
    tensor_height = tf.shape(tensor)[1]
    padded_tensor_height = tf.cast(
        tf.ceil(
            tf.cast(tensor_height, dtype=tf.float32) /
            tf.cast(multiple, dtype=tf.float32)),
        dtype=tf.int32) * multiple
  else:
    padded_tensor_height = int(
        math.ceil(float(tensor_height) / multiple) * multiple)

  if tensor_width is None:
    tensor_width = tf.shape(tensor)[2]
    padded_tensor_width = tf.cast(
        tf.ceil(
            tf.cast(tensor_width, dtype=tf.float32) /
            tf.cast(multiple, dtype=tf.float32)),
        dtype=tf.int32) * multiple
  else:
    padded_tensor_width = int(
        math.ceil(float(tensor_width) / multiple) * multiple)

  if tensor_depth is None:
    tensor_depth = tf.shape(tensor)[3]

  # Use tf.concat instead of tf.pad to preserve static shape
  if padded_tensor_height != tensor_height:
    height_pad = tf.zeros([
        batch_size, padded_tensor_height - tensor_height, tensor_width,
        tensor_depth
    ], dtype=tensor.dtype)
    tensor = tf.concat([tensor, height_pad], 1)
  if padded_tensor_width != tensor_width:
    width_pad = tf.zeros([
        batch_size, padded_tensor_height, padded_tensor_width - tensor_width,
        tensor_depth
    ], dtype=tensor.dtype)
    tensor = tf.concat([tensor, width_pad], 2)

  return tensor


def padded_one_hot_encoding(indices, depth, left_pad):
  """Returns a zero padded one-hot tensor.

  This function converts a sparse representation of indices (e.g., [4]) to a
  zero padded one-hot representation (e.g., [0, 0, 0, 0, 1] with depth = 4 and
  left_pad = 1). If `indices` is empty, the result will simply be a tensor of
  shape (0, depth + left_pad). If depth = 0, then this function just returns
  `None`.

  Args:
    indices: an integer tensor of shape [num_indices].
    depth: depth for the one-hot tensor (integer).
    left_pad: number of zeros to left pad the one-hot tensor with (integer).

  Returns:
    padded_onehot: a tensor with shape (num_indices, depth + left_pad). Returns
      `None` if the depth is zero.

  Raises:
    ValueError: if `indices` does not have rank 1 or if `left_pad` or `depth are
      either negative or non-integers.

  TODO(rathodv): add runtime checks for depth and indices.
  """
  if depth < 0 or not isinstance(depth, six.integer_types):
    raise ValueError('`depth` must be a non-negative integer.')
  if left_pad < 0 or not isinstance(left_pad, six.integer_types):
    raise ValueError('`left_pad` must be a non-negative integer.')
  if depth == 0:
    return None

  rank = len(indices.get_shape().as_list())
  if rank != 1:
    raise ValueError('`indices` must have rank 1, but has rank=%s' % rank)

  def one_hot_and_pad():
    one_hot = tf.cast(tf.one_hot(tf.cast(indices, tf.int64), depth,
                                 on_value=1, off_value=0), tf.float32)
    return tf.pad(one_hot, [[0, 0], [left_pad, 0]], mode='CONSTANT')
  result = tf.cond(tf.greater(tf.size(indices), 0), one_hot_and_pad,
                   lambda: tf.zeros((tf.size(indices), depth + left_pad)))
  return tf.reshape(result, [-1, depth + left_pad])


def dense_to_sparse_boxes(dense_locations, dense_num_boxes, num_classes):
  """Converts bounding boxes from dense to sparse form.

  Args:
    dense_locations:  a [max_num_boxes, 4] tensor in which only the first k rows
      are valid bounding box location coordinates, where k is the sum of
      elements in dense_num_boxes.
    dense_num_boxes: a [max_num_classes] tensor indicating the counts of
       various bounding box classes e.g. [1, 0, 0, 2] means that the first
       bounding box is of class 0 and the second and third bounding boxes are
       of class 3. The sum of elements in this tensor is the number of valid
       bounding boxes.
    num_classes: number of classes

  Returns:
    box_locations: a [num_boxes, 4] tensor containing only valid bounding
       boxes (i.e. the first num_boxes rows of dense_locations)
    box_classes: a [num_boxes] tensor containing the classes of each bounding
       box (e.g. dense_num_boxes = [1, 0, 0, 2] => box_classes = [0, 3, 3]
  """

  num_valid_boxes = tf.reduce_sum(dense_num_boxes)
  box_locations = tf.slice(dense_locations,
                           tf.constant([0, 0]), tf.stack([num_valid_boxes, 4]))
  tiled_classes = [tf.tile([i], tf.expand_dims(dense_num_boxes[i], 0))
                   for i in range(num_classes)]
  box_classes = tf.concat(tiled_classes, 0)
  box_locations.set_shape([None, 4])
  return box_locations, box_classes


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

  return tf.dynamic_stitch([tf.range(size), tf.cast(indices, dtype=tf.int32)],
                           [zeros, values])


def reduce_sum_trailing_dimensions(tensor, ndims):
  """Computes sum across all dimensions following first `ndims` dimensions."""
  return tf.reduce_sum(tensor, axis=tuple(range(ndims, tensor.shape.ndims)))


def retain_groundtruth(tensor_dict, valid_indices):
  """Retains groundtruth by valid indices.

  Args:
    tensor_dict: a dictionary of following groundtruth tensors -
      fields.InputDataFields.groundtruth_boxes
      fields.InputDataFields.groundtruth_classes
      fields.InputDataFields.groundtruth_confidences
      fields.InputDataFields.groundtruth_keypoints
      fields.InputDataFields.groundtruth_instance_masks
      fields.InputDataFields.groundtruth_is_crowd
      fields.InputDataFields.groundtruth_area
      fields.InputDataFields.groundtruth_label_types
      fields.InputDataFields.groundtruth_difficult
    valid_indices: a tensor with valid indices for the box-level groundtruth.

  Returns:
    a dictionary of tensors containing only the groundtruth for valid_indices.

  Raises:
    ValueError: If the shape of valid_indices is invalid.
    ValueError: field fields.InputDataFields.groundtruth_boxes is
      not present in tensor_dict.
  """
  input_shape = valid_indices.get_shape().as_list()
  if not (len(input_shape) == 1 or
          (len(input_shape) == 2 and input_shape[1] == 1)):
    raise ValueError('The shape of valid_indices is invalid.')
  valid_indices = tf.reshape(valid_indices, [-1])
  valid_dict = {}
  if fields.InputDataFields.groundtruth_boxes in tensor_dict:
    # Prevents reshape failure when num_boxes is 0.
    num_boxes = tf.maximum(tf.shape(
        tensor_dict[fields.InputDataFields.groundtruth_boxes])[0], 1)
    for key in tensor_dict:
      if key in [fields.InputDataFields.groundtruth_boxes,
                 fields.InputDataFields.groundtruth_classes,
                 fields.InputDataFields.groundtruth_confidences,
                 fields.InputDataFields.groundtruth_keypoints,
                 fields.InputDataFields.groundtruth_keypoint_visibilities,
                 fields.InputDataFields.groundtruth_instance_masks]:
        valid_dict[key] = tf.gather(tensor_dict[key], valid_indices)
      # Input decoder returns empty tensor when these fields are not provided.
      # Needs to reshape into [num_boxes, -1] for tf.gather() to work.
      elif key in [fields.InputDataFields.groundtruth_is_crowd,
                   fields.InputDataFields.groundtruth_area,
                   fields.InputDataFields.groundtruth_difficult,
                   fields.InputDataFields.groundtruth_label_types]:
        valid_dict[key] = tf.reshape(
            tf.gather(tf.reshape(tensor_dict[key], [num_boxes, -1]),
                      valid_indices), [-1])
      # Fields that are not associated with boxes.
      else:
        valid_dict[key] = tensor_dict[key]
  else:
    raise ValueError('%s not present in input tensor dict.' % (
        fields.InputDataFields.groundtruth_boxes))
  return valid_dict


def retain_groundtruth_with_positive_classes(tensor_dict):
  """Retains only groundtruth with positive class ids.

  Args:
    tensor_dict: a dictionary of following groundtruth tensors -
      fields.InputDataFields.groundtruth_boxes
      fields.InputDataFields.groundtruth_classes
      fields.InputDataFields.groundtruth_confidences
      fields.InputDataFields.groundtruth_keypoints
      fields.InputDataFields.groundtruth_instance_masks
      fields.InputDataFields.groundtruth_is_crowd
      fields.InputDataFields.groundtruth_area
      fields.InputDataFields.groundtruth_label_types
      fields.InputDataFields.groundtruth_difficult

  Returns:
    a dictionary of tensors containing only the groundtruth with positive
    classes.

  Raises:
    ValueError: If groundtruth_classes tensor is not in tensor_dict.
  """
  if fields.InputDataFields.groundtruth_classes not in tensor_dict:
    raise ValueError('`groundtruth classes` not in tensor_dict.')
  keep_indices = tf.where(tf.greater(
      tensor_dict[fields.InputDataFields.groundtruth_classes], 0))
  return retain_groundtruth(tensor_dict, keep_indices)


def replace_nan_groundtruth_label_scores_with_ones(label_scores):
  """Replaces nan label scores with 1.0.

  Args:
    label_scores: a tensor containing object annoation label scores.

  Returns:
    a tensor where NaN label scores have been replaced by ones.
  """
  return tf.where(
      tf.is_nan(label_scores), tf.ones(tf.shape(label_scores)), label_scores)


def filter_groundtruth_with_crowd_boxes(tensor_dict):
  """Filters out groundtruth with boxes corresponding to crowd.

  Args:
    tensor_dict: a dictionary of following groundtruth tensors -
      fields.InputDataFields.groundtruth_boxes
      fields.InputDataFields.groundtruth_classes
      fields.InputDataFields.groundtruth_confidences
      fields.InputDataFields.groundtruth_keypoints
      fields.InputDataFields.groundtruth_instance_masks
      fields.InputDataFields.groundtruth_is_crowd
      fields.InputDataFields.groundtruth_area
      fields.InputDataFields.groundtruth_label_types

  Returns:
    a dictionary of tensors containing only the groundtruth that have bounding
    boxes.
  """
  if fields.InputDataFields.groundtruth_is_crowd in tensor_dict:
    is_crowd = tensor_dict[fields.InputDataFields.groundtruth_is_crowd]
    is_not_crowd = tf.logical_not(is_crowd)
    is_not_crowd_indices = tf.where(is_not_crowd)
    tensor_dict = retain_groundtruth(tensor_dict, is_not_crowd_indices)
  return tensor_dict


def filter_groundtruth_with_nan_box_coordinates(tensor_dict):
  """Filters out groundtruth with no bounding boxes.

  Args:
    tensor_dict: a dictionary of following groundtruth tensors -
      fields.InputDataFields.groundtruth_boxes
      fields.InputDataFields.groundtruth_classes
      fields.InputDataFields.groundtruth_confidences
      fields.InputDataFields.groundtruth_keypoints
      fields.InputDataFields.groundtruth_instance_masks
      fields.InputDataFields.groundtruth_is_crowd
      fields.InputDataFields.groundtruth_area
      fields.InputDataFields.groundtruth_label_types

  Returns:
    a dictionary of tensors containing only the groundtruth that have bounding
    boxes.
  """
  groundtruth_boxes = tensor_dict[fields.InputDataFields.groundtruth_boxes]
  nan_indicator_vector = tf.greater(tf.reduce_sum(tf.cast(
      tf.is_nan(groundtruth_boxes), dtype=tf.int32), reduction_indices=[1]), 0)
  valid_indicator_vector = tf.logical_not(nan_indicator_vector)
  valid_indices = tf.where(valid_indicator_vector)

  return retain_groundtruth(tensor_dict, valid_indices)


def filter_unrecognized_classes(tensor_dict):
  """Filters out class labels that are not unrecognized by the labelmap.

  Decoder would parse unrecognized classes (not included in the labelmap) to
  a label of value -1. Such targets are unecessary for training, and causes
  issue for evaluation, due to labeling mapping logic. This function filters
  those labels out for both training and evaluation.

  Args:
    tensor_dict: dictionary containing input tensors keyed by
      fields.InputDataFields.

  Returns:
    A dictionary keyed by fields.InputDataFields containing the tensors
    obtained after applying the filtering.

  Raises:
    ValueError: If groundtruth_classes tensor is not in tensor_dict.
  """
  if fields.InputDataFields.groundtruth_classes not in tensor_dict:
    raise ValueError('`groundtruth classes` not in tensor_dict.')
  # Refer to tf_example_decoder for how unrecognized labels are handled.
  unrecognized_label = -1
  recognized_indices = tf.where(
      tf.greater(tensor_dict[fields.InputDataFields.groundtruth_classes],
                 unrecognized_label))

  return retain_groundtruth(tensor_dict, recognized_indices)


def normalize_to_target(inputs,
                        target_norm_value,
                        dim,
                        epsilon=1e-7,
                        trainable=True,
                        scope='NormalizeToTarget',
                        summarize=True):
  """L2 normalizes the inputs across the specified dimension to a target norm.

  This op implements the L2 Normalization layer introduced in
  Liu, Wei, et al. "SSD: Single Shot MultiBox Detector."
  and Liu, Wei, Andrew Rabinovich, and Alexander C. Berg.
  "Parsenet: Looking wider to see better." and is useful for bringing
  activations from multiple layers in a convnet to a standard scale.

  Note that the rank of `inputs` must be known and the dimension to which
  normalization is to be applied should be statically defined.

  TODO(jonathanhuang): Add option to scale by L2 norm of the entire input.

  Args:
    inputs: A `Tensor` of arbitrary size.
    target_norm_value: A float value that specifies an initial target norm or
      a list of floats (whose length must be equal to the depth along the
      dimension to be normalized) specifying a per-dimension multiplier
      after normalization.
    dim: The dimension along which the input is normalized.
    epsilon: A small value to add to the inputs to avoid dividing by zero.
    trainable: Whether the norm is trainable or not
    scope: Optional scope for variable_scope.
    summarize: Whether or not to add a tensorflow summary for the op.

  Returns:
    The input tensor normalized to the specified target norm.

  Raises:
    ValueError: If dim is smaller than the number of dimensions in 'inputs'.
    ValueError: If target_norm_value is not a float or a list of floats with
      length equal to the depth along the dimension to be normalized.
  """
  with tf.variable_scope(scope, 'NormalizeToTarget', [inputs]):
    if not inputs.get_shape():
      raise ValueError('The input rank must be known.')
    input_shape = inputs.get_shape().as_list()
    input_rank = len(input_shape)
    if dim < 0 or dim >= input_rank:
      raise ValueError(
          'dim must be non-negative but smaller than the input rank.')
    if not input_shape[dim]:
      raise ValueError('input shape should be statically defined along '
                       'the specified dimension.')
    depth = input_shape[dim]
    if not (isinstance(target_norm_value, float) or
            (isinstance(target_norm_value, list) and
             len(target_norm_value) == depth) and
            all([isinstance(val, float) for val in target_norm_value])):
      raise ValueError('target_norm_value must be a float or a list of floats '
                       'with length equal to the depth along the dimension to '
                       'be normalized.')
    if isinstance(target_norm_value, float):
      initial_norm = depth * [target_norm_value]
    else:
      initial_norm = target_norm_value
    target_norm = slim.model_variable(
        name='weights',
        dtype=tf.float32,
        initializer=tf.constant(initial_norm, dtype=tf.float32),
        trainable=trainable)
    if summarize:
      mean = tf.reduce_mean(target_norm)
      tf.summary.scalar(tf.get_variable_scope().name, mean)
    lengths = epsilon + tf.sqrt(tf.reduce_sum(tf.square(inputs), dim, True))
    mult_shape = input_rank*[1]
    mult_shape[dim] = depth
    return tf.reshape(target_norm, mult_shape) * tf.truediv(inputs, lengths)


def batch_position_sensitive_crop_regions(images,
                                          boxes,
                                          crop_size,
                                          num_spatial_bins,
                                          global_pool,
                                          parallel_iterations=64):
  """Position sensitive crop with batches of images and boxes.

  This op is exactly like `position_sensitive_crop_regions` below but operates
  on batches of images and boxes. See `position_sensitive_crop_regions` function
  below for the operation applied per batch element.

  Args:
    images: A `Tensor`. Must be one of the following types: `uint8`, `int8`,
      `int16`, `int32`, `int64`, `half`, `float32`, `float64`.
      A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
      Both `image_height` and `image_width` need to be positive.
    boxes: A `Tensor` of type `float32`.
      A 3-D tensor of shape `[batch, num_boxes, 4]`. Each box is specified in
      normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value
      of `y` is mapped to the image coordinate at `y * (image_height - 1)`, so
      as the `[0, 1]` interval of normalized image height is mapped to
      `[0, image_height - 1] in image height coordinates. We do allow y1 > y2,
      in which case the sampled crop is an up-down flipped version of the
      original image. The width dimension is treated similarly.
    crop_size: See `position_sensitive_crop_regions` below.
    num_spatial_bins: See `position_sensitive_crop_regions` below.
    global_pool: See `position_sensitive_crop_regions` below.
    parallel_iterations: Number of batch items to process in parallel.

  Returns:
  """
  def _position_sensitive_crop_fn(inputs):
    images, boxes = inputs
    return position_sensitive_crop_regions(
        images,
        boxes,
        crop_size=crop_size,
        num_spatial_bins=num_spatial_bins,
        global_pool=global_pool)

  return shape_utils.static_or_dynamic_map_fn(
      _position_sensitive_crop_fn,
      elems=[images, boxes],
      dtype=tf.float32,
      parallel_iterations=parallel_iterations)


def position_sensitive_crop_regions(image,
                                    boxes,
                                    crop_size,
                                    num_spatial_bins,
                                    global_pool):
  """Position-sensitive crop and pool rectangular regions from a feature grid.

  The output crops are split into `spatial_bins_y` vertical bins
  and `spatial_bins_x` horizontal bins. For each intersection of a vertical
  and a horizontal bin the output values are gathered by performing
  `tf.image.crop_and_resize` (bilinear resampling) on a a separate subset of
  channels of the image. This reduces `depth` by a factor of
  `(spatial_bins_y * spatial_bins_x)`.

  When global_pool is True, this function implements a differentiable version
  of position-sensitive RoI pooling used in
  [R-FCN detection system](https://arxiv.org/abs/1605.06409).

  When global_pool is False, this function implements a differentiable version
  of position-sensitive assembling operation used in
  [instance FCN](https://arxiv.org/abs/1603.08678).

  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `int8`,
      `int16`, `int32`, `int64`, `half`, `float32`, `float64`.
      A 3-D tensor of shape `[image_height, image_width, depth]`.
      Both `image_height` and `image_width` need to be positive.
    boxes: A `Tensor` of type `float32`.
      A 2-D tensor of shape `[num_boxes, 4]`. Each box is specified in
      normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value
      of `y` is mapped to the image coordinate at `y * (image_height - 1)`, so
      as the `[0, 1]` interval of normalized image height is mapped to
      `[0, image_height - 1] in image height coordinates. We do allow y1 > y2,
      in which case the sampled crop is an up-down flipped version of the
      original image. The width dimension is treated similarly.
    crop_size: A list of two integers `[crop_height, crop_width]`. All
      cropped image patches are resized to this size. The aspect ratio of the
      image content is not preserved. Both `crop_height` and `crop_width` need
      to be positive.
    num_spatial_bins: A list of two integers `[spatial_bins_y, spatial_bins_x]`.
      Represents the number of position-sensitive bins in y and x directions.
      Both values should be >= 1. `crop_height` should be divisible by
      `spatial_bins_y`, and similarly for width.
      The number of image channels should be divisible by
      (spatial_bins_y * spatial_bins_x).
      Suggested value from R-FCN paper: [3, 3].
    global_pool: A boolean variable.
      If True, we perform average global pooling on the features assembled from
        the position-sensitive score maps.
      If False, we keep the position-pooled features without global pooling
        over the spatial coordinates.
      Note that using global_pool=True is equivalent to but more efficient than
        running the function with global_pool=False and then performing global
        average pooling.

  Returns:
    position_sensitive_features: A 4-D tensor of shape
      `[num_boxes, K, K, crop_channels]`,
      where `crop_channels = depth / (spatial_bins_y * spatial_bins_x)`,
      where K = 1 when global_pool is True (Average-pooled cropped regions),
      and K = crop_size when global_pool is False.
  Raises:
    ValueError: Raised in four situations:
      `num_spatial_bins` is not >= 1;
      `num_spatial_bins` does not divide `crop_size`;
      `(spatial_bins_y*spatial_bins_x)` does not divide `depth`;
      `bin_crop_size` is not square when global_pool=False due to the
        constraint in function space_to_depth.
  """
  total_bins = 1
  bin_crop_size = []

  for (num_bins, crop_dim) in zip(num_spatial_bins, crop_size):
    if num_bins < 1:
      raise ValueError('num_spatial_bins should be >= 1')

    if crop_dim % num_bins != 0:
      raise ValueError('crop_size should be divisible by num_spatial_bins')

    total_bins *= num_bins
    bin_crop_size.append(crop_dim // num_bins)

  if not global_pool and bin_crop_size[0] != bin_crop_size[1]:
    raise ValueError('Only support square bin crop size for now.')

  ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
  spatial_bins_y, spatial_bins_x = num_spatial_bins

  # Split each box into spatial_bins_y * spatial_bins_x bins.
  position_sensitive_boxes = []
  for bin_y in range(spatial_bins_y):
    step_y = (ymax - ymin) / spatial_bins_y
    for bin_x in range(spatial_bins_x):
      step_x = (xmax - xmin) / spatial_bins_x
      box_coordinates = [ymin + bin_y * step_y,
                         xmin + bin_x * step_x,
                         ymin + (bin_y + 1) * step_y,
                         xmin + (bin_x + 1) * step_x,
                        ]
      position_sensitive_boxes.append(tf.stack(box_coordinates, axis=1))

  image_splits = tf.split(value=image, num_or_size_splits=total_bins, axis=2)

  image_crops = []
  for (split, box) in zip(image_splits, position_sensitive_boxes):
    if split.shape.is_fully_defined() and box.shape.is_fully_defined():
      crop = tf.squeeze(
          matmul_crop_and_resize(
              tf.expand_dims(split, axis=0), tf.expand_dims(box, axis=0),
              bin_crop_size),
          axis=0)
    else:
      crop = tf.image.crop_and_resize(
          tf.expand_dims(split, 0), box,
          tf.zeros(tf.shape(boxes)[0], dtype=tf.int32), bin_crop_size)
    image_crops.append(crop)

  if global_pool:
    # Average over all bins.
    position_sensitive_features = tf.add_n(image_crops) / len(image_crops)
    # Then average over spatial positions within the bins.
    position_sensitive_features = tf.reduce_mean(
        position_sensitive_features, [1, 2], keepdims=True)
  else:
    # Reorder height/width to depth channel.
    block_size = bin_crop_size[0]
    if block_size >= 2:
      image_crops = [tf.space_to_depth(
          crop, block_size=block_size) for crop in image_crops]

    # Pack image_crops so that first dimension is for position-senstive boxes.
    position_sensitive_features = tf.stack(image_crops, axis=0)

    # Unroll the position-sensitive boxes to spatial positions.
    position_sensitive_features = tf.squeeze(
        tf.batch_to_space_nd(position_sensitive_features,
                             block_shape=[1] + num_spatial_bins,
                             crops=tf.zeros((3, 2), dtype=tf.int32)),
        axis=[0])

    # Reorder back the depth channel.
    if block_size >= 2:
      position_sensitive_features = tf.depth_to_space(
          position_sensitive_features, block_size=block_size)

  return position_sensitive_features


def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width, resize_method='bilinear'):
  """Transforms the box masks back to full image masks.

  Embeds masks in bounding boxes of larger masks whose shapes correspond to
  image shape.

  Args:
    box_masks: A tensor of size [num_masks, mask_height, mask_width].
    boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
           corners. Row i contains [ymin, xmin, ymax, xmax] of the box
           corresponding to mask i. Note that the box corners are in
           normalized coordinates.
    image_height: Image height. The output mask will have the same height as
                  the image height.
    image_width: Image width. The output mask will have the same width as the
                 image width.
    resize_method: The resize method, either 'bilinear' or 'nearest'. Note that
      'bilinear' is only respected if box_masks is a float.

  Returns:
    A tensor of size [num_masks, image_height, image_width] with the same dtype
    as `box_masks`.
  """
  resize_method = 'nearest' if box_masks.dtype == tf.uint8 else resize_method
  # TODO(rathodv): Make this a public function.
  def reframe_box_masks_to_image_masks_default():
    """The default function when there are more than 0 box masks."""
    def transform_boxes_relative_to_boxes(boxes, reference_boxes):
      boxes = tf.reshape(boxes, [-1, 2, 2])
      min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
      max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
      denom = max_corner - min_corner
      # Prevent a divide by zero.
      denom = tf.math.maximum(denom, 1e-4)
      transformed_boxes = (boxes - min_corner) / denom
      return tf.reshape(transformed_boxes, [-1, 4])

    box_masks_expanded = tf.expand_dims(box_masks, axis=3)
    num_boxes = tf.shape(box_masks_expanded)[0]
    unit_boxes = tf.concat(
        [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
    reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)

    # TODO(vighneshb) Use matmul_crop_and_resize so that the output shape
    # is static. This will help us run and test on TPUs.
    resized_crops = tf.image.crop_and_resize(
        image=box_masks_expanded,
        boxes=reverse_boxes,
        box_ind=tf.range(num_boxes),
        crop_size=[image_height, image_width],
        method=resize_method,
        extrapolation_value=0)
    return tf.cast(resized_crops, box_masks.dtype)

  image_masks = tf.cond(
      tf.shape(box_masks)[0] > 0,
      reframe_box_masks_to_image_masks_default,
      lambda: tf.zeros([0, image_height, image_width, 1], box_masks.dtype))
  return tf.squeeze(image_masks, axis=3)


def merge_boxes_with_multiple_labels(boxes,
                                     classes,
                                     confidences,
                                     num_classes,
                                     quantization_bins=10000):
  """Merges boxes with same coordinates and returns K-hot encoded classes.

  Args:
    boxes: A tf.float32 tensor with shape [N, 4] holding N boxes. Only
      normalized coordinates are allowed.
    classes: A tf.int32 tensor with shape [N] holding class indices.
      The class index starts at 0.
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
  boxes_shape = tf.shape(boxes)
  classes_shape = tf.shape(classes)
  confidences_shape = tf.shape(confidences)
  box_class_shape_assert = shape_utils.assert_shape_equal_along_first_dimension(
      boxes_shape, classes_shape)
  box_confidence_shape_assert = (
      shape_utils.assert_shape_equal_along_first_dimension(
          boxes_shape, confidences_shape))
  box_dimension_assert = tf.assert_equal(boxes_shape[1], 4)
  box_normalized_assert = shape_utils.assert_box_normalized(boxes)

  with tf.control_dependencies(
      [box_class_shape_assert, box_confidence_shape_assert,
       box_dimension_assert, box_normalized_assert]):
    quantized_boxes = tf.to_int64(boxes * (quantization_bins - 1))
    ymin, xmin, ymax, xmax = tf.unstack(quantized_boxes, axis=1)
    hashcodes = (
        ymin +
        xmin * quantization_bins +
        ymax * quantization_bins * quantization_bins +
        xmax * quantization_bins * quantization_bins * quantization_bins)
    unique_hashcodes, unique_indices = tf.unique(hashcodes)
    num_boxes = tf.shape(boxes)[0]
    num_unique_boxes = tf.shape(unique_hashcodes)[0]
    merged_box_indices = tf.unsorted_segment_min(
        tf.range(num_boxes), unique_indices, num_unique_boxes)
    merged_boxes = tf.gather(boxes, merged_box_indices)
    unique_indices = tf.to_int64(unique_indices)
    classes = tf.to_int64(classes)

    def map_box_encodings(i):
      """Produces box K-hot and score encodings for each class index."""
      box_mask = tf.equal(
          unique_indices, i * tf.ones(num_boxes, dtype=tf.int64))
      box_mask = tf.reshape(box_mask, [-1])
      box_indices = tf.boolean_mask(classes, box_mask)
      box_confidences = tf.boolean_mask(confidences, box_mask)
      box_class_encodings = tf.sparse_to_dense(
          box_indices, [num_classes], tf.constant(1, dtype=tf.int64),
          validate_indices=False)
      box_confidence_encodings = tf.sparse_to_dense(
          box_indices, [num_classes], box_confidences, validate_indices=False)
      return box_class_encodings, box_confidence_encodings

    # Important to avoid int32 here since there is no GPU kernel for int32.
    # int64 and float32 are fine.
    class_encodings, confidence_encodings = tf.map_fn(
        map_box_encodings,
        tf.range(tf.to_int64(num_unique_boxes)),
        back_prop=False,
        dtype=(tf.int64, tf.float32))

    merged_boxes = tf.reshape(merged_boxes, [-1, 4])
    class_encodings = tf.cast(class_encodings, dtype=tf.int32)
    class_encodings = tf.reshape(class_encodings, [-1, num_classes])
    confidence_encodings = tf.reshape(confidence_encodings, [-1, num_classes])
    merged_box_indices = tf.reshape(merged_box_indices, [-1])
    return (merged_boxes, class_encodings, confidence_encodings,
            merged_box_indices)


def nearest_neighbor_upsampling(input_tensor, scale=None, height_scale=None,
                                width_scale=None):
  """Nearest neighbor upsampling implementation.

  Nearest neighbor upsampling function that maps input tensor with shape
  [batch_size, height, width, channels] to [batch_size, height * scale
  , width * scale, channels]. This implementation only uses reshape and
  broadcasting to make it TPU compatible.

  Args:
    input_tensor: A float32 tensor of size [batch, height_in, width_in,
      channels].
    scale: An integer multiple to scale resolution of input data in both height
      and width dimensions.
    height_scale: An integer multiple to scale the height of input image. This
      option when provided overrides `scale` option.
    width_scale: An integer multiple to scale the width of input image. This
      option when provided overrides `scale` option.
  Returns:
    data_up: A float32 tensor of size
      [batch, height_in*scale, width_in*scale, channels].

  Raises:
    ValueError: If both scale and height_scale or if both scale and width_scale
      are None.
  """
  if not scale and (height_scale is None or width_scale is None):
    raise ValueError('Provide either `scale` or `height_scale` and'
                     ' `width_scale`.')
  with tf.name_scope('nearest_neighbor_upsampling'):
    h_scale = scale if height_scale is None else height_scale
    w_scale = scale if width_scale is None else width_scale
    (batch_size, height, width,
     channels) = shape_utils.combined_static_and_dynamic_shape(input_tensor)
    output_tensor = tf.stack([input_tensor] * w_scale, axis=3)
    output_tensor = tf.stack([output_tensor] * h_scale, axis=2)
    return tf.reshape(output_tensor,
                      [batch_size, height * h_scale, width * w_scale, channels])


def matmul_gather_on_zeroth_axis(params, indices, scope=None):
  """Matrix multiplication based implementation of tf.gather on zeroth axis.

  TODO(rathodv, jonathanhuang): enable sparse matmul option.

  Args:
    params: A float32 Tensor. The tensor from which to gather values.
      Must be at least rank 1.
    indices: A Tensor. Must be one of the following types: int32, int64.
      Must be in range [0, params.shape[0])
    scope: A name for the operation (optional).

  Returns:
    A Tensor. Has the same type as params. Values from params gathered
    from indices given by indices, with shape indices.shape + params.shape[1:].
  """
  with tf.name_scope(scope, 'MatMulGather'):
    params_shape = shape_utils.combined_static_and_dynamic_shape(params)
    indices_shape = shape_utils.combined_static_and_dynamic_shape(indices)
    params2d = tf.reshape(params, [params_shape[0], -1])
    indicator_matrix = tf.one_hot(indices, params_shape[0])
    gathered_result_flattened = tf.matmul(indicator_matrix, params2d)
    return tf.reshape(gathered_result_flattened,
                      tf.stack(indices_shape + params_shape[1:]))


def fpn_feature_levels(num_levels, unit_scale_index, image_ratio, boxes):
  """Returns fpn feature level for each box based on its area.

  See section 4.2 of https://arxiv.org/pdf/1612.03144.pdf for details.

  Args:
    num_levels: An integer indicating the number of feature levels to crop boxes
      from.
    unit_scale_index: An 0-based integer indicating the index of feature map
      which most closely matches the resolution of the pretrained model.
    image_ratio: A float indicating the ratio of input image area to pretraining
      image area.
    boxes: A float tensor of shape [batch, num_boxes, 4] containing boxes of the
      form [ymin, xmin, ymax, xmax] in normalized coordinates.

  Returns:
    An int32 tensor of shape [batch_size, num_boxes] containing feature indices.
  """
  assert num_levels > 0, (
      '`num_levels` must be > 0. Found {}'.format(num_levels))
  assert unit_scale_index < num_levels and unit_scale_index >= 0, (
      '`unit_scale_index` must be in [0, {}). Found {}.'.format(
          num_levels, unit_scale_index))
  box_height_width = boxes[:, :, 2:4] - boxes[:, :, 0:2]
  areas_sqrt = tf.sqrt(tf.reduce_prod(box_height_width, axis=2))
  log_2 = tf.cast(tf.log(2.0), dtype=boxes.dtype)
  levels = tf.cast(
      tf.floordiv(tf.log(areas_sqrt * image_ratio), log_2)
      +
      unit_scale_index,
      dtype=tf.int32)
  levels = tf.maximum(0, tf.minimum(num_levels - 1, levels))
  return levels


def bfloat16_to_float32_nested(input_nested):
  """Convert float32 tensors in a nested structure to bfloat16.

  Args:
    input_nested: A Python dict, values being Tensor or Python list/tuple of
      Tensor or Non-Tensor.

  Returns:
    A Python dict with the same structure as `tensor_dict`,
    with all bfloat16 tensors converted to float32.
  """
  if isinstance(input_nested, tf.Tensor):
    if input_nested.dtype == tf.bfloat16:
      return tf.cast(input_nested, dtype=tf.float32)
    else:
      return input_nested
  elif isinstance(input_nested, (list, tuple)):
    out_tensor_dict = [bfloat16_to_float32_nested(t) for t in input_nested]
  elif isinstance(input_nested, dict):
    out_tensor_dict = {
        k: bfloat16_to_float32_nested(v) for k, v in input_nested.items()
    }
  else:
    return input_nested
  return out_tensor_dict


def gather_with_padding_values(input_tensor, indices, padding_value):
  """Gathers elements from tensor and pads `padding_value` for ignore indices.

  Gathers elements from `input_tensor` based on `indices`. If there are ignore
  indices (which are "-1"s) in `indices`, `padding_value` will be gathered for
  those positions.

  Args:
    input_tensor: A N-D tensor of shape [M, d_1, d_2 .. d_(N-1)] to gather
      values from.
    indices: A 1-D tensor in which each element is either an index in the
      first dimension of input_tensor or -1.
    padding_value: A (N-1)-D tensor of shape [d_1, d_2 .. d_(N-1)] which will be
      used as gathered value for each ignore index in `indices`.

  Returns:
    gathered_tensor: A tensor of shape [L, d_1, d_2 .. d_(N-1)] containing
      values gathered from input_tensor. The first dimension L is equal to the
      length of `indices`.
  """
  padding_value = tf.expand_dims(padding_value, axis=0)
  input_tensor = tf.concat([padding_value, input_tensor], axis=0)
  gather_indices = indices + 1
  gathered_tensor = tf.gather(input_tensor, gather_indices)
  return gathered_tensor





EqualizationLossConfig = collections.namedtuple('EqualizationLossConfig',
                                                ['weight', 'exclude_prefixes'])




def tile_context_tensors(tensor_dict):
  """Tiles context fields to have num_frames along 0-th dimension."""

  num_frames = tf.shape(tensor_dict[fields.InputDataFields.image])[0]

  for key in tensor_dict:
    if key not in fields.SEQUENCE_FIELDS:
      original_tensor = tensor_dict[key]
      tensor_shape = shape_utils.combined_static_and_dynamic_shape(
          original_tensor)
      tensor_dict[key] = tf.tile(
          tf.expand_dims(original_tensor, 0),
          tf.stack([num_frames] + [1] * len(tensor_shape), axis=0))
  return tensor_dict


def decode_image(tensor_dict):
  """Decodes images in a tensor dict."""

  tensor_dict[fields.InputDataFields.image] = tf.io.decode_image(
      tensor_dict[fields.InputDataFields.image], channels=3)
  tensor_dict[fields.InputDataFields.image].set_shape([None, None, 3])
  return tensor_dict


def giou(boxes1, boxes2):
  """Computes generalized IOU between two tensors.

  Each box should be represented as [ymin, xmin, ymax, xmax].

  Args:
    boxes1: a tensor with shape [num_boxes, 4]
    boxes2: a tensor with shape [num_boxes, 4]

  Returns:
    a tensor of shape [num_boxes] containing GIoUs

  """
  pred_ymin, pred_xmin, pred_ymax, pred_xmax = tf.unstack(boxes1, axis=1)
  gt_ymin, gt_xmin, gt_ymax, gt_xmax = tf.unstack(boxes2, axis=1)

  gt_area = (gt_ymax - gt_ymin) * (gt_xmax - gt_xmin)
  pred_area = (pred_ymax - pred_ymin) * (pred_xmax - pred_xmin)

  x1_i = tf.maximum(pred_xmin, gt_xmin)
  x2_i = tf.minimum(pred_xmax, gt_xmax)
  y1_i = tf.maximum(pred_ymin, gt_ymin)
  y2_i = tf.minimum(pred_ymax, gt_ymax)
  intersection_area = tf.maximum(0.0, y2_i - y1_i) * tf.maximum(0.0,
                                                                x2_i - x1_i)

  x1_c = tf.minimum(pred_xmin, gt_xmin)
  x2_c = tf.maximum(pred_xmax, gt_xmax)
  y1_c = tf.minimum(pred_ymin, gt_ymin)
  y2_c = tf.maximum(pred_ymax, gt_ymax)
  hull_area = (y2_c - y1_c) * (x2_c - x1_c)

  union_area = gt_area + pred_area - intersection_area
  iou = tf.where(tf.equal(union_area, 0.0),
                 tf.zeros_like(union_area), intersection_area / union_area)
  giou_ = iou - tf.where(hull_area > 0.0,
                         (hull_area - union_area) / hull_area, iou)
  return giou_


def center_to_corner_coordinate(input_tensor):
  """Converts input boxes from center to corner representation."""
  reshaped_encodings = tf.reshape(input_tensor, [-1, 4])
  ycenter = tf.gather(reshaped_encodings, [0], axis=1)
  xcenter = tf.gather(reshaped_encodings, [1], axis=1)
  h = tf.gather(reshaped_encodings, [2], axis=1)
  w = tf.gather(reshaped_encodings, [3], axis=1)
  ymin = ycenter - h / 2.
  xmin = xcenter - w / 2.
  ymax = ycenter + h / 2.
  xmax = xcenter + w / 2.
  return tf.squeeze(tf.stack([ymin, xmin, ymax, xmax], axis=1))
