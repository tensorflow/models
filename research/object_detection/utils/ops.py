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
import math
import numpy as np
import six

import tensorflow as tf

from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import standard_fields as fields
from object_detection.utils import shape_utils
from object_detection.utils import static_shape


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
    normalized_boxes: a float32 tensor of shape [None, num_boxes, 4] in
      normalized coordinates.
    image_shape: a float32 tensor of shape [4] containing the image shape.
    parallel_iterations: parallelism for the map_fn op.

  Returns:
    absolute_boxes: a float32 tensor of shape [None, num_boxes, 4] containg the
      boxes in image coordinates.
  """
  def _to_absolute_coordinates(normalized_boxes):
    return box_list_ops.to_absolute_coordinates(
        box_list.BoxList(normalized_boxes),
        image_shape[1], image_shape[2], check_range=False).get()

  absolute_boxes = shape_utils.static_or_dynamic_map_fn(
      _to_absolute_coordinates,
      elems=(normalized_boxes),
      dtype=tf.float32,
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
  tensor_shape = tensor.get_shape()
  batch_size = static_shape.get_batch_size(tensor_shape)
  tensor_height = static_shape.get_height(tensor_shape)
  tensor_width = static_shape.get_width(tensor_shape)
  tensor_depth = static_shape.get_depth(tensor_shape)

  if batch_size is None:
    batch_size = tf.shape(tensor)[0]

  if tensor_height is None:
    tensor_height = tf.shape(tensor)[1]
    padded_tensor_height = tf.to_int32(
        tf.ceil(tf.to_float(tensor_height) / tf.to_float(multiple))) * multiple
  else:
    padded_tensor_height = int(
        math.ceil(float(tensor_height) / multiple) * multiple)

  if tensor_width is None:
    tensor_width = tf.shape(tensor)[2]
    padded_tensor_width = tf.to_int32(
        tf.ceil(tf.to_float(tensor_width) / tf.to_float(multiple))) * multiple
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
    ])
    tensor = tf.concat([tensor, height_pad], 1)
  if padded_tensor_width != tensor_width:
    width_pad = tf.zeros([
        batch_size, padded_tensor_height, padded_tensor_width - tensor_width,
        tensor_depth
    ])
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
                   lambda: tf.zeros((depth + left_pad, 0)))
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
  size = tf.to_int32(size)
  zeros = tf.ones([size], dtype=dtype) * default_value
  values = tf.ones_like(indices, dtype=dtype) * indices_value

  return tf.dynamic_stitch([tf.range(size), tf.to_int32(indices)],
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
                 fields.InputDataFields.groundtruth_keypoints,
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
  nan_indicator_vector = tf.greater(tf.reduce_sum(tf.to_int32(
      tf.is_nan(groundtruth_boxes)), reduction_indices=[1]), 0)
  valid_indicator_vector = tf.logical_not(nan_indicator_vector)
  valid_indices = tf.where(valid_indicator_vector)

  return retain_groundtruth(tensor_dict, valid_indices)


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
    target_norm = tf.contrib.framework.model_variable(
        name='weights', dtype=tf.float32,
        initializer=tf.constant(initial_norm, dtype=tf.float32),
        trainable=trainable)
    if summarize:
      mean = tf.reduce_mean(target_norm)
      mean = tf.Print(mean, ['NormalizeToTarget:', mean])
      tf.summary.scalar(tf.get_variable_scope().name, mean)
    lengths = epsilon + tf.sqrt(tf.reduce_sum(tf.square(inputs), dim, True))
    mult_shape = input_rank*[1]
    mult_shape[dim] = depth
    return tf.reshape(target_norm, mult_shape) * tf.truediv(inputs, lengths)


def position_sensitive_crop_regions(image,
                                    boxes,
                                    box_ind,
                                    crop_size,
                                    num_spatial_bins,
                                    global_pool,
                                    extrapolation_value=None):
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
      A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
      Both `image_height` and `image_width` need to be positive.
    boxes: A `Tensor` of type `float32`.
      A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
      specifies the coordinates of a box in the `box_ind[i]` image and is
      specified in normalized coordinates `[y1, x1, y2, x2]`. A normalized
      coordinate value of `y` is mapped to the image coordinate at
      `y * (image_height - 1)`, so as the `[0, 1]` interval of normalized image
      height is mapped to `[0, image_height - 1] in image height coordinates.
      We do allow y1 > y2, in which case the sampled crop is an up-down flipped
      version of the original image. The width dimension is treated similarly.
      Normalized coordinates outside the `[0, 1]` range are allowed, in which
      case we use `extrapolation_value` to extrapolate the input image values.
    box_ind:  A `Tensor` of type `int32`.
      A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
      The value of `box_ind[i]` specifies the image that the `i`-th box refers
      to.
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
    extrapolation_value: An optional `float`. Defaults to `0`.
      Value used for extrapolation, when applicable.
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

  image_splits = tf.split(value=image, num_or_size_splits=total_bins, axis=3)

  image_crops = []
  for (split, box) in zip(image_splits, position_sensitive_boxes):
    crop = tf.image.crop_and_resize(split, box, box_ind, bin_crop_size,
                                    extrapolation_value=extrapolation_value)
    image_crops.append(crop)

  if global_pool:
    # Average over all bins.
    position_sensitive_features = tf.add_n(image_crops) / len(image_crops)
    # Then average over spatial positions within the bins.
    position_sensitive_features = tf.reduce_mean(
        position_sensitive_features, [1, 2], keep_dims=True)
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
        squeeze_dims=[0])

    # Reorder back the depth channel.
    if block_size >= 2:
      position_sensitive_features = tf.depth_to_space(
          position_sensitive_features, block_size=block_size)

  return position_sensitive_features


def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width):
  """Transforms the box masks back to full image masks.

  Embeds masks in bounding boxes of larger masks whose shapes correspond to
  image shape.

  Args:
    box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
    boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
           corners. Row i contains [ymin, xmin, ymax, xmax] of the box
           corresponding to mask i. Note that the box corners are in
           normalized coordinates.
    image_height: Image height. The output mask will have the same height as
                  the image height.
    image_width: Image width. The output mask will have the same width as the
                 image width.

  Returns:
    A tf.float32 tensor of size [num_masks, image_height, image_width].
  """
  # TODO(rathodv): Make this a public function.
  def reframe_box_masks_to_image_masks_default():
    """The default function when there are more than 0 box masks."""
    def transform_boxes_relative_to_boxes(boxes, reference_boxes):
      boxes = tf.reshape(boxes, [-1, 2, 2])
      min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
      max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
      transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
      return tf.reshape(transformed_boxes, [-1, 4])

    box_masks_expanded = tf.expand_dims(box_masks, axis=3)
    num_boxes = tf.shape(box_masks_expanded)[0]
    unit_boxes = tf.concat(
        [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
    reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
    return tf.image.crop_and_resize(
        image=box_masks_expanded,
        boxes=reverse_boxes,
        box_ind=tf.range(num_boxes),
        crop_size=[image_height, image_width],
        extrapolation_value=0.0)
  image_masks = tf.cond(
      tf.shape(box_masks)[0] > 0,
      reframe_box_masks_to_image_masks_default,
      lambda: tf.zeros([0, image_height, image_width, 1], dtype=tf.float32))
  return tf.squeeze(image_masks, axis=3)


def merge_boxes_with_multiple_labels(boxes, classes, num_classes):
  """Merges boxes with same coordinates and returns K-hot encoded classes.

  Args:
    boxes: A tf.float32 tensor with shape [N, 4] holding N boxes.
    classes: A tf.int32 tensor with shape [N] holding class indices.
      The class index starts at 0.
    num_classes: total number of classes to use for K-hot encoding.

  Returns:
    merged_boxes: A tf.float32 tensor with shape [N', 4] holding boxes,
      where N' <= N.
    class_encodings: A tf.int32 tensor with shape [N', num_classes] holding
      k-hot encodings for the merged boxes.
    merged_box_indices: A tf.int32 tensor with shape [N'] holding original
      indices of the boxes.
  """
  def merge_numpy_boxes(boxes, classes, num_classes):
    """Python function to merge numpy boxes."""
    if boxes.size < 1:
      return (np.zeros([0, 4], dtype=np.float32),
              np.zeros([0, num_classes], dtype=np.int32),
              np.zeros([0], dtype=np.int32))
    box_to_class_indices = {}
    for box_index in range(boxes.shape[0]):
      box = tuple(boxes[box_index, :].tolist())
      class_index = classes[box_index]
      if box not in box_to_class_indices:
        box_to_class_indices[box] = [box_index, np.zeros([num_classes])]
      box_to_class_indices[box][1][class_index] = 1
    merged_boxes = np.vstack(box_to_class_indices.keys()).astype(np.float32)
    class_encodings = [item[1] for item in box_to_class_indices.values()]
    class_encodings = np.vstack(class_encodings).astype(np.int32)
    merged_box_indices = [item[0] for item in box_to_class_indices.values()]
    merged_box_indices = np.array(merged_box_indices).astype(np.int32)
    return merged_boxes, class_encodings, merged_box_indices

  merged_boxes, class_encodings, merged_box_indices = tf.py_func(
      merge_numpy_boxes, [boxes, classes, num_classes],
      [tf.float32, tf.int32, tf.int32])
  merged_boxes = tf.reshape(merged_boxes, [-1, 4])
  class_encodings = tf.reshape(class_encodings, [-1, num_classes])
  merged_box_indices = tf.reshape(merged_box_indices, [-1])
  return merged_boxes, class_encodings, merged_box_indices


def nearest_neighbor_upsampling(input_tensor, scale):
  """Nearest neighbor upsampling implementation.

  Nearest neighbor upsampling function that maps input tensor with shape
  [batch_size, height, width, channels] to [batch_size, height * scale
  , width * scale, channels]. This implementation only uses reshape and
  broadcasting to make it TPU compatible.

  Args:
    input_tensor: A float32 tensor of size [batch, height_in, width_in,
      channels].
    scale: An integer multiple to scale resolution of input data.
  Returns:
    data_up: A float32 tensor of size
      [batch, height_in*scale, width_in*scale, channels].
  """
  with tf.name_scope('nearest_neighbor_upsampling'):
    (batch_size, height, width,
     channels) = shape_utils.combined_static_and_dynamic_shape(input_tensor)
    output_tensor = tf.reshape(
        input_tensor, [batch_size, height, 1, width, 1, channels]) * tf.ones(
            [1, 1, scale, 1, scale, 1], dtype=input_tensor.dtype)
    return tf.reshape(output_tensor,
                      [batch_size, height * scale, width * scale, channels])


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


def matmul_crop_and_resize(image, boxes, crop_size, scope=None):
  """Matrix multiplication based implementation of the crop and resize op.

  Extracts crops from the input image tensor and bilinearly resizes them
  (possibly with aspect ratio change) to a common output size specified by
  crop_size. This is more general than the crop_to_bounding_box op which
  extracts a fixed size slice from the input image and does not allow
  resizing or aspect ratio change.

  Returns a tensor with crops from the input image at positions defined at
  the bounding box locations in boxes. The cropped boxes are all resized
  (with bilinear interpolation) to a fixed size = `[crop_height, crop_width]`.
  The result is a 4-D tensor `[num_boxes, crop_height, crop_width, depth]`.

  Running time complexity:
    O((# channels) * (# boxes) * (crop_size)^2 * M), where M is the number
  of pixels of the longer edge of the image.

  Note that this operation is meant to replicate the behavior of the standard
  tf.image.crop_and_resize operation but there are a few differences.
  Specifically:
    1) The extrapolation value (the values that are interpolated from outside
      the bounds of the image window) is always zero
    2) Only XLA supported operations are used (e.g., matrix multiplication).
    3) There is no `box_indices` argument --- to run this op on multiple images,
      one must currently call this op independently on each image.
    4) All shapes and the `crop_size` parameter are assumed to be statically
      defined.  Moreover, the number of boxes must be strictly nonzero.

  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `int8`,
      `int16`, `int32`, `int64`, `half`, `float32`, `float64`.
      A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
      Both `image_height` and `image_width` need to be positive.
    boxes: A `Tensor` of type `float32`.
      A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
      specifies the coordinates of a box in the `box_ind[i]` image and is
      specified in normalized coordinates `[y1, x1, y2, x2]`. A normalized
      coordinate value of `y` is mapped to the image coordinate at
      `y * (image_height - 1)`, so as the `[0, 1]` interval of normalized image
      height is mapped to `[0, image_height - 1] in image height coordinates.
      We do allow y1 > y2, in which case the sampled crop is an up-down flipped
      version of the original image. The width dimension is treated similarly.
      Normalized coordinates outside the `[0, 1]` range are allowed, in which
      case we use `extrapolation_value` to extrapolate the input image values.
    crop_size: A list of two integers `[crop_height, crop_width]`. All
      cropped image patches are resized to this size. The aspect ratio of the
      image content is not preserved. Both `crop_height` and `crop_width` need
      to be positive.
    scope: A name for the operation (optional).

  Returns:
    A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`

  Raises:
    ValueError: if image tensor does not have shape
      `[1, image_height, image_width, depth]` and all dimensions statically
      defined.
    ValueError: if boxes tensor does not have shape `[num_boxes, 4]` where
      num_boxes > 0.
    ValueError: if crop_size is not a list of two positive integers
  """
  img_shape = image.shape.as_list()
  boxes_shape = boxes.shape.as_list()
  _, img_height, img_width, _ = img_shape
  if not isinstance(crop_size, list) or len(crop_size) != 2:
    raise ValueError('`crop_size` must be a list of length 2')
  dimensions = img_shape + crop_size + boxes_shape
  if not all([isinstance(dim, int) for dim in dimensions]):
    raise ValueError('all input shapes must be statically defined')
  if len(crop_size) != 2:
    raise ValueError('`crop_size` must be a list of length 2')
  if len(boxes_shape) != 2 or boxes_shape[1] != 4:
    raise ValueError('`boxes` should have shape `[num_boxes, 4]`')
  if len(img_shape) != 4 and img_shape[0] != 1:
    raise ValueError('image should have shape '
                     '`[1, image_height, image_width, depth]`')
  num_crops = boxes_shape[0]
  if not num_crops > 0:
    raise ValueError('number of boxes must be > 0')
  if not (crop_size[0] > 0 and crop_size[1] > 0):
    raise ValueError('`crop_size` must be a list of two positive integers.')

  def _lin_space_weights(num, img_size):
    if num > 1:
      alpha = (img_size - 1) / float(num - 1)
      indices = np.reshape(np.arange(num), (1, num))
      start_weights = alpha * (num - 1 - indices)
      stop_weights = alpha * indices
    else:
      start_weights = num * [.5 * (img_size - 1)]
      stop_weights = num * [.5 * (img_size - 1)]
    return (tf.constant(start_weights, dtype=tf.float32),
            tf.constant(stop_weights, dtype=tf.float32))

  with tf.name_scope(scope, 'MatMulCropAndResize'):
    y1_weights, y2_weights = _lin_space_weights(crop_size[0], img_height)
    x1_weights, x2_weights = _lin_space_weights(crop_size[1], img_width)
    [y1, x1, y2, x2] = tf.split(value=boxes, num_or_size_splits=4, axis=1)

    # Pixel centers of input image and grid points along height and width
    image_idx_h = tf.constant(
        np.reshape(np.arange(img_height), (1, 1, img_height)), dtype=tf.float32)
    image_idx_w = tf.constant(
        np.reshape(np.arange(img_width), (1, 1, img_width)), dtype=tf.float32)
    grid_pos_h = tf.expand_dims(y1 * y1_weights + y2 * y2_weights, 2)
    grid_pos_w = tf.expand_dims(x1 * x1_weights + x2 * x2_weights, 2)

    # Create kernel matrices of pairwise kernel evaluations between pixel
    # centers of image and grid points.
    kernel_h = tf.nn.relu(1 - tf.abs(image_idx_h - grid_pos_h))
    kernel_w = tf.nn.relu(1 - tf.abs(image_idx_w - grid_pos_w))

    # TODO(jonathanhuang): investigate whether all channels can be processed
    # without the explicit unstack --- possibly with a permute and map_fn call.
    result_channels = []
    for channel in tf.unstack(image, axis=3):
      result_channels.append(
          tf.matmul(
              tf.matmul(kernel_h, tf.tile(channel, [num_crops, 1, 1])),
              kernel_w, transpose_b=True))
    return tf.stack(result_channels, axis=3)
