# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Preprocess images and bounding boxes for detection.

We perform two sets of operations in preprocessing stage:
(a) operations that are applied to both training and testing data,
(b) operations that are applied only to training data for the purpose of
    data augmentation.

A preprocessing function receives a set of inputs,
e.g. an image and bounding boxes,
performs an operation on them, and returns them.
Some examples are: randomly cropping the image, randomly mirroring the image,
                   randomly changing the brightness, contrast, hue and
                   randomly jittering the bounding boxes.

The image is a rank 4 tensor: [1, height, width, channels] with
dtype=tf.float32. The groundtruth_boxes is a rank 2 tensor: [N, 4] where
in each row there is a box with [ymin xmin ymax xmax].
Boxes are in normalized coordinates meaning
their coordinate values range in [0, 1]

Important Note: In tensor_dict, images is a rank 4 tensor, but preprocessing
functions receive a rank 3 tensor for processing the image. Thus, inside the
preprocess function we squeeze the image to become a rank 3 tensor and then
we pass it to the functions. At the end of the preprocess we expand the image
back to rank 4.
"""

import numpy as np
import tensorflow as tf
from official.vision.utils.object_detection import box_list


def _flip_boxes_left_right(boxes):
  """Left-right flip the boxes.

  Args:
    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4]. Boxes
      are in normalized form meaning their coordinates vary between [0, 1]. Each
      row is in the form of [ymin, xmin, ymax, xmax].

  Returns:
    Flipped boxes.
  """
  ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
  flipped_xmin = tf.subtract(1.0, xmax)
  flipped_xmax = tf.subtract(1.0, xmin)
  flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], 1)
  return flipped_boxes


def _flip_masks_left_right(masks):
  """Left-right flip masks.

  Args:
    masks: rank 3 float32 tensor with shape [num_instances, height, width]
      representing instance masks.

  Returns:
    flipped masks: rank 3 float32 tensor with shape
      [num_instances, height, width] representing instance masks.
  """
  return masks[:, :, ::-1]


def keypoint_flip_horizontal(keypoints,
                             flip_point,
                             flip_permutation,
                             scope=None):
  """Flips the keypoints horizontally around the flip_point.

  This operation flips the x coordinate for each keypoint around the flip_point
  and also permutes the keypoints in a manner specified by flip_permutation.

  Args:
    keypoints: a tensor of shape [num_instances, num_keypoints, 2]
    flip_point:  (float) scalar tensor representing the x coordinate to flip the
      keypoints around.
    flip_permutation: rank 1 int32 tensor containing the keypoint flip
      permutation. This specifies the mapping from original keypoint indices to
      the flipped keypoint indices. This is used primarily for keypoints that
      are not reflection invariant. E.g. Suppose there are 3 keypoints
      representing ['head', 'right_eye', 'left_eye'], then a logical choice for
      flip_permutation might be [0, 2, 1] since we want to swap the 'left_eye'
      and 'right_eye' after a horizontal flip.
    scope: name scope.

  Returns:
    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]
  """
  if not scope:
    scope = 'FlipHorizontal'
  with tf.name_scope(scope):
    keypoints = tf.transpose(a=keypoints, perm=[1, 0, 2])
    keypoints = tf.gather(keypoints, flip_permutation)
    v, u = tf.split(value=keypoints, num_or_size_splits=2, axis=2)
    u = flip_point * 2.0 - u
    new_keypoints = tf.concat([v, u], 2)
    new_keypoints = tf.transpose(a=new_keypoints, perm=[1, 0, 2])
    return new_keypoints


def keypoint_change_coordinate_frame(keypoints, window, scope=None):
  """Changes coordinate frame of the keypoints to be relative to window's frame.

  Given a window of the form [y_min, x_min, y_max, x_max], changes keypoint
  coordinates from keypoints of shape [num_instances, num_keypoints, 2]
  to be relative to this window.

  An example use case is data augmentation: where we are given groundtruth
  keypoints and would like to randomly crop the image to some window. In this
  case we need to change the coordinate frame of each groundtruth keypoint to be
  relative to this new window.

  Args:
    keypoints: a tensor of shape [num_instances, num_keypoints, 2]
    window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]
      window we should change the coordinate frame to.
    scope: name scope.

  Returns:
    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]
  """
  if not scope:
    scope = 'ChangeCoordinateFrame'
  with tf.name_scope(scope):
    win_height = window[2] - window[0]
    win_width = window[3] - window[1]
    new_keypoints = box_list_ops.scale(keypoints - [window[0], window[1]],
                                       1.0 / win_height, 1.0 / win_width)
    return new_keypoints


def keypoint_prune_outside_window(keypoints, window, scope=None):
  """Prunes keypoints that fall outside a given window.

  This function replaces keypoints that fall outside the given window with nan.
  See also clip_to_window which clips any keypoints that fall outside the given
  window.

  Args:
    keypoints: a tensor of shape [num_instances, num_keypoints, 2]
    window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]
      window outside of which the op should prune the keypoints.
    scope: name scope.

  Returns:
    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]
  """
  if not scope:
    scope = 'PruneOutsideWindow'
  with tf.name_scope(scope):
    y, x = tf.split(value=keypoints, num_or_size_splits=2, axis=2)
    win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)

    valid_indices = tf.logical_and(
        tf.logical_and(y >= win_y_min, y <= win_y_max),
        tf.logical_and(x >= win_x_min, x <= win_x_max))

    new_y = tf.where(valid_indices, y, np.nan * tf.ones_like(y))
    new_x = tf.where(valid_indices, x, np.nan * tf.ones_like(x))
    new_keypoints = tf.concat([new_y, new_x], 2)

    return new_keypoints


def random_horizontal_flip(image,
                           boxes=None,
                           masks=None,
                           keypoints=None,
                           keypoint_flip_permutation=None,
                           seed=None):
  """Randomly flips the image and detections horizontally.

  The probability of flipping the image is 50%.

  Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    boxes: (optional) rank 2 float32 tensor with shape [N, 4] containing the
      bounding boxes. Boxes are in normalized form meaning their coordinates
      vary between [0, 1]. Each row is in the form of [ymin, xmin, ymax, xmax].
    masks: (optional) rank 3 float32 tensor with shape [num_instances, height,
      width] containing instance masks. The masks are of the same height, width
      as the input `image`.
    keypoints: (optional) rank 3 float32 tensor with shape [num_instances,
      num_keypoints, 2]. The keypoints are in y-x normalized coordinates.
    keypoint_flip_permutation: rank 1 int32 tensor containing the keypoint flip
      permutation.
    seed: random seed

  Returns:
    image: image which is the same shape as input image.

    If boxes, masks, keypoints, and keypoint_flip_permutation are not None,
    the function also returns the following tensors.

    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
    masks: rank 3 float32 tensor with shape [num_instances, height, width]
           containing instance masks.
    keypoints: rank 3 float32 tensor with shape
               [num_instances, num_keypoints, 2]

  Raises:
    ValueError: if keypoints are provided but keypoint_flip_permutation is not.
  """

  def _flip_image(image):
    # flip image
    image_flipped = tf.image.flip_left_right(image)
    return image_flipped

  if keypoints is not None and keypoint_flip_permutation is None:
    raise ValueError(
        'keypoints are provided but keypoints_flip_permutation is not provided')

  with tf.name_scope('RandomHorizontalFlip'):
    result = []
    # random variable defining whether to do flip or not
    do_a_flip_random = tf.greater(tf.random.uniform([], seed=seed), 0.5)

    # flip image
    image = tf.cond(
        pred=do_a_flip_random,
        true_fn=lambda: _flip_image(image),
        false_fn=lambda: image)
    result.append(image)

    # flip boxes
    if boxes is not None:
      boxes = tf.cond(
          pred=do_a_flip_random,
          true_fn=lambda: _flip_boxes_left_right(boxes),
          false_fn=lambda: boxes)
      result.append(boxes)

    # flip masks
    if masks is not None:
      masks = tf.cond(
          pred=do_a_flip_random,
          true_fn=lambda: _flip_masks_left_right(masks),
          false_fn=lambda: masks)
      result.append(masks)

    # flip keypoints
    if keypoints is not None and keypoint_flip_permutation is not None:
      permutation = keypoint_flip_permutation
      keypoints = tf.cond(
          pred=do_a_flip_random,
          true_fn=lambda: keypoint_flip_horizontal(keypoints, 0.5, permutation),
          false_fn=lambda: keypoints)
      result.append(keypoints)

    return tuple(result)


def _compute_new_static_size(image, min_dimension, max_dimension):
  """Compute new static shape for resize_to_range method."""
  image_shape = image.get_shape().as_list()
  orig_height = image_shape[0]
  orig_width = image_shape[1]
  num_channels = image_shape[2]
  orig_min_dim = min(orig_height, orig_width)
  # Calculates the larger of the possible sizes
  large_scale_factor = min_dimension / float(orig_min_dim)
  # Scaling orig_(height|width) by large_scale_factor will make the smaller
  # dimension equal to min_dimension, save for floating point rounding errors.
  # For reasonably-sized images, taking the nearest integer will reliably
  # eliminate this error.
  large_height = int(round(orig_height * large_scale_factor))
  large_width = int(round(orig_width * large_scale_factor))
  large_size = [large_height, large_width]
  if max_dimension:
    # Calculates the smaller of the possible sizes, use that if the larger
    # is too big.
    orig_max_dim = max(orig_height, orig_width)
    small_scale_factor = max_dimension / float(orig_max_dim)
    # Scaling orig_(height|width) by small_scale_factor will make the larger
    # dimension equal to max_dimension, save for floating point rounding
    # errors. For reasonably-sized images, taking the nearest integer will
    # reliably eliminate this error.
    small_height = int(round(orig_height * small_scale_factor))
    small_width = int(round(orig_width * small_scale_factor))
    small_size = [small_height, small_width]
    new_size = large_size
    if max(large_size) > max_dimension:
      new_size = small_size
  else:
    new_size = large_size
  return tf.constant(new_size + [num_channels])


def _compute_new_dynamic_size(image, min_dimension, max_dimension):
  """Compute new dynamic shape for resize_to_range method."""
  image_shape = tf.shape(input=image)
  orig_height = tf.cast(image_shape[0], dtype=tf.float32)
  orig_width = tf.cast(image_shape[1], dtype=tf.float32)
  num_channels = image_shape[2]
  orig_min_dim = tf.minimum(orig_height, orig_width)
  # Calculates the larger of the possible sizes
  min_dimension = tf.constant(min_dimension, dtype=tf.float32)
  large_scale_factor = min_dimension / orig_min_dim
  # Scaling orig_(height|width) by large_scale_factor will make the smaller
  # dimension equal to min_dimension, save for floating point rounding errors.
  # For reasonably-sized images, taking the nearest integer will reliably
  # eliminate this error.
  large_height = tf.cast(
      tf.round(orig_height * large_scale_factor), dtype=tf.int32)
  large_width = tf.cast(
      tf.round(orig_width * large_scale_factor), dtype=tf.int32)
  large_size = tf.stack([large_height, large_width])
  if max_dimension:
    # Calculates the smaller of the possible sizes, use that if the larger
    # is too big.
    orig_max_dim = tf.maximum(orig_height, orig_width)
    max_dimension = tf.constant(max_dimension, dtype=tf.float32)
    small_scale_factor = max_dimension / orig_max_dim
    # Scaling orig_(height|width) by small_scale_factor will make the larger
    # dimension equal to max_dimension, save for floating point rounding
    # errors. For reasonably-sized images, taking the nearest integer will
    # reliably eliminate this error.
    small_height = tf.cast(
        tf.round(orig_height * small_scale_factor), dtype=tf.int32)
    small_width = tf.cast(
        tf.round(orig_width * small_scale_factor), dtype=tf.int32)
    small_size = tf.stack([small_height, small_width])
    new_size = tf.cond(
        pred=tf.cast(tf.reduce_max(input_tensor=large_size), dtype=tf.float32) >
        max_dimension,
        true_fn=lambda: small_size,
        false_fn=lambda: large_size)
  else:
    new_size = large_size
  return tf.stack(tf.unstack(new_size) + [num_channels])


def resize_to_range(image,
                    masks=None,
                    min_dimension=None,
                    max_dimension=None,
                    method=tf.image.ResizeMethod.BILINEAR,
                    align_corners=False,
                    pad_to_max_dimension=False):
  """Resizes an image so its dimensions are within the provided value.

  The output size can be described by two cases:
  1. If the image can be rescaled so its minimum dimension is equal to the
     provided value without the other dimension exceeding max_dimension,
     then do so.
  2. Otherwise, resize so the largest dimension is equal to max_dimension.

  Args:
    image: A 3D tensor of shape [height, width, channels]
    masks: (optional) rank 3 float32 tensor with shape [num_instances, height,
      width] containing instance masks.
    min_dimension: (optional) (scalar) desired size of the smaller image
      dimension.
    max_dimension: (optional) (scalar) maximum allowed size of the larger image
      dimension.
    method: (optional) interpolation method used in resizing. Defaults to
      BILINEAR.
    align_corners: bool. If true, exactly align all 4 corners of the input and
      output. Defaults to False.
    pad_to_max_dimension: Whether to resize the image and pad it with zeros so
      the resulting image is of the spatial size [max_dimension, max_dimension].
      If masks are included they are padded similarly.

  Returns:
    Note that the position of the resized_image_shape changes based on whether
    masks are present.
    resized_image: A 3D tensor of shape [new_height, new_width, channels],
      where the image has been resized (with bilinear interpolation) so that
      min(new_height, new_width) == min_dimension or
      max(new_height, new_width) == max_dimension.
    resized_masks: If masks is not None, also outputs masks. A 3D tensor of
      shape [num_instances, new_height, new_width].
    resized_image_shape: A 1D tensor of shape [3] containing shape of the
      resized image.

  Raises:
    ValueError: if the image is not a 3D tensor.
  """
  if len(image.get_shape()) != 3:
    raise ValueError('Image should be 3D tensor')

  with tf.name_scope('ResizeToRange'):
    if image.get_shape().is_fully_defined():
      new_size = _compute_new_static_size(image, min_dimension, max_dimension)
    else:
      new_size = _compute_new_dynamic_size(image, min_dimension, max_dimension)
    new_image = tf.image.resize(image, new_size[:-1], method=method)

    if pad_to_max_dimension:
      new_image = tf.image.pad_to_bounding_box(new_image, 0, 0, max_dimension,
                                               max_dimension)

    result = [new_image]
    if masks is not None:
      new_masks = tf.expand_dims(masks, 3)
      new_masks = tf.image.resize(
          new_masks,
          new_size[:-1],
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      new_masks = tf.squeeze(new_masks, 3)
      if pad_to_max_dimension:
        new_masks = tf.image.pad_to_bounding_box(new_masks, 0, 0, max_dimension,
                                                 max_dimension)
      result.append(new_masks)

    result.append(new_size)
    return result


def _copy_extra_fields(boxlist_to_copy_to, boxlist_to_copy_from):
  """Copies the extra fields of boxlist_to_copy_from to boxlist_to_copy_to.

  Args:
    boxlist_to_copy_to: BoxList to which extra fields are copied.
    boxlist_to_copy_from: BoxList from which fields are copied.

  Returns:
    boxlist_to_copy_to with extra fields.
  """
  for field in boxlist_to_copy_from.get_extra_fields():
    boxlist_to_copy_to.add_field(field, boxlist_to_copy_from.get_field(field))
  return boxlist_to_copy_to


def box_list_scale(boxlist, y_scale, x_scale, scope=None):
  """scale box coordinates in x and y dimensions.

  Args:
    boxlist: BoxList holding N boxes
    y_scale: (float) scalar tensor
    x_scale: (float) scalar tensor
    scope: name scope.

  Returns:
    boxlist: BoxList holding N boxes
  """
  if not scope:
    scope = 'Scale'
  with tf.name_scope(scope):
    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist.get(), num_or_size_splits=4, axis=1)
    y_min = y_scale * y_min
    y_max = y_scale * y_max
    x_min = x_scale * x_min
    x_max = x_scale * x_max
    scaled_boxlist = box_list.BoxList(
        tf.concat([y_min, x_min, y_max, x_max], 1))
    return _copy_extra_fields(scaled_boxlist, boxlist)


def keypoint_scale(keypoints, y_scale, x_scale, scope=None):
  """Scales keypoint coordinates in x and y dimensions.

  Args:
    keypoints: a tensor of shape [num_instances, num_keypoints, 2]
    y_scale: (float) scalar tensor
    x_scale: (float) scalar tensor
    scope: name scope.

  Returns:
    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]
  """
  if not scope:
    scope = 'Scale'
  with tf.name_scope(scope):
    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    new_keypoints = keypoints * [[[y_scale, x_scale]]]
    return new_keypoints


def scale_boxes_to_pixel_coordinates(image, boxes, keypoints=None):
  """Scales boxes from normalized to pixel coordinates.

  Args:
    image: A 3D float32 tensor of shape [height, width, channels].
    boxes: A 2D float32 tensor of shape [num_boxes, 4] containing the bounding
      boxes in normalized coordinates. Each row is of the form [ymin, xmin,
      ymax, xmax].
    keypoints: (optional) rank 3 float32 tensor with shape [num_instances,
      num_keypoints, 2]. The keypoints are in y-x normalized coordinates.

  Returns:
    image: unchanged input image.
    scaled_boxes: a 2D float32 tensor of shape [num_boxes, 4] containing the
      bounding boxes in pixel coordinates.
    scaled_keypoints: a 3D float32 tensor with shape
      [num_instances, num_keypoints, 2] containing the keypoints in pixel
      coordinates.
  """
  boxlist = box_list.BoxList(boxes)
  image_height = tf.shape(input=image)[0]
  image_width = tf.shape(input=image)[1]
  scaled_boxes = box_list_scale(boxlist, image_height, image_width).get()
  result = [image, scaled_boxes]
  if keypoints is not None:
    scaled_keypoints = keypoint_scale(keypoints, image_height, image_width)
    result.append(scaled_keypoints)
  return tuple(result)
