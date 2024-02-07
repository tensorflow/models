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

"""Utility functions for input processing."""

import math

import tensorflow as tf, tf_keras

from official.legacy.detection.utils import box_utils
from official.vision.utils.object_detection import preprocessor


def pad_to_fixed_size(input_tensor, size, constant_values=0):
  """Pads data to a fixed length at the first dimension.

  Args:
    input_tensor: `Tensor` with any dimension.
    size: `int` number for the first dimension of output Tensor.
    constant_values: `int` value assigned to the paddings.

  Returns:
    `Tensor` with the first dimension padded to `size`.
  """
  input_shape = input_tensor.get_shape().as_list()
  padding_shape = []

  # Computes the padding length on the first dimension.
  padding_length = tf.maximum(0, size - tf.shape(input_tensor)[0])
  assert_length = tf.Assert(
      tf.greater_equal(padding_length, 0), [padding_length])
  with tf.control_dependencies([assert_length]):
    padding_shape.append(padding_length)

  # Copies shapes of the rest of input shape dimensions.
  for i in range(1, len(input_shape)):
    padding_shape.append(tf.shape(input=input_tensor)[i])

  # Pads input tensor to the fixed first dimension.
  paddings = tf.cast(constant_values * tf.ones(padding_shape),
                     input_tensor.dtype)
  padded_tensor = tf.concat([input_tensor, paddings], axis=0)
  output_shape = input_shape
  output_shape[0] = size
  padded_tensor.set_shape(output_shape)
  return padded_tensor


def normalize_image(image,
                    offset=(0.485, 0.456, 0.406),
                    scale=(0.229, 0.224, 0.225)):
  """Normalizes the image to zero mean and unit variance."""
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  offset = tf.constant(offset)
  offset = tf.expand_dims(offset, axis=0)
  offset = tf.expand_dims(offset, axis=0)
  image -= offset

  scale = tf.constant(scale)
  scale = tf.expand_dims(scale, axis=0)
  scale = tf.expand_dims(scale, axis=0)
  image /= scale
  return image


def compute_padded_size(desired_size, stride):
  """Compute the padded size given the desired size and the stride.

  The padded size will be the smallest rectangle, such that each dimension is
  the smallest multiple of the stride which is larger than the desired
  dimension. For example, if desired_size = (100, 200) and stride = 32,
  the output padded_size = (128, 224).

  Args:
    desired_size: a `Tensor` or `int` list/tuple of two elements representing
      [height, width] of the target output image size.
    stride: an integer, the stride of the backbone network.

  Returns:
    padded_size: a `Tensor` or `int` list/tuple of two elements representing
      [height, width] of the padded output image size.
  """
  if isinstance(desired_size, list) or isinstance(desired_size, tuple):
    padded_size = [
        int(math.ceil(d * 1.0 / stride) * stride) for d in desired_size
    ]
  else:
    padded_size = tf.cast(
        tf.math.ceil(tf.cast(desired_size, dtype=tf.float32) / stride) * stride,
        tf.int32)
  return padded_size


def resize_and_crop_image(image,
                          desired_size,
                          padded_size,
                          aug_scale_min=1.0,
                          aug_scale_max=1.0,
                          seed=1,
                          method=tf.image.ResizeMethod.BILINEAR):
  """Resizes the input image to output size.

  Resize and pad images given the desired output size of the image and
  stride size.

  Here are the preprocessing steps.
  1. For a given image, keep its aspect ratio and rescale the image to make it
     the largest rectangle to be bounded by the rectangle specified by the
     `desired_size`.
  2. Pad the rescaled image to the padded_size.

  Args:
    image: a `Tensor` of shape [height, width, 3] representing an image.
    desired_size: a `Tensor` or `int` list/tuple of two elements representing
      [height, width] of the desired actual output image size.
    padded_size: a `Tensor` or `int` list/tuple of two elements representing
      [height, width] of the padded output image size. Padding will be applied
      after scaling the image to the desired_size.
    aug_scale_min: a `float` with range between [0, 1.0] representing minimum
      random scale applied to desired_size for training scale jittering.
    aug_scale_max: a `float` with range between [1.0, inf] representing maximum
      random scale applied to desired_size for training scale jittering.
    seed: seed for random scale jittering.
    method: function to resize input image to scaled image.

  Returns:
    output_image: `Tensor` of shape [height, width, 3] where [height, width]
      equals to `output_size`.
    image_info: a 2D `Tensor` that encodes the information of the image and the
      applied preprocessing. It is in the format of
      [[original_height, original_width], [desired_height, desired_width],
       [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
      desireed_width] is the actual scaled image size, and [y_scale, x_scale] is
      the scaling factory, which is the ratio of
      scaled dimension / original dimension.
  """
  with tf.name_scope('resize_and_crop_image'):
    image_size = tf.cast(tf.shape(input=image)[0:2], tf.float32)

    random_jittering = (aug_scale_min != 1.0 or aug_scale_max != 1.0)

    if random_jittering:
      random_scale = tf.random.uniform([],
                                       aug_scale_min,
                                       aug_scale_max,
                                       seed=seed)
      scaled_size = tf.round(random_scale * desired_size)
    else:
      scaled_size = desired_size

    scale = tf.minimum(scaled_size[0] / image_size[0],
                       scaled_size[1] / image_size[1])
    scaled_size = tf.round(image_size * scale)

    # Computes 2D image_scale.
    image_scale = scaled_size / image_size

    # Selects non-zero random offset (x, y) if scaled image is larger than
    # desired_size.
    if random_jittering:
      max_offset = scaled_size - desired_size
      max_offset = tf.where(
          tf.less(max_offset, 0), tf.zeros_like(max_offset), max_offset)
      offset = max_offset * tf.random.uniform([
          2,
      ], 0, 1, seed=seed)
      offset = tf.cast(offset, tf.int32)
    else:
      offset = tf.zeros((2,), tf.int32)

    scaled_image = tf.image.resize(
        image, tf.cast(scaled_size, tf.int32), method=method)

    if random_jittering:
      scaled_image = scaled_image[offset[0]:offset[0] + desired_size[0],
                                  offset[1]:offset[1] + desired_size[1], :]

    output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0,
                                                padded_size[0], padded_size[1])

    image_info = tf.stack([
        image_size,
        tf.cast(desired_size, dtype=tf.float32), image_scale,
        tf.cast(offset, tf.float32)
    ])
    return output_image, image_info


def resize_and_crop_image_v2(image,
                             short_side,
                             long_side,
                             padded_size,
                             aug_scale_min=1.0,
                             aug_scale_max=1.0,
                             seed=1,
                             method=tf.image.ResizeMethod.BILINEAR):
  """Resizes the input image to output size (Faster R-CNN style).

  Resize and pad images given the specified short / long side length and the
  stride size.

  Here are the preprocessing steps.
  1. For a given image, keep its aspect ratio and first try to rescale the short
     side of the original image to `short_side`.
  2. If the scaled image after 1 has a long side that exceeds `long_side`, keep
     the aspect ratio and rescal the long side of the image to `long_side`.
  2. Pad the rescaled image to the padded_size.

  Args:
    image: a `Tensor` of shape [height, width, 3] representing an image.
    short_side: a scalar `Tensor` or `int` representing the desired short side
      to be rescaled to.
    long_side: a scalar `Tensor` or `int` representing the desired long side to
      be rescaled to.
    padded_size: a `Tensor` or `int` list/tuple of two elements representing
      [height, width] of the padded output image size. Padding will be applied
      after scaling the image to the desired_size.
    aug_scale_min: a `float` with range between [0, 1.0] representing minimum
      random scale applied to desired_size for training scale jittering.
    aug_scale_max: a `float` with range between [1.0, inf] representing maximum
      random scale applied to desired_size for training scale jittering.
    seed: seed for random scale jittering.
    method: function to resize input image to scaled image.

  Returns:
    output_image: `Tensor` of shape [height, width, 3] where [height, width]
      equals to `output_size`.
    image_info: a 2D `Tensor` that encodes the information of the image and the
      applied preprocessing. It is in the format of
      [[original_height, original_width], [desired_height, desired_width],
       [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
      desired_width] is the actual scaled image size, and [y_scale, x_scale] is
      the scaling factor, which is the ratio of
      scaled dimension / original dimension.
  """
  with tf.name_scope('resize_and_crop_image_v2'):
    image_size = tf.cast(tf.shape(image)[0:2], tf.float32)

    scale_using_short_side = (
        short_side / tf.math.minimum(image_size[0], image_size[1]))
    scale_using_long_side = (
        long_side / tf.math.maximum(image_size[0], image_size[1]))

    scaled_size = tf.math.round(image_size * scale_using_short_side)
    scaled_size = tf.where(
        tf.math.greater(
            tf.math.maximum(scaled_size[0], scaled_size[1]), long_side),
        tf.math.round(image_size * scale_using_long_side), scaled_size)
    desired_size = scaled_size

    random_jittering = (aug_scale_min != 1.0 or aug_scale_max != 1.0)

    if random_jittering:
      random_scale = tf.random.uniform([],
                                       aug_scale_min,
                                       aug_scale_max,
                                       seed=seed)
      scaled_size = tf.math.round(random_scale * scaled_size)

    # Computes 2D image_scale.
    image_scale = scaled_size / image_size

    # Selects non-zero random offset (x, y) if scaled image is larger than
    # desired_size.
    if random_jittering:
      max_offset = scaled_size - desired_size
      max_offset = tf.where(
          tf.math.less(max_offset, 0), tf.zeros_like(max_offset), max_offset)
      offset = max_offset * tf.random.uniform([
          2,
      ], 0, 1, seed=seed)
      offset = tf.cast(offset, tf.int32)
    else:
      offset = tf.zeros((2,), tf.int32)

    scaled_image = tf.image.resize(
        image, tf.cast(scaled_size, tf.int32), method=method)

    if random_jittering:
      scaled_image = scaled_image[offset[0]:offset[0] + desired_size[0],
                                  offset[1]:offset[1] + desired_size[1], :]

    output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0,
                                                padded_size[0], padded_size[1])

    image_info = tf.stack([
        image_size,
        tf.cast(desired_size, dtype=tf.float32), image_scale,
        tf.cast(offset, tf.float32)
    ])
    return output_image, image_info


def resize_and_crop_boxes(boxes, image_scale, output_size, offset):
  """Resizes boxes to output size with scale and offset.

  Args:
    boxes: `Tensor` of shape [N, 4] representing ground truth boxes.
    image_scale: 2D float `Tensor` representing scale factors that apply to
      [height, width] of input image.
    output_size: 2D `Tensor` or `int` representing [height, width] of target
      output image size.
    offset: 2D `Tensor` representing top-left corner [y0, x0] to crop scaled
      boxes.

  Returns:
    boxes: `Tensor` of shape [N, 4] representing the scaled boxes.
  """
  # Adjusts box coordinates based on image_scale and offset.
  boxes *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
  boxes -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
  # Clips the boxes.
  boxes = box_utils.clip_boxes(boxes, output_size)
  return boxes


def resize_and_crop_masks(masks, image_scale, output_size, offset):
  """Resizes boxes to output size with scale and offset.

  Args:
    masks: `Tensor` of shape [N, H, W, 1] representing ground truth masks.
    image_scale: 2D float `Tensor` representing scale factors that apply to
      [height, width] of input image.
    output_size: 2D `Tensor` or `int` representing [height, width] of target
      output image size.
    offset: 2D `Tensor` representing top-left corner [y0, x0] to crop scaled
      boxes.

  Returns:
    masks: `Tensor` of shape [N, H, W, 1] representing the scaled masks.
  """
  mask_size = tf.shape(input=masks)[1:3]
  scaled_size = tf.cast(image_scale * tf.cast(mask_size, image_scale.dtype),
                        tf.int32)
  scaled_masks = tf.image.resize(
      masks, scaled_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  offset = tf.cast(offset, tf.int32)
  scaled_masks = scaled_masks[:, offset[0]:offset[0] + output_size[0],
                              offset[1]:offset[1] + output_size[1], :]

  output_masks = tf.image.pad_to_bounding_box(scaled_masks, 0, 0,
                                              output_size[0], output_size[1])
  return output_masks


def random_horizontal_flip(image, boxes=None, masks=None):
  """Randomly flips input image and bounding boxes."""
  return preprocessor.random_horizontal_flip(image, boxes, masks)
