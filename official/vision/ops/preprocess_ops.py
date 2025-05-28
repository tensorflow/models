# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Preprocessing ops."""

import math
from typing import Optional, Sequence, Tuple, Union

from six.moves import range
import tensorflow as tf, tf_keras

from official.vision.ops import augment
from official.vision.ops import box_ops

CENTER_CROP_FRACTION = 0.875

# Calculated from the ImageNet training set
MEAN_NORM = (0.485, 0.456, 0.406)
STDDEV_NORM = (0.229, 0.224, 0.225)
MEAN_RGB = tuple(255 * i for i in MEAN_NORM)
STDDEV_RGB = tuple(255 * i for i in STDDEV_NORM)
MEDIAN_RGB = (128.0, 128.0, 128.0)

# Alias for convenience. PLEASE use `box_ops.horizontal_flip_boxes` directly.
horizontal_flip_boxes = box_ops.horizontal_flip_boxes
vertical_flip_boxes = box_ops.vertical_flip_boxes


def clip_or_pad_to_fixed_size(input_tensor, size, constant_values=0):
  """Pads data to a fixed length at the first dimension.

  Args:
    input_tensor: `Tensor` with any dimension.
    size: `int` number for the first dimension of output Tensor.
    constant_values: `int` or `str` value assigned to the paddings.

  Returns:
    `Tensor` with the first dimension padded to `size`.
  """
  input_shape = input_tensor.get_shape().as_list()
  padding_shape = []

  # Computes the padding length on the first dimension, clip input tensor if it
  # is longer than `size`.
  input_length = tf.shape(input_tensor)[0]
  input_length = tf.clip_by_value(input_length, 0, size)
  input_tensor = input_tensor[:input_length]

  padding_length = tf.maximum(0, size - input_length)
  padding_shape.append(padding_length)

  # Copies shapes of the rest of input shape dimensions.
  for i in range(1, len(input_shape)):
    padding_shape.append(tf.shape(input_tensor)[i])

  # Pads input tensor to the fixed first dimension.
  paddings = tf.cast(
      tf.fill(dims=padding_shape, value=constant_values), input_tensor.dtype
  )
  padded_tensor = tf.concat([input_tensor, paddings], axis=0)
  output_shape = input_shape
  output_shape[0] = size
  padded_tensor.set_shape(output_shape)
  return padded_tensor


def normalize_image(
    image: tf.Tensor,
    offset: Sequence[float] = MEAN_NORM,
    scale: Sequence[float] = STDDEV_NORM,
) -> tf.Tensor:
  """Normalizes the image to zero mean and unit variance.

  This function normalizes the input image by subtracting the `offset`
  and dividing by the `scale`.

  **Important Note about Input Types and Normalization:**

  * **Integer Images:** If the input `image` is an integer type (e.g., `uint8`),
    the provided `offset` and `scale` values should be already **normalized**
    to the range [0, 1]. This is because the function converts integer images to
    float32 with values in the range [0, 1] before the normalization happens.

  * **Float Images:** If the input `image` is a float type (e.g., `float32`),
    the `offset` and `scale` values should be in the **same range** as the
    image data.
      - If the image has values in [0, 1], the `offset` and `scale` should
        also be in [0, 1].
      - If the image has values in [0, 255], the `offset` and `scale` should
        also be in [0, 255].

  Args:
    image: A `tf.Tensor` in either:
           (1) float dtype with values in range [0, 1) or [0, 255], or
           (2) int type with values in range [0, 255].
    offset: A tuple of mean values to be subtracted from the image.
    scale: A tuple of normalization factors.

  Returns:
    A normalized image tensor.
  """
  with tf.name_scope('normalize_image'):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return normalize_scaled_float_image(image, offset, scale)


def normalize_scaled_float_image(
    image: tf.Tensor,
    offset: Sequence[float] = MEAN_NORM,
    scale: Sequence[float] = STDDEV_NORM,
):
  """Normalizes a scaled float image to zero mean and unit variance.

  It assumes the input image is float dtype with values in [0, 1) if offset is
  MEAN_NORM, values in [0, 255] if offset is MEAN_RGB.

  Args:
    image: A tf.Tensor in float32 dtype with values in range [0, 1) or [0, 255].
    offset: A tuple of mean values to be subtracted from the image.
    scale: A tuple of normalization factors.

  Returns:
    A normalized image tensor.
  """
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
        tf.int32,
    )
  return padded_size


def resize_and_crop_image(
    image,
    desired_size,
    padded_size,
    aug_scale_min=1.0,
    aug_scale_max=1.0,
    seed=1,
    method=tf.image.ResizeMethod.BILINEAR,
    keep_aspect_ratio=True,
    centered_crop=False,
):
  """Resizes the input image to output size (RetinaNet style).

  Resize and pad images given the desired output size of the image and
  stride size.

  Here are the preprocessing steps.
  1. For a given image, keep its aspect ratio and rescale the image to make it
     the largest rectangle to be bounded by the rectangle specified by the
     `desired_size`.
  2. Pad the rescaled image to the padded_size.

  Args:
    image: a `Tensor` of shape [height, width, c] representing an image.
    desired_size: a `Tensor` or `int` list/tuple of two elements representing
      [height, width] of the desired actual output image size.
    padded_size: a `Tensor` or `int` list/tuple of two elements representing
      [height, width] of the padded output image size. Padding will be applied
      after scaling the image to the desired_size. Can be None to disable
      padding.
    aug_scale_min: a `float` with range between [0, 1.0] representing minimum
      random scale applied to desired_size for training scale jittering.
    aug_scale_max: a `float` with range between [1.0, inf] representing maximum
      random scale applied to desired_size for training scale jittering.
    seed: seed for random scale jittering.
    method: function to resize input image to scaled image.
    keep_aspect_ratio: whether or not to keep the aspect ratio when resizing.
    centered_crop: If `centered_crop` is set to True, then resized crop (if
      smaller than padded size) is place in the center of the image. Default
      behaviour is to place it at left top corner.

  Returns:
    output_image: `Tensor` of shape [height, width, c] where [height, width]
      equals to `output_size`.
    image_info: a 2D `Tensor` that encodes the information of the image and the
      applied preprocessing. It is in the format of
      [[original_height, original_width], [desired_height, desired_width],
       [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
      desired_width] is the actual scaled image size, and [y_scale, x_scale] is
      the scaling factor, which is the ratio of
      scaled dimension / original dimension.
  """
  with tf.name_scope('resize_and_crop_image'):
    image_size = tf.cast(tf.shape(image)[0:2], tf.float32)

    random_jittering = (
        isinstance(aug_scale_min, tf.Tensor)
        or isinstance(aug_scale_max, tf.Tensor)
        or not math.isclose(aug_scale_min, 1.0)
        or not math.isclose(aug_scale_max, 1.0)
    )

    if random_jittering:
      random_scale = tf.random.uniform(
          [], aug_scale_min, aug_scale_max, seed=seed
      )
      scaled_size = tf.round(random_scale * tf.cast(desired_size, tf.float32))
    else:
      scaled_size = tf.cast(desired_size, tf.float32)

    if keep_aspect_ratio:
      scale = tf.minimum(
          scaled_size[0] / image_size[0], scaled_size[1] / image_size[1]
      )
      scaled_size = tf.round(image_size * scale)

    # Computes 2D image_scale.
    image_scale = scaled_size / image_size

    # Selects non-zero random offset (x, y) if scaled image is larger than
    # desired_size.
    if random_jittering:
      max_offset = scaled_size - tf.cast(desired_size, tf.float32)
      max_offset = tf.where(
          tf.less(max_offset, 0), tf.zeros_like(max_offset), max_offset
      )
      offset = max_offset * tf.random.uniform(
          [
              2,
          ],
          0,
          1,
          seed=seed,
      )
      offset = tf.cast(offset, tf.int32)
    else:
      offset = tf.zeros((2,), tf.int32)

    scaled_image = tf.image.resize(
        image, tf.cast(scaled_size, tf.int32), method=method
    )

    if random_jittering:
      scaled_image = scaled_image[
          offset[0] : offset[0] + desired_size[0],
          offset[1] : offset[1] + desired_size[1],
          :,
      ]

    output_image = scaled_image
    if padded_size is not None:
      if centered_crop:
        scaled_image_size = tf.cast(tf.shape(scaled_image)[0:2], tf.int32)
        output_image = tf.image.pad_to_bounding_box(
            scaled_image,
            tf.maximum((padded_size[0] - scaled_image_size[0]) // 2, 0),
            tf.maximum((padded_size[1] - scaled_image_size[1]) // 2, 0),
            padded_size[0],
            padded_size[1],
        )
      else:
        output_image = tf.image.pad_to_bounding_box(
            scaled_image, 0, 0, padded_size[0], padded_size[1]
        )

    image_info = tf.stack([
        image_size,
        tf.cast(desired_size, dtype=tf.float32),
        image_scale,
        tf.cast(offset, tf.float32),
    ])
    return output_image, image_info


def resize_and_crop_image_v2(
    image,
    short_side,
    long_side,
    padded_size,
    aug_scale_min=1.0,
    aug_scale_max=1.0,
    seed=1,
    method=tf.image.ResizeMethod.BILINEAR,
):
  """Resizes the input image to output size (Faster R-CNN style).

  Resize and pad images given the specified short / long side length and the
  stride size.

  Here are the preprocessing steps.
  1. For a given image, keep its aspect ratio and first try to rescale the short
     side of the original image to `short_side`.
  2. If the scaled image after 1 has a long side that exceeds `long_side`, keep
     the aspect ratio and rescale the long side of the image to `long_side`.
  3. (Optional) Apply random jittering according to `aug_scale_min` and
    `aug_scale_max`. By default this step is skipped.
  4. Pad the rescaled image to the padded_size.

  Args:
    image: a `Tensor` of shape [height, width, 3] representing an image.
    short_side: a scalar `Tensor` or `int` representing the desired short side
      to be rescaled to.
    long_side: a scalar `Tensor` or `int` representing the desired long side to
      be rescaled to.
    padded_size: a `Tensor` or `int` list/tuple of two elements representing
      [height, width] of the padded output image size.
    aug_scale_min: a `float` with range between [0, 1.0] representing minimum
      random scale applied for training scale jittering.
    aug_scale_max: a `float` with range between [1.0, inf] representing maximum
      random scale applied for training scale jittering.
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

    scale_using_short_side = short_side / tf.math.minimum(
        image_size[0], image_size[1]
    )
    scale_using_long_side = long_side / tf.math.maximum(
        image_size[0], image_size[1]
    )

    scaled_size = tf.math.round(image_size * scale_using_short_side)
    scaled_size = tf.where(
        tf.math.greater(
            tf.math.maximum(scaled_size[0], scaled_size[1]), long_side
        ),
        tf.math.round(image_size * scale_using_long_side),
        scaled_size,
    )
    desired_size = scaled_size

    random_jittering = (
        isinstance(aug_scale_min, tf.Tensor)
        or isinstance(aug_scale_max, tf.Tensor)
        or not math.isclose(aug_scale_min, 1.0)
        or not math.isclose(aug_scale_max, 1.0)
    )

    if random_jittering:
      random_scale = tf.random.uniform(
          [], aug_scale_min, aug_scale_max, seed=seed
      )
      scaled_size = tf.math.round(random_scale * scaled_size)

    # Computes 2D image_scale.
    image_scale = scaled_size / image_size

    # Selects non-zero random offset (x, y) if scaled image is larger than
    # desired_size.
    if random_jittering:
      max_offset = scaled_size - desired_size
      max_offset = tf.where(
          tf.math.less(max_offset, 0), tf.zeros_like(max_offset), max_offset
      )
      offset = max_offset * tf.random.uniform(
          [
              2,
          ],
          0,
          1,
          seed=seed,
      )
      offset = tf.cast(offset, tf.int32)
    else:
      offset = tf.zeros((2,), tf.int32)

    scaled_image = tf.image.resize(
        image, tf.cast(scaled_size, tf.int32), method=method
    )

    if random_jittering:
      scaled_image = scaled_image[
          offset[0] : offset[0] + desired_size[0],
          offset[1] : offset[1] + desired_size[1],
          :,
      ]

    output_image = tf.image.pad_to_bounding_box(
        scaled_image, 0, 0, padded_size[0], padded_size[1]
    )

    image_info = tf.stack([
        image_size,
        tf.cast(desired_size, dtype=tf.float32),
        image_scale,
        tf.cast(offset, tf.float32),
    ])
    return output_image, image_info


def resize_image(
    image: tf.Tensor,
    size: Union[Tuple[int, int], int],
    max_size: Optional[int] = None,
    method: tf.image.ResizeMethod = tf.image.ResizeMethod.BILINEAR,
):
  """Resize image with size and max_size.

  Args:
    image: the image to be resized.
    size: if list to tuple, resize to it. If scalar, we keep the same aspect
      ratio and resize the short side to the value.
    max_size: only used when size is a scalar. When the larger side is larger
      than max_size after resized with size we used max_size to keep the aspect
      ratio instead.
    method: the method argument passed to tf.image.resize.

  Returns:
    the resized image and image_info to be used for downstream processing.
    image_info: a 2D `Tensor` that encodes the information of the image and the
      applied preprocessing. It is in the format of
      [[original_height, original_width], [resized_height, resized_width],
      [y_scale, x_scale], [0, 0]], where [resized_height, resized_width]
      is the actual scaled image size, and [y_scale, x_scale] is the
      scaling factor, which is the ratio of
      scaled dimension / original dimension.
  """

  def get_size_with_aspect_ratio(image_size, size, max_size=None):
    h = image_size[0]
    w = image_size[1]
    if max_size is not None:
      min_original_size = tf.cast(tf.math.minimum(w, h), dtype=tf.float32)
      max_original_size = tf.cast(tf.math.maximum(w, h), dtype=tf.float32)
      if max_original_size / min_original_size * size > max_size:
        size = tf.cast(
            tf.math.floor(max_size * min_original_size / max_original_size),
            dtype=tf.int32,
        )
      else:
        size = tf.cast(size, tf.int32)

    else:
      size = tf.cast(size, tf.int32)
    if (w <= h and w == size) or (h <= w and h == size):
      return tf.stack([h, w])

    if w < h:
      ow = size
      oh = tf.cast(
          (
              tf.cast(size, dtype=tf.float32)
              * tf.cast(h, dtype=tf.float32)
              / tf.cast(w, dtype=tf.float32)
          ),
          dtype=tf.int32,
      )
    else:
      oh = size
      ow = tf.cast(
          (
              tf.cast(size, dtype=tf.float32)
              * tf.cast(w, dtype=tf.float32)
              / tf.cast(h, dtype=tf.float32)
          ),
          dtype=tf.int32,
      )

    return tf.stack([oh, ow])

  def get_size(image_size, size, max_size=None):
    if isinstance(size, (list, tuple)):
      return size[::-1]
    else:
      return get_size_with_aspect_ratio(image_size, size, max_size)

  orignal_size = tf.shape(image)[0:2]
  size = get_size(orignal_size, size, max_size)
  rescaled_image = tf.image.resize(
      image, tf.cast(size, tf.int32), method=method
  )
  image_scale = size / orignal_size
  image_info = tf.stack([
      tf.cast(orignal_size, dtype=tf.float32),
      tf.cast(size, dtype=tf.float32),
      tf.cast(image_scale, tf.float32),
      tf.constant([0.0, 0.0], dtype=tf.float32),
  ])
  return rescaled_image, image_info


def center_crop_image(
    image, center_crop_fraction: float = CENTER_CROP_FRACTION
):
  """Center crop a square shape slice from the input image.

  It crops a square shape slice from the image. The side of the actual crop
  is 224 / 256 = 0.875 of the short side of the original image. References:
  [1] Very Deep Convolutional Networks for Large-Scale Image Recognition
      https://arxiv.org/abs/1409.1556
  [2] Deep Residual Learning for Image Recognition
      https://arxiv.org/abs/1512.03385

  Args:
    image: a Tensor of shape [height, width, 3] representing the input image.
    center_crop_fraction: a float of ratio between the side of the cropped image
      and the short side of the original image

  Returns:
    cropped_image: a Tensor representing the center cropped image.
  """
  with tf.name_scope('center_crop_image'):
    image_size = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    crop_size = center_crop_fraction * tf.math.minimum(
        image_size[0], image_size[1]
    )
    crop_offset = tf.cast((image_size - crop_size) / 2.0, dtype=tf.int32)
    crop_size = tf.cast(crop_size, dtype=tf.int32)
    cropped_image = image[
        crop_offset[0] : crop_offset[0] + crop_size,
        crop_offset[1] : crop_offset[1] + crop_size,
        :,
    ]
    return cropped_image


def center_crop_image_v2(
    image_bytes, image_shape, center_crop_fraction: float = CENTER_CROP_FRACTION
):
  """Center crop a square shape slice from the input image.

  It crops a square shape slice from the image. The side of the actual crop
  is 224 / 256 = 0.875 of the short side of the original image. References:
  [1] Very Deep Convolutional Networks for Large-Scale Image Recognition
      https://arxiv.org/abs/1409.1556
  [2] Deep Residual Learning for Image Recognition
      https://arxiv.org/abs/1512.03385

  This is a faster version of `center_crop_image` which takes the original
  image bytes and image size as the inputs, and partially decode the JPEG
  bytes according to the center crop.

  Args:
    image_bytes: a Tensor of type string representing the raw image bytes.
    image_shape: a Tensor specifying the shape of the raw image.
    center_crop_fraction: a float of ratio between the side of the cropped image
      and the short side of the original image

  Returns:
    cropped_image: a Tensor representing the center cropped image.
  """
  with tf.name_scope('center_image_crop_v2'):
    image_shape = tf.cast(image_shape, tf.float32)
    crop_size = center_crop_fraction * tf.math.minimum(
        image_shape[0], image_shape[1]
    )
    crop_offset = tf.cast((image_shape - crop_size) / 2.0, dtype=tf.int32)
    crop_size = tf.cast(crop_size, dtype=tf.int32)
    crop_window = tf.stack(
        [crop_offset[0], crop_offset[1], crop_size, crop_size]
    )
    cropped_image = tf.image.decode_and_crop_jpeg(
        image_bytes, crop_window, channels=3
    )
    return cropped_image


def random_crop_image(
    image,
    aspect_ratio_range=(3.0 / 4.0, 4.0 / 3.0),
    area_range=(0.08, 1.0),
    max_attempts=10,
    seed=1,
):
  """Randomly crop an arbitrary shaped slice from the input image.

  Args:
    image: a Tensor of shape [height, width, 3] representing the input image.
    aspect_ratio_range: a list of floats. The cropped area of the image must
      have an aspect ratio = width / height within this range.
    area_range: a list of floats. The cropped reas of the image must contain a
      fraction of the input image within this range.
    max_attempts: the number of attempts at generating a cropped region of the
      image of the specified constraints. After max_attempts failures, return
      the entire image.
    seed: the seed of the random generator.

  Returns:
    cropped_image: a Tensor representing the random cropped image. Can be the
      original image if max_attempts is exhausted.
  """
  with tf.name_scope('random_crop_image'):
    crop_offset, crop_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]),
        seed=seed,
        min_object_covered=area_range[0],
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
    )
    cropped_image = tf.slice(image, crop_offset, crop_size)
    return cropped_image


def random_crop_image_v2(
    image_bytes,
    image_shape,
    aspect_ratio_range=(3.0 / 4.0, 4.0 / 3.0),
    area_range=(0.08, 1.0),
    max_attempts=10,
    seed=1,
):
  """Randomly crop an arbitrary shaped slice from the input image.

  This is a faster version of `random_crop_image` which takes the original
  image bytes and image size as the inputs, and partially decode the JPEG
  bytes according to the generated crop.

  Args:
    image_bytes: a Tensor of type string representing the raw image bytes.
    image_shape: a Tensor specifying the shape of the raw image.
    aspect_ratio_range: a list of floats. The cropped area of the image must
      have an aspect ratio = width / height within this range.
    area_range: a list of floats. The cropped reas of the image must contain a
      fraction of the input image within this range.
    max_attempts: the number of attempts at generating a cropped region of the
      image of the specified constraints. After max_attempts failures, return
      the entire image.
    seed: the seed of the random generator.

  Returns:
    cropped_image: a Tensor representing the random cropped image. Can be the
      original image if max_attempts is exhausted.
  """
  with tf.name_scope('random_crop_image_v2'):
    crop_offset, crop_size, _ = tf.image.sample_distorted_bounding_box(
        image_shape,
        tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]),
        seed=seed,
        min_object_covered=area_range[0],
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
    )
    offset_y, offset_x, _ = tf.unstack(crop_offset)
    crop_height, crop_width, _ = tf.unstack(crop_size)
    crop_window = tf.stack([offset_y, offset_x, crop_height, crop_width])
    cropped_image = tf.image.decode_and_crop_jpeg(
        image_bytes, crop_window, channels=3
    )
    return cropped_image


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
  with tf.name_scope('resize_and_crop_boxes'):
    # Adjusts box coordinates based on image_scale and offset.
    boxes *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
    boxes -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
    # Clips the boxes.
    boxes = box_ops.clip_boxes(boxes, output_size)
    return boxes


def resize_and_crop_masks(
    masks, image_scale, output_size, offset, centered_crop: bool = False
):
  """Resizes boxes to output size with scale and offset.

  Args:
    masks: `Tensor` of shape [N, H, W, C] representing ground truth masks.
    image_scale: 2D float `Tensor` representing scale factors that apply to
      [height, width] of input image.
    output_size: 2D `Tensor` or `int` representing [height, width] of target
      output image size.
    offset: 2D `Tensor` representing top-left corner [y0, x0] to crop scaled
      boxes.
    centered_crop: If `centered_crop` is set to True, then resized crop (if
      smaller than padded size) is place in the center of the image. Default
      behaviour is to place it at left top corner.

  Returns:
    masks: `Tensor` of shape [N, H, W, C] representing the scaled masks.
  """
  with tf.name_scope('resize_and_crop_masks'):
    mask_size = tf.cast(tf.shape(masks)[1:3], tf.float32)
    num_channels = tf.shape(masks)[3]
    # Pad masks to avoid empty mask annotations.
    masks = tf.concat(
        [
            tf.zeros(
                [1, mask_size[0], mask_size[1], num_channels], dtype=masks.dtype
            ),
            masks,
        ],
        axis=0,
    )

    scaled_size = tf.cast(image_scale * mask_size, tf.int32)
    scaled_masks = tf.image.resize(
        masks, scaled_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    offset = tf.cast(offset, tf.int32)
    scaled_masks = scaled_masks[
        :,
        offset[0] : offset[0] + output_size[0],
        offset[1] : offset[1] + output_size[1],
        :,
    ]

    if centered_crop:
      scaled_mask_size = tf.cast(tf.shape(scaled_masks)[1:3], tf.int32)
      output_masks = tf.image.pad_to_bounding_box(
          scaled_masks,
          tf.maximum((output_size[0] - scaled_mask_size[0]) // 2, 0),
          tf.maximum((output_size[1] - scaled_mask_size[1]) // 2, 0),
          output_size[0],
          output_size[1],
      )
    else:
      output_masks = tf.image.pad_to_bounding_box(
          scaled_masks, 0, 0, output_size[0], output_size[1]
      )

    # Remove padding.
    output_masks = output_masks[1::]
    return output_masks


def horizontal_flip_image(image):
  """Flips image horizontally."""
  return tf.image.flip_left_right(image)


def horizontal_flip_masks(masks):
  """Flips masks horizontally. Expects rank-3 input dimensions."""
  # For masks shape of [h, w, 1].
  if masks.shape[-1] == 1:
    return masks[:, ::-1, :]
  else:
    return masks[:, :, ::-1]


def random_horizontal_flip(
    image, normalized_boxes=None, masks=None, seed=1, prob=0.5
):
  """Randomly flips input image and bounding boxes and/or masks horizontally.

  Expects input tensors without the batch dimension; i.e. for RGB image assume
  rank-3 input like [h, w, c], for masks assume either [h, w, 1] or [1, h, w].

  Args:
    image: `tf.Tensor`, the image to apply the random flip, [h, w, channels].
    normalized_boxes: `tf.Tensor` or `None`, boxes corresponding to the image.
    masks: `tf.Tensor` or `None`, masks corresponding to the image, [h, w, 1] or
      [1, h, w].
    seed: Seed for Tensorflow's random number generator.
    prob: A float from 0 to 1 indicating the probability of flipping the input
      horizontally.

  Returns:
    image: `tf.Tensor`, flipped image.
    boxes: `tf.Tensor` or `None`, flipped normalized boxes corresponding to the
      image.
    masks: `tf.Tensor` or `None`, flipped masks corresponding to the image.
  """
  with tf.name_scope('random_horizontal_flip'):
    do_flip = tf.less(tf.random.uniform([], seed=seed), prob)

    image = tf.cond(
        do_flip, lambda: horizontal_flip_image(image), lambda: image
    )

    if normalized_boxes is not None:
      normalized_boxes = tf.cond(
          do_flip,
          lambda: horizontal_flip_boxes(normalized_boxes),
          lambda: normalized_boxes,
      )

    if masks is not None:
      masks = tf.cond(
          do_flip, lambda: horizontal_flip_masks(masks), lambda: masks
      )

    return image, normalized_boxes, masks


def random_horizontal_flip_with_roi(
    image: tf.Tensor,
    boxes: Optional[tf.Tensor] = None,
    masks: Optional[tf.Tensor] = None,
    roi_boxes: Optional[tf.Tensor] = None,
    seed: int = 1,
) -> Tuple[
    tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor], Optional[tf.Tensor]
]:
  """Randomly flips input image and bounding boxes horizontally.

  Extends preprocess_ops.random_horizontal_flip to also flip roi_boxes used
  by ViLD.

  Args:
    image: `tf.Tensor`, the image to apply the random flip.
    boxes: `tf.Tensor` or `None`, boxes corresponding to the image.
    masks: `tf.Tensor` or `None`, masks corresponding to the image.
    roi_boxes: `tf.Tensor` or `None`, RoIs corresponding to the image.
    seed: Seed for Tensorflow's random number generator.

  Returns:
    image: `tf.Tensor`, flipped image.
    boxes: `tf.Tensor` or `None`, flipped boxes corresponding to the image.
    masks: `tf.Tensor` or `None`, flipped masks corresponding to the image.
    roi_boxes: `tf.Tensor` or `None`, flipped RoIs corresponding to the image.
  """
  with tf.name_scope('random_horizontal_flip'):
    do_flip = tf.greater(tf.random.uniform([], seed=seed), 0.5)

    image = tf.cond(
        do_flip, lambda: horizontal_flip_image(image), lambda: image
    )

    if boxes is not None:
      boxes = tf.cond(
          do_flip, lambda: horizontal_flip_boxes(boxes), lambda: boxes
      )

    if masks is not None:
      masks = tf.cond(
          do_flip, lambda: horizontal_flip_masks(masks), lambda: masks
      )

    if roi_boxes is not None:
      roi_boxes = tf.cond(
          do_flip, lambda: horizontal_flip_boxes(roi_boxes), lambda: roi_boxes
      )

    return image, boxes, masks, roi_boxes


def random_vertical_flip(
    image, normalized_boxes=None, masks=None, seed=1, prob=0.5
):
  """Randomly flips input image and bounding boxes vertically."""
  with tf.name_scope('random_vertical_flip'):
    do_flip = tf.less(tf.random.uniform([], seed=seed), prob)

    image = tf.cond(
        do_flip, lambda: tf.image.flip_up_down(image), lambda: image
    )

    if normalized_boxes is not None:
      normalized_boxes = tf.cond(
          do_flip,
          lambda: vertical_flip_boxes(normalized_boxes),
          lambda: normalized_boxes,
      )

    if masks is not None:
      masks = tf.cond(
          do_flip,
          lambda: tf.image.flip_up_down(masks[..., None])[..., 0],
          lambda: masks,
      )

    return image, normalized_boxes, masks


def color_jitter(
    image: tf.Tensor,
    brightness: Optional[float] = 0.0,
    contrast: Optional[float] = 0.0,
    saturation: Optional[float] = 0.0,
    seed: Optional[int] = None,
) -> tf.Tensor:
  """Applies color jitter to an image, similarly to torchvision`s ColorJitter.

  Args:
    image (tf.Tensor): Of shape [height, width, 3] and type uint8.
    brightness (float, optional): Magnitude for brightness jitter. Defaults to
      0.
    contrast (float, optional): Magnitude for contrast jitter. Defaults to 0.
    saturation (float, optional): Magnitude for saturation jitter. Defaults to
      0.
    seed (int, optional): Random seed. Defaults to None.

  Returns:
    tf.Tensor: The augmented `image` of type uint8.
  """
  image = tf.cast(image, dtype=tf.uint8)
  image = random_brightness(image, brightness, seed=seed)
  image = random_contrast(image, contrast, seed=seed)
  image = random_saturation(image, saturation, seed=seed)
  return image


def random_brightness(
    image: tf.Tensor, brightness: float = 0.0, seed: Optional[int] = None
) -> tf.Tensor:
  """Jitters brightness of an image.

  Args:
      image (tf.Tensor): Of shape [height, width, 3] and type uint8.
      brightness (float, optional): Magnitude for brightness jitter. Defaults to
        0.
      seed (int, optional): Random seed. Defaults to None.

  Returns:
      tf.Tensor: The augmented `image` of type uint8.
  """
  assert brightness >= 0, '`brightness` must be positive'
  brightness = tf.random.uniform(
      [], max(0, 1 - brightness), 1 + brightness, seed=seed, dtype=tf.float32
  )
  return augment.brightness(image, brightness)


def random_contrast(
    image: tf.Tensor, contrast: float = 0.0, seed: Optional[int] = None
) -> tf.Tensor:
  """Jitters contrast of an image, similarly to torchvision`s ColorJitter.

  Args:
      image (tf.Tensor): Of shape [height, width, 3] and type uint8.
      contrast (float, optional): Magnitude for contrast jitter. Defaults to 0.
      seed (int, optional): Random seed. Defaults to None.

  Returns:
      tf.Tensor: The augmented `image` of type uint8.
  """
  assert contrast >= 0, '`contrast` must be positive'
  contrast = tf.random.uniform(
      [], max(0, 1 - contrast), 1 + contrast, seed=seed, dtype=tf.float32
  )
  return augment.contrast(image, contrast)


def random_saturation(
    image: tf.Tensor, saturation: float = 0.0, seed: Optional[int] = None
) -> tf.Tensor:
  """Jitters saturation of an image, similarly to torchvision`s ColorJitter.

  Args:
      image (tf.Tensor): Of shape [height, width, 3] and type uint8.
      saturation (float, optional): Magnitude for saturation jitter. Defaults to
        0.
      seed (int, optional): Random seed. Defaults to None.

  Returns:
      tf.Tensor: The augmented `image` of type uint8.
  """
  assert saturation >= 0, '`saturation` must be positive'
  saturation = tf.random.uniform(
      [], max(0, 1 - saturation), 1 + saturation, seed=seed, dtype=tf.float32
  )
  return _saturation(image, saturation)


def _saturation(
    image: tf.Tensor, saturation: Optional[float] = 0.0
) -> tf.Tensor:
  return augment.blend(
      tf.repeat(tf.image.rgb_to_grayscale(image), 3, axis=-1), image, saturation
  )


def random_crop_image_with_boxes_and_labels(
    img,
    boxes,
    labels,
    min_scale,
    aspect_ratio_range,
    min_overlap_params,
    max_retry,
):
  """Crops a random slice from the input image.

  The function will correspondingly recompute the bounding boxes and filter out
  outside boxes and their labels.

  References:
  [1] End-to-End Object Detection with Transformers
  https://arxiv.org/abs/2005.12872

  The preprocessing steps:
  1. Sample a minimum IoU overlap.
  2. For each trial, sample the new image width, height, and top-left corner.
  3. Compute the IoUs of bounding boxes with the cropped image and retry if
    the maximum IoU is below the sampled threshold.
  4. Find boxes whose centers are in the cropped image.
  5. Compute new bounding boxes in the cropped region and only select those
    boxes' labels.

  Args:
    img: a 'Tensor' of shape [height, width, 3] representing the input image.
    boxes: a 'Tensor' of shape [N, 4] representing the ground-truth bounding
      boxes with (ymin, xmin, ymax, xmax).
    labels: a 'Tensor' of shape [N,] representing the class labels of the boxes.
    min_scale: a 'float' in [0.0, 1.0) indicating the lower bound of the random
      scale variable.
    aspect_ratio_range: a list of two 'float' that specifies the lower and upper
      bound of the random aspect ratio.
    min_overlap_params: a list of four 'float' representing the min value, max
      value, step size, and offset for the minimum overlap sample.
    max_retry: an 'int' representing the number of trials for cropping. If it is
      exhausted, no cropping will be performed.

  Returns:
    img: a Tensor representing the random cropped image. Can be the
      original image if max_retry is exhausted.
    boxes: a Tensor representing the bounding boxes in the cropped image.
    labels: a Tensor representing the new bounding boxes' labels.
  """

  shape = tf.shape(img)
  original_h = shape[0]
  original_w = shape[1]

  minval, maxval, step, offset = min_overlap_params

  min_overlap = (
      tf.math.floordiv(
          tf.random.uniform([], minval=minval, maxval=maxval), step
      )
      * step
      - offset
  )

  min_overlap = tf.clip_by_value(min_overlap, 0.0, 1.1)

  if min_overlap > 1.0:
    return img, boxes, labels

  aspect_ratio_low = aspect_ratio_range[0]
  aspect_ratio_high = aspect_ratio_range[1]

  for _ in tf.range(max_retry):
    scale_h = tf.random.uniform([], min_scale, 1.0)
    scale_w = tf.random.uniform([], min_scale, 1.0)
    new_h = tf.cast(
        scale_h * tf.cast(original_h, dtype=tf.float32), dtype=tf.int32
    )
    new_w = tf.cast(
        scale_w * tf.cast(original_w, dtype=tf.float32), dtype=tf.int32
    )

    # Aspect ratio has to be in the prespecified range
    aspect_ratio = new_h / new_w
    if aspect_ratio_low > aspect_ratio or aspect_ratio > aspect_ratio_high:
      continue

    left = tf.random.uniform([], 0, original_w - new_w, dtype=tf.int32)
    right = left + new_w
    top = tf.random.uniform([], 0, original_h - new_h, dtype=tf.int32)
    bottom = top + new_h

    normalized_left = tf.cast(left, dtype=tf.float32) / tf.cast(
        original_w, dtype=tf.float32
    )
    normalized_right = tf.cast(right, dtype=tf.float32) / tf.cast(
        original_w, dtype=tf.float32
    )
    normalized_top = tf.cast(top, dtype=tf.float32) / tf.cast(
        original_h, dtype=tf.float32
    )
    normalized_bottom = tf.cast(bottom, dtype=tf.float32) / tf.cast(
        original_h, dtype=tf.float32
    )

    cropped_box = tf.expand_dims(
        tf.stack([
            normalized_top,
            normalized_left,
            normalized_bottom,
            normalized_right,
        ]),
        axis=0,
    )
    iou = box_ops.bbox_overlap(
        tf.expand_dims(cropped_box, axis=0), tf.expand_dims(boxes, axis=0)
    )  # (1, 1, n_ground_truth)
    iou = tf.squeeze(iou, axis=[0, 1])

    # If not a single bounding box has a Jaccard overlap of greater than
    # the minimum, try again
    if tf.reduce_max(iou) < min_overlap:
      continue

    centroids = box_ops.yxyx_to_cycxhw(boxes)
    mask = tf.math.logical_and(
        tf.math.logical_and(
            centroids[:, 0] > normalized_top,
            centroids[:, 0] < normalized_bottom,
        ),
        tf.math.logical_and(
            centroids[:, 1] > normalized_left,
            centroids[:, 1] < normalized_right,
        ),
    )
    # If not a single bounding box has its center in the crop, try again.
    if tf.reduce_sum(tf.cast(mask, dtype=tf.int32)) > 0:
      indices = tf.squeeze(tf.where(mask), axis=1)

      filtered_boxes = tf.gather(boxes, indices)

      boxes = tf.clip_by_value(
          (
              filtered_boxes[..., :]
              * tf.cast(
                  tf.stack([original_h, original_w, original_h, original_w]),
                  dtype=tf.float32,
              )
              - tf.cast(tf.stack([top, left, top, left]), dtype=tf.float32)
          )
          / tf.cast(tf.stack([new_h, new_w, new_h, new_w]), dtype=tf.float32),
          0.0,
          1.0,
      )

      img = tf.image.crop_to_bounding_box(
          img, top, left, bottom - top, right - left
      )

      labels = tf.gather(labels, indices)
      break

  return img, boxes, labels


def random_crop(
    image,
    boxes,
    labels,
    min_scale=0.3,
    aspect_ratio_range=(0.5, 2.0),
    min_overlap_params=(0.0, 1.4, 0.2, 0.1),
    max_retry=50,
    seed=None,
):
  """Randomly crop the image and boxes, filtering labels.

  Args:
    image: a 'Tensor' of shape [height, width, 3] representing the input image.
    boxes: a 'Tensor' of shape [N, 4] representing the ground-truth bounding
      boxes with (ymin, xmin, ymax, xmax).
    labels: a 'Tensor' of shape [N,] representing the class labels of the boxes.
    min_scale: a 'float' in [0.0, 1.0) indicating the lower bound of the random
      scale variable.
    aspect_ratio_range: a list of two 'float' that specifies the lower and upper
      bound of the random aspect ratio.
    min_overlap_params: a list of four 'float' representing the min value, max
      value, step size, and offset for the minimum overlap sample.
    max_retry: an 'int' representing the number of trials for cropping. If it is
      exhausted, no cropping will be performed.
    seed: the random number seed of int, but could be None.

  Returns:
    image: a Tensor representing the random cropped image. Can be the
      original image if max_retry is exhausted.
    boxes: a Tensor representing the bounding boxes in the cropped image.
    labels: a Tensor representing the new bounding boxes' labels.
  """
  with tf.name_scope('random_crop'):
    do_crop = tf.greater(tf.random.uniform([], seed=seed), 0.5)
    if do_crop:
      return random_crop_image_with_boxes_and_labels(
          image,
          boxes,
          labels,
          min_scale,
          aspect_ratio_range,
          min_overlap_params,
          max_retry,
      )
    else:
      return image, boxes, labels


def random_jpeg_quality(
    image: tf.Tensor,
    min_quality: int | tf.Tensor = 20,
    max_quality: int | tf.Tensor = 100,
    prob_to_apply: float | tf.Tensor = 0.6,
) -> tf.Tensor:
  """Randomly encode the image as jpeg and decode it.

  Args:
    image: a uint8 'Tensor' of shape [height, width, 3] representing the input
      image.
    min_quality: minimum jpeg quality in range of [0, 100].
    max_quality: maximum jpeg quality in range of [0, 100].
    prob_to_apply: probability to apply this augmentation.

  Returns:
    image with jpeg quality changed
  """
  if tf.random.uniform(shape=[], maxval=1.0) > prob_to_apply:
    return image
  quality = tf.random.uniform(
      [], minval=min_quality, maxval=max_quality, dtype=tf.int32
  )
  return tf.image.adjust_jpeg_quality(image, quality)
