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

"""Augmentation policies for enhanced image/video preprocessing.

AutoAugment Reference:
  - AutoAugment Reference: https://arxiv.org/abs/1805.09501
  - AutoAugment for Object Detection Reference: https://arxiv.org/abs/1906.11172
RandAugment Reference: https://arxiv.org/abs/1909.13719
RandomErasing Reference: https://arxiv.org/abs/1708.04896
MixupAndCutmix:
  - Mixup: https://arxiv.org/abs/1710.09412
  - Cutmix: https://arxiv.org/abs/1905.04899

RandomErasing, Mixup and Cutmix are inspired by
https://github.com/rwightman/pytorch-image-models

"""
import inspect
import math
from typing import Any, List, Iterable, Optional, Text, Tuple

from keras.layers.preprocessing import image_preprocessing as image_ops
import numpy as np
import tensorflow as tf


# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.


def to_4d(image: tf.Tensor) -> tf.Tensor:
  """Converts an input Tensor to 4 dimensions.

  4D image => [N, H, W, C] or [N, C, H, W]
  3D image => [1, H, W, C] or [1, C, H, W]
  2D image => [1, H, W, 1]

  Args:
    image: The 2/3/4D input tensor.

  Returns:
    A 4D image tensor.

  Raises:
    `TypeError` if `image` is not a 2/3/4D tensor.

  """
  shape = tf.shape(image)
  original_rank = tf.rank(image)
  left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
  right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
  new_shape = tf.concat(
      [
          tf.ones(shape=left_pad, dtype=tf.int32),
          shape,
          tf.ones(shape=right_pad, dtype=tf.int32),
      ],
      axis=0,
  )
  return tf.reshape(image, new_shape)


def from_4d(image: tf.Tensor, ndims: tf.Tensor) -> tf.Tensor:
  """Converts a 4D image back to `ndims` rank."""
  shape = tf.shape(image)
  begin = tf.cast(tf.less_equal(ndims, 3), dtype=tf.int32)
  end = 4 - tf.cast(tf.equal(ndims, 2), dtype=tf.int32)
  new_shape = shape[begin:end]
  return tf.reshape(image, new_shape)


def _convert_translation_to_transform(translations: tf.Tensor) -> tf.Tensor:
  """Converts translations to a projective transform.

  The translation matrix looks like this:
    [[1 0 -dx]
     [0 1 -dy]
     [0 0 1]]

  Args:
    translations: The 2-element list representing [dx, dy], or a matrix of
      2-element lists representing [dx dy] to translate for each image. The
      shape must be static.

  Returns:
    The transformation matrix of shape (num_images, 8).

  Raises:
    `TypeError` if
      - the shape of `translations` is not known or
      - the shape of `translations` is not rank 1 or 2.

  """
  translations = tf.convert_to_tensor(translations, dtype=tf.float32)
  if translations.get_shape().ndims is None:
    raise TypeError('translations rank must be statically known')
  elif len(translations.get_shape()) == 1:
    translations = translations[None]
  elif len(translations.get_shape()) != 2:
    raise TypeError('translations should have rank 1 or 2.')
  num_translations = tf.shape(translations)[0]

  return tf.concat(
      values=[
          tf.ones((num_translations, 1), tf.dtypes.float32),
          tf.zeros((num_translations, 1), tf.dtypes.float32),
          -translations[:, 0, None],
          tf.zeros((num_translations, 1), tf.dtypes.float32),
          tf.ones((num_translations, 1), tf.dtypes.float32),
          -translations[:, 1, None],
          tf.zeros((num_translations, 2), tf.dtypes.float32),
      ],
      axis=1,
  )


def _convert_angles_to_transform(angles: tf.Tensor, image_width: tf.Tensor,
                                 image_height: tf.Tensor) -> tf.Tensor:
  """Converts an angle or angles to a projective transform.

  Args:
    angles: A scalar to rotate all images, or a vector to rotate a batch of
      images. This must be a scalar.
    image_width: The width of the image(s) to be transformed.
    image_height: The height of the image(s) to be transformed.

  Returns:
    A tensor of shape (num_images, 8).

  Raises:
    `TypeError` if `angles` is not rank 0 or 1.

  """
  angles = tf.convert_to_tensor(angles, dtype=tf.float32)
  if len(angles.get_shape()) == 0:  # pylint:disable=g-explicit-length-test
    angles = angles[None]
  elif len(angles.get_shape()) != 1:
    raise TypeError('Angles should have a rank 0 or 1.')
  x_offset = ((image_width - 1) -
              (tf.math.cos(angles) * (image_width - 1) - tf.math.sin(angles) *
               (image_height - 1))) / 2.0
  y_offset = ((image_height - 1) -
              (tf.math.sin(angles) * (image_width - 1) + tf.math.cos(angles) *
               (image_height - 1))) / 2.0
  num_angles = tf.shape(angles)[0]
  return tf.concat(
      values=[
          tf.math.cos(angles)[:, None],
          -tf.math.sin(angles)[:, None],
          x_offset[:, None],
          tf.math.sin(angles)[:, None],
          tf.math.cos(angles)[:, None],
          y_offset[:, None],
          tf.zeros((num_angles, 2), tf.dtypes.float32),
      ],
      axis=1,
  )


def transform(image: tf.Tensor, transforms) -> tf.Tensor:
  """Prepares input data for `image_ops.transform`."""
  original_ndims = tf.rank(image)
  transforms = tf.convert_to_tensor(transforms, dtype=tf.float32)
  if transforms.shape.rank == 1:
    transforms = transforms[None]
  image = to_4d(image)
  image = image_ops.transform(
      images=image, transforms=transforms, interpolation='nearest')
  return from_4d(image, original_ndims)


def translate(image: tf.Tensor, translations) -> tf.Tensor:
  """Translates image(s) by provided vectors.

  Args:
    image: An image Tensor of type uint8.
    translations: A vector or matrix representing [dx dy].

  Returns:
    The translated version of the image.

  """
  transforms = _convert_translation_to_transform(translations)
  return transform(image, transforms=transforms)


def rotate(image: tf.Tensor, degrees: float) -> tf.Tensor:
  """Rotates the image by degrees either clockwise or counterclockwise.

  Args:
    image: An image Tensor of type uint8.
    degrees: Float, a scalar angle in degrees to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.

  Returns:
    The rotated version of image.

  """
  # Convert from degrees to radians.
  degrees_to_radians = math.pi / 180.0
  radians = tf.cast(degrees * degrees_to_radians, tf.float32)

  original_ndims = tf.rank(image)
  image = to_4d(image)

  image_height = tf.cast(tf.shape(image)[1], tf.float32)
  image_width = tf.cast(tf.shape(image)[2], tf.float32)
  transforms = _convert_angles_to_transform(
      angles=radians, image_width=image_width, image_height=image_height)
  # In practice, we should randomize the rotation degrees by flipping
  # it negatively half the time, but that's done on 'degrees' outside
  # of the function.
  image = transform(image, transforms=transforms)
  return from_4d(image, original_ndims)


def blend(image1: tf.Tensor, image2: tf.Tensor, factor: float) -> tf.Tensor:
  """Blend image1 and image2 using 'factor'.

  Factor can be above 0.0.  A value of 0.0 means only image1 is used.
  A value of 1.0 means only image2 is used.  A value between 0.0 and
  1.0 means we linearly interpolate the pixel values between the two
  images.  A value greater than 1.0 "extrapolates" the difference
  between the two pixel values, and we clip the results to values
  between 0 and 255.

  Args:
    image1: An image Tensor of type uint8.
    image2: An image Tensor of type uint8.
    factor: A floating point value above 0.0.

  Returns:
    A blended image Tensor of type uint8.
  """
  if factor == 0.0:
    return tf.convert_to_tensor(image1)
  if factor == 1.0:
    return tf.convert_to_tensor(image2)

  image1 = tf.cast(image1, tf.float32)
  image2 = tf.cast(image2, tf.float32)

  difference = image2 - image1
  scaled = factor * difference

  # Do addition in float.
  temp = tf.cast(image1, tf.float32) + scaled

  # Interpolate
  if factor > 0.0 and factor < 1.0:
    # Interpolation means we always stay within 0 and 255.
    return tf.cast(temp, tf.uint8)

  # Extrapolate:
  #
  # We need to clip and then cast.
  return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)


def cutout(image: tf.Tensor, pad_size: int, replace: int = 0) -> tf.Tensor:
  """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

  This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
  a random location within `image`. The pixel values filled in will be of the
  value `replace`. The location where the mask will be applied is randomly
  chosen uniformly over the whole image.

  Args:
    image: An image Tensor of type uint8.
    pad_size: Specifies how big the zero mask that will be generated is that is
      applied to the image. The mask will be of size (2*pad_size x 2*pad_size).
    replace: What pixel value to fill in the image in the area that has the
      cutout mask applied to it.

  Returns:
    An image Tensor that is of type uint8.
  """
  if image.shape.rank not in [3, 4]:
    raise ValueError('Bad image rank: {}'.format(image.shape.rank))

  if image.shape.rank == 4:
    return cutout_video(image, replace=replace)

  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  # Sample the center location in the image where the zero mask will be applied.
  cutout_center_height = tf.random.uniform(
      shape=[], minval=0, maxval=image_height, dtype=tf.int32)

  cutout_center_width = tf.random.uniform(
      shape=[], minval=0, maxval=image_width, dtype=tf.int32)

  image = _fill_rectangle(image, cutout_center_width, cutout_center_height,
                          pad_size, pad_size, replace)

  return image


def _fill_rectangle(image,
                    center_width,
                    center_height,
                    half_width,
                    half_height,
                    replace=None):
  """Fill blank area."""
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  lower_pad = tf.maximum(0, center_height - half_height)
  upper_pad = tf.maximum(0, image_height - center_height - half_height)
  left_pad = tf.maximum(0, center_width - half_width)
  right_pad = tf.maximum(0, image_width - center_width - half_width)

  cutout_shape = [
      image_height - (lower_pad + upper_pad),
      image_width - (left_pad + right_pad)
  ]
  padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype),
      padding_dims,
      constant_values=1)
  mask = tf.expand_dims(mask, -1)
  mask = tf.tile(mask, [1, 1, 3])

  if replace is None:
    fill = tf.random.normal(tf.shape(image), dtype=image.dtype)
  elif isinstance(replace, tf.Tensor):
    fill = replace
  else:
    fill = tf.ones_like(image, dtype=image.dtype) * replace
  image = tf.where(tf.equal(mask, 0), fill, image)

  return image


def cutout_video(image: tf.Tensor, replace: int = 0) -> tf.Tensor:
  """Apply cutout (https://arxiv.org/abs/1708.04552) to a video.

  This operation applies a random size 3D mask of zeros to a random location
  within `image`. The mask is padded The pixel values filled in will be of the
  value `replace`. The location where the mask will be applied is randomly
  chosen uniformly over the whole image. The size of the mask is randomly
  sampled uniformly from [0.25*height, 0.5*height], [0.25*width, 0.5*width],
  and [1, 0.25*depth], which represent the height, width, and number of frames
  of the input video tensor respectively.

  Args:
    image: A video Tensor of type uint8.
    replace: What pixel value to fill in the image in the area that has the
      cutout mask applied to it.

  Returns:
    An video Tensor that is of type uint8.
  """
  image_depth = tf.shape(image)[0]
  image_height = tf.shape(image)[1]
  image_width = tf.shape(image)[2]

  # Sample the center location in the image where the zero mask will be applied.
  cutout_center_height = tf.random.uniform(
      shape=[], minval=0, maxval=image_height, dtype=tf.int32)

  cutout_center_width = tf.random.uniform(
      shape=[], minval=0, maxval=image_width, dtype=tf.int32)

  cutout_center_depth = tf.random.uniform(
      shape=[], minval=0, maxval=image_depth, dtype=tf.int32)

  pad_size_height = tf.random.uniform(
      shape=[],
      minval=tf.maximum(1, tf.cast(image_height / 4, tf.int32)),
      maxval=tf.maximum(2, tf.cast(image_height / 2, tf.int32)),
      dtype=tf.int32)
  pad_size_width = tf.random.uniform(
      shape=[],
      minval=tf.maximum(1, tf.cast(image_width / 4, tf.int32)),
      maxval=tf.maximum(2, tf.cast(image_width / 2, tf.int32)),
      dtype=tf.int32)
  pad_size_depth = tf.random.uniform(
      shape=[],
      minval=1,
      maxval=tf.maximum(2, tf.cast(image_depth / 4, tf.int32)),
      dtype=tf.int32)

  lower_pad = tf.maximum(0, cutout_center_height - pad_size_height)
  upper_pad = tf.maximum(
      0, image_height - cutout_center_height - pad_size_height)
  left_pad = tf.maximum(0, cutout_center_width - pad_size_width)
  right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size_width)
  back_pad = tf.maximum(0, cutout_center_depth - pad_size_depth)
  forward_pad = tf.maximum(
      0, image_depth - cutout_center_depth - pad_size_depth)

  cutout_shape = [
      image_depth - (back_pad + forward_pad),
      image_height - (lower_pad + upper_pad),
      image_width - (left_pad + right_pad),
  ]
  padding_dims = [[back_pad, forward_pad],
                  [lower_pad, upper_pad],
                  [left_pad, right_pad]]
  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype),
      padding_dims,
      constant_values=1)
  mask = tf.expand_dims(mask, -1)
  mask = tf.tile(mask, [1, 1, 1, 3])
  image = tf.where(
      tf.equal(mask, 0),
      tf.ones_like(image, dtype=image.dtype) * replace, image)
  return image


def solarize(image: tf.Tensor, threshold: int = 128) -> tf.Tensor:
  """Solarize the input image(s)."""
  # For each pixel in the image, select the pixel
  # if the value is less than the threshold.
  # Otherwise, subtract 255 from the pixel.
  return tf.where(image < threshold, image, 255 - image)


def solarize_add(image: tf.Tensor,
                 addition: int = 0,
                 threshold: int = 128) -> tf.Tensor:
  """Additive solarize the input image(s)."""
  # For each pixel in the image less than threshold
  # we add 'addition' amount to it and then clip the
  # pixel value to be between 0 and 255. The value
  # of 'addition' is between -128 and 128.
  added_image = tf.cast(image, tf.int64) + addition
  added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
  return tf.where(image < threshold, added_image, image)


def color(image: tf.Tensor, factor: float) -> tf.Tensor:
  """Equivalent of PIL Color."""
  degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
  return blend(degenerate, image, factor)


def contrast(image: tf.Tensor, factor: float) -> tf.Tensor:
  """Equivalent of PIL Contrast."""
  degenerate = tf.image.rgb_to_grayscale(image)
  # Cast before calling tf.histogram.
  degenerate = tf.cast(degenerate, tf.int32)

  # Compute the grayscale histogram, then compute the mean pixel value,
  # and create a constant image size of that value.  Use that as the
  # blending degenerate target of the original image.
  hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
  mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
  degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
  return blend(degenerate, image, factor)


def brightness(image: tf.Tensor, factor: float) -> tf.Tensor:
  """Equivalent of PIL Brightness."""
  degenerate = tf.zeros_like(image)
  return blend(degenerate, image, factor)


def posterize(image: tf.Tensor, bits: int) -> tf.Tensor:
  """Equivalent of PIL Posterize."""
  shift = 8 - bits
  return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


def wrapped_rotate(image: tf.Tensor, degrees: float, replace: int) -> tf.Tensor:
  """Applies rotation with wrap/unwrap."""
  image = rotate(wrap(image), degrees=degrees)
  return unwrap(image, replace)


def translate_x(image: tf.Tensor, pixels: int, replace: int) -> tf.Tensor:
  """Equivalent of PIL Translate in X dimension."""
  image = translate(wrap(image), [-pixels, 0])
  return unwrap(image, replace)


def translate_y(image: tf.Tensor, pixels: int, replace: int) -> tf.Tensor:
  """Equivalent of PIL Translate in Y dimension."""
  image = translate(wrap(image), [0, -pixels])
  return unwrap(image, replace)


def shear_x(image: tf.Tensor, level: float, replace: int) -> tf.Tensor:
  """Equivalent of PIL Shearing in X dimension."""
  # Shear parallel to x axis is a projective transform
  # with a matrix form of:
  # [1  level
  #  0  1].
  image = transform(
      image=wrap(image), transforms=[1., level, 0., 0., 1., 0., 0., 0.])
  return unwrap(image, replace)


def shear_y(image: tf.Tensor, level: float, replace: int) -> tf.Tensor:
  """Equivalent of PIL Shearing in Y dimension."""
  # Shear parallel to y axis is a projective transform
  # with a matrix form of:
  # [1  0
  #  level  1].
  image = transform(
      image=wrap(image), transforms=[1., 0., 0., level, 1., 0., 0., 0.])
  return unwrap(image, replace)


def autocontrast(image: tf.Tensor) -> tf.Tensor:
  """Implements Autocontrast function from PIL using TF ops.

  Args:
    image: A 3D uint8 tensor.

  Returns:
    The image after it has had autocontrast applied to it and will be of type
    uint8.
  """

  def scale_channel(image: tf.Tensor) -> tf.Tensor:
    """Scale the 2D image using the autocontrast rule."""
    # A possibly cheaper version can be done using cumsum/unique_with_counts
    # over the histogram values, rather than iterating over the entire image.
    # to compute mins and maxes.
    lo = tf.cast(tf.reduce_min(image), tf.float32)
    hi = tf.cast(tf.reduce_max(image), tf.float32)

    # Scale the image, making the lowest value 0 and the highest value 255.
    def scale_values(im):
      scale = 255.0 / (hi - lo)
      offset = -lo * scale
      im = tf.cast(im, tf.float32) * scale + offset
      im = tf.clip_by_value(im, 0.0, 255.0)
      return tf.cast(im, tf.uint8)

    result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
    return result

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image[..., 0])
  s2 = scale_channel(image[..., 1])
  s3 = scale_channel(image[..., 2])
  image = tf.stack([s1, s2, s3], -1)

  return image


def sharpness(image: tf.Tensor, factor: float) -> tf.Tensor:
  """Implements Sharpness function from PIL using TF ops."""
  orig_image = image
  image = tf.cast(image, tf.float32)
  # Make image 4D for conv operation.
  image = tf.expand_dims(image, 0)
  # SMOOTH PIL Kernel.
  if orig_image.shape.rank == 3:
    kernel = tf.constant([[1, 1, 1], [1, 5, 1], [1, 1, 1]],
                         dtype=tf.float32,
                         shape=[3, 3, 1, 1]) / 13.
    # Tile across channel dimension.
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]
    degenerate = tf.nn.depthwise_conv2d(
        image, kernel, strides, padding='VALID', dilations=[1, 1])
  elif orig_image.shape.rank == 4:
    kernel = tf.constant([[1, 1, 1], [1, 5, 1], [1, 1, 1]],
                         dtype=tf.float32,
                         shape=[1, 3, 3, 1, 1]) / 13.
    strides = [1, 1, 1, 1, 1]
    # Run the kernel across each channel
    channels = tf.split(image, 3, axis=-1)
    degenerates = [
        tf.nn.conv3d(channel, kernel, strides, padding='VALID',
                     dilations=[1, 1, 1, 1, 1])
        for channel in channels
    ]
    degenerate = tf.concat(degenerates, -1)
  else:
    raise ValueError('Bad image rank: {}'.format(image.shape.rank))
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

  # For the borders of the resulting image, fill in the values of the
  # original image.
  mask = tf.ones_like(degenerate)
  paddings = [[0, 0]] * (orig_image.shape.rank - 3)
  padded_mask = tf.pad(mask, paddings + [[1, 1], [1, 1], [0, 0]])
  padded_degenerate = tf.pad(degenerate, paddings + [[1, 1], [1, 1], [0, 0]])
  result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

  # Blend the final result.
  return blend(result, orig_image, factor)


def equalize(image: tf.Tensor) -> tf.Tensor:
  """Implements Equalize function from PIL using TF ops."""

  def scale_channel(im, c):
    """Scale the data in the channel to implement equalize."""
    im = tf.cast(im[..., c], tf.int32)
    # Compute the histogram of the image channel.
    histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

    # For the purposes of computing the step, filter out the nonzeros.
    nonzero = tf.where(tf.not_equal(histo, 0))
    nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

    def build_lut(histo, step):
      # Compute the cumulative sum, shifting by step // 2
      # and then normalization by step.
      lut = (tf.cumsum(histo) + (step // 2)) // step
      # Shift lut, prepending with 0.
      lut = tf.concat([[0], lut[:-1]], 0)
      # Clip the counts to be in range.  This is done
      # in the C code for image.point.
      return tf.clip_by_value(lut, 0, 255)

    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    result = tf.cond(
        tf.equal(step, 0), lambda: im,
        lambda: tf.gather(build_lut(histo, step), im))

    return tf.cast(result, tf.uint8)

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image, 0)
  s2 = scale_channel(image, 1)
  s3 = scale_channel(image, 2)
  image = tf.stack([s1, s2, s3], -1)
  return image


def invert(image: tf.Tensor) -> tf.Tensor:
  """Inverts the image pixels."""
  image = tf.convert_to_tensor(image)
  return 255 - image


def wrap(image: tf.Tensor) -> tf.Tensor:
  """Returns 'image' with an extra channel set to all 1s."""
  shape = tf.shape(image)
  extended_channel = tf.expand_dims(tf.ones(shape[:-1], image.dtype), -1)
  extended = tf.concat([image, extended_channel], axis=-1)
  return extended


def unwrap(image: tf.Tensor, replace: int) -> tf.Tensor:
  """Unwraps an image produced by wrap.

  Where there is a 0 in the last channel for every spatial position,
  the rest of the three channels in that spatial dimension are grayed
  (set to 128).  Operations like translate and shear on a wrapped
  Tensor will leave 0s in empty locations.  Some transformations look
  at the intensity of values to do preprocessing, and we want these
  empty pixels to assume the 'average' value, rather than pure black.


  Args:
    image: A 3D Image Tensor with 4 channels.
    replace: A one or three value 1D tensor to fill empty pixels.

  Returns:
    image: A 3D image Tensor with 3 channels.
  """
  image_shape = tf.shape(image)
  # Flatten the spatial dimensions.
  flattened_image = tf.reshape(image, [-1, image_shape[-1]])

  # Find all pixels where the last channel is zero.
  alpha_channel = tf.expand_dims(flattened_image[..., 3], axis=-1)

  replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)

  # Where they are zero, fill them in with 'replace'.
  flattened_image = tf.where(
      tf.equal(alpha_channel, 0),
      tf.ones_like(flattened_image, dtype=image.dtype) * replace,
      flattened_image)

  image = tf.reshape(flattened_image, image_shape)
  image = tf.slice(
      image,
      [0] * image.shape.rank,
      tf.concat([image_shape[:-1], [3]], -1))
  return image


def _scale_bbox_only_op_probability(prob):
  """Reduce the probability of the bbox-only operation.

  Probability is reduced so that we do not distort the content of too many
  bounding boxes that are close to each other. The value of 3.0 was a chosen
  hyper parameter when designing the autoaugment algorithm that we found
  empirically to work well.

  Args:
    prob: Float that is the probability of applying the bbox-only operation.

  Returns:
    Reduced probability.
  """
  return prob / 3.0


def _apply_bbox_augmentation(image, bbox, augmentation_func, *args):
  """Applies augmentation_func to the subsection of image indicated by bbox.

  Args:
    image: 3D uint8 Tensor.
    bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
      of type float that represents the normalized coordinates between 0 and 1.
    augmentation_func: Augmentation function that will be applied to the
      subsection of image.
    *args: Additional parameters that will be passed into augmentation_func
      when it is called.

  Returns:
    A modified version of image, where the bbox location in the image will
    have `ugmentation_func applied to it.
  """
  image_height = tf.cast(tf.shape(image)[0], tf.float32)
  image_width = tf.cast(tf.shape(image)[1], tf.float32)
  min_y = tf.cast(image_height * bbox[0], tf.int32)
  min_x = tf.cast(image_width * bbox[1], tf.int32)
  max_y = tf.cast(image_height * bbox[2], tf.int32)
  max_x = tf.cast(image_width * bbox[3], tf.int32)
  image_height = tf.cast(image_height, tf.int32)
  image_width = tf.cast(image_width, tf.int32)

  # Clip to be sure the max values do not fall out of range.
  max_y = tf.minimum(max_y, image_height - 1)
  max_x = tf.minimum(max_x, image_width - 1)

  # Get the sub-tensor that is the image within the bounding box region.
  bbox_content = image[min_y:max_y + 1, min_x:max_x + 1, :]

  # Apply the augmentation function to the bbox portion of the image.
  augmented_bbox_content = augmentation_func(bbox_content, *args)

  # Pad the augmented_bbox_content and the mask to match the shape of original
  # image.
  augmented_bbox_content = tf.pad(augmented_bbox_content,
                                  [[min_y, (image_height - 1) - max_y],
                                   [min_x, (image_width - 1) - max_x],
                                   [0, 0]])

  # Create a mask that will be used to zero out a part of the original image.
  mask_tensor = tf.zeros_like(bbox_content)

  mask_tensor = tf.pad(mask_tensor,
                       [[min_y, (image_height - 1) - max_y],
                        [min_x, (image_width - 1) - max_x],
                        [0, 0]],
                       constant_values=1)
  # Replace the old bbox content with the new augmented content.
  image = image * mask_tensor + augmented_bbox_content
  return image


def _concat_bbox(bbox, bboxes):
  """Helper function that concates bbox to bboxes along the first dimension."""

  # Note if all elements in bboxes are -1 (_INVALID_BOX), then this means
  # we discard bboxes and start the bboxes Tensor with the current bbox.
  bboxes_sum_check = tf.reduce_sum(bboxes)
  bbox = tf.expand_dims(bbox, 0)
  # This check will be true when it is an _INVALID_BOX
  bboxes = tf.cond(tf.equal(bboxes_sum_check, -4.0),
                   lambda: bbox,
                   lambda: tf.concat([bboxes, bbox], 0))
  return bboxes


def _apply_bbox_augmentation_wrapper(image, bbox, new_bboxes, prob,
                                     augmentation_func, func_changes_bbox,
                                     *args):
  """Applies _apply_bbox_augmentation with probability prob.

  Args:
    image: 3D uint8 Tensor.
    bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
      of type float that represents the normalized coordinates between 0 and 1.
    new_bboxes: 2D Tensor that is a list of the bboxes in the image after they
      have been altered by aug_func. These will only be changed when
      func_changes_bbox is set to true. Each bbox has 4 elements
      (min_y, min_x, max_y, max_x) of type float that are the normalized
      bbox coordinates between 0 and 1.
    prob: Float that is the probability of applying _apply_bbox_augmentation.
    augmentation_func: Augmentation function that will be applied to the
      subsection of image.
    func_changes_bbox: Boolean. Does augmentation_func return bbox in addition
      to image.
    *args: Additional parameters that will be passed into augmentation_func
      when it is called.

  Returns:
    A tuple. Fist element is a modified version of image, where the bbox
    location in the image will have augmentation_func applied to it if it is
    chosen to be called with probability `prob`. The second element is a
    Tensor of Tensors of length 4 that will contain the altered bbox after
    applying augmentation_func.
  """
  should_apply_op = tf.cast(
      tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
  if func_changes_bbox:
    augmented_image, bbox = tf.cond(
        should_apply_op,
        lambda: augmentation_func(image, bbox, *args),
        lambda: (image, bbox))
  else:
    augmented_image = tf.cond(
        should_apply_op,
        lambda: _apply_bbox_augmentation(image, bbox, augmentation_func, *args),
        lambda: image)
  new_bboxes = _concat_bbox(bbox, new_bboxes)
  return augmented_image, new_bboxes


def _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob, aug_func,
                                           func_changes_bbox, *args):
  """Checks to be sure num bboxes > 0 before calling inner function."""
  num_bboxes = tf.shape(bboxes)[0]
  image, bboxes = tf.cond(
      tf.equal(num_bboxes, 0),
      lambda: (image, bboxes),
      # pylint:disable=g-long-lambda
      lambda: _apply_multi_bbox_augmentation(
          image, bboxes, prob, aug_func, func_changes_bbox, *args))
  # pylint:enable=g-long-lambda
  return image, bboxes


# Represents an invalid bounding box that is used for checking for padding
# lists of bounding box coordinates for a few augmentation operations
_INVALID_BOX = [[-1.0, -1.0, -1.0, -1.0]]


def _apply_multi_bbox_augmentation(image, bboxes, prob, aug_func,
                                   func_changes_bbox, *args):
  """Applies aug_func to the image for each bbox in bboxes.

  Args:
    image: 3D uint8 Tensor.
    bboxes: 2D Tensor that is a list of the bboxes in the image. Each bbox
      has 4 elements (min_y, min_x, max_y, max_x) of type float.
    prob: Float that is the probability of applying aug_func to a specific
      bounding box within the image.
    aug_func: Augmentation function that will be applied to the
      subsections of image indicated by the bbox values in bboxes.
    func_changes_bbox: Boolean. Does augmentation_func return bbox in addition
      to image.
    *args: Additional parameters that will be passed into augmentation_func
      when it is called.

  Returns:
    A modified version of image, where each bbox location in the image will
    have augmentation_func applied to it if it is chosen to be called with
    probability prob independently across all bboxes. Also the final
    bboxes are returned that will be unchanged if func_changes_bbox is set to
    false and if true, the new altered ones will be returned.

  Raises:
    ValueError if applied to video.
  """
  if image.shape.rank == 4:
    raise ValueError('Image rank 4 is not supported')

  # Will keep track of the new altered bboxes after aug_func is repeatedly
  # applied. The -1 values are a dummy value and this first Tensor will be
  # removed upon appending the first real bbox.
  new_bboxes = tf.constant(_INVALID_BOX)

  # If the bboxes are empty, then just give it _INVALID_BOX. The result
  # will be thrown away.
  bboxes = tf.cond(tf.equal(tf.size(bboxes), 0),
                   lambda: tf.constant(_INVALID_BOX),
                   lambda: bboxes)

  bboxes = tf.ensure_shape(bboxes, (None, 4))

  # pylint:disable=g-long-lambda
  wrapped_aug_func = (
      lambda _image, bbox, _new_bboxes: _apply_bbox_augmentation_wrapper(
          _image, bbox, _new_bboxes, prob, aug_func, func_changes_bbox, *args))
  # pylint:enable=g-long-lambda

  # Setup the while_loop.
  num_bboxes = tf.shape(bboxes)[0]  # We loop until we go over all bboxes.
  idx = tf.constant(0)  # Counter for the while loop.

  # Conditional function when to end the loop once we go over all bboxes
  # images_and_bboxes contain (_image, _new_bboxes)
  cond = lambda _idx, _images_and_bboxes: tf.less(_idx, num_bboxes)

  # Shuffle the bboxes so that the augmentation order is not deterministic if
  # we are not changing the bboxes with aug_func.
  if not func_changes_bbox:
    loop_bboxes = tf.random.shuffle(bboxes)
  else:
    loop_bboxes = bboxes

  # Main function of while_loop where we repeatedly apply augmentation on the
  # bboxes in the image.
  # pylint:disable=g-long-lambda
  body = lambda _idx, _images_and_bboxes: [
      _idx + 1, wrapped_aug_func(_images_and_bboxes[0],
                                 loop_bboxes[_idx],
                                 _images_and_bboxes[1])]
  # pylint:enable=g-long-lambda

  _, (image, new_bboxes) = tf.while_loop(
      cond, body, [idx, (image, new_bboxes)],
      shape_invariants=[idx.get_shape(),
                        (image.get_shape(), tf.TensorShape([None, 4]))])

  # Either return the altered bboxes or the original ones depending on if
  # we altered them in anyway.
  if func_changes_bbox:
    final_bboxes = new_bboxes
  else:
    final_bboxes = bboxes
  return image, final_bboxes


def _clip_bbox(min_y, min_x, max_y, max_x):
  """Clip bounding box coordinates between 0 and 1.

  Args:
    min_y: Normalized bbox coordinate of type float between 0 and 1.
    min_x: Normalized bbox coordinate of type float between 0 and 1.
    max_y: Normalized bbox coordinate of type float between 0 and 1.
    max_x: Normalized bbox coordinate of type float between 0 and 1.

  Returns:
    Clipped coordinate values between 0 and 1.
  """
  min_y = tf.clip_by_value(min_y, 0.0, 1.0)
  min_x = tf.clip_by_value(min_x, 0.0, 1.0)
  max_y = tf.clip_by_value(max_y, 0.0, 1.0)
  max_x = tf.clip_by_value(max_x, 0.0, 1.0)
  return min_y, min_x, max_y, max_x


def _check_bbox_area(min_y, min_x, max_y, max_x, delta=0.05):
  """Adjusts bbox coordinates to make sure the area is > 0.

  Args:
    min_y: Normalized bbox coordinate of type float between 0 and 1.
    min_x: Normalized bbox coordinate of type float between 0 and 1.
    max_y: Normalized bbox coordinate of type float between 0 and 1.
    max_x: Normalized bbox coordinate of type float between 0 and 1.
    delta: Float, this is used to create a gap of size 2 * delta between
      bbox min/max coordinates that are the same on the boundary.
      This prevents the bbox from having an area of zero.

  Returns:
    Tuple of new bbox coordinates between 0 and 1 that will now have a
    guaranteed area > 0.
  """
  height = max_y - min_y
  width = max_x - min_x
  def _adjust_bbox_boundaries(min_coord, max_coord):
    # Make sure max is never 0 and min is never 1.
    max_coord = tf.maximum(max_coord, 0.0 + delta)
    min_coord = tf.minimum(min_coord, 1.0 - delta)
    return min_coord, max_coord
  min_y, max_y = tf.cond(tf.equal(height, 0.0),
                         lambda: _adjust_bbox_boundaries(min_y, max_y),
                         lambda: (min_y, max_y))
  min_x, max_x = tf.cond(tf.equal(width, 0.0),
                         lambda: _adjust_bbox_boundaries(min_x, max_x),
                         lambda: (min_x, max_x))
  return min_y, min_x, max_y, max_x


def _rotate_bbox(bbox, image_height, image_width, degrees):
  """Rotates the bbox coordinated by degrees.

  Args:
    bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
      of type float that represents the normalized coordinates between 0 and 1.
    image_height: Int, height of the image.
    image_width: Int, height of the image.
    degrees: Float, a scalar angle in degrees to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.

  Returns:
    A tensor of the same shape as bbox, but now with the rotated coordinates.
  """
  image_height, image_width = (
      tf.cast(image_height, tf.float32), tf.cast(image_width, tf.float32))

  # Convert from degrees to radians.
  degrees_to_radians = math.pi / 180.0
  radians = degrees * degrees_to_radians

  # Translate the bbox to the center of the image and turn the normalized 0-1
  # coordinates to absolute pixel locations.
  # Y coordinates are made negative as the y axis of images goes down with
  # increasing pixel values, so we negate to make sure x axis and y axis points
  # are in the traditionally positive direction.
  min_y = -tf.cast(image_height * (bbox[0] - 0.5), tf.int32)
  min_x = tf.cast(image_width * (bbox[1] - 0.5), tf.int32)
  max_y = -tf.cast(image_height * (bbox[2] - 0.5), tf.int32)
  max_x = tf.cast(image_width * (bbox[3] - 0.5), tf.int32)
  coordinates = tf.stack(
      [[min_y, min_x], [min_y, max_x], [max_y, min_x], [max_y, max_x]])
  coordinates = tf.cast(coordinates, tf.float32)
  # Rotate the coordinates according to the rotation matrix clockwise if
  # radians is positive, else negative
  rotation_matrix = tf.stack(
      [[tf.cos(radians), tf.sin(radians)],
       [-tf.sin(radians), tf.cos(radians)]])
  new_coords = tf.cast(
      tf.matmul(rotation_matrix, tf.transpose(coordinates)), tf.int32)
  # Find min/max values and convert them back to normalized 0-1 floats.
  min_y = -(
      tf.cast(tf.reduce_max(new_coords[0, :]), tf.float32) / image_height - 0.5)
  min_x = tf.cast(tf.reduce_min(new_coords[1, :]),
                  tf.float32) / image_width + 0.5
  max_y = -(
      tf.cast(tf.reduce_min(new_coords[0, :]), tf.float32) / image_height - 0.5)
  max_x = tf.cast(tf.reduce_max(new_coords[1, :]),
                  tf.float32) / image_width + 0.5

  # Clip the bboxes to be sure the fall between [0, 1].
  min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
  min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
  return tf.stack([min_y, min_x, max_y, max_x])


def rotate_with_bboxes(image, bboxes, degrees, replace):
  """Equivalent of PIL Rotate that rotates the image and bbox.

  Args:
    image: 3D uint8 Tensor.
    bboxes: 2D Tensor that is a list of the bboxes in the image. Each bbox
      has 4 elements (min_y, min_x, max_y, max_x) of type float.
    degrees: Float, a scalar angle in degrees to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.
    replace: A one or three value 1D tensor to fill empty pixels.

  Returns:
    A tuple containing a 3D uint8 Tensor that will be the result of rotating
    image by degrees. The second element of the tuple is bboxes, where now
    the coordinates will be shifted to reflect the rotated image.

  Raises:
    ValueError: If applied to video.
  """
  if image.shape.rank == 4:
    raise ValueError('Image rank 4 is not supported')

  # Rotate the image.
  image = wrapped_rotate(image, degrees, replace)

  # Convert bbox coordinates to pixel values.
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  # pylint:disable=g-long-lambda
  wrapped_rotate_bbox = lambda bbox: _rotate_bbox(
      bbox, image_height, image_width, degrees)
  # pylint:enable=g-long-lambda
  bboxes = tf.map_fn(wrapped_rotate_bbox, bboxes)
  return image, bboxes


def _shear_bbox(bbox, image_height, image_width, level, shear_horizontal):
  """Shifts the bbox according to how the image was sheared.

  Args:
    bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
      of type float that represents the normalized coordinates between 0 and 1.
    image_height: Int, height of the image.
    image_width: Int, height of the image.
    level: Float. How much to shear the image.
    shear_horizontal: If true then shear in X dimension else shear in
      the Y dimension.

  Returns:
    A tensor of the same shape as bbox, but now with the shifted coordinates.
  """
  image_height, image_width = (
      tf.cast(image_height, tf.float32), tf.cast(image_width, tf.float32))

  # Change bbox coordinates to be pixels.
  min_y = tf.cast(image_height * bbox[0], tf.int32)
  min_x = tf.cast(image_width * bbox[1], tf.int32)
  max_y = tf.cast(image_height * bbox[2], tf.int32)
  max_x = tf.cast(image_width * bbox[3], tf.int32)
  coordinates = tf.stack(
      [[min_y, min_x], [min_y, max_x], [max_y, min_x], [max_y, max_x]])
  coordinates = tf.cast(coordinates, tf.float32)

  # Shear the coordinates according to the translation matrix.
  if shear_horizontal:
    translation_matrix = tf.stack(
        [[1, 0], [-level, 1]])
  else:
    translation_matrix = tf.stack(
        [[1, -level], [0, 1]])
  translation_matrix = tf.cast(translation_matrix, tf.float32)
  new_coords = tf.cast(
      tf.matmul(translation_matrix, tf.transpose(coordinates)), tf.int32)

  # Find min/max values and convert them back to floats.
  min_y = tf.cast(tf.reduce_min(new_coords[0, :]), tf.float32) / image_height
  min_x = tf.cast(tf.reduce_min(new_coords[1, :]), tf.float32) / image_width
  max_y = tf.cast(tf.reduce_max(new_coords[0, :]), tf.float32) / image_height
  max_x = tf.cast(tf.reduce_max(new_coords[1, :]), tf.float32) / image_width

  # Clip the bboxes to be sure the fall between [0, 1].
  min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
  min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
  return tf.stack([min_y, min_x, max_y, max_x])


def shear_with_bboxes(image, bboxes, level, replace, shear_horizontal):
  """Applies Shear Transformation to the image and shifts the bboxes.

  Args:
    image: 3D uint8 Tensor.
    bboxes: 2D Tensor that is a list of the bboxes in the image. Each bbox
      has 4 elements (min_y, min_x, max_y, max_x) of type float with values
      between [0, 1].
    level: Float. How much to shear the image. This value will be between
      -0.3 to 0.3.
    replace: A one or three value 1D tensor to fill empty pixels.
    shear_horizontal: Boolean. If true then shear in X dimension else shear in
      the Y dimension.

  Returns:
    A tuple containing a 3D uint8 Tensor that will be the result of shearing
    image by level. The second element of the tuple is bboxes, where now
    the coordinates will be shifted to reflect the sheared image.

  Raises:
    ValueError: If applied to video.
  """
  if image.shape.rank == 4:
    raise ValueError('Image rank 4 is not supported')

  if shear_horizontal:
    image = shear_x(image, level, replace)
  else:
    image = shear_y(image, level, replace)

  # Convert bbox coordinates to pixel values.
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  # pylint:disable=g-long-lambda
  wrapped_shear_bbox = lambda bbox: _shear_bbox(
      bbox, image_height, image_width, level, shear_horizontal)
  # pylint:enable=g-long-lambda
  bboxes = tf.map_fn(wrapped_shear_bbox, bboxes)
  return image, bboxes


def _shift_bbox(bbox, image_height, image_width, pixels, shift_horizontal):
  """Shifts the bbox coordinates by pixels.

  Args:
    bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
      of type float that represents the normalized coordinates between 0 and 1.
    image_height: Int, height of the image.
    image_width: Int, width of the image.
    pixels: An int. How many pixels to shift the bbox.
    shift_horizontal: Boolean. If true then shift in X dimension else shift in
      Y dimension.

  Returns:
    A tensor of the same shape as bbox, but now with the shifted coordinates.
  """
  pixels = tf.cast(pixels, tf.int32)
  # Convert bbox to integer pixel locations.
  min_y = tf.cast(tf.cast(image_height, tf.float32) * bbox[0], tf.int32)
  min_x = tf.cast(tf.cast(image_width, tf.float32) * bbox[1], tf.int32)
  max_y = tf.cast(tf.cast(image_height, tf.float32) * bbox[2], tf.int32)
  max_x = tf.cast(tf.cast(image_width, tf.float32) * bbox[3], tf.int32)

  if shift_horizontal:
    min_x = tf.maximum(0, min_x - pixels)
    max_x = tf.minimum(image_width, max_x - pixels)
  else:
    min_y = tf.maximum(0, min_y - pixels)
    max_y = tf.minimum(image_height, max_y - pixels)

  # Convert bbox back to floats.
  min_y = tf.cast(min_y, tf.float32) / tf.cast(image_height, tf.float32)
  min_x = tf.cast(min_x, tf.float32) / tf.cast(image_width, tf.float32)
  max_y = tf.cast(max_y, tf.float32) / tf.cast(image_height, tf.float32)
  max_x = tf.cast(max_x, tf.float32) / tf.cast(image_width, tf.float32)

  # Clip the bboxes to be sure the fall between [0, 1].
  min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
  min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
  return tf.stack([min_y, min_x, max_y, max_x])


def translate_bbox(image, bboxes, pixels, replace, shift_horizontal):
  """Equivalent of PIL Translate in X/Y dimension that shifts image and bbox.

  Args:
    image: 3D uint8 Tensor.
    bboxes: 2D Tensor that is a list of the bboxes in the image. Each bbox
      has 4 elements (min_y, min_x, max_y, max_x) of type float with values
      between [0, 1].
    pixels: An int. How many pixels to shift the image and bboxes
    replace: A one or three value 1D tensor to fill empty pixels.
    shift_horizontal: Boolean. If true then shift in X dimension else shift in
      Y dimension.

  Returns:
    A tuple containing a 3D uint8 Tensor that will be the result of translating
    image by pixels. The second element of the tuple is bboxes, where now
    the coordinates will be shifted to reflect the shifted image.

  Raises:
    ValueError if applied to video.
  """
  if image.shape.rank == 4:
    raise ValueError('Image rank 4 is not supported')

  if shift_horizontal:
    image = translate_x(image, pixels, replace)
  else:
    image = translate_y(image, pixels, replace)

  # Convert bbox coordinates to pixel values.
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  # pylint:disable=g-long-lambda
  wrapped_shift_bbox = lambda bbox: _shift_bbox(
      bbox, image_height, image_width, pixels, shift_horizontal)
  # pylint:enable=g-long-lambda
  bboxes = tf.map_fn(wrapped_shift_bbox, bboxes)
  return image, bboxes


def translate_y_only_bboxes(
    image: tf.Tensor, bboxes: tf.Tensor, prob: float, pixels: int, replace):
  """Apply translate_y to each bbox in the image with probability prob."""
  if bboxes.shape.rank == 4:
    raise ValueError('translate_y_only_bboxes does not support rank 4 boxes')

  func_changes_bbox = False
  prob = _scale_bbox_only_op_probability(prob)
  return _apply_multi_bbox_augmentation_wrapper(
      image, bboxes, prob, translate_y, func_changes_bbox, pixels, replace)


def _randomly_negate_tensor(tensor):
  """With 50% prob turn the tensor negative."""
  should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
  final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
  return final_tensor


def _rotate_level_to_arg(level: float):
  level = (level / _MAX_LEVEL) * 30.
  level = _randomly_negate_tensor(level)
  return (level,)


def _shrink_level_to_arg(level: float):
  """Converts level to ratio by which we shrink the image content."""
  if level == 0:
    return (1.0,)  # if level is zero, do not shrink the image
  # Maximum shrinking ratio is 2.9.
  level = 2. / (_MAX_LEVEL / level) + 0.9
  return (level,)


def _enhance_level_to_arg(level: float):
  return ((level / _MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level: float):
  level = (level / _MAX_LEVEL) * 0.3
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _translate_level_to_arg(level: float, translate_const: float):
  level = (level / _MAX_LEVEL) * float(translate_const)
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _mult_to_arg(level: float, multiplier: float = 1.):
  return (int((level / _MAX_LEVEL) * multiplier),)


def _apply_func_with_prob(func: Any, image: tf.Tensor,
                          bboxes: Optional[tf.Tensor], args: Any, prob: float):
  """Apply `func` to image w/ `args` as input with probability `prob`."""
  assert isinstance(args, tuple)
  assert inspect.getfullargspec(func)[0][1] == 'bboxes'

  # Apply the function with probability `prob`.
  should_apply_op = tf.cast(
      tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
  augmented_image, augmented_bboxes = tf.cond(
      should_apply_op,
      lambda: func(image, bboxes, *args),
      lambda: (image, bboxes))
  return augmented_image, augmented_bboxes


def select_and_apply_random_policy(policies: Any,
                                   image: tf.Tensor,
                                   bboxes: Optional[tf.Tensor] = None):
  """Select a random policy from `policies` and apply it to `image`."""
  policy_to_select = tf.random.uniform([], maxval=len(policies), dtype=tf.int32)
  # Note that using tf.case instead of tf.conds would result in significantly
  # larger graphs and would even break export for some larger policies.
  for (i, policy) in enumerate(policies):
    image, bboxes = tf.cond(
        tf.equal(i, policy_to_select),
        lambda selected_policy=policy: selected_policy(image, bboxes),
        lambda: (image, bboxes))
  return image, bboxes


NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': wrapped_rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Cutout': cutout,
    'Rotate_BBox': rotate_with_bboxes,
    # pylint:disable=g-long-lambda
    'ShearX_BBox': lambda image, bboxes, level, replace: shear_with_bboxes(
        image, bboxes, level, replace, shear_horizontal=True),
    'ShearY_BBox': lambda image, bboxes, level, replace: shear_with_bboxes(
        image, bboxes, level, replace, shear_horizontal=False),
    'TranslateX_BBox': lambda image, bboxes, pixels, replace: translate_bbox(
        image, bboxes, pixels, replace, shift_horizontal=True),
    'TranslateY_BBox': lambda image, bboxes, pixels, replace: translate_bbox(
        image, bboxes, pixels, replace, shift_horizontal=False),
    # pylint:enable=g-long-lambda
    'TranslateY_Only_BBoxes': translate_y_only_bboxes,
}

# Functions that require a `bboxes` parameter.
REQUIRE_BOXES_FUNCS = frozenset({
    'Rotate_BBox',
    'ShearX_BBox',
    'ShearY_BBox',
    'TranslateX_BBox',
    'TranslateY_BBox',
    'TranslateY_Only_BBoxes',
})

# Functions that have a 'prob' parameter
PROB_FUNCS = frozenset({
    'TranslateY_Only_BBoxes',
})

# Functions that have a 'replace' parameter
REPLACE_FUNCS = frozenset({
    'Rotate',
    'TranslateX',
    'ShearX',
    'ShearY',
    'TranslateY',
    'Cutout',
    'Rotate_BBox',
    'ShearX_BBox',
    'ShearY_BBox',
    'TranslateX_BBox',
    'TranslateY_BBox',
    'TranslateY_Only_BBoxes',
})


def level_to_arg(cutout_const: float, translate_const: float):
  """Creates a dict mapping image operation names to their arguments."""

  no_arg = lambda level: ()
  posterize_arg = lambda level: _mult_to_arg(level, 4)
  solarize_arg = lambda level: _mult_to_arg(level, 256)
  solarize_add_arg = lambda level: _mult_to_arg(level, 110)
  cutout_arg = lambda level: _mult_to_arg(level, cutout_const)
  translate_arg = lambda level: _translate_level_to_arg(level, translate_const)
  translate_bbox_arg = lambda level: _translate_level_to_arg(level, 120)

  args = {
      'AutoContrast': no_arg,
      'Equalize': no_arg,
      'Invert': no_arg,
      'Rotate': _rotate_level_to_arg,
      'Posterize': posterize_arg,
      'Solarize': solarize_arg,
      'SolarizeAdd': solarize_add_arg,
      'Color': _enhance_level_to_arg,
      'Contrast': _enhance_level_to_arg,
      'Brightness': _enhance_level_to_arg,
      'Sharpness': _enhance_level_to_arg,
      'ShearX': _shear_level_to_arg,
      'ShearY': _shear_level_to_arg,
      'Cutout': cutout_arg,
      'TranslateX': translate_arg,
      'TranslateY': translate_arg,
      'Rotate_BBox': _rotate_level_to_arg,
      'ShearX_BBox': _shear_level_to_arg,
      'ShearY_BBox': _shear_level_to_arg,
      # pylint:disable=g-long-lambda
      'TranslateX_BBox': lambda level: _translate_level_to_arg(
          level, translate_const),
      'TranslateY_BBox': lambda level: _translate_level_to_arg(
          level, translate_const),
      # pylint:enable=g-long-lambda
      'TranslateY_Only_BBoxes': translate_bbox_arg,
  }
  return args


def bbox_wrapper(func):
  """Adds a bboxes function argument to func and returns unchanged bboxes."""
  def wrapper(images, bboxes, *args, **kwargs):
    return (func(images, *args, **kwargs), bboxes)
  return wrapper


def _parse_policy_info(name: Text,
                       prob: float,
                       level: float,
                       replace_value: List[int],
                       cutout_const: float,
                       translate_const: float,
                       level_std: float = 0.) -> Tuple[Any, float, Any]:
  """Return the function that corresponds to `name` and update `level` param."""
  func = NAME_TO_FUNC[name]

  if level_std > 0:
    level += tf.random.normal([], dtype=tf.float32)
    level = tf.clip_by_value(level, 0., _MAX_LEVEL)

  args = level_to_arg(cutout_const, translate_const)[name](level)

  if name in PROB_FUNCS:
    # Add in the prob arg if it is required for the function that is called.
    args = tuple([prob] + list(args))

  if name in REPLACE_FUNCS:
    # Add in replace arg if it is required for the function that is called.
    args = tuple(list(args) + [replace_value])

  # Add bboxes as the second positional argument for the function if it does
  # not already exist.
  if 'bboxes' not in inspect.getfullargspec(func)[0]:
    func = bbox_wrapper(func)

  return func, prob, args


class ImageAugment(object):
  """Image augmentation class for applying image distortions."""

  def distort(
      self,
      image: tf.Tensor
  ) -> tf.Tensor:
    """Given an image tensor, returns a distorted image with the same shape.

    Args:
      image: `Tensor` of shape [height, width, 3] or
        [num_frames, height, width, 3] representing an image or image sequence.

    Returns:
      The augmented version of `image`.
    """
    raise NotImplementedError()

  def distort_with_boxes(
      self,
      image: tf.Tensor,
      bboxes: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Distorts the image and bounding boxes.

    Args:
      image: `Tensor` of shape [height, width, 3] or
        [num_frames, height, width, 3] representing an image or image sequence.
      bboxes: `Tensor` of shape [num_boxes, 4] or [num_frames, num_boxes, 4]
        representing bounding boxes for an image or image sequence.

    Returns:
      The augmented version of `image` and `bboxes`.
    """
    raise NotImplementedError


class AutoAugment(ImageAugment):
  """Applies the AutoAugment policy to images.

    AutoAugment is from the paper: https://arxiv.org/abs/1805.09501.
  """

  def __init__(self,
               augmentation_name: Text = 'v0',
               policies: Optional[Iterable[Iterable[Tuple[Text, float,
                                                          float]]]] = None,
               cutout_const: float = 100,
               translate_const: float = 250):
    """Applies the AutoAugment policy to images.

    Args:
      augmentation_name: The name of the AutoAugment policy to use. The
        available options are `v0`, `test`, `reduced_cifar10`, `svhn` and
        `reduced_imagenet`. `v0` is the policy used for all
        of the results in the paper and was found to achieve the best results on
        the COCO dataset. `v1`, `v2` and `v3` are additional good policies found
        on the COCO dataset that have slight variation in what operations were
        used during the search procedure along with how many operations are
        applied in parallel to a single image (2 vs 3). Make sure to set
        `policies` to `None` (the default) if you want to set options using
        `augmentation_name`.
      policies: list of lists of tuples in the form `(func, prob, level)`,
        `func` is a string name of the augmentation function, `prob` is the
        probability of applying the `func` operation, `level` (or magnitude) is
        the input argument for `func`. For example:
        ```
        [[('Equalize', 0.9, 3), ('Color', 0.7, 8)],
         [('Invert', 0.6, 5), ('Rotate', 0.2, 9), ('ShearX', 0.1, 2)], ...]
        ```
        The outer-most list must be 3-d. The number of operations in a
        sub-policy can vary from one sub-policy to another.
        If you provide `policies` as input, any option set with
        `augmentation_name` will get overriden as they are mutually exclusive.
      cutout_const: multiplier for applying cutout.
      translate_const: multiplier for applying translation.

    Raises:
      ValueError if `augmentation_name` is unsupported.
    """
    super(AutoAugment, self).__init__()

    self.augmentation_name = augmentation_name
    self.cutout_const = float(cutout_const)
    self.translate_const = float(translate_const)
    self.available_policies = {
        'detection_v0': self.detection_policy_v0(),
        'v0': self.policy_v0(),
        'test': self.policy_test(),
        'simple': self.policy_simple(),
        'reduced_cifar10': self.policy_reduced_cifar10(),
        'svhn': self.policy_svhn(),
        'reduced_imagenet': self.policy_reduced_imagenet(),
    }

    if not policies:
      if augmentation_name not in self.available_policies:
        raise ValueError(
            'Invalid augmentation_name: {}'.format(augmentation_name))

      self.policies = self.available_policies[augmentation_name]

    else:
      self._check_policy_shape(policies)
      self.policies = policies

  def _check_policy_shape(self, policies):
    """Checks dimension and shape of the custom policy.

    Args:
      policies: List of list of tuples in the form `(func, prob, level)`. Must
        have shape of `(:, :, 3)`.

    Raises:
      ValueError if the shape of `policies` is unexpected.
    """
    in_shape = np.array(policies).shape
    if len(in_shape) != 3 or in_shape[-1:] != (3,):
      raise ValueError('Wrong shape detected for custom policy. Expected '
                       '(:, :, 3) but got {}.'.format(in_shape))

  def _make_tf_policies(self):
    """Prepares the TF functions for augmentations based on the policies."""
    replace_value = [128] * 3

    # func is the string name of the augmentation function, prob is the
    # probability of applying the operation and level is the parameter
    # associated with the tf op.

    # tf_policies are functions that take in an image and return an augmented
    # image.
    tf_policies = []
    for policy in self.policies:
      tf_policy = []
      assert_ranges = []
      # Link string name to the correct python function and make sure the
      # correct argument is passed into that function.
      for policy_info in policy:
        _, prob, level = policy_info
        assert_ranges.append(tf.Assert(tf.less_equal(prob, 1.), [prob]))
        assert_ranges.append(
            tf.Assert(tf.less_equal(level, int(_MAX_LEVEL)), [level]))

        policy_info = list(policy_info) + [
            replace_value, self.cutout_const, self.translate_const
        ]
        tf_policy.append(_parse_policy_info(*policy_info))
      # Now build the tf policy that will apply the augmentation procedue
      # on image.
      def make_final_policy(tf_policy_):

        def final_policy(image_, bboxes_):
          for func, prob, args in tf_policy_:
            image_, bboxes_ = _apply_func_with_prob(func, image_, bboxes_, args,
                                                    prob)
          return image_, bboxes_

        return final_policy

      with tf.control_dependencies(assert_ranges):
        tf_policies.append(make_final_policy(tf_policy))

    return tf_policies

  def distort(self, image: tf.Tensor) -> tf.Tensor:
    """See base class."""
    input_image_type = image.dtype
    if input_image_type != tf.uint8:
      image = tf.clip_by_value(image, 0.0, 255.0)
      image = tf.cast(image, dtype=tf.uint8)

    tf_policies = self._make_tf_policies()
    image, _ = select_and_apply_random_policy(tf_policies, image, bboxes=None)
    return image

  def distort_with_boxes(self, image: tf.Tensor,
                         bboxes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """See base class."""
    input_image_type = image.dtype
    if input_image_type != tf.uint8:
      image = tf.clip_by_value(image, 0.0, 255.0)
      image = tf.cast(image, dtype=tf.uint8)

    tf_policies = self._make_tf_policies()
    image, bboxes = select_and_apply_random_policy(tf_policies, image, bboxes)
    return image, bboxes

  @staticmethod
  def detection_policy_v0():
    """Autoaugment policy that was used in AutoAugment Paper for Detection.

    https://arxiv.org/pdf/1906.11172

    Each tuple is an augmentation operation of the form
    (operation, probability, magnitude). Each element in policy is a
    sub-policy that will be applied sequentially on the image.

    Returns:
      the policy.
    """
    policy = [
        [('TranslateX_BBox', 0.6, 4), ('Equalize', 0.8, 10)],
        [('TranslateY_Only_BBoxes', 0.2, 2), ('Cutout', 0.8, 8)],
        [('Sharpness', 0.0, 8), ('ShearX_BBox', 0.4, 0)],
        [('ShearY_BBox', 1.0, 2), ('TranslateY_Only_BBoxes', 0.6, 6)],
        [('Rotate_BBox', 0.6, 10), ('Color', 1.0, 6)],
    ]
    return policy

  @staticmethod
  def policy_v0():
    """Autoaugment policy that was used in AutoAugment Paper.

    Each tuple is an augmentation operation of the form
    (operation, probability, magnitude). Each element in policy is a
    sub-policy that will be applied sequentially on the image.

    Returns:
      the policy.
    """

    policy = [
        [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
        [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
        [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
        [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
        [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
        [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
        [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
        [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
        [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
        [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
        [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
        [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
        [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
        [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
        [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
        [('Rotate', 1.0, 7), ('TranslateY', 0.8, 9)],
        [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
        [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
        [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
        [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
        [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
        [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
        [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
        [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
        [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
    ]
    return policy

  @staticmethod
  def policy_reduced_cifar10():
    """Autoaugment policy for reduced CIFAR-10 dataset.

    Result is from the AutoAugment paper: https://arxiv.org/abs/1805.09501.

    Each tuple is an augmentation operation of the form
    (operation, probability, magnitude). Each element in policy is a
    sub-policy that will be applied sequentially on the image.

    Returns:
      the policy.
    """
    policy = [
        [('Invert', 0.1, 7), ('Contrast', 0.2, 6)],
        [('Rotate', 0.7, 2), ('TranslateX', 0.3, 9)],
        [('Sharpness', 0.8, 1), ('Sharpness', 0.9, 3)],
        [('ShearY', 0.5, 8), ('TranslateY', 0.7, 9)],
        [('AutoContrast', 0.5, 8), ('Equalize', 0.9, 2)],
        [('ShearY', 0.2, 7), ('Posterize', 0.3, 7)],
        [('Color', 0.4, 3), ('Brightness', 0.6, 7)],
        [('Sharpness', 0.3, 9), ('Brightness', 0.7, 9)],
        [('Equalize', 0.6, 5), ('Equalize', 0.5, 1)],
        [('Contrast', 0.6, 7), ('Sharpness', 0.6, 5)],
        [('Color', 0.7, 7), ('TranslateX', 0.5, 8)],
        [('Equalize', 0.3, 7), ('AutoContrast', 0.4, 8)],
        [('TranslateY', 0.4, 3), ('Sharpness', 0.2, 6)],
        [('Brightness', 0.9, 6), ('Color', 0.2, 8)],
        [('Solarize', 0.5, 2), ('Invert', 0.0, 3)],
        [('Equalize', 0.2, 0), ('AutoContrast', 0.6, 0)],
        [('Equalize', 0.2, 8), ('Equalize', 0.6, 4)],
        [('Color', 0.9, 9), ('Equalize', 0.6, 6)],
        [('AutoContrast', 0.8, 4), ('Solarize', 0.2, 8)],
        [('Brightness', 0.1, 3), ('Color', 0.7, 0)],
        [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 3)],
        [('TranslateY', 0.9, 9), ('TranslateY', 0.7, 9)],
        [('AutoContrast', 0.9, 2), ('Solarize', 0.8, 3)],
        [('Equalize', 0.8, 8), ('Invert', 0.1, 3)],
        [('TranslateY', 0.7, 9), ('AutoContrast', 0.9, 1)],
    ]
    return policy

  @staticmethod
  def policy_svhn():
    """Autoaugment policy for SVHN dataset.

    Result is from the AutoAugment paper: https://arxiv.org/abs/1805.09501.

    Each tuple is an augmentation operation of the form
    (operation, probability, magnitude). Each element in policy is a
    sub-policy that will be applied sequentially on the image.

    Returns:
      the policy.
    """
    policy = [
        [('ShearX', 0.9, 4), ('Invert', 0.2, 3)],
        [('ShearY', 0.9, 8), ('Invert', 0.7, 5)],
        [('Equalize', 0.6, 5), ('Solarize', 0.6, 6)],
        [('Invert', 0.9, 3), ('Equalize', 0.6, 3)],
        [('Equalize', 0.6, 1), ('Rotate', 0.9, 3)],
        [('ShearX', 0.9, 4), ('AutoContrast', 0.8, 3)],
        [('ShearY', 0.9, 8), ('Invert', 0.4, 5)],
        [('ShearY', 0.9, 5), ('Solarize', 0.2, 6)],
        [('Invert', 0.9, 6), ('AutoContrast', 0.8, 1)],
        [('Equalize', 0.6, 3), ('Rotate', 0.9, 3)],
        [('ShearX', 0.9, 4), ('Solarize', 0.3, 3)],
        [('ShearY', 0.8, 8), ('Invert', 0.7, 4)],
        [('Equalize', 0.9, 5), ('TranslateY', 0.6, 6)],
        [('Invert', 0.9, 4), ('Equalize', 0.6, 7)],
        [('Contrast', 0.3, 3), ('Rotate', 0.8, 4)],
        [('Invert', 0.8, 5), ('TranslateY', 0.0, 2)],
        [('ShearY', 0.7, 6), ('Solarize', 0.4, 8)],
        [('Invert', 0.6, 4), ('Rotate', 0.8, 4)],
        [('ShearY', 0.3, 7), ('TranslateX', 0.9, 3)],
        [('ShearX', 0.1, 6), ('Invert', 0.6, 5)],
        [('Solarize', 0.7, 2), ('TranslateY', 0.6, 7)],
        [('ShearY', 0.8, 4), ('Invert', 0.8, 8)],
        [('ShearX', 0.7, 9), ('TranslateY', 0.8, 3)],
        [('ShearY', 0.8, 5), ('AutoContrast', 0.7, 3)],
        [('ShearX', 0.7, 2), ('Invert', 0.1, 5)],
    ]
    return policy

  @staticmethod
  def policy_reduced_imagenet():
    """Autoaugment policy for reduced ImageNet dataset.

    Result is from the AutoAugment paper: https://arxiv.org/abs/1805.09501.

    Each tuple is an augmentation operation of the form
    (operation, probability, magnitude). Each element in policy is a
    sub-policy that will be applied sequentially on the image.

    Returns:
      the policy.
    """
    policy = [
        [('Posterize', 0.4, 8), ('Rotate', 0.6, 9)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
        [('Posterize', 0.6, 7), ('Posterize', 0.6, 6)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)],
        [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
        [('Posterize', 0.8, 5), ('Equalize', 1.0, 2)],
        [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
        [('Equalize', 0.6, 8), ('Posterize', 0.4, 6)],
        [('Rotate', 0.8, 8), ('Color', 0.4, 0)],
        [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)],
        [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Rotate', 0.8, 8), ('Color', 1.0, 2)],
        [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
        [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
        [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
        [('Color', 0.4, 0), ('Equalize', 0.6, 3)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)]
    ]
    return policy

  @staticmethod
  def policy_simple():
    """Same as `policy_v0`, except with custom ops removed."""

    policy = [
        [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
        [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
        [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
        [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
        [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
        [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
        [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
        [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
        [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
        [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
        [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
        [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
        [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
    ]
    return policy

  @staticmethod
  def policy_test():
    """Autoaugment test policy for debugging."""
    policy = [
        [('TranslateX', 1.0, 4), ('Equalize', 1.0, 10)],
    ]
    return policy


def _maybe_identity(x: Optional[tf.Tensor]) -> Optional[tf.Tensor]:
  return tf.identity(x) if x is not None else None


class RandAugment(ImageAugment):
  """Applies the RandAugment policy to images.

  RandAugment is from the paper https://arxiv.org/abs/1909.13719,
  """

  def __init__(self,
               num_layers: int = 2,
               magnitude: float = 10.,
               cutout_const: float = 40.,
               translate_const: float = 100.,
               magnitude_std: float = 0.0,
               prob_to_apply: Optional[float] = None,
               exclude_ops: Optional[List[str]] = None):
    """Applies the RandAugment policy to images.

    Args:
      num_layers: Integer, the number of augmentation transformations to apply
        sequentially to an image. Represented as (N) in the paper. Usually best
        values will be in the range [1, 3].
      magnitude: Integer, shared magnitude across all augmentation operations.
        Represented as (M) in the paper. Usually best values are in the range
        [5, 10].
      cutout_const: multiplier for applying cutout.
      translate_const: multiplier for applying translation.
      magnitude_std: randomness of the severity as proposed by the authors of
        the timm library.
      prob_to_apply: The probability to apply the selected augmentation at each
        layer.
      exclude_ops: exclude selected operations.
    """
    super(RandAugment, self).__init__()

    self.num_layers = num_layers
    self.magnitude = float(magnitude)
    self.cutout_const = float(cutout_const)
    self.translate_const = float(translate_const)
    self.prob_to_apply = (
        float(prob_to_apply) if prob_to_apply is not None else None)
    self.available_ops = [
        'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize', 'Solarize',
        'Color', 'Contrast', 'Brightness', 'Sharpness', 'ShearX', 'ShearY',
        'TranslateX', 'TranslateY', 'Cutout', 'SolarizeAdd'
    ]
    self.magnitude_std = magnitude_std
    if exclude_ops:
      self.available_ops = [
          op for op in self.available_ops if op not in exclude_ops
      ]

  @classmethod
  def build_for_detection(cls,
                          num_layers: int = 2,
                          magnitude: float = 10.,
                          cutout_const: float = 40.,
                          translate_const: float = 100.,
                          magnitude_std: float = 0.0,
                          prob_to_apply: Optional[float] = None,
                          exclude_ops: Optional[List[str]] = None):
    """Builds a RandAugment that modifies bboxes for geometric transforms."""
    augmenter = cls(
        num_layers=num_layers,
        magnitude=magnitude,
        cutout_const=cutout_const,
        translate_const=translate_const,
        magnitude_std=magnitude_std,
        prob_to_apply=prob_to_apply,
        exclude_ops=exclude_ops)
    box_aware_ops_by_base_name = {
        'Rotate': 'Rotate_BBox',
        'ShearX': 'ShearX_BBox',
        'ShearY': 'ShearY_BBox',
        'TranslateX': 'TranslateX_BBox',
        'TranslateY': 'TranslateY_BBox',
    }
    augmenter.available_ops = [
        box_aware_ops_by_base_name.get(op_name) or op_name
        for op_name in augmenter.available_ops
    ]
    return augmenter

  def _distort_common(
      self,
      image: tf.Tensor,
      bboxes: Optional[tf.Tensor] = None
  ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
    """Distorts the image and optionally bounding boxes."""
    input_image_type = image.dtype

    if input_image_type != tf.uint8:
      image = tf.clip_by_value(image, 0.0, 255.0)
      image = tf.cast(image, dtype=tf.uint8)

    replace_value = [128] * 3
    min_prob, max_prob = 0.2, 0.8

    aug_image = image
    aug_bboxes = bboxes

    for _ in range(self.num_layers):
      op_to_select = tf.random.uniform([],
                                       maxval=len(self.available_ops) + 1,
                                       dtype=tf.int32)

      branch_fns = []
      for (i, op_name) in enumerate(self.available_ops):
        prob = tf.random.uniform([],
                                 minval=min_prob,
                                 maxval=max_prob,
                                 dtype=tf.float32)
        func, _, args = _parse_policy_info(op_name, prob, self.magnitude,
                                           replace_value, self.cutout_const,
                                           self.translate_const,
                                           self.magnitude_std)
        branch_fns.append((
            i,
            # pylint:disable=g-long-lambda
            lambda selected_func=func, selected_args=args: selected_func(
                image, bboxes, *selected_args)))
        # pylint:enable=g-long-lambda

      aug_image, aug_bboxes = tf.switch_case(
          branch_index=op_to_select,
          branch_fns=branch_fns,
          default=lambda: (tf.identity(image), _maybe_identity(bboxes)))

      if self.prob_to_apply is not None:
        aug_image, aug_bboxes = tf.cond(
            tf.random.uniform(shape=[], dtype=tf.float32) < self.prob_to_apply,
            lambda: (tf.identity(aug_image), _maybe_identity(aug_bboxes)),
            lambda: (tf.identity(image), _maybe_identity(bboxes)))
      image = aug_image
      bboxes = aug_bboxes

    image = tf.cast(image, dtype=input_image_type)
    return image, bboxes

  def distort(self, image: tf.Tensor) -> tf.Tensor:
    """See base class."""
    image, _ = self._distort_common(image)
    return image

  def distort_with_boxes(self, image: tf.Tensor,
                         bboxes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """See base class."""
    image, bboxes = self._distort_common(image, bboxes)
    return image, bboxes


class RandomErasing(ImageAugment):
  """Applies RandomErasing to a single image.

  Reference: https://arxiv.org/abs/1708.04896

  Implementaion is inspired by https://github.com/rwightman/pytorch-image-models
  """

  def __init__(self,
               probability: float = 0.25,
               min_area: float = 0.02,
               max_area: float = 1 / 3,
               min_aspect: float = 0.3,
               max_aspect=None,
               min_count=1,
               max_count=1,
               trials=10):
    """Applies RandomErasing to a single image.

    Args:
      probability (float, optional): Probability of augmenting the image.
        Defaults to 0.25.
      min_area (float, optional): Minimum area of the random erasing rectangle.
        Defaults to 0.02.
      max_area (float, optional): Maximum area of the random erasing rectangle.
        Defaults to 1/3.
      min_aspect (float, optional): Minimum aspect rate of the random erasing
        rectangle. Defaults to 0.3.
      max_aspect ([type], optional): Maximum aspect rate of the random erasing
        rectangle. Defaults to None.
      min_count (int, optional): Minimum number of erased rectangles. Defaults
        to 1.
      max_count (int, optional):  Maximum number of erased rectangles. Defaults
        to 1.
      trials (int, optional): Maximum number of trials to randomly sample a
        rectangle that fulfills constraint. Defaults to 10.
    """
    self._probability = probability
    self._min_area = float(min_area)
    self._max_area = float(max_area)
    self._min_log_aspect = math.log(min_aspect)
    self._max_log_aspect = math.log(max_aspect or 1 / min_aspect)
    self._min_count = min_count
    self._max_count = max_count
    self._trials = trials

  def distort(self, image: tf.Tensor) -> tf.Tensor:
    """Applies RandomErasing to single `image`.

    Args:
      image (tf.Tensor): Of shape [height, width, 3] representing an image.

    Returns:
      tf.Tensor: The augmented version of `image`.
    """
    uniform_random = tf.random.uniform(shape=[], minval=0., maxval=1.0)
    mirror_cond = tf.less(uniform_random, self._probability)
    image = tf.cond(mirror_cond, lambda: self._erase(image), lambda: image)
    return image

  @tf.function
  def _erase(self, image: tf.Tensor) -> tf.Tensor:
    """Erase an area."""
    if self._min_count == self._max_count:
      count = self._min_count
    else:
      count = tf.random.uniform(
          shape=[],
          minval=int(self._min_count),
          maxval=int(self._max_count - self._min_count + 1),
          dtype=tf.int32)

    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    area = tf.cast(image_width * image_height, tf.float32)

    for _ in range(count):
      # Work around since break is not supported in tf.function
      is_trial_successfull = False
      for _ in range(self._trials):
        if not is_trial_successfull:
          erase_area = tf.random.uniform(
              shape=[],
              minval=area * self._min_area,
              maxval=area * self._max_area)
          aspect_ratio = tf.math.exp(
              tf.random.uniform(
                  shape=[],
                  minval=self._min_log_aspect,
                  maxval=self._max_log_aspect))

          half_height = tf.cast(
              tf.math.round(tf.math.sqrt(erase_area * aspect_ratio) / 2),
              dtype=tf.int32)
          half_width = tf.cast(
              tf.math.round(tf.math.sqrt(erase_area / aspect_ratio) / 2),
              dtype=tf.int32)

          if 2 * half_height < image_height and 2 * half_width < image_width:
            center_height = tf.random.uniform(
                shape=[],
                minval=0,
                maxval=int(image_height - 2 * half_height),
                dtype=tf.int32)
            center_width = tf.random.uniform(
                shape=[],
                minval=0,
                maxval=int(image_width - 2 * half_width),
                dtype=tf.int32)

            image = _fill_rectangle(
                image,
                center_width,
                center_height,
                half_width,
                half_height,
                replace=None)

            is_trial_successfull = True

    return image


class MixupAndCutmix:
  """Applies Mixup and/or Cutmix to a batch of images.

  - Mixup: https://arxiv.org/abs/1710.09412
  - Cutmix: https://arxiv.org/abs/1905.04899

  Implementaion is inspired by https://github.com/rwightman/pytorch-image-models
  """

  def __init__(self,
               mixup_alpha: float = .8,
               cutmix_alpha: float = 1.,
               prob: float = 1.0,
               switch_prob: float = 0.5,
               label_smoothing: float = 0.1,
               num_classes: int = 1001):
    """Applies Mixup and/or Cutmix to a batch of images.

    Args:
      mixup_alpha (float, optional): For drawing a random lambda (`lam`) from a
        beta distribution (for each image). If zero Mixup is deactivated.
        Defaults to .8.
      cutmix_alpha (float, optional): For drawing a random lambda (`lam`) from a
        beta distribution (for each image). If zero Cutmix is deactivated.
        Defaults to 1..
      prob (float, optional): Of augmenting the batch. Defaults to 1.0.
      switch_prob (float, optional): Probability of applying Cutmix for the
        batch. Defaults to 0.5.
      label_smoothing (float, optional): Constant for label smoothing. Defaults
        to 0.1.
      num_classes (int, optional): Number of classes. Defaults to 1001.
    """
    self.mixup_alpha = mixup_alpha
    self.cutmix_alpha = cutmix_alpha
    self.mix_prob = prob
    self.switch_prob = switch_prob
    self.label_smoothing = label_smoothing
    self.num_classes = num_classes
    self.mode = 'batch'
    self.mixup_enabled = True

    if self.mixup_alpha and not self.cutmix_alpha:
      self.switch_prob = -1
    elif not self.mixup_alpha and self.cutmix_alpha:
      self.switch_prob = 1

  def __call__(self, images: tf.Tensor,
               labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return self.distort(images, labels)

  def distort(self, images: tf.Tensor,
              labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies Mixup and/or Cutmix to batch of images and transforms labels.

    Args:
      images (tf.Tensor): Of shape [batch_size,height, width, 3] representing a
        batch of image.
      labels (tf.Tensor): Of shape [batch_size, ] representing the class id for
        each image of the batch.

    Returns:
      Tuple[tf.Tensor, tf.Tensor]: The augmented version of `image` and
        `labels`.
    """
    augment_cond = tf.less(
        tf.random.uniform(shape=[], minval=0., maxval=1.0), self.mix_prob)
    # pylint: disable=g-long-lambda
    augment_a = lambda: self._update_labels(*tf.cond(
        tf.less(
            tf.random.uniform(shape=[], minval=0., maxval=1.0), self.switch_prob
        ), lambda: self._cutmix(images, labels), lambda: self._mixup(
            images, labels)))
    augment_b = lambda: (images, self._smooth_labels(labels))
    # pylint: enable=g-long-lambda

    return tf.cond(augment_cond, augment_a, augment_b)

  @staticmethod
  def _sample_from_beta(alpha, beta, shape):
    sample_alpha = tf.random.gamma(shape, 1., beta=alpha)
    sample_beta = tf.random.gamma(shape, 1., beta=beta)
    return sample_alpha / (sample_alpha + sample_beta)

  def _cutmix(self, images: tf.Tensor,
              labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Apply cutmix."""
    lam = MixupAndCutmix._sample_from_beta(self.cutmix_alpha, self.cutmix_alpha,
                                           labels.shape)

    ratio = tf.math.sqrt(1 - lam)

    batch_size = tf.shape(images)[0]
    image_height, image_width = tf.shape(images)[1], tf.shape(images)[2]

    cut_height = tf.cast(
        ratio * tf.cast(image_height, dtype=tf.float32), dtype=tf.int32)
    cut_width = tf.cast(
        ratio * tf.cast(image_height, dtype=tf.float32), dtype=tf.int32)

    random_center_height = tf.random.uniform(
        shape=[batch_size], minval=0, maxval=image_height, dtype=tf.int32)
    random_center_width = tf.random.uniform(
        shape=[batch_size], minval=0, maxval=image_width, dtype=tf.int32)

    bbox_area = cut_height * cut_width
    lam = 1. - bbox_area / (image_height * image_width)
    lam = tf.cast(lam, dtype=tf.float32)

    images = tf.map_fn(
        lambda x: _fill_rectangle(*x),
        (images, random_center_width, random_center_height, cut_width // 2,
         cut_height // 2, tf.reverse(images, [0])),
        dtype=(tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32),
        fn_output_signature=tf.TensorSpec(images.shape[1:], dtype=tf.float32))

    return images, labels, lam

  def _mixup(self, images: tf.Tensor,
             labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    lam = MixupAndCutmix._sample_from_beta(self.mixup_alpha, self.mixup_alpha,
                                           labels.shape)
    lam = tf.reshape(lam, [-1, 1, 1, 1])
    images = lam * images + (1. - lam) * tf.reverse(images, [0])

    return images, labels, tf.squeeze(lam)

  def _smooth_labels(self, labels: tf.Tensor) -> tf.Tensor:
    off_value = self.label_smoothing / self.num_classes
    on_value = 1. - self.label_smoothing + off_value

    smooth_labels = tf.one_hot(
        labels, self.num_classes, on_value=on_value, off_value=off_value)
    return smooth_labels

  def _update_labels(self, images: tf.Tensor, labels: tf.Tensor,
                     lam: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    labels_1 = self._smooth_labels(labels)
    labels_2 = tf.reverse(labels_1, [0])

    lam = tf.reshape(lam, [-1, 1])
    labels = lam * labels_1 + (1. - lam) * labels_2

    return images, labels
