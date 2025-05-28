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

"""Preprocessing ops imported from OD API."""

import functools

import tensorflow as tf, tf_keras

from official.projects.centernet.ops import box_list
from official.projects.centernet.ops import box_list_ops


def _get_or_create_preprocess_rand_vars(generator_func,
                                        function_id,
                                        preprocess_vars_cache,
                                        key=''):
  """Returns a tensor stored in preprocess_vars_cache or using generator_func.

  If the tensor was previously generated and appears in the PreprocessorCache,
  the previously generated tensor will be returned. Otherwise, a new tensor
  is generated using generator_func and stored in the cache.

  Args:
    generator_func: A 0-argument function that generates a tensor.
    function_id: identifier for the preprocessing function used.
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.
    key: identifier for the variable stored.

  Returns:
    The generated tensor.
  """
  if preprocess_vars_cache is not None:
    var = preprocess_vars_cache.get(function_id, key)
    if var is None:
      var = generator_func()
      preprocess_vars_cache.update(function_id, key, var)
  else:
    var = generator_func()
  return var


def _random_integer(minval, maxval, seed):
  """Returns a random 0-D tensor between minval and maxval.

  Args:
    minval: minimum value of the random tensor.
    maxval: maximum value of the random tensor.
    seed: random seed.

  Returns:
    A random 0-D tensor between minval and maxval.
  """
  return tf.random.uniform(
      [], minval=minval, maxval=maxval, dtype=tf.int32, seed=seed)


def _get_crop_border(border, size):
  """Get the border of cropping."""

  border = tf.cast(border, tf.float32)
  size = tf.cast(size, tf.float32)

  i = tf.math.ceil(tf.math.log(2.0 * border / size) / tf.math.log(2.0))
  divisor = tf.pow(2.0, i)
  divisor = tf.clip_by_value(divisor, 1, border)
  divisor = tf.cast(divisor, tf.int32)

  return tf.cast(border, tf.int32) // divisor


def random_square_crop_by_scale(image,
                                boxes,
                                labels,
                                max_border=128,
                                scale_min=0.6,
                                scale_max=1.3,
                                num_scales=8,
                                seed=None,
                                preprocess_vars_cache=None):
  """Randomly crop a square in proportion to scale and image size.

   Extract a square sized crop from an image whose side length is sampled by
   randomly scaling the maximum spatial dimension of the image. If part of
   the crop falls outside the image, it is filled with zeros.
   The augmentation is borrowed from [1]
   [1]: https://arxiv.org/abs/1904.07850

  Args:
    image: rank 3 float32 tensor containing 1 image ->
           [height, width, channels].
    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1]. Each row is in the form of [ymin, xmin, ymax, xmax].
           Boxes on the crop boundary are clipped to the boundary and boxes
           falling outside the crop are ignored.
    labels: rank 1 int32 tensor containing the object classes.
    max_border: The maximum size of the border. The border defines distance in
      pixels to the image boundaries that will not be considered as a center of
      a crop. To make sure that the border does not go over the center of the
      image, we chose the border value by computing the minimum k, such that
      (max_border / (2**k)) < image_dimension/2.
    scale_min: float, the minimum value for scale.
    scale_max: float, the maximum value for scale.
    num_scales: int, the number of discrete scale values to sample between
      [scale_min, scale_max]
    seed: random seed.
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.


  Returns:
    image: image which is the same rank as input image.
    boxes: boxes which is the same rank as input boxes.
           Boxes are in normalized form.
    labels: new labels.

  """

  img_shape = tf.shape(image)
  height, width = img_shape[0], img_shape[1]
  scales = tf.linspace(scale_min, scale_max, num_scales)

  scale = _get_or_create_preprocess_rand_vars(
      lambda: scales[_random_integer(0, num_scales, seed)],
      'square_crop_scale',
      preprocess_vars_cache, 'scale')

  image_size = scale * tf.cast(tf.maximum(height, width), tf.float32)
  image_size = tf.cast(image_size, tf.int32)
  h_border = _get_crop_border(max_border, height)
  w_border = _get_crop_border(max_border, width)

  def y_function():
    y = _random_integer(h_border,
                        tf.cast(height, tf.int32) - h_border + 1,
                        seed)
    return y

  def x_function():
    x = _random_integer(w_border,
                        tf.cast(width, tf.int32) - w_border + 1,
                        seed)
    return x

  y_center = _get_or_create_preprocess_rand_vars(
      y_function,
      'square_crop_scale',
      preprocess_vars_cache, 'y_center')

  x_center = _get_or_create_preprocess_rand_vars(
      x_function,
      'square_crop_scale',
      preprocess_vars_cache, 'x_center')

  half_size = tf.cast(image_size / 2, tf.int32)
  crop_ymin, crop_ymax = y_center - half_size, y_center + half_size
  crop_xmin, crop_xmax = x_center - half_size, x_center + half_size

  ymin = tf.maximum(crop_ymin, 0)
  xmin = tf.maximum(crop_xmin, 0)
  ymax = tf.minimum(crop_ymax, height - 1)
  xmax = tf.minimum(crop_xmax, width - 1)

  cropped_image = image[ymin:ymax, xmin:xmax]
  offset_y = tf.maximum(0, ymin - crop_ymin)
  offset_x = tf.maximum(0, xmin - crop_xmin)

  oy_i = offset_y
  ox_i = offset_x

  output_image = tf.image.pad_to_bounding_box(
      cropped_image, offset_height=oy_i, offset_width=ox_i,
      target_height=image_size, target_width=image_size)

  if ymin == 0:
    # We might be padding the image.
    box_ymin = -offset_y
  else:
    box_ymin = crop_ymin

  if xmin == 0:
    # We might be padding the image.
    box_xmin = -offset_x
  else:
    box_xmin = crop_xmin

  box_ymax = box_ymin + image_size
  box_xmax = box_xmin + image_size

  image_box = [box_ymin / height, box_xmin / width,
               box_ymax / height, box_xmax / width]
  boxlist = box_list.BoxList(boxes)
  boxlist = box_list_ops.change_coordinate_frame(boxlist, image_box)
  boxlist, indices = box_list_ops.prune_completely_outside_window(
      boxlist, [0.0, 0.0, 1.0, 1.0])
  boxlist = box_list_ops.clip_to_window(boxlist, [0.0, 0.0, 1.0, 1.0],
                                        filter_nonoverlapping=False)

  return_values = [output_image,
                   boxlist.get(),
                   tf.gather(labels, indices)]

  return return_values


def resize_to_range(image,
                    masks=None,
                    min_dimension=None,
                    max_dimension=None,
                    method=tf.image.ResizeMethod.BILINEAR,
                    pad_to_max_dimension=False,
                    per_channel_pad_value=(0, 0, 0)):
  """Resizes an image so its dimensions are within the provided value.

  The output size can be described by two cases:
  1. If the image can be rescaled so its minimum dimension is equal to the
     provided value without the other dimension exceeding max_dimension,
     then do so.
  2. Otherwise, resize so the largest dimension is equal to max_dimension.

  Args:
    image: A 3D tensor of shape [height, width, channels]
    masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width] containing instance masks.
    min_dimension: (optional) (scalar) desired size of the smaller image
                   dimension.
    max_dimension: (optional) (scalar) maximum allowed size
                   of the larger image dimension.
    method: (optional) interpolation method used in resizing. Defaults to
            BILINEAR.
    pad_to_max_dimension: Whether to resize the image and pad it with zeros
      so the resulting image is of the spatial size
      [max_dimension, max_dimension]. If masks are included they are padded
      similarly.
    per_channel_pad_value: A tuple of per-channel scalar value to use for
      padding. By default pads zeros.

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

  def _resize_landscape_image(image):
    # resize a landscape image
    return tf.image.resize(
        image, tf.stack([min_dimension, max_dimension]), method=method,
        preserve_aspect_ratio=True)

  def _resize_portrait_image(image):
    # resize a portrait image
    return tf.image.resize(
        image, tf.stack([max_dimension, min_dimension]), method=method,
        preserve_aspect_ratio=True)

  with tf.name_scope('ResizeToRange'):
    if image.get_shape().is_fully_defined():
      if image.get_shape()[0] < image.get_shape()[1]:
        new_image = _resize_landscape_image(image)
      else:
        new_image = _resize_portrait_image(image)
      new_size = tf.constant(new_image.get_shape().as_list())
    else:
      new_image = tf.cond(
          tf.less(tf.shape(image)[0], tf.shape(image)[1]),
          lambda: _resize_landscape_image(image),
          lambda: _resize_portrait_image(image))
      new_size = tf.shape(new_image)

    if pad_to_max_dimension:
      channels = tf.unstack(new_image, axis=2)
      if len(channels) != len(per_channel_pad_value):
        raise ValueError('Number of channels must be equal to the length of '
                         'per-channel pad value.')
      new_image = tf.stack(
          [
              tf.pad(  # pylint: disable=g-complex-comprehension
                  channels[i], [[0, max_dimension - new_size[0]],
                                [0, max_dimension - new_size[1]]],
                  constant_values=per_channel_pad_value[i])
              for i in range(len(channels))
          ],
          axis=2)
      new_image.set_shape([max_dimension, max_dimension, len(channels)])

    result = [new_image, new_size]
    if masks is not None:
      new_masks = tf.expand_dims(masks, 3)
      new_masks = tf.image.resize(
          new_masks,
          new_size[:-1],
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      if pad_to_max_dimension:
        new_masks = tf.image.pad_to_bounding_box(
            new_masks, 0, 0, max_dimension, max_dimension)
      new_masks = tf.squeeze(new_masks, 3)
      result.append(new_masks)

    return result


def _augment_only_rgb_channels(image, augment_function):
  """Augments only the RGB slice of an image with additional channels."""
  rgb_slice = image[:, :, :3]
  augmented_rgb_slice = augment_function(rgb_slice)
  image = tf.concat([augmented_rgb_slice, image[:, :, 3:]], -1)
  return image


def random_adjust_brightness(image,
                             max_delta=0.2,
                             seed=None,
                             preprocess_vars_cache=None):
  """Randomly adjusts brightness.

  Makes sure the output image is still between 0 and 255.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    max_delta: how much to change the brightness. A value between [0, 1).
    seed: random seed.
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustBrightness'):
    generator_func = functools.partial(tf.random.uniform, [],
                                       -max_delta, max_delta, seed=seed)
    delta = _get_or_create_preprocess_rand_vars(
        generator_func,
        'adjust_brightness',
        preprocess_vars_cache)

    def _adjust_brightness(image):
      image = tf.image.adjust_brightness(image / 255, delta) * 255
      image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
      return image

    image = _augment_only_rgb_channels(image, _adjust_brightness)
    return image


def random_adjust_contrast(image,
                           min_delta=0.8,
                           max_delta=1.25,
                           seed=None,
                           preprocess_vars_cache=None):
  """Randomly adjusts contrast.

  Makes sure the output image is still between 0 and 255.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    min_delta: see max_delta.
    max_delta: how much to change the contrast. Contrast will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current contrast of the image.
    seed: random seed.
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustContrast'):
    generator_func = functools.partial(tf.random.uniform, [],
                                       min_delta, max_delta, seed=seed)
    contrast_factor = _get_or_create_preprocess_rand_vars(
        generator_func,
        'adjust_contrast',
        preprocess_vars_cache)

    def _adjust_contrast(image):
      image = tf.image.adjust_contrast(image / 255, contrast_factor) * 255
      image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
      return image

    image = _augment_only_rgb_channels(image, _adjust_contrast)
    return image


def random_adjust_hue(image,
                      max_delta=0.02,
                      seed=None,
                      preprocess_vars_cache=None):
  """Randomly adjusts hue.

  Makes sure the output image is still between 0 and 255.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    max_delta: change hue randomly with a value between 0 and max_delta.
    seed: random seed.
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustHue'):
    generator_func = functools.partial(tf.random.uniform, [],
                                       -max_delta, max_delta, seed=seed)
    delta = _get_or_create_preprocess_rand_vars(
        generator_func,
        'adjust_hue',
        preprocess_vars_cache)

    def _adjust_hue(image):
      image = tf.image.adjust_hue(image / 255, delta) * 255
      image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
      return image

    image = _augment_only_rgb_channels(image, _adjust_hue)
    return image


def random_adjust_saturation(image,
                             min_delta=0.8,
                             max_delta=1.25,
                             seed=None,
                             preprocess_vars_cache=None):
  """Randomly adjusts saturation.

  Makes sure the output image is still between 0 and 255.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    min_delta: see max_delta.
    max_delta: how much to change the saturation. Saturation will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current saturation of the image.
    seed: random seed.
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustSaturation'):
    generator_func = functools.partial(tf.random.uniform, [],
                                       min_delta, max_delta, seed=seed)
    saturation_factor = _get_or_create_preprocess_rand_vars(
        generator_func,
        'adjust_saturation',
        preprocess_vars_cache)

    def _adjust_saturation(image):
      image = tf.image.adjust_saturation(image / 255, saturation_factor) * 255
      image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
      return image

    image = _augment_only_rgb_channels(image, _adjust_saturation)
    return image
