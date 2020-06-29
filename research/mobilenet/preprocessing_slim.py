# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Provides utilities to preprocess images for the Inception networks.
Adapted from research/slim/preprocessing/inception_preprocessing.py
Modification is made to ensure compatibility of TF2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, Tuple, Optional

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops


def _resize_image(image_bytes: tf.Tensor,
                  height: int,
                  width: int) -> tf.Tensor:
  """Resizes an image to a given height and width.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    height: image height dimension.
    width: image width dimension.

  Returns:
    A tensor containing the resized image.

  """
  return tf.compat.v1.image.resize(
    image_bytes, [height, width], method=tf.image.ResizeMethod.BILINEAR,
    align_corners=False)


def apply_with_random_selector(x: tf.Tensor,
                               func: Callable,
                               num_cases: int) -> tf.Tensor:
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
    func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
    for case in range(num_cases)])[0]


def distort_color(image: tf.Tensor,
                  color_ordering: int = 0,
                  fast_mode: bool = True) -> tf.Tensor:
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  if fast_mode:
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    else:
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
  else:
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 2:
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif color_ordering == 3:
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
  return tf.clip_by_value(image, 0.0, 1.0)


def distorted_bounding_box_crop(
    image: tf.Tensor,
    bbox: tf.Tensor,
    min_object_covered: float = 0.1,
    aspect_ratio_range: Tuple[float, float] = (0.75, 1.33),
    area_range: Tuple[float, float] = (0.05, 1.0),
    max_attempts: int = 100) -> Tuple[tf.Tensor, tf.Tensor]:
  """Generates cropped_image using a one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
      image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
      area of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `floats`. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `floats`. The cropped area of the image
      must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
  Returns:
    A tuple, a 3-D Tensor cropped_image and the distorted bbox
  """
  # Each bounding box has shape [1, num_boxes, box coords] and
  # the coordinates are ordered [ymin, xmin, ymax, xmax].

  # A large fraction of image datasets contain a human-annotated bounding
  # box delineating the region of the image containing the object of interest.
  # We choose to create a new bounding box for the object which is a randomly
  # distorted version of the human-annotated bounding box that obeys an
  # allowed range of aspect ratios, sizes and overlap with the human-annotated
  # bounding box. If no box is supplied, then we assume the bounding box is
  # the entire image.
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
    tf.shape(image),
    bounding_boxes=bbox,
    min_object_covered=min_object_covered,
    aspect_ratio_range=aspect_ratio_range,
    area_range=area_range,
    max_attempts=max_attempts,
    use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

  # Crop the image to the specified bounding box.
  cropped_image = tf.slice(image, bbox_begin, bbox_size)
  return cropped_image, distort_bbox


def preprocess_for_train(image: tf.Tensor,
                         image_size: int,
                         bbox: Optional[tf.Tensor] = None,
                         fast_mode: bool = True,
                         add_image_summaries: bool = True,
                         random_crop: bool = True,
                         use_grayscale: bool = False) -> tf.Tensor:
  """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Additionally it would create image_summaries to display the different
  transformations applied to the image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    image_size: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    fast_mode: Optional boolean, if True avoids slower transformations (i.e.
      bi-cubic resizing, random_hue or random_contrast).
    add_image_summaries: Enable image summaries.
    random_crop: Enable random cropping of images during preprocessing for
      training.
    use_grayscale: Whether to convert the image from RGB to grayscale.
  Returns:
    3-D float Tensor of distorted image used for training with range [-1, 1].
  """
  if bbox is None:
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                       dtype=tf.float32,
                       shape=[1, 1, 4])
  if image.dtype == tf.string:
    image = tf.image.decode_jpeg(image, channels=3)
  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  # Each bounding box has shape [1, num_boxes, box coords] and
  # the coordinates are ordered [ymin, xmin, ymax, xmax].
  image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                bbox,
                                                colors=None)
  if add_image_summaries:
    tf.summary.image('image_with_bounding_boxes', image_with_box)

  if not random_crop:
    distorted_image = image
  else:
    distorted_image, distorted_bbox = distorted_bounding_box_crop(image, bbox)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([None, None, 3])
    image_with_distorted_box = tf.image.draw_bounding_boxes(
      tf.expand_dims(image, 0), distorted_bbox, colors=None)
    if add_image_summaries:
      tf.summary.image('images_with_distorted_bounding_box',
                       image_with_distorted_box)

  # This resizing operation may distort the images because the aspect
  # ratio is not respected. We select a resize method in a round robin
  # fashion based on the thread number.
  # Note that ResizeMethod contains 4 enumerated resizing methods.

  # We select only 1 case for fast_mode bilinear.
  num_resize_cases = 1 if fast_mode else 4
  distorted_image = apply_with_random_selector(
    distorted_image,
    lambda x, method: tf.compat.v1.image.resize(x, [image_size, image_size],
                                                method),
    num_cases=num_resize_cases)

  if add_image_summaries:
    tf.summary.image(('cropped_' if random_crop else '') + 'resized_image',
                     tf.expand_dims(distorted_image, 0))

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Randomly distort the colors. There are 1 or 4 ways to do it.
  num_distort_cases = 1 if fast_mode else 4
  distorted_image = apply_with_random_selector(
    distorted_image,
    lambda x, ordering: distort_color(x, ordering, fast_mode),
    num_cases=num_distort_cases)

  if use_grayscale:
    distorted_image = tf.image.rgb_to_grayscale(distorted_image)

  if add_image_summaries:
    tf.summary.image('final_distorted_image',
                     tf.expand_dims(distorted_image, 0))
  distorted_image = tf.subtract(distorted_image, 0.5)
  distorted_image = tf.multiply(distorted_image, 2.0)
  return distorted_image


def preprocess_for_eval(image: tf.Tensor,
                        image_size: int,
                        central_fraction: float = 0.875,
                        central_crop: bool = True,
                        use_grayscale: bool = False):
  """Prepare one image for evaluation.

  If height and width are specified it would output an image with that size by
  applying resize_bilinear.

  If central_fraction is specified it would crop the central fraction of the
  input image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    image_size: integer
    central_fraction: Optional Float, fraction of the image to crop.
    central_crop: Enable central cropping of images during preprocessing for
      evaluation.
    use_grayscale: Whether to convert the image from RGB to grayscale.
  Returns:
    3-D float Tensor of prepared image.
  """
  if image.dtype == tf.string:
    image = tf.image.decode_jpeg(image, channels=3)
  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  if use_grayscale:
    image = tf.image.rgb_to_grayscale(image)
  # Crop the central region of the image with an area containing 87.5% of
  # the original image.
  if central_crop and central_fraction:
    image = tf.image.central_crop(image, central_fraction=central_fraction)

  image = _resize_image(image_bytes=image,
                        height=image_size,
                        width=image_size)
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image
