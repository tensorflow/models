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

"""Preprocessing functions for images."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import tensorflow as tf
from typing import List, Optional, Text, Tuple

from official.vision.image_classification import augment


# Calculated from the ImageNet training set
MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)

IMAGE_SIZE = 224
CROP_PADDING = 32


def mean_image_subtraction(
    image_bytes: tf.Tensor,
    means: Tuple[float, ...],
    num_channels: int = 3,
    dtype: tf.dtypes.DType = tf.float32,
) ->  tf.Tensor:
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image_bytes = mean_image_subtraction(image_bytes, means)

  Note that the rank of `image` must be known.

  Args:
    image_bytes: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
    num_channels: number of color channels in the image that will be distorted.
    dtype: the dtype to convert the images to. Set to `None` to skip conversion.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image_bytes.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')

  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  # We have a 1-D tensor of means; convert to 3-D.
  # Note(b/130245863): we explicitly call `broadcast` instead of simply
  # expanding dimensions for better performance.
  means = tf.broadcast_to(means, tf.shape(image_bytes))
  if dtype is not None:
    means = tf.cast(means, dtype=dtype)

  return image_bytes - means


def standardize_image(
    image_bytes: tf.Tensor,
    stddev: Tuple[float, ...],
    num_channels: int = 3,
    dtype: tf.dtypes.DType = tf.float32,
) ->  tf.Tensor:
  """Divides the given stddev from each image channel.

  For example:
    stddev = [123.68, 116.779, 103.939]
    image_bytes = standardize_image(image_bytes, stddev)

  Note that the rank of `image` must be known.

  Args:
    image_bytes: a tensor of size [height, width, C].
    stddev: a C-vector of values to divide from each channel.
    num_channels: number of color channels in the image that will be distorted.
    dtype: the dtype to convert the images to. Set to `None` to skip conversion.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `stddev`.
  """
  if image_bytes.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')

  if len(stddev) != num_channels:
    raise ValueError('len(stddev) must match the number of channels')

  # We have a 1-D tensor of stddev; convert to 3-D.
  # Note(b/130245863): we explicitly call `broadcast` instead of simply
  # expanding dimensions for better performance.
  stddev = tf.broadcast_to(stddev, tf.shape(image_bytes))
  if dtype is not None:
    stddev = tf.cast(stddev, dtype=dtype)

  return image_bytes / stddev


def normalize_images(features: tf.Tensor,
                     mean_rgb: Tuple[float, ...] = MEAN_RGB,
                     stddev_rgb: Tuple[float, ...] = STDDEV_RGB,
                     num_channels: int = 3,
                     dtype: tf.dtypes.DType = tf.float32,
                     data_format: Text = 'channels_last') -> tf.Tensor:
  """Normalizes the input image channels with the given mean and stddev.

  Args:
    features: `Tensor` representing decoded images in float format.
    mean_rgb: the mean of the channels to subtract.
    stddev_rgb: the stddev of the channels to divide.
    num_channels: the number of channels in the input image tensor.
    dtype: the dtype to convert the images to. Set to `None` to skip conversion.
    data_format: the format of the input image tensor
                 ['channels_first', 'channels_last'].

  Returns:
    A normalized image `Tensor`.
  """
  # TODO(allencwang) - figure out how to use mean_image_subtraction and
  # standardize_image on batches of images and replace the following.
  if data_format == 'channels_first':
    stats_shape = [num_channels, 1, 1]
  else:
    stats_shape = [1, 1, num_channels]

  if dtype is not None:
    features = tf.image.convert_image_dtype(features, dtype=dtype)

  if mean_rgb is not None:
    mean_rgb = tf.constant(mean_rgb,
                           shape=stats_shape,
                           dtype=features.dtype)
    mean_rgb = tf.broadcast_to(mean_rgb, tf.shape(features))
    features = features - mean_rgb

  if stddev_rgb is not None:
    stddev_rgb = tf.constant(stddev_rgb,
                             shape=stats_shape,
                             dtype=features.dtype)
    stddev_rgb = tf.broadcast_to(stddev_rgb, tf.shape(features))
    features = features / stddev_rgb

  return features


def decode_and_center_crop(image_bytes: tf.Tensor,
                           image_size: int = IMAGE_SIZE,
                           crop_padding: int = CROP_PADDING) -> tf.Tensor:
  """Crops to center of image with padding then scales image_size.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    image_size: image height/width dimension.
    crop_padding: the padding size to use when centering the crop.

  Returns:
    A decoded and cropped image `Tensor`.
  """
  decoded = image_bytes.dtype != tf.string
  shape = (tf.shape(image_bytes) if decoded
           else tf.image.extract_jpeg_shape(image_bytes))
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + crop_padding)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_size, padded_center_crop_size])
  if decoded:
    image = tf.image.crop_to_bounding_box(
        image_bytes,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=padded_center_crop_size,
        target_width=padded_center_crop_size)
  else:
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

  image = resize_image(image_bytes=image,
                       height=image_size,
                       width=image_size)

  return image


def decode_crop_and_flip(image_bytes: tf.Tensor) -> tf.Tensor:
  """Crops an image to a random part of the image, then randomly flips.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.

  Returns:
    A decoded and cropped image `Tensor`.

  """
  decoded = image_bytes.dtype != tf.string
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  shape = (tf.shape(image_bytes) if decoded
           else tf.image.extract_jpeg_shape(image_bytes))
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      shape,
      bounding_boxes=bbox,
      min_object_covered=0.1,
      aspect_ratio_range=[0.75, 1.33],
      area_range=[0.05, 1.0],
      max_attempts=100,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box

  # Reassemble the bounding box in the format the crop op requires.
  offset_height, offset_width, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_height, offset_width,
                          target_height, target_width])
  if decoded:
    cropped = tf.image.crop_to_bounding_box(
        image_bytes,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=target_height,
        target_width=target_width)
  else:
    cropped = tf.image.decode_and_crop_jpeg(image_bytes,
                                            crop_window,
                                            channels=3)

  # Flip to add a little more random distortion in.
  cropped = tf.image.random_flip_left_right(cropped)
  return cropped


def resize_image(image_bytes: tf.Tensor,
                 height: int = IMAGE_SIZE,
                 width: int = IMAGE_SIZE) -> tf.Tensor:
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


def preprocess_for_eval(
    image_bytes: tf.Tensor,
    image_size: int = IMAGE_SIZE,
    num_channels: int = 3,
    mean_subtract: bool = False,
    standardize: bool = False,
    dtype: tf.dtypes.DType = tf.float32
) -> tf.Tensor:
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    image_size: image height/width dimension.
    num_channels: number of image input channels.
    mean_subtract: whether or not to apply mean subtraction.
    standardize: whether or not to apply standardization.
    dtype: the dtype to convert the images to. Set to `None` to skip conversion.

  Returns:
    A preprocessed and normalized image `Tensor`.
  """
  images = decode_and_center_crop(image_bytes, image_size)
  images = tf.reshape(images, [image_size, image_size, num_channels])

  if mean_subtract:
    images = mean_image_subtraction(image_bytes=images, means=MEAN_RGB)
  if standardize:
    images = standardize_image(image_bytes=images, stddev=STDDEV_RGB)
  if dtype is not None:
    images = tf.image.convert_image_dtype(images, dtype=dtype)

  return images


def load_eval_image(filename: Text, image_size: int = IMAGE_SIZE) -> tf.Tensor:
  """Reads an image from the filesystem and applies image preprocessing.

  Args:
    filename: a filename path of an image.
    image_size: image height/width dimension.

  Returns:
    A preprocessed and normalized image `Tensor`.
  """
  image_bytes = tf.io.read_file(filename)
  image = preprocess_for_eval(image_bytes, image_size)

  return image


def build_eval_dataset(filenames: List[Text],
                       labels: List[int] = None,
                       image_size: int = IMAGE_SIZE,
                       batch_size: int = 1) -> tf.Tensor:
  """Builds a tf.data.Dataset from a list of filenames and labels.

  Args:
    filenames: a list of filename paths of images.
    labels: a list of labels corresponding to each image.
    image_size: image height/width dimension.
    batch_size: the batch size used by the dataset

  Returns:
    A preprocessed and normalized image `Tensor`.
  """
  if labels is None:
    labels = [0] * len(filenames)

  filenames = tf.constant(filenames)
  labels = tf.constant(labels)
  dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

  dataset = dataset.map(
      lambda filename, label: (load_eval_image(filename, image_size), label))
  dataset = dataset.batch(batch_size)

  return dataset


def preprocess_for_train(image_bytes: tf.Tensor,
                         image_size: int = IMAGE_SIZE,
                         augmenter: Optional[augment.ImageAugment] = None,
                         mean_subtract: bool = False,
                         standardize: bool = False,
                         dtype: tf.dtypes.DType = tf.float32) -> tf.Tensor:
  """Preprocesses the given image for training.

  Args:
    image_bytes: `Tensor` representing an image binary of
      arbitrary size of dtype tf.uint8.
    image_size: image height/width dimension.
    augmenter: the image augmenter to apply.
    mean_subtract: whether or not to apply mean subtraction.
    standardize: whether or not to apply standardization.
    dtype: the dtype to convert the images to. Set to `None` to skip conversion.

  Returns:
    A preprocessed and normalized image `Tensor`.
  """
  images = decode_crop_and_flip(image_bytes=image_bytes)
  images = resize_image(images, height=image_size, width=image_size)
  if augmenter is not None:
    images = augmenter.distort(images)
  if mean_subtract:
    images = mean_image_subtraction(image_bytes=images, means=MEAN_RGB)
  if standardize:
    images = standardize_image(image_bytes=images, stddev=STDDEV_RGB)
  if dtype is not None:
    images = tf.image.convert_image_dtype(images, dtype)

  return images
