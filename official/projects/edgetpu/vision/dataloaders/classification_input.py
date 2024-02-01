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

"""Classification decoder and parser."""
# Import libraries
import tensorflow as tf, tf_keras

from official.vision.dataloaders import classification_input
from official.vision.ops import preprocess_ops

MEAN_RGB = (0.5 * 255, 0.5 * 255, 0.5 * 255)
STDDEV_RGB = (0.5 * 255, 0.5 * 255, 0.5 * 255)


def random_crop_image(image,
                      aspect_ratio_range=(0.75, 1.33),
                      area_range=(0.05, 1.0),
                      max_attempts=100):
  """Randomly crop an arbitrary shaped slice from the input image.

  Args:
    image: a Tensor of shape [height, width, 3] representing the input image.
    aspect_ratio_range: a list of floats. The cropped area of the image must
      have an aspect ratio = width / height within this range.
    area_range: a list of floats. The cropped reas of the image must contain
      a fraction of the input image within this range.
    max_attempts: the number of attempts at generating a cropped region of the
      image of the specified constraints. After max_attempts failures, return
      the entire image.

  Returns:
    cropped_image: a Tensor representing the random cropped image. Can be the
      original image if max_attempts is exhausted.
  """
  with tf.name_scope('random_crop_image'):
    crop_offset, crop_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]),
        min_object_covered=0.1,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    cropped_image = tf.slice(image, crop_offset, crop_size)
    return cropped_image


def random_crop_image_v2(image_bytes,
                         image_shape,
                         aspect_ratio_range=(0.75, 1.33),
                         area_range=(0.05, 1.0),
                         max_attempts=100):
  """Randomly crop an arbitrary shaped slice from the input image.

  This is a faster version of `random_crop_image` which takes the original
  image bytes and image size as the inputs, and partially decode the JPEG
  bytes according to the generated crop.

  Args:
    image_bytes: a Tensor of type string representing the raw image bytes.
    image_shape: a Tensor specifying the shape of the raw image.
    aspect_ratio_range: a list of floats. The cropped area of the image must
      have an aspect ratio = width / height within this range.
    area_range: a list of floats. The cropped reas of the image must contain
      a fraction of the input image within this range.
    max_attempts: the number of attempts at generating a cropped region of the
      image of the specified constraints. After max_attempts failures, return
      the entire image.

  Returns:
    cropped_image: a Tensor representing the random cropped image. Can be the
      original image if max_attempts is exhausted.
  """
  with tf.name_scope('random_crop_image_v2'):
    crop_offset, crop_size, _ = tf.image.sample_distorted_bounding_box(
        image_shape,
        tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]),
        min_object_covered=0.1,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    offset_y, offset_x, _ = tf.unstack(crop_offset)
    crop_height, crop_width, _ = tf.unstack(crop_size)
    crop_window = tf.stack([offset_y, offset_x, crop_height, crop_width])
    cropped_image = tf.image.decode_and_crop_jpeg(
        image_bytes, crop_window, channels=3)
    return cropped_image


class Decoder(classification_input.Decoder):
  """A tf.Example decoder for classification task."""
  pass


class Parser(classification_input.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def _parse_train_image(self, decoded_tensors):
    """Parses image data for training."""
    image_bytes = decoded_tensors[self._image_field_key]

    if self._decode_jpeg_only:
      image_shape = tf.image.extract_jpeg_shape(image_bytes)

      # Crops image.
      cropped_image = random_crop_image_v2(
          image_bytes, image_shape)
      image = tf.cond(
          tf.reduce_all(tf.equal(tf.shape(cropped_image), image_shape)),
          lambda: preprocess_ops.center_crop_image_v2(image_bytes, image_shape),
          lambda: cropped_image)
    else:
      # Decodes image.
      image = tf.io.decode_image(image_bytes, channels=3)
      image.set_shape([None, None, 3])

      # Crops image.
      cropped_image = random_crop_image(image)

      image = tf.cond(
          tf.reduce_all(tf.equal(tf.shape(cropped_image), tf.shape(image))),
          lambda: preprocess_ops.center_crop_image(image),
          lambda: cropped_image)

    if self._aug_rand_hflip:
      image = tf.image.random_flip_left_right(image)

    # Resizes image.
    image = tf.image.resize(
        image, self._output_size, method=tf.image.ResizeMethod.BILINEAR)

    # Apply autoaug or randaug.
    if self._augmenter is not None:
      image = self._augmenter.distort(image)

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image,
                                           offset=MEAN_RGB,
                                           scale=STDDEV_RGB)

    # Convert image to self._dtype.
    image = tf.image.convert_image_dtype(image, self._dtype)

    return image

  def _parse_eval_image(self, decoded_tensors):
    """Parses image data for evaluation."""
    image_bytes = decoded_tensors[self._image_field_key]

    if self._decode_jpeg_only:
      image_shape = tf.image.extract_jpeg_shape(image_bytes)

      # Center crops.
      image = preprocess_ops.center_crop_image_v2(image_bytes, image_shape)
    else:
      # Decodes image.
      image = tf.io.decode_image(image_bytes, channels=3)
      image.set_shape([None, None, 3])

      # Center crops.
      image = preprocess_ops.center_crop_image(image)

    image = tf.image.resize(
        image, self._output_size, method=tf.image.ResizeMethod.BILINEAR)

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image,
                                           offset=MEAN_RGB,
                                           scale=STDDEV_RGB)

    # Convert image to self._dtype.
    image = tf.image.convert_image_dtype(image, self._dtype)

    return image
