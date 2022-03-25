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

"""Classification decoder and parser."""
import tensorflow as tf
from official.vision.dataloaders import classification_input
from official.vision.ops import preprocess_ops


class Parser(classification_input.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def _parse_train_image(self, decoded_tensors):
    """Parses image data for training."""
    image_bytes = decoded_tensors[self._image_field_key]

    if self._decode_jpeg_only:
      image_shape = tf.image.extract_jpeg_shape(image_bytes)

      # Crops image.
      cropped_image = preprocess_ops.random_crop_image_v2(
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
      cropped_image = preprocess_ops.random_crop_image(image)

      image = tf.cond(
          tf.reduce_all(tf.equal(tf.shape(cropped_image), tf.shape(image))),
          lambda: preprocess_ops.center_crop_image(image),
          lambda: cropped_image)

    if self._aug_rand_hflip:
      image = tf.image.random_flip_left_right(image)

    # Resizes image.
    image = tf.image.resize(
        image, self._output_size, method=tf.image.ResizeMethod.BILINEAR)
    image.set_shape([self._output_size[0], self._output_size[1], 3])

    # Apply autoaug or randaug.
    if self._augmenter is not None:
      image = self._augmenter.distort(image)

    # Convert image to self._dtype.
    image = tf.image.convert_image_dtype(image, self._dtype)
    image = image / 255.0
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
    image.set_shape([self._output_size[0], self._output_size[1], 3])

    # Convert image to self._dtype.
    image = tf.image.convert_image_dtype(image, self._dtype)
    image = image / 255.0
    return image
