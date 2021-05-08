# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Data parser and processing for segmentation datasets."""

import tensorflow as tf
from official.vision.beta.dataloaders import decoder
from official.vision.beta.dataloaders import parser
from official.vision.beta.ops import preprocess_ops



class Decoder(decoder.Decoder):
  """A tf.Example decoder for basnet task."""

  def __init__(self):
    self._keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/height': tf.io.FixedLenFeature((), tf.int64, default_value=0),
        'image/width': tf.io.FixedLenFeature((), tf.int64, default_value=0),
        'image/segmentation/class/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value='')
    }

  def decode(self, serialized_example):
    return tf.io.parse_single_example(
        serialized_example, self._keys_to_features)


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               output_size,
               aug_rand_hflip=False,
               dtype='float32'):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      aug_rand_hflip: `bool`, if True, augment training with random
        horizontal flip.
      dtype: `str`, data type. One of {`bfloat16`, `float32`, `float16`}.
    """
    self._output_size = output_size

    # Data augmentation.
    self._aug_rand_hflip = aug_rand_hflip

    # dtype.
    self._dtype = dtype

  def _prepare_image_and_label(self, data):
    """Prepare normalized image and label."""
    image = tf.io.decode_image(data['image/encoded'], channels=3)
    label = tf.io.decode_image(data['image/segmentation/class/encoded'],
                               channels=1)
    height = data['image/height']
    width = data['image/width']
    image = tf.reshape(image, (height, width, 3))
    
    label = tf.reshape(label, (height, width, 1))
    label = tf.image.convert_image_dtype(label, dtype=tf.float32)

    image = preprocess_ops.normalize_image(image)

    return image, label

  def _parse_train_data(self, data):
    """Parses data for training and evaluation."""
    image, label = self._prepare_image_and_label(data)

    # Flips image randomly during training.
    if self._aug_rand_hflip:
      image, _, label = preprocess_ops.random_horizontal_flip(image, masks=label)



    image = tf.image.resize(image, tf.cast([256, 256], tf.int32))
    label = tf.image.resize(label, tf.cast([256, 256], tf.int32))

    # Random crop both image and mask on training.
    image_mask = tf.concat([image, label], 2)
    image_mask_crop = tf.image.random_crop(image_mask,
                                           self._output_size + [4])
    image = image_mask_crop[:, :, :-1]
    label = image_mask_crop[:, :,-1]


    # Cast image as self._dtype
    image = tf.cast(image, self._dtype)

    return image, label

  def _parse_eval_data(self, data):
    """Parses data for training and evaluation."""
    image, label = self._prepare_image_and_label(data)

    image = tf.image.resize(image, tf.cast([256, 256], tf.int32))
    label = tf.image.resize(label, tf.cast([256, 256], tf.int32))

    # Cast image as self._dtype
    image = tf.cast(image, self._dtype)

    return image, label
