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

"""Data parser and processing for SimCLR.

For pre-training:
- Preprocessing:
  -> random cropping
  -> resize back to the original size
  -> random color distortions
  -> random Gaussian blur (sequential)
- Each image need to be processed randomly twice

```snippets
      if train_mode == 'pretrain':
        xs = []
        for _ in range(2):  # Two transformations
          xs.append(preprocess_fn_pretrain(image))
        image = tf.concat(xs, -1)
      else:
        image = preprocess_fn_finetune(image)
```

For fine-tuning:
typical image classification input
"""

from typing import List

import tensorflow as tf

from official.projects.simclr.dataloaders import preprocess_ops as simclr_preprocess_ops
from official.projects.simclr.modeling import simclr_model
from official.vision.dataloaders import decoder
from official.vision.dataloaders import parser
from official.vision.ops import preprocess_ops


class Decoder(decoder.Decoder):
  """A tf.Example decoder for classification task."""

  def __init__(self, decode_label=True):
    self._decode_label = decode_label

    self._keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
    }
    if self._decode_label:
      self._keys_to_features.update({
          'image/class/label': (
              tf.io.FixedLenFeature((), tf.int64, default_value=-1))
      })

  def decode(self, serialized_example):
    return tf.io.parse_single_example(
        serialized_example, self._keys_to_features)


class TFDSDecoder(decoder.Decoder):
  """A TFDS decoder for classification task."""

  def __init__(self, decode_label=True):
    self._decode_label = decode_label

  def decode(self, serialized_example):
    sample_dict = {
        'image/encoded': tf.io.encode_jpeg(
            serialized_example['image'], quality=100),
    }
    if self._decode_label:
      sample_dict.update({
          'image/class/label': serialized_example['label'],
      })
    return sample_dict


class Parser(parser.Parser):
  """Parser for SimCLR training."""

  def __init__(self,
               output_size: List[int],
               aug_rand_crop: bool = True,
               aug_rand_hflip: bool = True,
               aug_color_distort: bool = True,
               aug_color_jitter_strength: float = 1.0,
               aug_color_jitter_impl: str = 'simclrv2',
               aug_rand_blur: bool = True,
               parse_label: bool = True,
               test_crop: bool = True,
               mode: str = simclr_model.PRETRAIN,
               dtype: str = 'float32'):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      aug_rand_crop: `bool`, if Ture, augment training with random cropping.
      aug_rand_hflip: `bool`, if True, augment training with random
        horizontal flip.
      aug_color_distort: `bool`, if True augment training with color distortion.
      aug_color_jitter_strength: `float`, the floating number for the strength
        of the color augmentation
      aug_color_jitter_impl: `str`, 'simclrv1' or 'simclrv2'. Define whether
        to use simclrv1 or simclrv2's version of random brightness.
      aug_rand_blur: `bool`, if True, augment training with random blur.
      parse_label: `bool`, if True, parse label together with image.
      test_crop: `bool`, if True, augment eval with center cropping.
      mode: `str`, 'pretain' or 'finetune'. Define training mode.
      dtype: `str`, cast output image in dtype. It can be 'float32', 'float16',
        or 'bfloat16'.
    """
    self._output_size = output_size
    self._aug_rand_crop = aug_rand_crop
    self._aug_rand_hflip = aug_rand_hflip
    self._aug_color_distort = aug_color_distort
    self._aug_color_jitter_strength = aug_color_jitter_strength
    self._aug_color_jitter_impl = aug_color_jitter_impl
    self._aug_rand_blur = aug_rand_blur
    self._parse_label = parse_label
    self._mode = mode
    self._test_crop = test_crop
    if max(self._output_size[0], self._output_size[1]) <= 32:
      self._test_crop = False

    if dtype == 'float32':
      self._dtype = tf.float32
    elif dtype == 'float16':
      self._dtype = tf.float16
    elif dtype == 'bfloat16':
      self._dtype = tf.bfloat16
    else:
      raise ValueError('dtype {!r} is not supported!'.format(dtype))

  def _parse_one_train_image(self, image_bytes):

    image = tf.image.decode_jpeg(image_bytes, channels=3)
    # This line convert the image to float 0.0 - 1.0
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    if self._aug_rand_crop:
      image = simclr_preprocess_ops.random_crop_with_resize(
          image, self._output_size[0], self._output_size[1])

    if self._aug_rand_hflip:
      image = tf.image.random_flip_left_right(image)

    if self._aug_color_distort and self._mode == simclr_model.PRETRAIN:
      image = simclr_preprocess_ops.random_color_jitter(
          image=image,
          color_jitter_strength=self._aug_color_jitter_strength,
          impl=self._aug_color_jitter_impl)

    if self._aug_rand_blur and self._mode == simclr_model.PRETRAIN:
      image = simclr_preprocess_ops.random_blur(
          image, self._output_size[0], self._output_size[1])

    image = tf.image.resize(
        image, self._output_size, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.reshape(image, [self._output_size[0], self._output_size[1], 3])

    image = tf.clip_by_value(image, 0., 1.)
    # Convert image to self._dtype.
    image = tf.image.convert_image_dtype(image, self._dtype)

    return image

  def _parse_train_data(self, decoded_tensors):
    """Parses data for training."""
    image_bytes = decoded_tensors['image/encoded']

    if self._mode == simclr_model.FINETUNE:
      image = self._parse_one_train_image(image_bytes)

    elif self._mode == simclr_model.PRETRAIN:
      # Transform each example twice using a combination of
      # simple augmentations, resulting in 2N data points
      xs = []
      for _ in range(2):
        xs.append(self._parse_one_train_image(image_bytes))
      image = tf.concat(xs, -1)

    else:
      raise ValueError('The mode {} is not supported by the Parser.'
                       .format(self._mode))

    if self._parse_label:
      label = tf.cast(decoded_tensors['image/class/label'], dtype=tf.int32)
      return image, label

    return image

  def _parse_eval_data(self, decoded_tensors):
    """Parses data for evaluation."""
    image_bytes = decoded_tensors['image/encoded']
    image_shape = tf.image.extract_jpeg_shape(image_bytes)

    if self._test_crop:
      image = preprocess_ops.center_crop_image_v2(image_bytes, image_shape)
    else:
      image = tf.image.decode_jpeg(image_bytes, channels=3)
    # This line convert the image to float 0.0 - 1.0
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = tf.image.resize(
        image, self._output_size, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.reshape(image, [self._output_size[0], self._output_size[1], 3])

    image = tf.clip_by_value(image, 0., 1.)

    # Convert image to self._dtype.
    image = tf.image.convert_image_dtype(image, self._dtype)

    if self._parse_label:
      label = tf.cast(decoded_tensors['image/class/label'], dtype=tf.int32)
      return image, label

    return image
