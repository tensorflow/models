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

"""Classification decoder and parser."""
from typing import Dict, List, Optional
# Import libraries
import tensorflow as tf

from official.vision.beta.configs import common
from official.vision.beta.dataloaders import decoder
from official.vision.beta.dataloaders import parser
from official.vision.beta.ops import augment
from official.vision.beta.ops import preprocess_ops

MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)


class Decoder(decoder.Decoder):
  """A tf.Example decoder for classification task."""

  def __init__(self,
               image_field_key: str = 'image/encoded',
               label_field_key: str = 'image/class/label'):
    self._keys_to_features = {
        image_field_key: tf.io.FixedLenFeature((), tf.string, default_value=''),
        label_field_key: (tf.io.FixedLenFeature((), tf.int64, default_value=-1))
    }

  def decode(self,
             serialized_example: tf.train.Example) -> Dict[str, tf.Tensor]:
    return tf.io.parse_single_example(
        serialized_example, self._keys_to_features)


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               output_size: List[int],
               num_classes: float,
               image_field_key: str = 'image/encoded',
               label_field_key: str = 'image/class/label',
               aug_rand_hflip: bool = True,
               aug_type: Optional[common.Augmentation] = None,
               dtype: str = 'float32'):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      num_classes: `float`, number of classes.
      image_field_key: A `str` of the key name to encoded image in TFExample.
      label_field_key: A `str` of the key name to label in TFExample.
      aug_rand_hflip: `bool`, if True, augment training with random
        horizontal flip.
      aug_type: An optional Augmentation object to choose from AutoAugment and
        RandAugment.
      dtype: `str`, cast output image in dtype. It can be 'float32', 'float16',
        or 'bfloat16'.
    """
    self._output_size = output_size
    self._aug_rand_hflip = aug_rand_hflip
    self._num_classes = num_classes
    self._image_field_key = image_field_key
    self._label_field_key = label_field_key

    if dtype == 'float32':
      self._dtype = tf.float32
    elif dtype == 'float16':
      self._dtype = tf.float16
    elif dtype == 'bfloat16':
      self._dtype = tf.bfloat16
    else:
      raise ValueError('dtype {!r} is not supported!'.format(dtype))
    if aug_type:
      if aug_type.type == 'autoaug':
        self._augmenter = augment.AutoAugment(
            augmentation_name=aug_type.autoaug.augmentation_name,
            cutout_const=aug_type.autoaug.cutout_const,
            translate_const=aug_type.autoaug.translate_const)
      elif aug_type.type == 'randaug':
        self._augmenter = augment.RandAugment(
            num_layers=aug_type.randaug.num_layers,
            magnitude=aug_type.randaug.magnitude,
            cutout_const=aug_type.randaug.cutout_const,
            translate_const=aug_type.randaug.translate_const)
      else:
        raise ValueError('Augmentation policy {} not supported.'.format(
            aug_type.type))
    else:
      self._augmenter = None

  def _parse_train_data(self, decoded_tensors):
    """Parses data for training."""
    label = tf.cast(decoded_tensors[self._label_field_key], dtype=tf.int32)
    image_bytes = decoded_tensors[self._image_field_key]
    image_shape = tf.image.extract_jpeg_shape(image_bytes)

    # Crops image.
    # TODO(pengchong): support image format other than JPEG.
    cropped_image = preprocess_ops.random_crop_image_v2(
        image_bytes, image_shape)
    image = tf.cond(
        tf.reduce_all(tf.equal(tf.shape(cropped_image), image_shape)),
        lambda: preprocess_ops.center_crop_image_v2(image_bytes, image_shape),
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

    return image, label

  def _parse_eval_data(self, decoded_tensors):
    """Parses data for evaluation."""
    label = tf.cast(decoded_tensors[self._label_field_key], dtype=tf.int32)
    image_bytes = decoded_tensors[self._image_field_key]
    image_shape = tf.image.extract_jpeg_shape(image_bytes)

    # Center crops and resizes image.
    image = preprocess_ops.center_crop_image_v2(image_bytes, image_shape)

    image = tf.image.resize(
        image, self._output_size, method=tf.image.ResizeMethod.BILINEAR)

    image = tf.reshape(image, [self._output_size[0], self._output_size[1], 3])

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image,
                                           offset=MEAN_RGB,
                                           scale=STDDEV_RGB)

    # Convert image to self._dtype.
    image = tf.image.convert_image_dtype(image, self._dtype)

    return image, label
