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
from typing import Any, Dict, List, Optional
# Import libraries
import tensorflow as tf

from official.vision.beta.configs import common
from official.vision.beta.dataloaders import decoder
from official.vision.beta.dataloaders import parser
from official.vision.beta.ops import augment
from official.vision.beta.ops import preprocess_ops

MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)

DEFAULT_IMAGE_FIELD_KEY = 'image/encoded'
DEFAULT_LABEL_FIELD_KEY = 'image/class/label'


class Decoder(decoder.Decoder):
  """A tf.Example decoder for classification task."""

  def __init__(self,
               image_field_key: str = DEFAULT_IMAGE_FIELD_KEY,
               label_field_key: str = DEFAULT_LABEL_FIELD_KEY,
               is_multilabel: bool = False,
               keys_to_features: Optional[Dict[str, Any]] = None):
    if not keys_to_features:
      keys_to_features = {
          image_field_key:
              tf.io.FixedLenFeature((), tf.string, default_value=''),
      }
      if is_multilabel:
        keys_to_features.update(
            {label_field_key: tf.io.VarLenFeature(dtype=tf.int64)})
      else:
        keys_to_features.update({
            label_field_key:
                tf.io.FixedLenFeature((), tf.int64, default_value=-1)
        })
    self._keys_to_features = keys_to_features

  def decode(self, serialized_example):
    return tf.io.parse_single_example(
        serialized_example, self._keys_to_features)


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               output_size: List[int],
               num_classes: float,
               image_field_key: str = DEFAULT_IMAGE_FIELD_KEY,
               label_field_key: str = DEFAULT_LABEL_FIELD_KEY,
               decode_jpeg_only: bool = True,
               aug_rand_hflip: bool = True,
               aug_type: Optional[common.Augmentation] = None,
               color_jitter: float = 0.,
               random_erasing: Optional[common.RandomErasing] = None,
               is_multilabel: bool = False,
               dtype: str = 'float32'):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      num_classes: `float`, number of classes.
      image_field_key: `str`, the key name to encoded image in tf.Example.
      label_field_key: `str`, the key name to label in tf.Example.
      decode_jpeg_only: `bool`, if True, only JPEG format is decoded, this is
        faster than decoding other types. Default is True.
      aug_rand_hflip: `bool`, if True, augment training with random
        horizontal flip.
      aug_type: An optional Augmentation object to choose from AutoAugment and
        RandAugment.
      color_jitter: Magnitude of color jitter. If > 0, the value is used to
        generate random scale factor for brightness, contrast and saturation.
        See `preprocess_ops.color_jitter` for more details.
      random_erasing: if not None, augment input image by random erasing. See
        `augment.RandomErasing` for more details.
      is_multilabel: A `bool`, whether or not each example has multiple labels.
      dtype: `str`, cast output image in dtype. It can be 'float32', 'float16',
        or 'bfloat16'.
    """
    self._output_size = output_size
    self._aug_rand_hflip = aug_rand_hflip
    self._num_classes = num_classes
    self._image_field_key = image_field_key
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
            translate_const=aug_type.randaug.translate_const,
            prob_to_apply=aug_type.randaug.prob_to_apply,
            exclude_ops=aug_type.randaug.exclude_ops)
      else:
        raise ValueError('Augmentation policy {} not supported.'.format(
            aug_type.type))
    else:
      self._augmenter = None
    self._label_field_key = label_field_key
    self._color_jitter = color_jitter
    if random_erasing:
      self._random_erasing = augment.RandomErasing(
          probability=random_erasing.probability,
          min_area=random_erasing.min_area,
          max_area=random_erasing.max_area,
          min_aspect=random_erasing.min_aspect,
          max_aspect=random_erasing.max_aspect,
          min_count=random_erasing.min_count,
          max_count=random_erasing.max_count,
          trials=random_erasing.trials)
    else:
      self._random_erasing = None
    self._is_multilabel = is_multilabel
    self._decode_jpeg_only = decode_jpeg_only

  def _parse_train_data(self, decoded_tensors):
    """Parses data for training."""
    image = self._parse_train_image(decoded_tensors)
    label = tf.cast(decoded_tensors[self._label_field_key], dtype=tf.int32)
    if self._is_multilabel:
      if isinstance(label, tf.sparse.SparseTensor):
        label = tf.sparse.to_dense(label)
      label = tf.reduce_sum(tf.one_hot(label, self._num_classes), axis=0)
    return image, label

  def _parse_eval_data(self, decoded_tensors):
    """Parses data for evaluation."""
    image = self._parse_eval_image(decoded_tensors)
    label = tf.cast(decoded_tensors[self._label_field_key], dtype=tf.int32)
    if self._is_multilabel:
      if isinstance(label, tf.sparse.SparseTensor):
        label = tf.sparse.to_dense(label)
      label = tf.reduce_sum(tf.one_hot(label, self._num_classes), axis=0)
    return image, label

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

    # Color jitter.
    if self._color_jitter > 0:
      image = preprocess_ops.color_jitter(image, self._color_jitter,
                                          self._color_jitter,
                                          self._color_jitter)

    # Resizes image.
    image = tf.image.resize(
        image, self._output_size, method=tf.image.ResizeMethod.BILINEAR)
    image.set_shape([self._output_size[0], self._output_size[1], 3])

    # Apply autoaug or randaug.
    if self._augmenter is not None:
      image = self._augmenter.distort(image)

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image,
                                           offset=MEAN_RGB,
                                           scale=STDDEV_RGB)

    # Random erasing after the image has been normalized
    if self._random_erasing is not None:
      image = self._random_erasing.distort(image)

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
    image.set_shape([self._output_size[0], self._output_size[1], 3])

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image,
                                           offset=MEAN_RGB,
                                           scale=STDDEV_RGB)

    # Convert image to self._dtype.
    image = tf.image.convert_image_dtype(image, self._dtype)

    return image
