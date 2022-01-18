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

"""Example classification decoder and parser.

This file defines the Decoder and Parser to load data. The example is shown on
loading standard tf.Example data but non-standard tf.Example or other data
format can be supported by implementing proper decoder and parser.
"""
from typing import Mapping, List, Tuple
# Import libraries
import tensorflow as tf

from official.vision.beta.dataloaders import decoder
from official.vision.beta.dataloaders import parser
from official.vision.beta.ops import preprocess_ops

MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)


class Decoder(decoder.Decoder):
  """A tf.Example decoder for classification task."""

  def __init__(self):
    """Initializes the decoder.

    The constructor defines the mapping between the field name and the value
    from an input tf.Example. For example, we define two fields for image bytes
    and labels. There is no limit on the number of fields to decode.
    """
    self._keys_to_features = {
        'image/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/class/label':
            tf.io.FixedLenFeature((), tf.int64, default_value=-1)
    }

  def decode(self,
             serialized_example: tf.train.Example) -> Mapping[str, tf.Tensor]:
    """Decodes a tf.Example to a dictionary.

    This function decodes a serialized tf.Example to a dictionary. The output
    will be consumed by `_parse_train_data` and `_parse_validation_data` in
    Parser.

    Args:
      serialized_example: A serialized tf.Example.

    Returns:
      A dictionary of field key name and decoded tensor mapping.
    """
    return tf.io.parse_single_example(
        serialized_example, self._keys_to_features)


class Parser(parser.Parser):
  """Parser to parse an image and its annotations.

  To define own Parser, client should override _parse_train_data and
  _parse_eval_data functions, where decoded tensors are parsed with optional
  pre-processing steps. The output from the two functions can be any structure
  like tuple, list or dictionary.
  """

  def __init__(self, output_size: List[int], num_classes: float):
    """Initializes parameters for parsing annotations in the dataset.

    This example only takes two arguments but one can freely add as many
    arguments as needed. For example, pre-processing and augmentations usually
    happen in Parser, and related parameters can be passed in by this
    constructor.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image.
      num_classes: `float`, number of classes.
    """
    self._output_size = output_size
    self._num_classes = num_classes
    self._dtype = tf.float32

  def _parse_data(
      self, decoded_tensors: Mapping[str,
                                     tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    label = tf.cast(decoded_tensors['image/class/label'], dtype=tf.int32)
    image_bytes = decoded_tensors['image/encoded']
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(
        image, self._output_size, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.ensure_shape(image, self._output_size + [3])

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(
        image, offset=MEAN_RGB, scale=STDDEV_RGB)

    image = tf.image.convert_image_dtype(image, self._dtype)
    return image, label

  def _parse_train_data(
      self, decoded_tensors: Mapping[str,
                                     tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Parses data for training.

    Args:
      decoded_tensors: A dictionary of field key name and decoded tensor mapping
        from Decoder.

    Returns:
      A tuple of (image, label) tensors.

    """
    return self._parse_data(decoded_tensors)

  def _parse_eval_data(
      self, decoded_tensors: Mapping[str,
                                     tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Parses data for evaluation.

    Args:
      decoded_tensors: A dictionary of field key name and decoded tensor mapping
        from Decoder.

    Returns:
      A tuple of (image, label) tensors.
    """
    return self._parse_data(decoded_tensors)
