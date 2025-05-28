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

"""Data parser and processing for 3D segmentation datasets."""

from typing import Any, Dict, Sequence, Tuple
import tensorflow as tf, tf_keras
from official.vision.dataloaders import decoder
from official.vision.dataloaders import parser


class Decoder(decoder.Decoder):
  """A tf.Example decoder for segmentation task."""

  def __init__(self,
               image_field_key: str = 'image/encoded',
               label_field_key: str = 'image/class/label'):
    self._keys_to_features = {
        image_field_key: tf.io.FixedLenFeature([], tf.string, default_value=''),
        label_field_key: tf.io.FixedLenFeature([], tf.string, default_value='')
    }

  def decode(self, serialized_example: tf.string) -> Dict[str, tf.Tensor]:
    return tf.io.parse_single_example(serialized_example,
                                      self._keys_to_features)


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               input_size: Sequence[int],
               num_classes: int,
               num_channels: int = 3,
               image_field_key: str = 'image/encoded',
               label_field_key: str = 'image/class/label',
               dtype: str = 'float32',
               label_dtype: str = 'float32'):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      input_size: The input tensor size of [height, width, volume] of input
        image.
      num_classes: The number of classes to be segmented.
      num_channels: The channel of input images.
      image_field_key: A `str` of the key name to encoded image in TFExample.
      label_field_key: A `str` of the key name to label in TFExample.
      dtype: The data type. One of {`bfloat16`, `float32`, `float16`}.
      label_dtype: The data type of input label.
    """
    self._input_size = input_size
    self._num_classes = num_classes
    self._num_channels = num_channels
    self._image_field_key = image_field_key
    self._label_field_key = label_field_key
    self._dtype = dtype
    self._label_dtype = label_dtype

  def _prepare_image_and_label(
      self, data: Dict[str, Any]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Prepares normalized image and label."""
    image = tf.io.decode_raw(data[self._image_field_key],
                             tf.as_dtype(tf.float32))
    label = tf.io.decode_raw(data[self._label_field_key],
                             tf.as_dtype(self._label_dtype))
    image_size = list(self._input_size) + [self._num_channels]
    image = tf.reshape(image, image_size)
    label_size = list(self._input_size) + [self._num_classes]
    label = tf.reshape(label, label_size)

    image = tf.cast(image, dtype=self._dtype)
    label = tf.cast(label, dtype=self._dtype)

    # TPU doesn't support tf.int64 well, use tf.int32 directly.
    if label.dtype == tf.int64:
      label = tf.cast(label, dtype=tf.int32)
    return image, label

  def _parse_train_data(self, data: Dict[str,
                                         Any]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Parses data for training and evaluation."""
    image, labels = self._prepare_image_and_label(data)
    # Cast image as self._dtype
    image = tf.cast(image, dtype=self._dtype)

    return image, labels

  def _parse_eval_data(self, data: Dict[str,
                                        Any]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Parses data for training and evaluation."""
    image, labels = self._prepare_image_and_label(data)
    # Cast image as self._dtype
    image = tf.cast(image, dtype=self._dtype)

    return image, labels
