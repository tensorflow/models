# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""COCO data loader for Pix2Seq."""

from typing import Tuple
import tensorflow as tf

from official.vision.dataloaders import parser
from official.vision.ops import preprocess_ops

class Parser(parser.Parser):
  """Parse an image and its annotations into a dictionary of tensors."""

  def __init__(
      self,
      eos_token_weight: float = 0.1,
      output_size: Tuple[int, int] = (1333, 1333),
      max_num_boxes: int = 100,
      aug_rand_hflip=True,
      aug_scale_min=0.3,
      aug_scale_max=2.0,
      aug_color_jitter_strength: float = 0.5,
      aug_color_jitter_impl='simclrv2',
      coord_vocab_shift=1000,
      quantization_bins=1000,
      skip_crowd_during_training=True,
      label_shift: int = 0,
  ):
    self._eos_token_weight = eos_token_weight
    self._output_size = output_size
    self._max_num_boxes = max_num_boxes
    self._aug_rand_hflip = aug_rand_hflip
    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max
    self._aug_color_jitter_strength = aug_color_jitter_strength
    self._aug_color_jitter_impl = aug_color_jitter_impl
    self._coord_vocab_shift = coord_vocab_shift
    self._quantization_bins = quantization_bins
    self._skip_crowd_during_training = skip_crowd_during_training
    self._label_shift = label_shift
    
  def parse_fn(self, is_training):
    """Returns a parse fn that reads and parses raw tensors from the decoder.

    Args:
      is_training: a `bool` to indicate whether it is in training mode.

    Returns:
      parse: a `callable` that takes the serialized example and generate the
        images, labels tuple where labels is a dict of Tensors that contains
        labels.
    """
    def parse(decoded_tensors):
      """Parses the serialized example data."""
      if is_training:
        return self._parse_train_data(decoded_tensors)
      else:
        return self._parse_eval_data(decoded_tensors)

    return parse

  def _parse_train_data(self, data):
    """Parses data for training and evaluation."""

    # Gets original image.
    image = data['image']

    # Normalizes image with mean and std pixel values.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=self._output_size,
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)
    
    out = tf.stack([image, image], 0)

    return out
  
  def _parse_eval_data(self, data):
    """Parses data for training and evaluation."""

    # Gets original image.
    image = data['image']

    # Normalizes image with mean and std pixel values.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=self._output_size,
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)

    return image