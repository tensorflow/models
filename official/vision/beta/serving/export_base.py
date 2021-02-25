# Lint as: python3
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
"""Base class for model export."""

import abc
from typing import Optional, Sequence, Mapping

import tensorflow as tf

from official.modeling.hyperparams import config_definitions as cfg


class ExportModule(tf.Module, metaclass=abc.ABCMeta):
  """Base Export Module."""

  def __init__(self,
               params: cfg.ExperimentConfig,
               batch_size: int,
               input_image_size: Sequence[int],
               num_channels: int = 3,
               model: Optional[tf.keras.Model] = None):
    """Initializes a module for export.

    Args:
      params: Experiment params.
      batch_size: The batch size of the model input. Can be `int` or None.
      input_image_size: List or Tuple of size of the input image. For 2D image,
        it is [height, width].
      num_channels: The number of the image channels.
      model: A tf.keras.Model instance to be exported.
    """

    super(ExportModule, self).__init__()
    self._params = params
    self._batch_size = batch_size
    self._input_image_size = input_image_size
    self._num_channels = num_channels
    self._model = model

  def _decode_image(self, encoded_image_bytes: str) -> tf.Tensor:
    """Decodes an image bytes to an image tensor.

    Use `tf.image.decode_image` to decode an image if input is expected to be 2D
    image; otherwise use `tf.io.decode_raw` to convert the raw bytes to tensor
    and reshape it to desire shape.

    Args:
      encoded_image_bytes: An encoded image string to be decoded.

    Returns:
      A decoded image tensor.
    """
    if len(self._input_image_size) == 2:
      # Decode an image if 2D input is expected.
      image_tensor = tf.image.decode_image(
          encoded_image_bytes, channels=self._num_channels)
      image_tensor.set_shape((None, None, self._num_channels))
    else:
      # Convert raw bytes into a tensor and reshape it, if not 2D input.
      image_tensor = tf.io.decode_raw(encoded_image_bytes, out_type=tf.uint8)
      image_tensor = tf.reshape(image_tensor,
                                self._input_image_size + [self._num_channels])
    return image_tensor

  def _decode_tf_example(
      self, tf_example_string_tensor: tf.train.Example) -> tf.Tensor:
    """Decodes a TF Example to an image tensor.

    Args:
      tf_example_string_tensor: A tf.train.Example of encoded image and other
        information.

    Returns:
      A decoded image tensor.
    """
    keys_to_features = {'image/encoded': tf.io.FixedLenFeature((), tf.string)}
    parsed_tensors = tf.io.parse_single_example(
        serialized=tf_example_string_tensor, features=keys_to_features)
    image_tensor = self._decode_image(parsed_tensors['image/encoded'])
    return image_tensor

  @abc.abstractmethod
  def build_model(self, **kwargs):
    """Builds model and sets self._model."""

  @abc.abstractmethod
  def _run_inference_on_image_tensors(
      self, images: tf.Tensor) -> Mapping[str, tf.Tensor]:
    """Runs inference on images."""

  @tf.function
  def inference_from_image_tensors(
      self, input_tensor: tf.Tensor) -> Mapping[str, tf.Tensor]:
    return self._run_inference_on_image_tensors(input_tensor)

  @tf.function
  def inference_from_image_bytes(self, input_tensor: str):
    with tf.device('cpu:0'):
      images = tf.nest.map_structure(
          tf.identity,
          tf.map_fn(
              self._decode_image,
              elems=input_tensor,
              fn_output_signature=tf.TensorSpec(
                  shape=[None] * len(self._input_image_size) +
                  [self._num_channels],
                  dtype=tf.uint8),
              parallel_iterations=32))
      images = tf.stack(images)
    return self._run_inference_on_image_tensors(images)

  @tf.function
  def inference_from_tf_example(
      self, input_tensor: tf.train.Example) -> Mapping[str, tf.Tensor]:
    with tf.device('cpu:0'):
      images = tf.nest.map_structure(
          tf.identity,
          tf.map_fn(
              self._decode_tf_example,
              elems=input_tensor,
              # Height/width of the shape of input images is unspecified (None)
              # at the time of decoding the example, but the shape will
              # be adjusted to conform to the input layer of the model,
              # by _run_inference_on_image_tensors() below.
              fn_output_signature=tf.TensorSpec(
                  shape=[None] * len(self._input_image_size) +
                  [self._num_channels],
                  dtype=tf.uint8),
              dtype=tf.uint8,
              parallel_iterations=32))
      images = tf.stack(images)
    return self._run_inference_on_image_tensors(images)
