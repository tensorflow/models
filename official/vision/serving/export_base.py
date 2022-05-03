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

"""Base class for model export."""

import abc
from typing import Dict, List, Mapping, Optional, Text

import tensorflow as tf
from official.core import config_definitions as cfg
from official.core import export_base


class ExportModule(export_base.ExportModule, metaclass=abc.ABCMeta):
  """Base Export Module."""

  def __init__(self,
               params: cfg.ExperimentConfig,
               *,
               batch_size: int,
               input_image_size: List[int],
               input_type: str = 'image_tensor',
               num_channels: int = 3,
               model: Optional[tf.keras.Model] = None,
               input_name: Optional[str] = None):
    """Initializes a module for export.

    Args:
      params: Experiment params.
      batch_size: The batch size of the model input. Can be `int` or None.
      input_image_size: List or Tuple of size of the input image. For 2D image,
        it is [height, width].
      input_type: The input signature type.
      num_channels: The number of the image channels.
      model: A tf.keras.Model instance to be exported.
      input_name: A customized input tensor name.
    """
    self.params = params
    self._batch_size = batch_size
    self._input_image_size = input_image_size
    self._num_channels = num_channels
    self._input_type = input_type
    self._input_name = input_name
    if model is None:
      model = self._build_model()  # pylint: disable=assignment-from-none
    super().__init__(params=params, model=model)

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

  def _build_model(self, **kwargs):
    """Returns a model built from the params."""
    return None

  @tf.function
  def inference_from_image_tensors(
      self, inputs: tf.Tensor) -> Mapping[str, tf.Tensor]:
    return self.serve(inputs)

  @tf.function
  def inference_for_tflite(self, inputs: tf.Tensor) -> Mapping[str, tf.Tensor]:
    return self.serve(inputs)

  @tf.function
  def inference_from_image_bytes(self, inputs: tf.Tensor):
    with tf.device('cpu:0'):
      images = tf.nest.map_structure(
          tf.identity,
          tf.map_fn(
              self._decode_image,
              elems=inputs,
              fn_output_signature=tf.TensorSpec(
                  shape=[None] * len(self._input_image_size) +
                  [self._num_channels],
                  dtype=tf.uint8),
              parallel_iterations=32))
      images = tf.stack(images)
    return self.serve(images)

  @tf.function
  def inference_from_tf_example(self,
                                inputs: tf.Tensor) -> Mapping[str, tf.Tensor]:
    with tf.device('cpu:0'):
      images = tf.nest.map_structure(
          tf.identity,
          tf.map_fn(
              self._decode_tf_example,
              elems=inputs,
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
    return self.serve(images)

  def get_inference_signatures(self, function_keys: Dict[Text, Text]):
    """Gets defined function signatures.

    Args:
      function_keys: A dictionary with keys as the function to create signature
        for and values as the signature keys when returns.

    Returns:
      A dictionary with key as signature key and value as concrete functions
        that can be used for tf.saved_model.save.
    """
    signatures = {}
    for key, def_name in function_keys.items():
      if key == 'image_tensor':
        input_signature = tf.TensorSpec(
            shape=[self._batch_size] + [None] * len(self._input_image_size) +
            [self._num_channels],
            dtype=tf.uint8,
            name=self._input_name)
        signatures[
            def_name] = self.inference_from_image_tensors.get_concrete_function(
                input_signature)
      elif key == 'image_bytes':
        input_signature = tf.TensorSpec(
            shape=[self._batch_size], dtype=tf.string, name=self._input_name)
        signatures[
            def_name] = self.inference_from_image_bytes.get_concrete_function(
                input_signature)
      elif key == 'serve_examples' or key == 'tf_example':
        input_signature = tf.TensorSpec(
            shape=[self._batch_size], dtype=tf.string, name=self._input_name)
        signatures[
            def_name] = self.inference_from_tf_example.get_concrete_function(
                input_signature)
      elif key == 'tflite':
        input_signature = tf.TensorSpec(
            shape=[self._batch_size] + self._input_image_size +
            [self._num_channels],
            dtype=tf.float32,
            name=self._input_name)
        signatures[def_name] = self.inference_for_tflite.get_concrete_function(
            input_signature)
      else:
        raise ValueError('Unrecognized `input_type`')
    return signatures
