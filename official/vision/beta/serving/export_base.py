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
import tensorflow as tf


def _decode_image(encoded_image_bytes):
  image_tensor = tf.image.decode_image(encoded_image_bytes, channels=3)
  image_tensor.set_shape((None, None, 3))
  return image_tensor


def _decode_tf_example(tf_example_string_tensor):
  keys_to_features = {'image/encoded': tf.io.FixedLenFeature((), tf.string)}
  parsed_tensors = tf.io.parse_single_example(
      serialized=tf_example_string_tensor, features=keys_to_features)
  image_tensor = _decode_image(parsed_tensors['image/encoded'])
  return image_tensor


class ExportModule(tf.Module, metaclass=abc.ABCMeta):
  """Base Export Module."""

  def __init__(self, params, batch_size, input_image_size, model=None):
    """Initializes a module for export.

    Args:
      params: Experiment params.
      batch_size: Int or None.
      input_image_size: List or Tuple of height, width of the input image.
      model: A tf.keras.Model instance to be exported.
    """

    super(ExportModule, self).__init__()
    self._params = params
    self._batch_size = batch_size
    self._input_image_size = input_image_size
    self._model = model

  @abc.abstractmethod
  def build_model(self):
    """Builds model and sets self._model."""

  @abc.abstractmethod
  def _run_inference_on_image_tensors(self, images):
    """Runs inference on images."""

  @tf.function
  def inference_from_image_tensors(self, input_tensor):
    return self._run_inference_on_image_tensors(input_tensor)

  @tf.function
  def inference_from_image_bytes(self, input_tensor):
    with tf.device('cpu:0'):
      images = tf.nest.map_structure(
          tf.identity,
          tf.map_fn(
              _decode_image,
              elems=input_tensor,
              fn_output_signature=tf.TensorSpec(
                  shape=self._input_image_size + [3], dtype=tf.uint8),
              parallel_iterations=32))
      images = tf.stack(images)
    return self._run_inference_on_image_tensors(images)

  @tf.function
  def inference_from_tf_example(self, input_tensor):
    with tf.device('cpu:0'):
      images = tf.nest.map_structure(
          tf.identity,
          tf.map_fn(
              _decode_tf_example,
              elems=input_tensor,
              fn_output_signature=tf.TensorSpec(
                  shape=self._input_image_size + [3], dtype=tf.uint8),
              dtype=tf.uint8,
              parallel_iterations=32))
      images = tf.stack(images)
    return self._run_inference_on_image_tensors(images)
