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

"""Helper utils for export library."""

from typing import List, Optional
import tensorflow as tf

# pylint: disable=g-long-lambda


def get_image_input_signatures(input_type: str,
                               batch_size: Optional[int],
                               input_image_size: List[int],
                               num_channels: int = 3):
  """Gets input signatures for an image.

  Args:
    input_type: A `str`, can be either tf_example, image_bytes, or image_tensor.
    batch_size: `int` for batch size or None.
    input_image_size: List[int] for the height and width of the input image.
    num_channels: `int` for number of channels in the input image.
  Returns:
    tf.TensorSpec of the input tensor.
  """
  if input_type == 'image_tensor':
    input_signature = tf.TensorSpec(
        shape=[batch_size] + [None] * len(input_image_size) + [num_channels],
        dtype=tf.uint8)
  elif input_type in ['image_bytes', 'serve_examples', 'tf_example']:
    input_signature = tf.TensorSpec(shape=[batch_size], dtype=tf.string)
  elif input_type == 'tflite':
    input_signature = tf.TensorSpec(
        shape=[1] + input_image_size + [num_channels], dtype=tf.float32)
  else:
    raise ValueError('Unrecognized `input_type`')
  return input_signature


def decode_image(encoded_image_bytes: str,
                 input_image_size: List[int],
                 num_channels: int = 3,) -> tf.Tensor:
  """Decodes an image bytes to an image tensor.

  Use `tf.image.decode_image` to decode an image if input is expected to be 2D
  image; otherwise use `tf.io.decode_raw` to convert the raw bytes to tensor
  and reshape it to desire shape.

  Args:
    encoded_image_bytes: An encoded image string to be decoded.
    input_image_size: List[int] for the desired input size. This will be used to
      infer whether the image is 2d or 3d.
    num_channels: `int` for number of image channels.

  Returns:
    A decoded image tensor.
  """
  if len(input_image_size) == 2:
    # Decode an image if 2D input is expected.
    image_tensor = tf.image.decode_image(
        encoded_image_bytes, channels=num_channels)
  else:
    # Convert raw bytes into a tensor and reshape it, if not 2D input.
    image_tensor = tf.io.decode_raw(encoded_image_bytes, out_type=tf.uint8)
  image_tensor.set_shape([None] * len(input_image_size) + [num_channels])
  return image_tensor


def decode_image_tf_example(tf_example_string_tensor: tf.train.Example,
                            input_image_size: List[int],
                            num_channels: int = 3,
                            encoded_key: str = 'image/encoded'
                            ) -> tf.Tensor:
  """Decodes a TF Example to an image tensor."""

  keys_to_features = {
      encoded_key: tf.io.FixedLenFeature((), tf.string, default_value=''),
  }
  parsed_tensors = tf.io.parse_single_example(
      serialized=tf_example_string_tensor, features=keys_to_features)
  image_tensor = decode_image(
      parsed_tensors[encoded_key],
      input_image_size=input_image_size,
      num_channels=num_channels)
  return image_tensor


def parse_image(
    inputs, input_type: str, input_image_size: List[int], num_channels: int):
  """Parses image."""
  if input_type in ['tf_example', 'serve_examples']:
    decode_image_tf_example_fn = (
        lambda x: decode_image_tf_example(x, input_image_size, num_channels))
    image_tensor = tf.map_fn(
        decode_image_tf_example_fn,
        elems=inputs,
        fn_output_signature=tf.TensorSpec(
            shape=[None] * len(input_image_size) + [num_channels],
            dtype=tf.uint8),
    )
  elif input_type == 'image_bytes':
    decode_image_fn = lambda x: decode_image(x, input_image_size, num_channels)
    image_tensor = tf.map_fn(
        decode_image_fn, elems=inputs,
        fn_output_signature=tf.TensorSpec(
            shape=[None] * len(input_image_size) + [num_channels],
            dtype=tf.uint8),)
  else:
    image_tensor = inputs
  return image_tensor
