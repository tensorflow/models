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

"""Image-related utilities that are useful to prepare dataset."""

import dataclasses
import imghdr
import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image


@dataclasses.dataclass
class ImageFormat:
  """Supported image formats.

  For model development, this library should support the same image formats as
  `tf.io.decode_image`[1].

  [1]: https://www.tensorflow.org/api_docs/python/tf/io/decode_image
  """
  bmp: str = 'BMP'
  png: str = 'PNG'
  jpeg: str = 'JPEG'
  raw: str = 'RAW'


def validate_image_format(format_str: str) -> str:
  """Validates `format_str` and returns canonical format.

  This function accepts image format in lower case and will returns the upper
  case string as canonical format.

  Args:
    format_str: Image format string.

  Returns:
    Canonical image format string.

  Raises:
    ValueError: If the canonical format is not listed in `ImageFormat`.
  """
  canonical_format = format_str.upper()
  if canonical_format in dataclasses.asdict(ImageFormat()).values():
    return canonical_format
  raise ValueError(f'Image format is invalid: {format_str}')


def encode_image(image_np: np.ndarray, image_format: str) -> bytes:
  """Encodes `image_np` specified by `image_format`.

  Args:
    image_np: Numpy image array.
    image_format: An enum specifying the format of the generated image.

  Returns:
    Encoded image string.
  """
  if image_format == 'RAW':
    return image_np.tobytes()

  if len(image_np.shape) > 2 and image_np.shape[2] == 1:
    image_pil = Image.fromarray(np.squeeze(image_np), 'L')
  else:
    image_pil = Image.fromarray(image_np)
  with io.BytesIO() as output:
    image_pil.save(output, format=validate_image_format(image_format))
    return output.getvalue()


def decode_image(image_bytes: bytes,
                 image_format: Optional[str] = None,
                 image_dtype: str = 'uint8') -> np.ndarray:
  """Decodes image_bytes into numpy array."""
  if image_format == 'RAW':
    return np.frombuffer(image_bytes, dtype=image_dtype)
  image_pil = Image.open(io.BytesIO(image_bytes))
  image_np = np.array(image_pil)
  if len(image_np.shape) < 3:
    image_np = image_np[..., np.newaxis]
  return image_np


def decode_image_metadata(image_bytes: bytes) -> Tuple[int, int, int, str]:
  """Decodes image metadata from encoded image string.

  Note that if the image is encoded in RAW format, the metadata cannot be
  inferred from the image bytes.

  Args:
    image_bytes: Encoded image string.

  Returns:
    A tuple of height, width, number of channels, and encoding format.
  """
  image_np = decode_image(image_bytes)
  # https://pillow.readthedocs.io/en/stable/reference/Image.html#image-attributes
  height, width, num_channels = image_np.shape
  image_format = imghdr.what(file=None, h=image_bytes)
  return height, width, num_channels, validate_image_format(image_format)
