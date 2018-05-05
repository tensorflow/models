# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Builder function for image resizing operations."""
import functools
import tensorflow as tf

from object_detection.core import preprocessor
from object_detection.protos import image_resizer_pb2


def _tf_resize_method(resize_method):
  """Maps image resize method from enumeration type to TensorFlow.

  Args:
    resize_method: The resize_method attribute of keep_aspect_ratio_resizer or
      fixed_shape_resizer.

  Returns:
    method: The corresponding TensorFlow ResizeMethod.

  Raises:
    ValueError: if `resize_method` is of unknown type.
  """
  dict_method = {
      image_resizer_pb2.BILINEAR:
          tf.image.ResizeMethod.BILINEAR,
      image_resizer_pb2.NEAREST_NEIGHBOR:
          tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      image_resizer_pb2.BICUBIC:
          tf.image.ResizeMethod.BICUBIC,
      image_resizer_pb2.AREA:
          tf.image.ResizeMethod.AREA
  }
  if resize_method in dict_method:
    return dict_method[resize_method]
  else:
    raise ValueError('Unknown resize_method')


def build(image_resizer_config):
  """Builds callable for image resizing operations.

  Args:
    image_resizer_config: image_resizer.proto object containing parameters for
      an image resizing operation.

  Returns:
    image_resizer_fn: Callable for image resizing.  This callable always takes
      a rank-3 image tensor (corresponding to a single image) and returns a
      rank-3 image tensor, possibly with new spatial dimensions.

  Raises:
    ValueError: if `image_resizer_config` is of incorrect type.
    ValueError: if `image_resizer_config.image_resizer_oneof` is of expected
      type.
    ValueError: if min_dimension > max_dimension when keep_aspect_ratio_resizer
      is used.
  """
  if not isinstance(image_resizer_config, image_resizer_pb2.ImageResizer):
    raise ValueError('image_resizer_config not of type '
                     'image_resizer_pb2.ImageResizer.')

  image_resizer_oneof = image_resizer_config.WhichOneof('image_resizer_oneof')
  if image_resizer_oneof == 'keep_aspect_ratio_resizer':
    keep_aspect_ratio_config = image_resizer_config.keep_aspect_ratio_resizer
    if not (keep_aspect_ratio_config.min_dimension <=
            keep_aspect_ratio_config.max_dimension):
      raise ValueError('min_dimension > max_dimension')
    method = _tf_resize_method(keep_aspect_ratio_config.resize_method)
    image_resizer_fn = functools.partial(
        preprocessor.resize_to_range,
        min_dimension=keep_aspect_ratio_config.min_dimension,
        max_dimension=keep_aspect_ratio_config.max_dimension,
        method=method,
        pad_to_max_dimension=keep_aspect_ratio_config.pad_to_max_dimension)
    if not keep_aspect_ratio_config.convert_to_grayscale:
      return image_resizer_fn
  elif image_resizer_oneof == 'fixed_shape_resizer':
    fixed_shape_resizer_config = image_resizer_config.fixed_shape_resizer
    method = _tf_resize_method(fixed_shape_resizer_config.resize_method)
    image_resizer_fn = functools.partial(
        preprocessor.resize_image,
        new_height=fixed_shape_resizer_config.height,
        new_width=fixed_shape_resizer_config.width,
        method=method)
    if not fixed_shape_resizer_config.convert_to_grayscale:
      return image_resizer_fn
  else:
    raise ValueError(
        'Invalid image resizer option: \'%s\'.' % image_resizer_oneof)

  def grayscale_image_resizer(image):
    [resized_image, resized_image_shape] = image_resizer_fn(image)
    grayscale_image = preprocessor.rgb_to_gray(resized_image)
    grayscale_image_shape = tf.concat([resized_image_shape[:-1], [1]], 0)
    return [grayscale_image, grayscale_image_shape]

  return functools.partial(grayscale_image_resizer)
