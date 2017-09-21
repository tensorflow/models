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

from object_detection.core import preprocessor
from object_detection.protos import image_resizer_pb2


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

  if image_resizer_config.WhichOneof(
      'image_resizer_oneof') == 'keep_aspect_ratio_resizer':
    keep_aspect_ratio_config = image_resizer_config.keep_aspect_ratio_resizer
    if not (keep_aspect_ratio_config.min_dimension
            <= keep_aspect_ratio_config.max_dimension):
      raise ValueError('min_dimension > max_dimension')
    return functools.partial(
        preprocessor.resize_to_range,
        min_dimension=keep_aspect_ratio_config.min_dimension,
        max_dimension=keep_aspect_ratio_config.max_dimension)
  if image_resizer_config.WhichOneof(
      'image_resizer_oneof') == 'fixed_shape_resizer':
    fixed_shape_resizer_config = image_resizer_config.fixed_shape_resizer
    return functools.partial(preprocessor.resize_image,
                             new_height=fixed_shape_resizer_config.height,
                             new_width=fixed_shape_resizer_config.width)
  raise ValueError('Invalid image resizer option.')
