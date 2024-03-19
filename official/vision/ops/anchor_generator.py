# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Multi scale anchor generator definition."""

import tensorflow as tf, tf_keras


# (TODO/tanzheny): consider having customized anchor offset.
class _SingleAnchorGenerator:
  """Utility to generate anchors for a single feature map.

  Example:
  ```python
  anchor_gen = _SingleAnchorGenerator(32, [.5, 1., 2.], stride=16)
  anchors = anchor_gen([512, 512, 3])
  ```
  """

  def __init__(self,
               anchor_size,
               scales,
               aspect_ratios,
               stride,
               clip_boxes=False):
    """Constructs single scale anchor.

    Args:
      anchor_size: A single int represents the base anchor size. The anchor
        height will be `anchor_size / sqrt(aspect_ratio)`, anchor width will be
        `anchor_size * sqrt(aspect_ratio)`.
      scales: A list/tuple, or a list/tuple of a list/tuple of positive
        floats representing the actual anchor size to the base `anchor_size`.
      aspect_ratios: a list/tuple of positive floats representing the ratio of
        anchor width to anchor height.
      stride: A single int represents the anchor stride size between center of
        each anchor.
      clip_boxes: Boolean to represent whether the anchor coordinates should be
        clipped to the image size. Defaults to `False`.
    Input shape: the size of the image, `[H, W, C]`
    Output shape: the size of anchors, `[(H / stride) * (W / stride), 4]`
    """
    self.anchor_size = anchor_size
    self.scales = scales
    self.aspect_ratios = aspect_ratios
    self.stride = stride
    self.clip_boxes = clip_boxes

  def __call__(self, image_size):
    image_height = tf.cast(image_size[0], tf.float32)
    image_width = tf.cast(image_size[1], tf.float32)

    k = len(self.scales) * len(self.aspect_ratios)
    aspect_ratios_sqrt = tf.cast(tf.sqrt(self.aspect_ratios), dtype=tf.float32)
    anchor_size = tf.cast(self.anchor_size, tf.float32)

    # [K]
    anchor_heights = []
    anchor_widths = []
    for scale in self.scales:
      anchor_size_t = anchor_size * scale
      anchor_height = anchor_size_t / aspect_ratios_sqrt
      anchor_width = anchor_size_t * aspect_ratios_sqrt
      anchor_heights.append(anchor_height)
      anchor_widths.append(anchor_width)
    anchor_heights = tf.concat(anchor_heights, axis=0)
    anchor_widths = tf.concat(anchor_widths, axis=0)
    half_anchor_heights = tf.reshape(0.5 * anchor_heights, [1, 1, k])
    half_anchor_widths = tf.reshape(0.5 * anchor_widths, [1, 1, k])

    stride = tf.cast(self.stride, tf.float32)
    # [W]
    cx = tf.range(0.5 * stride, image_width + 0.5 * stride, stride)
    # [H]
    cy = tf.range(0.5 * stride, image_height + 0.5 * stride, stride)
    # [H, W]
    cx_grid, cy_grid = tf.meshgrid(cx, cy)
    # [H, W, 1]
    cx_grid = tf.expand_dims(cx_grid, axis=-1)
    cy_grid = tf.expand_dims(cy_grid, axis=-1)

    # [H, W, K, 1]
    y_min = tf.expand_dims(cy_grid - half_anchor_heights, axis=-1)
    y_max = tf.expand_dims(cy_grid + half_anchor_heights, axis=-1)
    x_min = tf.expand_dims(cx_grid - half_anchor_widths, axis=-1)
    x_max = tf.expand_dims(cx_grid + half_anchor_widths, axis=-1)

    if self.clip_boxes:
      y_min = tf.maximum(tf.minimum(y_min, image_height), 0.)
      y_max = tf.maximum(tf.minimum(y_max, image_height), 0.)
      x_min = tf.maximum(tf.minimum(x_min, image_width), 0.)
      x_max = tf.maximum(tf.minimum(x_max, image_width), 0.)

    # [H, W, K, 4]
    result = tf.concat([y_min, x_min, y_max, x_max], axis=-1)
    shape = result.shape.as_list()
    # [H, W, K * 4]
    return tf.reshape(result, [shape[0], shape[1], shape[2] * shape[3]])


class AnchorGeneratorv1():
  """Utility to generate anchors for a multiple feature maps.

  Example:
  ```python
  anchor_gen = AnchorGenerator([32, 64], [.5, 1., 2.],
    strides=[16, 32])
  anchors = anchor_gen([512, 512, 3])
  ```

  """

  def __init__(self,
               anchor_sizes,
               scales,
               aspect_ratios,
               strides,
               clip_boxes=False):
    """Constructs multiscale anchors.

    Args:
      anchor_sizes: A list of int represents the anchor size for each scale. The
        anchor height will be `anchor_size / sqrt(aspect_ratio)`, anchor width
        will be `anchor_size * sqrt(aspect_ratio)` for each scale.
      scales: A list/tuple, or a list/tuple of a list/tuple of positive
        floats representing the actual anchor size to the base `anchor_size`.
      aspect_ratios: A list/tuple, or a list/tuple of a list/tuple of positive
        floats representing the ratio of anchor width to anchor height.
      strides: A list/tuple of ints represent the anchor stride size between
        center of anchors at each scale.
      clip_boxes: Boolean to represents whether the anchor coordinates should be
        clipped to the image size. Defaults to `False`.
    Input shape: the size of the image, `[H, W, C]`
    Output shape: the size of anchors concat on each level, `[(H /
      strides) * (W / strides), K * 4]`
    """
    # aspect_ratio is a single list that is the same across all levels.
    aspect_ratios = maybe_map_structure_for_anchor(aspect_ratios, anchor_sizes)
    scales = maybe_map_structure_for_anchor(scales, anchor_sizes)
    if isinstance(anchor_sizes, dict):
      self.anchor_generators = {}
      for k in anchor_sizes.keys():
        self.anchor_generators[k] = _SingleAnchorGenerator(
            anchor_sizes[k], scales[k], aspect_ratios[k], strides[k],
            clip_boxes)
    elif isinstance(anchor_sizes, (list, tuple)):
      self.anchor_generators = []
      for anchor_size, scale_list, ar_list, stride in zip(
          anchor_sizes, scales, aspect_ratios, strides):
        self.anchor_generators.append(
            _SingleAnchorGenerator(anchor_size, scale_list, ar_list, stride,
                                   clip_boxes))

  def __call__(self, image_size):
    anchor_generators = tf.nest.flatten(self.anchor_generators)
    results = [anchor_gen(image_size) for anchor_gen in anchor_generators]
    return tf.nest.pack_sequence_as(self.anchor_generators, results)


def maybe_map_structure_for_anchor(params, anchor_sizes):
  """broadcast the params to match anchor_sizes."""
  if all(isinstance(param, (int, float)) for param in params):
    if isinstance(anchor_sizes, (tuple, list)):
      return [params] * len(anchor_sizes)
    elif isinstance(anchor_sizes, dict):
      return tf.nest.map_structure(lambda _: params, anchor_sizes)
    else:
      raise ValueError("the structure of `anchor_sizes` must be a tuple, "
                       "list, or dict, given {}".format(anchor_sizes))
  else:
    return params
