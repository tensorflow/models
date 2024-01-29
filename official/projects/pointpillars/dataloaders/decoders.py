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

"""Data decoder and parser for Pointpillars."""

from typing import Any, Mapping, Tuple

import tensorflow as tf

from official.projects.pointpillars.configs import pointpillars as cfg
from official.vision.dataloaders import decoder


class ExampleDecoder(decoder.Decoder):
  """The class to decode preprocessed tf.example to tensors.

  Notations:
    P: number of pillars in an example
    N: number of points in a pillar
    D: number of features in a point
    M: number of labeled boxes in an example
  """

  def __init__(self,
               image_config: cfg.ImageConfig,
               pillars_config: cfg.PillarsConfig):
    """Initialize the decoder."""
    self._feature_description = {
        'frame_id': tf.io.FixedLenFeature([], tf.int64),
        'pillars': tf.io.FixedLenFeature([], tf.string),
        'indices': tf.io.FixedLenFeature([], tf.string),
        'bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'bbox/class': tf.io.VarLenFeature(tf.int64),
        'bbox/heading': tf.io.VarLenFeature(tf.float32),
        'bbox/z': tf.io.VarLenFeature(tf.float32),
        'bbox/height': tf.io.VarLenFeature(tf.float32),
        'bbox/difficulty': tf.io.VarLenFeature(tf.int64),
    }
    self._pillars_config = pillars_config

  def _decode_pillars(
      self, parsed_tensors: Mapping[str, tf.Tensor]
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Decode pillars from parsed tensors.

    Args:
      parsed_tensors: A {name: tensor} dict of parsed tensors.

    Returns:
      pillars: A tensor with shape [P, N, D]
      indices: A tensor with shape [P, 2]
    """
    pillars = tf.io.decode_raw(parsed_tensors['pillars'], tf.float32)
    pillars = tf.reshape(pillars, [
        self._pillars_config.num_pillars,
        self._pillars_config.num_points_per_pillar,
        self._pillars_config.num_features_per_point
    ])
    indices = tf.io.decode_raw(parsed_tensors['indices'], tf.int32)
    indices = tf.reshape(indices, [self._pillars_config.num_pillars, 2])
    return pillars, indices

  def _decode_boxes(self, parsed_tensors: Mapping[str, tf.Tensor]) -> tf.Tensor:
    """Decode boxes from parsed tensors.

    Args:
      parsed_tensors: A {name: tensor} dict of parsed tensors.

    Returns:
      boxes: A tensor with shape [M, 4], the last dim represents box yxyx
    """
    ymin = parsed_tensors['bbox/ymin']
    xmin = parsed_tensors['bbox/xmin']
    ymax = parsed_tensors['bbox/ymax']
    xmax = parsed_tensors['bbox/xmax']
    boxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return boxes

  def decode(self, serialized_example: Any) -> Mapping[str, Any]:
    """Decode the serialized example.

    Args:
      serialized_example: a single serialized tf.Example string.

    Returns:
      decoded_tensors: a dictionary of tensors with the following fields:
        - frame_id: an int64 scalar tensor to identify an example.
        - pillars: a float32 tensor of shape [P, N, D].
        - indices: an int32 tensor of shape [P, 2].
        - gt_classes: an int32 tensor of shape [M].
        - gt_boxes: a float32 tensor of shape [M, 4].
        - gt_attributes: a dict of (name, [M, 1]) float32 pairs.
        - gt_difficulty: an int32 tensor of shape [M].
    """
    parsed_tensors = tf.io.parse_single_example(
        serialized=serialized_example, features=self._feature_description)

    # Convert sparse tensor to dense tensor.
    for k in parsed_tensors:
      if isinstance(parsed_tensors[k], tf.SparseTensor):
        parsed_tensors[k] = tf.sparse.to_dense(
            parsed_tensors[k], default_value=0)

    # Decode features and labels.
    frame_id = parsed_tensors['frame_id']
    pillars, indices = self._decode_pillars(parsed_tensors)
    classes = tf.cast(parsed_tensors['bbox/class'], tf.int32)
    boxes = self._decode_boxes(parsed_tensors)
    attr_heading = tf.expand_dims(parsed_tensors['bbox/heading'], axis=1)
    attr_z = tf.expand_dims(parsed_tensors['bbox/z'], axis=1)
    attr_height = tf.expand_dims(parsed_tensors['bbox/height'], axis=1)
    difficulty = tf.cast(parsed_tensors['bbox/difficulty'], tf.int32)

    decoded_tensors = {
        'frame_id': frame_id,
        'pillars': pillars,
        'indices': indices,
        'gt_classes': classes,
        'gt_boxes': boxes,
        'gt_attributes': {
            'heading': attr_heading,
            'z': attr_z,
            'height': attr_height,
        },
        'gt_difficulty': difficulty,
    }
    return decoded_tensors
