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

"""YOLO input and model functions for serving/inference."""

from typing import List, Tuple

import tensorflow as tf

from official.projects.yolo.ops import preprocessing_ops
from official.vision.ops import box_ops


def letterbox(image: tf.Tensor,
              desired_size: List[int],
              letter_box: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
  """Letter box an image for image serving."""

  with tf.name_scope('letter_box'):
    image_size = tf.cast(preprocessing_ops.get_image_shape(image), tf.float32)

    scaled_size = tf.cast(desired_size, image_size.dtype)
    if letter_box:
      scale = tf.minimum(scaled_size[0] / image_size[0],
                         scaled_size[1] / image_size[1])
      scaled_size = tf.round(image_size * scale)
    else:
      scale = 1.0

    # Computes 2D image_scale.
    image_scale = scaled_size / image_size
    image_offset = tf.cast((desired_size - scaled_size) * 0.5, tf.int32)
    offset = (scaled_size - desired_size) * 0.5
    scaled_image = tf.image.resize(
        image, tf.cast(scaled_size, tf.int32), method='nearest')

    output_image = tf.image.pad_to_bounding_box(scaled_image, image_offset[0],
                                                image_offset[1],
                                                desired_size[0],
                                                desired_size[1])

    image_info = tf.stack([
        image_size,
        tf.cast(desired_size, dtype=tf.float32), image_scale,
        tf.cast(offset, tf.float32)
    ])
    return output_image, image_info


def undo_info(boxes: tf.Tensor,
              num_detections: int,
              info: tf.Tensor,
              expand: bool = True) -> tf.Tensor:
  """Clip and normalize boxes for serving."""

  mask = tf.sequence_mask(num_detections, maxlen=tf.shape(boxes)[1])
  boxes = tf.cast(tf.expand_dims(mask, axis=-1), boxes.dtype) * boxes

  if expand:
    info = tf.cast(tf.expand_dims(info, axis=0), boxes.dtype)
  inshape = tf.expand_dims(info[:, 1, :], axis=1)
  ogshape = tf.expand_dims(info[:, 0, :], axis=1)
  scale = tf.expand_dims(info[:, 2, :], axis=1)
  offset = tf.expand_dims(info[:, 3, :], axis=1)

  boxes = box_ops.denormalize_boxes(boxes, inshape)
  boxes += tf.tile(offset, [1, 1, 2])
  boxes /= tf.tile(scale, [1, 1, 2])
  boxes = box_ops.clip_boxes(boxes, ogshape)
  boxes = box_ops.normalize_boxes(boxes, ogshape)
  return boxes
