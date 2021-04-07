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
"""Preprocessing ops."""

import tensorflow as tf



def normalize_image(image,
                    offset=(0.485, 0.456, 0.406),
                    scale=(0.229, 0.224, 0.225)):
  """Normalizes the image to zero mean and unit variance."""
  with tf.name_scope('normalize_image'):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    offset = tf.constant(offset)
    offset = tf.expand_dims(offset, axis=0)
    offset = tf.expand_dims(offset, axis=0)
    image -= offset

    scale = tf.constant(scale)
    scale = tf.expand_dims(scale, axis=0)
    scale = tf.expand_dims(scale, axis=0)
    image /= scale
    return image


def random_horizontal_flip(image, normalized_boxes=None, masks=None, seed=1):
  """Randomly flips input image and bounding boxes."""
  with tf.name_scope('random_horizontal_flip'):
    do_flip = tf.greater(tf.random.uniform([], seed=seed, dtype=tf.float32),
                         0.5)

    image = tf.cond(
        do_flip,
        lambda: horizontal_flip_image(image),
        lambda: image)

    if normalized_boxes is not None:
      normalized_boxes = tf.cond(
          do_flip,
          lambda: horizontal_flip_boxes(normalized_boxes),
          lambda: normalized_boxes)

    if masks is not None:
      masks = tf.cond(
          do_flip,
          lambda: horizontal_flip_masks(masks),
          lambda: masks)

    return image, normalized_boxes, masks
