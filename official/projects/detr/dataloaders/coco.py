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

"""COCO data loader for DETR."""

import dataclasses
from typing import Optional, Tuple
import tensorflow as tf, tf_keras

from official.core import config_definitions as cfg
from official.core import input_reader
from official.vision.ops import box_ops
from official.vision.ops import preprocess_ops


@dataclasses.dataclass
class COCODataConfig(cfg.DataConfig):
  """Data config for COCO."""
  output_size: Tuple[int, int] = (1333, 1333)
  max_num_boxes: int = 100
  resize_scales: Tuple[int, ...] = (
      480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)


class COCODataLoader():
  """A class to load dataset for COCO detection task."""

  def __init__(self, params: COCODataConfig):
    self._params = params

  def preprocess(self, inputs):
    """Preprocess COCO for DETR."""
    image = inputs['image']
    boxes = inputs['objects']['bbox']
    classes = inputs['objects']['label'] + 1
    is_crowd = inputs['objects']['is_crowd']

    image = preprocess_ops.normalize_image(image)
    if self._params.is_training:
      image, boxes, _ = preprocess_ops.random_horizontal_flip(image, boxes)

      do_crop = tf.greater(tf.random.uniform([]), 0.5)
      if do_crop:
        # Rescale
        boxes = box_ops.denormalize_boxes(boxes, tf.shape(image)[:2])
        index = tf.random.categorical(tf.zeros([1, 3]), 1)[0]
        scales = tf.gather([400.0, 500.0, 600.0], index, axis=0)
        short_side = scales[0]
        image, image_info = preprocess_ops.resize_image(image, short_side)
        boxes = preprocess_ops.resize_and_crop_boxes(boxes,
                                                     image_info[2, :],
                                                     image_info[1, :],
                                                     image_info[3, :])
        boxes = box_ops.normalize_boxes(boxes, image_info[1, :])

        # Do croping
        shape = tf.cast(image_info[1], dtype=tf.int32)
        h = tf.random.uniform(
            [], 384, tf.math.minimum(shape[0], 600), dtype=tf.int32)
        w = tf.random.uniform(
            [], 384, tf.math.minimum(shape[1], 600), dtype=tf.int32)
        i = tf.random.uniform([], 0, shape[0] - h + 1, dtype=tf.int32)
        j = tf.random.uniform([], 0, shape[1] - w + 1, dtype=tf.int32)
        image = tf.image.crop_to_bounding_box(image, i, j, h, w)
        boxes = tf.clip_by_value(
            (boxes[..., :] * tf.cast(
                tf.stack([shape[0], shape[1], shape[0], shape[1]]),
                dtype=tf.float32) -
             tf.cast(tf.stack([i, j, i, j]), dtype=tf.float32)) /
            tf.cast(tf.stack([h, w, h, w]), dtype=tf.float32), 0.0, 1.0)
      scales = tf.constant(
          self._params.resize_scales,
          dtype=tf.float32)
      index = tf.random.categorical(tf.zeros([1, 11]), 1)[0]
      scales = tf.gather(scales, index, axis=0)
    else:
      scales = tf.constant([self._params.resize_scales[-1]], tf.float32)

    image_shape = tf.shape(image)[:2]
    boxes = box_ops.denormalize_boxes(boxes, image_shape)
    gt_boxes = boxes
    short_side = scales[0]
    image, image_info = preprocess_ops.resize_image(
        image,
        short_side,
        max(self._params.output_size))
    boxes = preprocess_ops.resize_and_crop_boxes(boxes,
                                                 image_info[2, :],
                                                 image_info[1, :],
                                                 image_info[3, :])
    boxes = box_ops.normalize_boxes(boxes, image_info[1, :])

    # Filters out ground truth boxes that are all zeros.
    indices = box_ops.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)
    is_crowd = tf.gather(is_crowd, indices)
    boxes = box_ops.yxyx_to_cycxhw(boxes)

    image = tf.image.pad_to_bounding_box(
        image, 0, 0, self._params.output_size[0], self._params.output_size[1])
    labels = {
        'classes':
            preprocess_ops.clip_or_pad_to_fixed_size(
                classes, self._params.max_num_boxes),
        'boxes':
            preprocess_ops.clip_or_pad_to_fixed_size(
                boxes, self._params.max_num_boxes)
    }
    if not self._params.is_training:
      labels.update({
          'id':
              inputs['image/id'],
          'image_info':
              image_info,
          'is_crowd':
              preprocess_ops.clip_or_pad_to_fixed_size(
                  is_crowd, self._params.max_num_boxes),
          'gt_boxes':
              preprocess_ops.clip_or_pad_to_fixed_size(
                  gt_boxes, self._params.max_num_boxes),
      })

    return image, labels

  def _transform_and_batch_fn(
      self,
      dataset,
      input_context: Optional[tf.distribute.InputContext] = None):
    """Preprocess and batch."""
    dataset = dataset.map(
        self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    per_replica_batch_size = input_context.get_per_replica_batch_size(
        self._params.global_batch_size
    ) if input_context else self._params.global_batch_size
    dataset = dataset.batch(
        per_replica_batch_size, drop_remainder=self._params.drop_remainder)
    return dataset

  def load(self, input_context: Optional[tf.distribute.InputContext] = None):
    """Returns a tf.dataset.Dataset."""
    reader = input_reader.InputReader(
        params=self._params,
        decoder_fn=None,
        transform_and_batch_fn=self._transform_and_batch_fn)
    return reader.read(input_context)
